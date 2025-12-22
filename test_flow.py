import numpy as np
import torch
import sys
import os
import argparse
import random
from matplotlib import pyplot as plt
from tqdm import tqdm as tqdm
import imageio
import io  # 添加这行

import dill
from utils.util import *
import torch.nn.functional as F

def NMSE(x, x_hat):
    power = np.sum(abs(x) ** 2, axis=0)
    mse = np.sum(abs(x - x_hat) ** 2, axis=0)
    nmse = np.mean(mse / power)
    return nmse


def main(args):

    # logger
    logger = get_logger()

    if args.save_constellation_gif:
        if not os.path.exists('./figures/'):
            os.makedirs('./figures/')

    # Cuda config
    torch.cuda.empty_cache()
    device = 'cuda:0'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32       = False
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)   

    logger.info(f"Device set to {device}.")

    # Load configs
    test_seed = 4321
    # 设置随机种子
    random.seed(test_seed)
    np.random.seed(test_seed)
    torch.manual_seed(test_seed)
    torch.cuda.manual_seed_all(test_seed)

    result_dir = 'results_joint_seed%d' % test_seed
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    logger.info(f"Results will be saved to {result_dir}.")
    


    model_path = './model.dill'
    # Get and load the model
    with open(model_path, 'rb') as f:
        diffuser = dill.load(f)

    diffuser.eval()

    # Set some paramaters
    # snr_range = np.arange(-10, 17.5, 2.5)
    num_steps = 30
    num_repeat_H = args.num_repeat_H
    num_repeat_X = args.num_repeat_X
    # snr_range = np.array([0,2,4,6,8,10])
    snr_range = np.array([10])
    noise_range = 10 ** (-snr_range / 10.) 
    alphas = torch.linspace(0.01, 0.99, steps=num_steps,device=device)
    # alphas = torch.linspace(0.001, 0.999, steps=num_steps,device=device)
    sigmas_H = 1-alphas
    sigmas_x = 1-alphas
    step_H = 1 / num_steps
    step_x = 1 / num_steps
    
    c_H = args.c_H
    c_grad_start = args.c_grad_start
    c_grad_end = c_grad_start
    # c_grad_end = args.c_grad_end
    c_prior_start = args.c_prior_start
    c_prior_end = args.c_prior_end

    c_grad = torch.linspace(c_grad_start, c_grad_end, steps=num_steps,device=device)
    c_prior = torch.linspace(c_prior_start, c_prior_end, steps=num_steps,device=device)
    # 定义QPSK星座点
    sqrt2_inv = 1.0 / torch.sqrt(torch.tensor(2.0))
    QPSK_REAL = torch.tensor([sqrt2_inv, -sqrt2_inv, -sqrt2_inv, sqrt2_inv], dtype=torch.float32,device=device)
    QPSK_IMAG = torch.tensor([sqrt2_inv, sqrt2_inv, -sqrt2_inv, -sqrt2_inv], dtype=torch.float32,device=device)

    H_test_complex   = torch.squeeze(torch.tensor(np.load('./data/CDLC_test.npy'),device=device))
    print(f"num_repeat: {num_repeat_H}, c_H:{c_H}, c_grad_start: {c_grad_start}, c_grad_end: {c_grad_end}, c_prior_start: {c_prior_start}, c_prior_end: {c_prior_end}")

    H_test_complex = H_test_complex / (H_test_complex.var()) ** 0.5
    H_raw_var = H_test_complex.var(dim=(1,2,3),keepdims = True)

    # H_test_complex = H_test_complex[-100:]
    batch_size = len(H_test_complex)  # 使用所有测试数据作为一个batch

    # Set some hyperparameters
    SER_langevin = []
    
    pilot_indices = torch.randint(0,2,(batch_size,1,48,12,2),device=device)
    pilot_complex = qpsk_modulator(pilot_indices, real_const=QPSK_REAL, imag_const=QPSK_IMAG)


    # 添加新的监控数组
    h_d_nmse_history = []
    x_d_nmse_history = []
    score_var_history = []
    meas_var_history = []
    prior_var_history = []
    grad_var_history = []
    H_var_history = []
    X_var_history = []

    # Start the loop for all SNRs
    for snr_idx, local_noise in enumerate(noise_range):
        # 添加列表存储每步的图像
        constellation_images = [] if args.save_constellation_gif else None
        
        # Setting parameters for each SNR
        local_noise_x = torch.tensor(local_noise,dtype=torch.float32, device=device)# Sigma_0 = Local noise

        # 直接使用H_test_complex替代batch循环
        H_batch = H_test_complex
        tx_indices = torch.randint(0,2,(batch_size,1,48,12,2),device=device)
        # tx_indices = torch.ones((batch_size,1,48,12,2),device=device,dtype=torch.int)
        tx_j_indices = tx_indices[:,0,:,:, 0] * 2 + tx_indices[:,0,:,:, 1]
        tx_symbols_complex = qpsk_modulator(tx_indices, real_const=QPSK_REAL, imag_const=QPSK_IMAG)

        ones_matrix = torch.ones_like(tx_symbols_complex,device=device)
        V = torch.sqrt(torch.tensor(0.6, dtype=torch.float32, device=device)) * ones_matrix
        W = torch.sqrt(torch.tensor(0.4, dtype=torch.float32, device=device)) * ones_matrix
        
        # rand_matrix = torch.rand((batch_size,1,48,12), dtype=torch.float32, device=device)
        # V = torch.sqrt(rand_matrix)
        # W = torch.sqrt(1-rand_matrix)

        S = pilot_complex* W + tx_symbols_complex * V

        Y = H_batch * S
        Y = Y + torch.sqrt(torch.tensor(local_noise, dtype=Y.dtype, device=device)) * torch.randn_like(Y)
        x_current = torch.zeros_like(Y).to(device=device)
        # x_current = torch.randn_like(Y[:,0]).to(device=device)
        # x_current = x_current.unsqueeze(1).expand(-1,2,-1,-1)
        # x_current = torch.rand_like(Y).to(device=device)*2-1-1j
        H_current = torch.randn_like(H_batch) *(2** 0.5)

        oracle = H_batch.clone().detach()
        
        

        with torch.no_grad():
            for step_idx in tqdm(range(num_steps)):
                # Compute current step size and noise power
                current_sigma = sigmas_H[step_idx].item()
                current_sigma_x = sigmas_x[step_idx].item()
                current_alpha = alphas[step_idx]
                current_lambda = 1/current_alpha

                # Labels for diffusion model
                # labels = torch.ones(H_current.shape[0]).cuda() * (num_steps - step_idx-1)
                labels = torch.ones(H_current.shape[0]).cuda() * current_sigma
                # labels = labels.long()

                # Step1 无条件更新H
                H_real = torch.cat([H_current.real, H_current.imag], dim=1).to(dtype=torch.float32)
                score = diffuser.model(H_real, labels)
                score = torch.complex(score[:, :2, :, :], score[:, 2:, :, :])
                H_current = H_current + step_H * score

                x_indices, _ = sym_detection(torch.sum(x_current,dim=1), tx_j_indices, real_const=QPSK_REAL, imag_const=QPSK_IMAG)
                xh_current = torch.complex(QPSK_REAL[x_indices], QPSK_IMAG[x_indices])
                xh_current = xh_current.unsqueeze(1).expand(-1,2,-1,-1)

                # x_indices= sym_detection2(x_current, real_const=QPSK_REAL, imag_const=QPSK_IMAG)
                # xh_current = torch.complex(QPSK_REAL[x_indices], QPSK_IMAG[x_indices])


                # x_current = x_current * (1+step_x * current_lambda)
                for inner_idx in range(num_repeat_H):
                    
                    # Step2 使用X_Current和Y更新H
                    # 使用实际X更新
                    xdxp = (V * x_current + W * pilot_complex)
                    # xdxp = (V * xh_current + W * pilot_complex)
                    # 使用理想X更新
                    # xdxp = (V * tx_symbols_complex + W * pilot_complex)
                    meas_grad = xdxp.conj() * (Y- H_current* xdxp / current_alpha) / (local_noise_x + (current_sigma * torch.abs(xdxp) /current_alpha)**2) / current_alpha
                    meas_grad = step_H * current_lambda * current_sigma * meas_grad * c_H
                    H_current = H_current + meas_grad
                    
                    # H_gen_var = H_current.var(dim=(1,2,3),keepdims = True)
                    # H_current = H_current * (H_raw_var/H_gen_var)**0.5

                    # snr_linear = 10**(snr_range[snr_idx]/10.0)
                    # P_Y = Y.var(dim=(1,2,3),keepdims=True)
                    # P_H = snr_linear / (1.0 + snr_linear) * P_Y * (1.5)
                    # H_current = H_current * (P_H/H_gen_var)**0.5

                # for inner_idx in range(num_repeat_X):
                    # Step3使用H_current和Y更新x_current

                    Zi_hat = gaussian2(x_current, current_sigma_x, device)
                    prior =  (current_alpha * Zi_hat - x_current) / current_sigma_x ** 2 #[50,2,48,12] Complex        
                    # Score of the likelihood
                    # 使用实际迭代的H
                    # grad_likelihood =V* H_current.conj()*(Y - H_current*(V * x_current / current_alpha+ W * pilot_complex)) / (local_noise_x+ (current_sigma_x * torch.abs(H_current) / current_alpha)**2) / current_alpha 
                    grad_likelihood =V* H_current.conj()*(Y - H_current*(V * xh_current+ W * pilot_complex)) / (local_noise_x+ (current_sigma_x * torch.abs(H_current) / current_alpha)**2) / current_alpha 
                    # 使用理想Ground Truth的H
                    # grad_likelihood =V* H_batch.conj()*(Y - H_batch*(V * x_current / current_alpha+ W * pilot_complex)) / (local_noise_x+ (current_sigma_x * torch.abs(H_batch) / current_alpha)**2) / current_alpha  
                    
                    grad =  grad_likelihood * c_grad[step_idx] + prior* c_prior[step_idx] 
                    grad = step_x * current_lambda * current_sigma * grad
                    # # Update
                    x_current = x_current + grad

                # 记录每个step结束时的指标
                H_D_NMSE = 10*np.log10(NMSE(oracle.cpu().numpy(),H_current.cpu().numpy()))
                X_D_NMSE = 10*np.log10(NMSE(tx_symbols_complex.cpu().numpy(),x_current.cpu().numpy()))
                

                h_d_nmse_history.append(H_D_NMSE)
                x_d_nmse_history.append(X_D_NMSE)
                
                score_var_history.append((step_H* score).var().cpu().item())
                meas_var_history.append((meas_grad).var().cpu().item())

                prior_var_history.append(prior.var().cpu().item())
                grad_var_history.append(grad.var().cpu().item())

                H_var_history.append(H_current.var().cpu().item())
                X_var_history.append(x_current.var().cpu().item())

                # 在每个step结束时绘制星座图
                if args.save_constellation_gif:
                    plt.figure(figsize=(8, 8))
                    x_flat = x_current.detach().cpu().view(-1).numpy()
                    plt.scatter(x_flat.real, x_flat.imag, alpha=0.1, s=1)
                    plt.xlim(-2, 2)
                    plt.ylim(-2, 2)
                    plt.grid(True)
                    plt.title(f'Step {step_idx}')
                    
                    # 保存图像到内存
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    constellation_images.append(imageio.imread(buf))
                    plt.close()
                    buf.close()

            # 循环结束后生成GIF
            if args.save_constellation_gif:
                imageio.mimsave(f'./figures/constellation_evolution_snr_{snr_range[snr_idx]}dB.gif',
                              constellation_images,
                              duration=0.1)  # 每帧持续0.1秒

        # x_hat = V*H_batch.conj()*(Y-W*H_batch*pilot_complex)/((V*torch.abs(H_batch))**2+local_noise_x)
        x_hat = V*H_current.conj()*(Y-W*H_current*pilot_complex)/((V*torch.abs(H_current))**2+local_noise_x)

        x_indices1, s_accuracy1, b_accuracy1, bl_accuracy1= sym_detection_full(torch.sum(x_current,dim=1), tx_j_indices, real_const=QPSK_REAL, imag_const=QPSK_IMAG)
        ser1,ber1,bler1 = 1-s_accuracy1, 1-b_accuracy1, 1-bl_accuracy1
        x_indices2, s_accuracy2, b_accuracy2, bl_accuracy2= sym_detection_full(torch.sum(x_hat,dim=1), tx_j_indices, real_const=QPSK_REAL, imag_const=QPSK_IMAG)
        ser2,ber2,bler2 = 1-s_accuracy2, 1-b_accuracy2, 1-bl_accuracy2
        SER_langevin.append(1 - s_accuracy1)
        nmse = h_d_nmse_history[-1]
        print(f'SNR = {snr_range[snr_idx]}dB, NMSE = {nmse:.4f}dB')
        print(f'SER_langevin1: {ser1.item():8f}, BER_langevin1: {ber1.item():.8f}, BLER_langevin1: {bler1.item():.8f}')
        print(f'SER_langevin2: {ser2.item():8f}, BER_langevin2: {ber2.item():.8f}, BLER_langevin2: {bler2.item():.8f}')



    # 保存生成的数据
    np.save('./data/H_generated.npy',H_current.cpu().numpy())
    np.save('./data/last_score_grad.npy',score.cpu().numpy())
    np.save('./data/last_meas_grad.npy',meas_grad.cpu().numpy())
    np.save('./data/X_generated.npy',x_current.cpu().numpy())
    np.save('./data/Y.npy',Y.cpu().numpy())
    # 绘制NMSE变化图
    plt.figure(figsize=(10,6))
    steps = range(len(h_d_nmse_history))
    plt.plot(steps, h_d_nmse_history, label='H_D NMSE')
    plt.plot(steps, x_d_nmse_history, label='X_D NMSE')
    plt.xlabel('Step')
    plt.ylabel('NMSE')
    plt.title('NMSE vs Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig('./figures/nmse.png')
    plt.close()

    # 绘制梯度方差变化图
    plt.figure(figsize=(10,6))
    plt.subplot(2, 1, 1)
    plt.plot(steps, score_var_history, label='Score Gradient Variance')
    plt.plot(steps, meas_var_history, label='Measurement Gradient Variance')
    plt.xlabel('Step')
    plt.ylabel('Variance')
    plt.title('Gradient Variance vs Steps')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # 使用对数刻度以更好地显示方差变化
    plt.subplot(2, 1, 2)
    plt.plot(steps, prior_var_history, label='Prior Gradient Variance')
    plt.plot(steps, grad_var_history, label='Gradient Variance')
    plt.xlabel('Step')
    plt.ylabel('Variance')
    plt.title('Gradient Variance vs Steps')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # 使用对数刻度以更好地显示方差变化
    plt.tight_layout()

    plt.savefig('./figures/gradient_variance.png')
    plt.close()

    # 绘制H和X的方差变化图
    plt.figure(figsize=(10,6))
    plt.plot(steps, H_var_history, label='H Variance')
    plt.plot(steps, X_var_history, label='X Variance')
    plt.xlabel('Step')
    plt.ylabel('Variance')
    plt.title('H and X Variance vs Steps')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # 使用对数刻度以更好地显示方差变化
    plt.savefig('./figures/h_x_variance.png')


    return nmse, s_accuracy1


if __name__ == "__main__":
    # Args  
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_repeat_H', type=int, default=1)
    parser.add_argument('--num_repeat_X', type=int, default=1)
    parser.add_argument('--c_H', type=float, default= 1)
    parser.add_argument('--c_grad_start', type=float, default= 0.1)
    parser.add_argument('--c_grad_end', type=float, default= 1)
    parser.add_argument('--c_prior_start', type=float, default=5)
    parser.add_argument('--c_prior_end', type=float, default=0)
    parser.add_argument('--save_constellation_gif', action='store_true', 
                       help='Whether to save constellation diagram animation as GIF')

    args = parser.parse_args()
    main(args)