"""
Train and test script for the DMCE.
"""
from CFM_RX import utils, DiffusionModel, Trainer, Tester, CNN
import os
import os.path as path
import argparse
import modules.utils as ut
import datetime
import csv
import matplotlib.pyplot as plt
import numpy as np
import torch
from CFM_RX.utils import cmplx2real
import dill
from thop import profile

CUDA_DEFAULT_ID = 0
def load_and_preprocess(path, max_samples,device):
    arr = np.load(path)
    t = torch.tensor(arr, dtype=torch.complex64, device=device).squeeze()
    real_imag = torch.cat([t.real, t.imag], dim=1).float()
    return real_imag[:max_samples]

def main():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # date_time = date_time_now.strftime('%Y-%m-%d_%H-%M-%S')  # convert to str compatible with all OSs
    os.environ['PYTHONHASHSEED']=str(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    print(torch.randint(10,[1]))
    # torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_printoptions(precision=8)
    # n_dim = 64 # RX antennas
    # n_dim2 = 16 # TX antennas
    
    load_model = False
    model_path = './model.dill'

    ##########DATA#LOAD#######################

    #####3km/h
    data_train = load_and_preprocess('./data/CDLC_train.npy',8000,device)
    data_val = load_and_preprocess('./data/CDLC_val.npy',1000,device)

    print('data_train.shape:', data_train.shape)
    print('data_val.shape:', data_val.shape)
    print(data_train.var())
    print(data_val.var())

    # set Diffusion model params
    num_timesteps = 100 #int(np.random.choice([100, 300, 500, 1_000, 2_000]))
    loss_type = 'l2'
    which_schedule = 'linear'

    # max_snr_dB = 40
    # beta_start = 1 - 10**(max_snr_dB/10) / (1 + 10**(max_snr_dB/10))
    beta_start=5e-3
    beta_end = 0.1

    objective = 'pred_noise'  # one of 'pred_noise' (L_n), 'pred_x_0' (L_h), 'pred_post_mean' (L_mu)
    loss_weighting = False # bool(np.random.choice([True, False]))
    clipping = False
    reverse_method = 'reverse_mean'  # either 'reverse_mean' or 'ground_truth'
    reverse_add_random = False  # True: PDF Sampling method | False: Reverse Mean Forwarding method
    complex_data = False
    data_shape = tuple(data_train.shape[1:])
    mode = '2D'
    cwd = os.getcwd()
    return_all_timesteps = False # evaluates all intermediate MSEs
    fft_pre = False 
    data_start_t = -1 # 如果是未加噪数据，这个值填-1！
    method = 'No_schedule' # one of 'Schedule_A', 'Schedule_B', 'Schedule_C', 'No_schedule'


    # diffusion model parameter dictionary, which is saved in 'sim_params.json'
    diff_model_dict = {
        'data_shape': data_shape,
        'complex_data': complex_data,
        'loss_type': loss_type,
        'which_schedule': which_schedule,
        'num_timesteps': num_timesteps,
        'beta_start': beta_start,   
        'beta_end': beta_end,
        'objective': objective,
        'loss_weighting': loss_weighting,
        'clipping': clipping,
        'reverse_method': reverse_method,
        'reverse_add_random': reverse_add_random,
        'data_start_t': data_start_t,
        'method': method
    }

    # kernel_size = [(13,17), (9, 13), (5, 9), (3, 3)]
    kernel_size = [(13,13), (9, 9), (7, 7), (5, 5), (3, 3), (1, 1)]
    n_layers_pre = 5
    max_filter = 64
    ch_layers_pre = np.linspace(start=1, stop=max_filter, num=n_layers_pre+1, dtype=int)
    ch_layers_pre[0] = 4
    ch_layers_pre = tuple(ch_layers_pre)
    ch_layers_pre = tuple(int(x) for x in ch_layers_pre)
    n_layers_post = 6
    ch_layers_post = np.linspace(start=1, stop=max_filter, num=n_layers_post+1, dtype=int)
    ch_layers_post[0] = 4
    ch_layers_post = ch_layers_post[::-1]
    ch_layers_post = tuple(ch_layers_post)
    ch_layers_post = tuple(int(x) for x in ch_layers_post)
    n_layers_time = 1
    ch_init_time = 16
    batch_norm = False
    downsamp_fac = 1

    cnn_dict = {
        'data_shape': data_shape,
        'n_layers_pre': n_layers_pre,
        'n_layers_post': n_layers_post,
        'ch_layers_pre': ch_layers_pre,
        'ch_layers_post': ch_layers_post,
        'n_layers_time': n_layers_time,
        'ch_init_time': ch_init_time,
        'kernel_size': kernel_size,
        'mode': mode,
        'batch_norm': batch_norm,
        'downsamp_fac': downsamp_fac,
        'device': device,
    }

    # set Trainer params
    batch_size = 1024
    lr_init = 1e-3
    lr_step_multiplier = 0.98
    epochs_until_lr_step = 50
    num_epochs = 5000   
    val_every_n_batches = 2000
    num_min_epochs = 50
    num_epochs_no_improve = 20
    track_val_loss = True
    track_fid_score = False
    track_mmd = False
    use_fixed_gen_noise = True
    use_ray = False
    save_mode = 'best' # newest, all
    dir_result = path.join(cwd, 'results')
    timestamp = utils.get_timestamp()
    dir_result = path.join(dir_result, timestamp)
    use_ssim = False
    # Trainer parameter dictionary, which is saved in 'sim_params.json'
    trainer_dict = {
        'batch_size': batch_size,
        'lr_init': lr_init,
        'lr_step_multiplier': lr_step_multiplier,
        'epochs_until_lr_step': epochs_until_lr_step,
        'num_epochs': num_epochs,
        'val_every_n_batches': val_every_n_batches,
        'track_val_loss': track_val_loss,
        'track_fid_score': track_fid_score,
        'track_mmd': track_mmd,
        'use_fixed_gen_noise': use_fixed_gen_noise,
        'save_mode': save_mode,
        'mode': mode,
        'dir_result': str(dir_result),
        'use_ray': use_ray,
        'complex_data': complex_data,
        'num_min_epochs': num_min_epochs,
        'num_epochs_no_improve': num_epochs_no_improve,
        'fft_pre': fft_pre,
        'use_ssim': use_ssim
    }

    # create result directory
    os.makedirs(dir_result, exist_ok=True)

    # instantiate CNN, DiffusionModel, Trainer and Tester
    cnn = CNN(**cnn_dict)
    # 
    if load_model:
        with open(model_path, 'rb') as f:
            diffusion_model = dill.load(f)
    else:
        diffusion_model = DiffusionModel(cnn, **diff_model_dict)
    trainer = Trainer(diffusion_model, data_train, data_val, **trainer_dict)
    # Print number of trainable parameters
    input = torch.randn((1, 4, 12, 48,),device = device)
    flops, params = profile(diffusion_model, inputs=(input,))

    
    print(f'Number of trainable model parameters: {params}')
    print(f'Number of Floating-Point Operations: {flops}')

    

    # run training routine
    train_dict = trainer.train()

if __name__ == '__main__':
    main()
