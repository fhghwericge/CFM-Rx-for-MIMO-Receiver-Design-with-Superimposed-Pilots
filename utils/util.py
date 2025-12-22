from unicodedata import name
import torch
import numpy as np
# from numpy import linalg as LA
# from torchmetrics import SNR
import logging
import math

"""
utils.py Utils functions

This class handle the sample generator module, such as the symbol detection
"""


def get_logger(stream_handler = True):
    logger = logging.getLogger(name='JED_MAP_Langevin§')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter("%(asctime)s [%(name)s] >> %(message)s")
    if stream_handler:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    else:
        fh = logging.FileHandler('logging_file.log')
        logger.addHandler(fh)

    return logger


# 定义QPSK星座点
sqrt2_inv = 1.0 / torch.sqrt(torch.tensor(2.0))
QPSK_REAL = torch.tensor([sqrt2_inv, -sqrt2_inv, -sqrt2_inv, sqrt2_inv], dtype=torch.float32)
QPSK_IMAG = torch.tensor([sqrt2_inv, sqrt2_inv, -sqrt2_inv, -sqrt2_inv], dtype=torch.float32)

# 定义16QAM星座点 (Gray编码)
sqrt10 = torch.sqrt(torch.tensor(10.0))
QAM16_REAL = torch.tensor([-3, -1, 3, 1, -3, -1, 3, 1, -3, -1, 3, 1, -3, -1, 3, 1], dtype=torch.float32) / sqrt10
QAM16_IMAG = torch.tensor([-3, -3, -3, -3, -1, -1, -1, -1, 1, 1, 1, 1, 3, 3, 3, 3], dtype=torch.float32) / sqrt10

def qpsk_modulator(bits, real_const=QPSK_REAL, imag_const=QPSK_IMAG):
    """
    QPSK调制器
    参数：
      bits: 输入比特流，最后一维必须为2（二进制位），类型建议为 torch.long
    返回：
      complex_symbols: 复数调制符号，类型为 torch.complex64
    """
    # bits[..., 0] 和 bits[..., 1] 需要为整数
    indices = bits[..., 0] * 2 + bits[..., 1]
    # 根据索引获取对应的实部和虚部
    symbols_real = real_const[indices]
    symbols_imag = imag_const[indices]
    # 组合为复数符号（转换为复数张量）
    complex_symbols = torch.complex(symbols_real, symbols_imag)
    return complex_symbols

def qam16_modulator(bits, real_const=QAM16_REAL, imag_const=QAM16_IMAG):
    """
    16QAM调制器
    参数：
      bits: 输入比特流，最后一维必须为4（每个16QAM符号4个比特），类型建议为 torch.long
      real_const: 16QAM实部星座点
      imag_const: 16QAM虚部星座点
    返回：
      complex_symbols: 复数调制符号，类型为 torch.complex64
    """
    assert bits.shape[-1] == 4, "最后一维必须为4（每个16QAM符号4比特）"
    # 使用 Gray 编码映射，将4位二进制转换为索引
    indices = (bits[..., 0] << 3) + (bits[..., 1] << 2) + (bits[..., 2] << 1) + bits[..., 3]
    # 根据索引获取对应的实部和虚部
    symbols_real = real_const[indices]
    symbols_imag = imag_const[indices]
    # 组合为复数符号
    complex_symbols = torch.complex(symbols_real, symbols_imag)
    return complex_symbols

def sym_detection(x_hat, j_indices, real_const=QPSK_REAL, imag_const=QPSK_IMAG):
    """
    QPSK符号检测器
    参数：
      x_hat: 接收到的符号张量，形状 [..., 2]，最后一维为实部和虚部
      j_indices: 正确的符号索引
      real_QAM_const: QPSK实部星座点
      imag_QAM_const: QPSK虚部星座点
    返回：
      accuracy: 检测准确率
    """
    # 分离出实部和虚部
    x_real = x_hat.real
    x_imag = x_hat.imag

    # 扩展维度便于广播计算
    x_real = x_real.unsqueeze(-1)
    x_imag = x_imag.unsqueeze(-1)

    # 计算欧氏距离的平方
    x_real_dist = (x_real - real_const)**2
    x_imag_dist = (x_imag - imag_const)**2
    x_dist = x_real_dist + x_imag_dist

    # 找到距离最小的索引
    x_indices = torch.argmin(x_dist, dim=-1)

    # 计算准确率
    accuracy = (x_indices == j_indices).sum().float() / x_indices.numel()
    return x_indices, accuracy

def sym_detection_BER(x_hat, j_indices, real_const=QPSK_REAL, imag_const=QPSK_IMAG):
    """
    QPSK符号检测器（计算比特误码率版本）
    参数：
      x_hat: 接收到的符号张量，形状 [..., 2]，最后一维为实部和虚部
      j_indices: 正确的符号索引
      real_const: QPSK实部星座点
      imag_const: QPSK虚部星座点
    返回：
      x_indices: 检测出的符号索引
      ber: 比特误码率
    """
    # 分离出实部和虚部
    x_real = x_hat.real
    x_imag = x_hat.imag

    # 扩展维度便于广播计算
    x_real = x_real.unsqueeze(-1)
    x_imag = x_imag.unsqueeze(-1)

    # 计算欧氏距离的平方
    x_real_dist = (x_real - real_const)**2
    x_imag_dist = (x_imag - imag_const)**2
    x_dist = x_real_dist + x_imag_dist

    # 找到距离最小的索引
    x_indices = torch.argmin(x_dist, dim=-1)
    
    # 将索引转换为比特（对于QPSK，每个符号2位）
    pred_bits = torch.zeros((x_indices.numel(), 2), dtype=torch.long, device=x_indices.device)
    pred_bits[:, 0] = x_indices.flatten() // 2
    pred_bits[:, 1] = x_indices.flatten() % 2
    
    true_bits = torch.zeros((j_indices.numel(), 2), dtype=torch.long, device=j_indices.device)
    true_bits[:, 0] = j_indices.flatten() // 2
    true_bits[:, 1] = j_indices.flatten() % 2

    # 计算比特误码率
    bit_accuracy = (pred_bits == true_bits).sum().float()
    total_bits = pred_bits.numel()
    bit_accuracy_rate = bit_accuracy / total_bits

    return x_indices, bit_accuracy_rate

def sym_detection2(x_hat, real_const=QPSK_REAL, imag_const=QPSK_IMAG):
    """
    QPSK符号检测器
    参数：
      x_hat: 接收到的符号张量，形状 [..., 2]，最后一维为实部和虚部
      j_indices: 正确的符号索引
      real_QAM_const: QPSK实部星座点
      imag_QAM_const: QPSK虚部星座点
    返回：
      accuracy: 检测准确率
    """
    # 分离出实部和虚部
    x_real = x_hat.real
    x_imag = x_hat.imag

    # 扩展维度便于广播计算
    x_real = x_real.unsqueeze(-1)
    x_imag = x_imag.unsqueeze(-1)

    # 计算欧氏距离的平方
    x_real_dist = (x_real - real_const)**2
    x_imag_dist = (x_imag - imag_const)**2
    x_dist = x_real_dist + x_imag_dist

    # 找到距离最小的索引
    x_indices = torch.argmin(x_dist, dim=-1)

    return x_indices

def sym_detection_qam16(x_hat, j_indices, real_const=QAM16_REAL, imag_const=QAM16_IMAG):
    """
    16QAM符号检测器
    参数：
      x_hat: 接收到的符号张量，形状 [..., 2]，最后一维为实部和虚部
      j_indices: 正确的符号索引
      real_const: 16QAM实部星座点
      imag_const: 16QAM虚部星座点
    返回：
      accuracy: 检测准确率
    """
    # 分离出实部和虚部
    x_real = x_hat[..., 0]
    x_imag = x_hat[..., 1]

    # 扩展维度便于广播计算
    x_real = x_real.unsqueeze(-1)
    x_imag = x_imag.unsqueeze(-1)

    # 计算欧氏距离的平方
    x_real_dist = (x_real - real_const)**2
    x_imag_dist = (x_imag - imag_const)**2
    x_dist = x_real_dist + x_imag_dist

    # 找到距离最小的索引
    x_indices = torch.argmin(x_dist, dim=-1)

    # 计算准确率
    accuracy = (x_indices == j_indices).sum().float() / x_indices.numel()
    return accuracy



def batch_matvec_mul(A,b):
    '''Multiplies a matrix A of size batch_sizexNxK
       with a vector b of size batch_sizexK
       to produce the output of size batch_sizexN
    '''    
    C = torch.matmul(A, torch.unsqueeze(b, dim=2))
    return torch.squeeze(C, -1) 

def batch_identity_matrix(row, cols, batch_size):
    eye = torch.eye(row, cols)
    eye = eye.reshape((1, row, cols))
    
    return eye.repeat(batch_size, 1, 1)

def dict2namespace(config):
    namespace = type('new', (object,), config)
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    
    namespace = addAttr(namespace)

    return namespace

def addAttr(config):
    M = int(np.sqrt(config.mod_n))
    sigConst = np.linspace(-M+1, M-1, M) 
    sigConst /= np.sqrt((sigConst ** 2).mean())
    sigConst /= np.sqrt(2.) 
    setattr(config, 'M', M)
    setattr(config, 'sigConst', sigConst)
    
    SNR_dBs = np.arange( config.SNR_db_min, config.SNR_db_max, config.SNR_step)
    setattr(config, 'snr_range', SNR_dBs)
    return config


def gaussian(zt, generator, noise_sigma, NT, M, device):
    argr = torch.reshape(zt[:,0:NT],[-1,1]) - generator.QAM_const()[0].to(device=device)
    argi = torch.reshape(zt[:,NT:],[-1,1]) - generator.QAM_const()[1].to(device=device)

    argr = torch.reshape(argr, [-1, NT, M **2]) 
    argi = torch.reshape(argi, [-1, NT, M **2]) 

    zt = torch.pow(argr,2) + torch.pow(argi,2)
    exp = -1.0 * (zt/(2.0 * noise_sigma))
    exp = exp.softmax(dim=-1)

    xr = torch.mul(torch.reshape(exp,[-1,M **2]).float(), generator.QAM_const()[0].to(device=device))
    xi = torch.mul(torch.reshape(exp,[-1,M **2 ]).float(), generator.QAM_const()[1].to(device=device))

    xr = torch.reshape(xr, [-1, NT, M **2]).sum(dim=-1)
    xi = torch.reshape(xi, [-1, NT, M **2]).sum(dim=-1)
    x_out = torch.cat((xr, xi), dim=-1)

    return x_out

def gaussian2_VE(zt,noise_sigma, device,M = 2, star_real = QPSK_REAL, star_imag = QPSK_IMAG,):
    shape = zt.shape
    argr = torch.reshape(zt.real,[-1,1]) - star_real.to(device=device)
    argi = torch.reshape(zt.imag,[-1,1]) - star_imag.to(device=device)

    argr = torch.reshape(argr, shape+(M**2,)) 
    argi = torch.reshape(argi, shape+(M**2,)) 

    zt = torch.pow(argr,2) + torch.pow(argi,2)
    exp = -1.0 * (zt/(2.0 * (noise_sigma**2)))
    exp = exp.softmax(dim=-1)

    xr = torch.mul(torch.reshape(exp,[-1,M **2]).float(), star_real.to(device=device))
    xi = torch.mul(torch.reshape(exp,[-1,M **2 ]).float(), star_imag.to(device=device))

    xr = torch.reshape(xr, shape+(M**2,)).sum(dim=-1)
    xi = torch.reshape(xi, shape+(M**2,)).sum(dim=-1)
    x_out = xr+ 1j*xi

    return x_out

def gaussian2(zt,noise_sigma, device,M = 2, star_real = QPSK_REAL, star_imag = QPSK_IMAG,):
    shape = zt.shape
    argr = torch.reshape(zt.real,[-1,1]) - (1-noise_sigma) * star_real.to(device=device)
    argi = torch.reshape(zt.imag,[-1,1]) - (1-noise_sigma) * star_imag.to(device=device)

    argr = torch.reshape(argr, shape+(M**2,)) 
    argi = torch.reshape(argi, shape+(M**2,)) 

    zt = torch.pow(argr,2) + torch.pow(argi,2)
    exp = -1.0 * (zt/(2.0 * (noise_sigma**2)))
    exp = exp.softmax(dim=-1)

    xr = torch.mul(torch.reshape(exp,[-1,M **2]).float(), star_real.to(device=device))
    xi = torch.mul(torch.reshape(exp,[-1,M **2 ]).float(), star_imag.to(device=device))

    xr = torch.reshape(xr, shape+(M**2,)).sum(dim=-1)
    xi = torch.reshape(xi, shape+(M**2,)).sum(dim=-1)
    x_out = xr+ 1j*xi

    return x_out


def generate_fixed_pilot(batch_size, pilot_shape, device):
    """
    生成确定性的导频序列，尺寸为 pilot_shape = (48,12)。
    这里将总长度的元素（共 pilot_length 个）均匀散布在 0~2pi 上，
    并通过固定随机排列得到较好的相关特性。
    最后生成的导频为复数表示，并以 [batch_size, 1, 48, 12, 2] (实/虚部) 返回。
    """
    # pilot_shape: (48,12)
    pilot_length = pilot_shape[0] * pilot_shape[1]  # 576
    device = device

    # 固定种子，保证可重复
    torch.manual_seed(12345)
    # 生成一个从 0 到 2*pi 均匀分布的相位序列（连续值）
    step = (2 * math.pi) / pilot_length
    phases = torch.arange(0, 2 * math.pi, step, device=device)

    # 打乱相位顺序（固定随机排列）
    perm = torch.randperm(pilot_length, device=device)
    phases = phases[perm]
    # 生成复数导频：单位圆上的点
    pilot_seq = torch.exp(1j * phases)  # 形状: [pilot_length]
    
    # 将一维序列 reshape 回 pilot_shape
    pilot_seq = pilot_seq.reshape(pilot_shape)  # [48, 12]

    # 将导频序列扩展到 batch_size，并添加一个 channel 维度
    pilot_seq = pilot_seq.unsqueeze(0).unsqueeze(1)  # shape: [1, 1, 48, 12]
    pilot_seq = pilot_seq.repeat(batch_size, 1, 1, 1)  # [batch_size, 1, 48, 12]

    return pilot_seq

def sym_detection_full(x_hat, j_indices, real_const=QPSK_REAL, imag_const=QPSK_IMAG):
    """
    QPSK符号检测器 (综合版本)
    参数：
      x_hat: 接收到的符号张量，形状 [batch_size, ...]，最后一维为实部和虚部
      j_indices: 正确的符号索引
      real_const: QPSK实部星座点
      imag_const: QPSK虚部星座点
    返回：
      x_indices: 检测出的符号索引
      symbol_accuracy: 符号准确率
      bit_accuracy_rate: 比特准确率
      block_accuracy_rate: 块准确率 (样本级别的准确率)
    """
    # 分离出实部和虚部
    x_real = x_hat.real
    x_imag = x_hat.imag

    # 扩展维度便于广播计算
    x_real = x_real.unsqueeze(-1)
    x_imag = x_imag.unsqueeze(-1)

    # 计算欧氏距离的平方
    x_real_dist = (x_real - real_const)**2
    x_imag_dist = (x_imag - imag_const)**2
    x_dist = x_real_dist + x_imag_dist

    # 找到距离最小的索引
    x_indices = torch.argmin(x_dist, dim=-1)

    # 计算符号准确率
    symbol_accuracy = (x_indices == j_indices).sum().float() / x_indices.numel()
    
    # 计算比特准确率
    pred_bits = torch.zeros((x_indices.numel(), 2), dtype=torch.long, device=x_indices.device)
    pred_bits[:, 0] = x_indices.flatten() // 2
    pred_bits[:, 1] = x_indices.flatten() % 2
    
    true_bits = torch.zeros((j_indices.numel(), 2), dtype=torch.long, device=j_indices.device)
    true_bits[:, 0] = j_indices.flatten() // 2
    true_bits[:, 1] = j_indices.flatten() % 2

    bit_accuracy = (pred_bits == true_bits).sum().float()
    total_bits = pred_bits.numel()
    bit_accuracy_rate = bit_accuracy / total_bits

    # 计算块准确率 (每个样本所有比特都正确才算正确)
    batch_size = x_hat.shape[0]
    symbols_per_sample = x_indices.numel() // batch_size
    bits_per_sample = symbols_per_sample * 2
    
    # 重组比特以计算每个样本的准确率
    pred_bits = pred_bits.reshape(batch_size, -1)  # [batch_size, bits_per_sample]
    true_bits = true_bits.reshape(batch_size, -1)  # [batch_size, bits_per_sample]
    
    # 计算每个样本是否完全正确
    sample_correct = (pred_bits == true_bits).all(dim=1)  # [batch_size]
    block_accuracy_rate = sample_correct.sum().float() / batch_size

    return x_indices, symbol_accuracy, bit_accuracy_rate, block_accuracy_rate

