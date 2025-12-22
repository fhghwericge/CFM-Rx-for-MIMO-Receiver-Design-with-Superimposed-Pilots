import torch
from torch import nn
import math
from typing import Union, Optional
from DMCE import utils

# def get_positional_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
#     """
#     Creates the DM time embedding from an integer time step

#     Parameters
#     ----------
#     t : Tensor of shape [batch_size]
#         timesteps of the corresponding data samples
#     dim : int
#         dimension of the resulting embedding
#     Returns
#     -------
#     t_emb : Tensor of shape [batch_size, dim]
#         time embeddings for each data sample
#     """

#     half_dim = dim // 2
#     emb = math.log(10000) / (half_dim - 1)
#     emb = torch.exp(- emb * torch.arange(half_dim, device=t.device))
#     emb = t[:, None] * emb[None, :]
#     emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

#     # if dim is an odd number, pad the last entry of the embedding vector with zeros
#     if dim % 2 != 0:
#         emb = torch.nn.functional.pad(emb, (0, 1), 'constant', 0)
#     return emb

def get_positional_embedding(
    t: torch.Tensor,
    dim: int,
    max_freq_log2: float = 9.0,
    num_bands: Optional[int] = None,
) -> torch.Tensor:
    """
    对连续时间步 t∈[0,1] 做傅里叶特征编码（sin/cos）。

    Parameters
    ----------
    t : Tensor of shape [batch_size]
        连续时间步，范围在 [0,1]
    dim : int
        输出 embedding 的维度（通常取 2*num_bands，若为奇数则在末尾补 0）
    max_freq_log2 : float, optional
        最高频率采用 2^max_freq_log2, 默认为 2^9=512
    num_bands : int, optional
        频带数量；如果为 None，则自动设为 dim//2

    Returns
    -------
    Tensor of shape [batch_size, dim]
        对每个 t 的高维傅里叶编码
    """
    # 选择频带个数
    if num_bands is None:
        num_bands = dim // 2

    # 生成频率：[1, 2, 4, …, 2^max_freq_log2]
    # 线性扫描 log2 频率
    freq_bands = 2.0 ** torch.linspace(0.0, max_freq_log2, num_bands, device=t.device)

    # 将 t 从 [batch] -> [batch, num_bands]
    args = t[:, None] * freq_bands[None, :] * math.pi

    # 拼接 sin/cos
    emb = torch.cat([args.sin(), args.cos()], dim=-1)

    # 若 dim 为奇数，则补 0
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1), 'constant', 0)

    return emb

class CNN(nn.Module):
    def __init__(self,
                 data_shape: tuple,
                 n_layers_pre: int = 1,
                 n_layers_post: int = 1,
                 ch_layers_pre: tuple = (2, 2),
                 ch_layers_post: tuple = (2, 2),
                 n_layers_time: int = 1,
                 ch_init_time: int = 16,
                 kernel_size: tuple = (3, ),
                 mode: str = '1D',
                 batch_norm: bool = False,
                 downsamp_fac: int = 1,
                 stride: int = 1,
                 padding_mode: str = 'zeros',
                 device: Union[str, torch.device] = 'cuda'):
        super().__init__()
        self.data_shape = data_shape
        self.n_layers_pre = n_layers_pre
        self.n_layers_post = n_layers_post
        self.ch_layers_pre = ch_layers_pre
        self.ch_layers_post = ch_layers_post
        self.n_layers_time = n_layers_time
        self.ch_init_time = ch_init_time
        self.kernel_size = kernel_size
        self.mode = mode
        self.batch_norm = batch_norm
        self.downsamp_fac = downsamp_fac
        self.stride = stride
        self.padding_mode = padding_mode
        self.device = utils.set_device(device)

        #self.dim_time = np.prod(data_shape[1:])
        self.dim_time = ch_layers_pre[-1]
        ch_time = None
        if n_layers_time == 0:
            ch_time = (2*self.dim_time, )
        elif n_layers_time == 1:
            ch_time = (ch_init_time, 2*self.dim_time)
        elif n_layers_time == 2:
            ch_time = (ch_init_time, self.dim_time, 2*self.dim_time)
        elif n_layers_time == 3:
            ch_time = (ch_init_time, self.dim_time, self.dim_time, 2 * self.dim_time)
        else:
            raise NotImplementedError

        # Time embedding related functionalities, computing the base time embedding
        self.time_embedding_func = lambda t: get_positional_embedding(t, ch_time[0])
        self.time_mlp = nn.Sequential().to(device=device)
        for i in range(n_layers_time):
            self.time_mlp.add_module(f'time_linear{i+1}', nn.Linear(ch_time[i], ch_time[i+1], device=self.device))
            if i < n_layers_time - 1:
                self.time_mlp.add_module(f'act_time{i+1}', nn.ReLU())


        self.cnn_pre = nn.Sequential().to(device=device)
        for i in range(n_layers_pre):
            if mode == '1D':
                self.cnn_pre.add_module(f'conv_pre{i}', nn.Conv1d(ch_layers_pre[i], ch_layers_pre[i+1], stride=stride,
                                   kernel_size=kernel_size[i], padding='same',device=device))
            else:
                self.cnn_pre.add_module(f'conv_pre{i}', nn.Conv2d(ch_layers_pre[i], ch_layers_pre[i+1], stride=stride,
                                   kernel_size=kernel_size[i], padding='same', device=device))
            if i < n_layers_pre - 1:
                if batch_norm and mode == '2D':
                    self.cnn_pre.add_module(f'batchnorm_pre{i+1}', nn.BatchNorm2d(num_features=ch_layers_pre[i+1], device=device))
                self.cnn_pre.add_module(f'act_pre{i+1}', nn.ReLU())

        self.cnn_post = nn.Sequential().to(device=device)
        for i in range(n_layers_post):
            if mode == '1D':
                self.cnn_post.add_module(f'conv_post{i}', nn.Conv1d(ch_layers_post[i], ch_layers_post[i + 1],
                                                         stride=stride,kernel_size=kernel_size[i], padding='same',
                                                         device=device))
            else:
                self.cnn_post.add_module(f'conv_post{i}', nn.Conv2d(ch_layers_post[i], ch_layers_post[i + 1],
                                                        stride=stride, kernel_size=kernel_size[i], padding='same',
                                                        device=device))
            if i < n_layers_post - 1:
                if batch_norm and mode == '2D':
                    self.cnn_post.add_module(f'batchnorm_post{i+1}', nn.BatchNorm2d(num_features=ch_layers_post[i+1], device=device))
                self.cnn_post.add_module(f'act_post{i+1}', nn.ReLU())


    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # compute the time embedding for all timesteps
        t_emb = self.time_mlp(self.time_embedding_func(t))
        scale = t_emb[:, :self.dim_time]
        shift = t_emb[:, self.dim_time:]

        # print(x.shape)
        x = self.cnn_pre(x)
        # print(x.shape)
        if self.mode == '1D':
            x = x + scale[:, :, None] * x + shift[:, :, None]
        else:
            x = x + scale[:, :, None, None] * x + shift[:, :, None, None]

        x = self.cnn_post(x)

        # print(x.shape)
        return x

