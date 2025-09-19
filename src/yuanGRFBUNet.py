from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import  List
from thop import profile
from math import sqrt
import math
import numpy as np
from torch import Tensor
from einops import rearrange
from torch.nn.modules.utils import _single, _pair, _triple
# 1。我可以对这个GRFB再次改进优化。本身就涨点，那再试试。他是特征提取，我就特征增强，特征融合 Down  ok边缘图像特征
#2，自己再改改、双卷积改改卷积，conv  ARConv自适应矩形卷积和长条带卷积，变成自适应长条带卷积，ARConv也是将标准卷积层替换为ARConv
#3,。还有一个我来发一个递归门控注意力机制 up  ok
# 先插入一个模块，然后看涨不涨点，涨点再改进
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None,use_attention=False):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
class ELA(nn.Module):
    """Constructs an Efficient Local Attention module.
    Args:
        channel: Number of channels of the input feature map
        kernel_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, kernel_size=7):
        super(ELA, self).__init__()

        self.conv = nn.Conv1d(channel, channel, kernel_size=kernel_size, padding=kernel_size // 2,
                              groups=channel, bias=False)
        self.gn = nn.GroupNorm(16, channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).view(B, C, H)
        x_w = torch.mean(x, dim=2, keepdim=True).view(B, C, W)
        x_h = self.sigmoid(self.gn(self.conv(x_h))).view(B, C, H, 1)
        x_w = self.sigmoid(self.gn(self.conv(x_w))).view(B, C, 1, W)

        return x * x_h * x_w
class ScharrConv(nn.Module):
    def __init__(self, channel):
        super(ScharrConv, self).__init__()
        
        # 定义Scharr算子的水平和垂直权重核
        scharr_kernel_x = np.array([[3,  0, -3],
                                    [10, 0, -10],
                                    [3,  0, -3]], dtype=np.float32)
        
        scharr_kernel_y = np.array([[3, 10, 3],
                                    [0,  0, 0],
                                    [-3, -10, -3]], dtype=np.float32)
        
        # 将Scharr核转换为PyTorch张量并扩展为通道数,unsqueeze(0)两次：添加两个维度，形状从(3, 3)变为(1, 1, 3, 3)，符合PyTorch卷积核的维度要求(out_channels, in_channels, H, W)。
        scharr_kernel_x = torch.tensor(scharr_kernel_x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
        scharr_kernel_y = torch.tensor(scharr_kernel_y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
        
        # 扩展为多通道，使用expand将单通道核扩展为channel个通道，形状变为(channel, 1, 3, 3)。
        self.scharr_kernel_x = scharr_kernel_x.expand(channel, 1, 3, 3)  # (channel, 1, 3, 3)
        self.scharr_kernel_y = scharr_kernel_y.expand(channel, 1, 3, 3)  # (channel, 1, 3, 3)

        # 定义卷积层，但不学习卷积核，直接使用Scharr核，所以先定义一个卷积层
        self.scharr_kernel_x_conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.scharr_kernel_y_conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        
        # 将卷积核的权重设置为Scharr算子的核，权重初始化
        self.scharr_kernel_x_conv.weight.data = self.scharr_kernel_x.clone()
        self.scharr_kernel_y_conv.weight.data = self.scharr_kernel_y.clone()

        # 禁用梯度更新
        self.scharr_kernel_x_conv.requires_grad = False
        self.scharr_kernel_y_conv.requires_grad = False

    def forward(self, x):
        # 对输入的特征图进行Scharr卷积（水平和垂直方向）
        grad_x = self.scharr_kernel_x_conv(x)
        grad_y = self.scharr_kernel_y_conv(x)
        # 得到x方向和Y方向的卷积
        # # 计算边缘轻度
        edge_strength = torch.sqrt( grad_x ** 2 + grad_y ** 2)
        # #生成边缘注意力权重
        # edge_attention = self.activation(self.norm(edge_strength)
        # # 应用注意力机制增强特征
        # enhanced_feature =x*edge_attention
        # # 计算梯度幅值
        # edge_magnitude = grad_x * 0.5 + grad_y * 0.5
        
        return edge_strength

class SobelConv(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()
        
        sobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        sobel_kernel_y = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)
        sobel_kernel_x = torch.tensor(sobel.T, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)
        
        self.sobel_kernel_x_conv3d = nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.sobel_kernel_y_conv3d = nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        
        self.sobel_kernel_x_conv3d.weight.data = sobel_kernel_x.clone()
        self.sobel_kernel_y_conv3d.weight.data = sobel_kernel_y.clone()
        
        self.sobel_kernel_x_conv3d.requires_grad = False
        self.sobel_kernel_y_conv3d.requires_grad = False

    def forward(self, x):
        return (self.sobel_kernel_x_conv3d(x[:, :, None, :, :]) + self.sobel_kernel_y_conv3d(x[:, :, None, :, :]))[:, :, 0]
    
# class wConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, den, stride=1, padding=1, groups=1, dilation=1, bias=False):
#         super(wConv2d, self).__init__()       
#         self.stride = _pair(stride)
#         self.padding = _pair(padding)
#         self.kernel_size = _pair(kernel_size)
#         self.groups = groups
#         self.dilation = _pair(dilation)      
#         self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
#         nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')        
#         self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

#         device = torch.device('cpu')  
#         self.register_buffer('alfa', torch.cat([torch.tensor(den, device=device),torch.tensor([1.0], device=device),torch.flip(torch.tensor(den, device=device), dims=[0])]))
#         self.register_buffer('Phi', torch.outer(self.alfa, self.alfa))

#         if self.Phi.shape != self.kernel_size:
#             raise ValueError(f"Phi shape {self.Phi.shape} must match kernel size {self.kernel_size}")

#     def forward(self, x):
#         Phi = self.Phi.to(x.device)
#         weight_Phi = self.weight * Phi
#         return F.conv2d(x, weight_Phi, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups, dilation=self.dilation)
class wConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, den, stride=1, padding=1, groups=1, dilation=1, bias=False):
        super(wConv2d, self).__init__()       
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.kernel_size = _pair(kernel_size)
        self.groups = groups
        self.dilation = _pair(dilation)      
        
        # 初始化卷积权重
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')        
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # 可学习的输出缩放因子
        self.alpha = nn.Parameter(torch.tensor(1.0))  # 初始化为1.0

        # 构建密度权重矩阵
        device = torch.device('cpu')  
        self.register_buffer('alfa', torch.cat([
            torch.tensor(den, device=device),
            torch.tensor([1.0], device=device),
            torch.flip(torch.tensor(den, device=device), dims=[0])
        ]))
        self.register_buffer('Phi', torch.outer(self.alfa, self.alfa))

        if self.Phi.shape != self.kernel_size:
            raise ValueError(f"Phi shape {self.Phi.shape} must match kernel size {self.kernel_size}")

    def forward(self, x):
        Phi = self.Phi.to(x.device)
        weight_Phi = self.weight * Phi
        return F.conv2d(x, weight_Phi, bias=self.bias, 
                       stride=self.stride, padding=self.padding, 
                       groups=self.groups, dilation=self.dilation) * self.alpha



class HEGDC(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, den=0.5):
        super(HEGDC, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        
        # 混合边缘检测模块 (Scharr+Sobel融合)
        self.edge_conv = nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False)
        self._init_hybrid_edge_detector()
        
        # 动态密度权重矩阵
        self.den = nn.Parameter(torch.tensor([den]))
        self.alpha = nn.Parameter(torch.tensor(1.0))
        
        # 优化的双卷积结构
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # 边缘特征融合与转换
        self.edge_fusion = nn.Sequential(
            nn.Conv2d(5, 8, kernel_size=1),  # 4原始边缘+1融合特征
            nn.ReLU(inplace=True),
            nn.Conv2d(8, mid_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 预定义密度权重矩阵
        self.register_buffer('phi_base', torch.tensor([[1.0, 1.0, 1.0],
                                                     [1.0, 1.0, 1.0],
                                                     [1.0, 1.0, 1.0]]).view(1, 1, 3, 3))

    def _init_hybrid_edge_detector(self):
        # Scharr算子
        scharr_x = torch.tensor([[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32) / 16.0
        scharr_y = torch.tensor([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32) / 16.0
        
        # Sobel算子
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32) / 4.0
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32) / 4.0
        
        # 将4个核组合到单个卷积的4个输出通道
        kernel = torch.stack([scharr_x, scharr_y, sobel_x, sobel_y], dim=0).unsqueeze(1)
        self.edge_conv.weight.data = kernel
        self.edge_conv.weight.requires_grad = False

    def _get_density_matrix(self):
        """动态密度权重矩阵"""
        den = torch.sigmoid(self.den)  # 限制在0-1范围内
        return self.phi_base * den

    def forward(self, x):
        # 1. 混合边缘检测 (单次卷积计算4种边缘)
        with torch.no_grad():
            x_mean = x.mean(dim=1, keepdim=True)  # 通道均值
            edges = self.edge_conv(x_mean)  # [B,4,H,W]
            
            # Scharr和Sobel特征融合
            # scharr = torch.abs(edges[:,0:1]) + torch.abs(edges[:,1:2])
            # sobel = torch.abs(edges[:,2:3]) + torch.abs(edges[:,3:4])
            # edge_mag = 0.6*scharr + 0.4*sobel  # 自适应混合权重
            scharr_x, scharr_y = edges[:,0:1], edges[:,1:2]
            sobel_x, sobel_y = edges[:,2:3], edges[:,3:4]
            def dynamic_norm_fusion(s_x, s_y, so_x, so_y):
                # Scharr特征处理
                scharr_mag = torch.sqrt(s_x**2 + s_y**2 + 1e-6)
                scharr_mag = (scharr_mag - scharr_mag.min()) / (scharr_mag.max() - scharr_mag.min() + 1e-6)
                scharr_mag = torch.pow(scharr_mag, 0.5)  # 伽马校正

                # Sobel特征处理
                sobel_mag = torch.abs(so_x) + torch.abs(so_y)  # L1范数
                sobel_mag = (sobel_mag - sobel_mag.min()) / (sobel_mag.max() - sobel_mag.min() + 1e-6)

                # 动态融合 (基于特征对比度)
                alpha = torch.sigmoid(scharr_mag.mean() - sobel_mag.mean())  # 自动调整权重
                return alpha * scharr_mag + (1-alpha) * sobel_mag
#             def directional_fusion(s_x, s_y, so_x, so_y):
#                 # 计算梯度方向
#                 grad_dir = torch.atan2(s_y + so_y, s_x + so_x)  # 综合方向

#                 # Scharr对对角线更敏感
#                 scharr_weight = torch.abs(torch.sin(grad_dir * 2))  # 45°方向增强
#                 scharr_part = scharr_weight * (torch.abs(s_x) + torch.abs(s_y))

#                 # Sobel对水平/垂直更敏感
#                 sobel_weight = (torch.abs(torch.cos(grad_dir * 2)))  # 0/90°方向增强
#                 sobel_part = sobel_weight * (torch.abs(so_x) + torch.abs(so_y))

#                 return 0.5 * (scharr_part + sobel_part)
#             def multiscale_select(s_x, s_y, so_x, so_y):
#                 # Scharr检测精细边缘
#                 scharr_fine = torch.sqrt(s_x**2 + s_y**2)

#                 # Sobel检测主要边缘
#                 sobel_coarse = torch.abs(so_x) + torch.abs(so_y)

#                 # 自适应阈值选择
#                 threshold = 0.5 * (scharr_fine.mean() + sobel_coarse.mean())
#                 mask = (sobel_coarse > threshold).float()

#                 return mask * sobel_coarse + (1-mask) * scharr_fine
            # 合并所有边缘特征 (4原始 + 1融合)
            edge_mag1 = dynamic_norm_fusion(scharr_x, scharr_y, sobel_x, sobel_y)
            # edge_mag2 = directional_fusion(scharr_x, scharr_y, sobel_x, sobel_y)
            # edge_mag3 = multiscale_select(scharr_x, scharr_y, sobel_x, sobel_y)
            all_edge_feats = torch.cat([edges, edge_mag1], dim=1)  # [B,5,H,W]
        
        # 2. 边缘特征转换 (现在输入5通道)
        edge_weights = self.edge_fusion(all_edge_feats)
        
        # 3. 动态密度卷积
        Phi = self._get_density_matrix()
        conv1_weight = self.conv1[0].weight * Phi
        conv1_out = F.conv2d(x, conv1_weight, padding=1)
        conv1_out = self.conv1[1](conv1_out)
        conv1_out = F.relu(conv1_out, inplace=True)
        
        # 4. 边缘引导特征调制
        conv1_out = conv1_out * edge_weights * self.alpha
        
        # 5. 第二层卷积
        conv2_out = self.conv2[0](conv1_out)
        conv2_out = self.conv2[1](conv2_out)
        return F.relu(conv2_out, inplace=True)
# class Up(nn.Module):
#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super(Up, self).__init__()
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = HEGDC(in_channels, out_channels)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = HEGDC(in_channels, out_channels)

#     def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
#         x1 = self.up(x1)
#         diff_y = x2.size()[2] - x1.size()[2]
#         diff_x = x2.size()[3] - x1.size()[3]
#         x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
#                         diff_y // 2, diff_y - diff_y // 2])
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)
def build_normalization(norm_cfg, channels):
    """根据配置构建归一化层
    
    Args:
        norm_cfg (dict): 归一化层配置字典，需包含type和requires_grad字段
        channels (int): 输入特征通道数
    
    Returns:
        tuple: (归一化类型字符串, 归一化层实例)
    """
    norm_type = norm_cfg['type']
    requires_grad = norm_cfg.get('requires_grad', True)
    
    # 提取额外参数（如eps、momentum等）
    params = {k: v for k, v in norm_cfg.items() 
             if k not in ['type', 'requires_grad']}
    
    # 根据类型创建归一化层
    if norm_type == 'BN':
        layer = nn.BatchNorm2d(channels, **params)
    elif norm_type == 'LN':
        layer = nn.LayerNorm([channels], **params)  # 假设处理2D特征
    elif norm_type == 'IN':
        layer = nn.InstanceNorm2d(channels, **params)
    else:
        raise ValueError(f"Unsupported normalization type: {norm_type}")
    
    # 控制参数是否需要梯度
    if not requires_grad:
        for param in layer.parameters():
            param.requires_grad = False
    
    return (norm_type, layer)
# class GaussianAttention(nn.Module):
#     def __init__(self, channels, kernel_size=3, sigma=0.5):
#         super().__init__()
#         # 配置归一化层参数
#         norm_cfg = dict(type='BN', requires_grad=True)
#         #生成高斯核并初始化卷积层
#         gaussian_kernel = self.create_gaussian_kernel(kernel_size, sigma)
#         gaussian_kernel = nn.Parameter(gaussian_kernel, requires_grad=False).clone()
#         #初始化分组卷积实现高斯滤波
#         self.gaussian_filter = nn.Conv2d(
#         channels, channels,
#         kernel_size=kernel_size,
#         padding=kernel_size //2,
#         groups=channels,
#         bias=False
#         )
#         self.gaussian_filter.weight.data = gaussian_kernel.repeat(channels, 1, 1, 1)

#         self.norm = build_normalization(norm_cfg, channels)[1]
#         self.activation = nn.GELU()
#     def create_gaussian_kernel(self,kernel_size, sigma):
#         return torch.FloatTensor([
#             [(1/(2 *math.pi * sigma ** 2))* math.exp(-(x** 2 +y** 2)/(2 * sigma ** 2))
#              for x in range(-kernel_size //2 +1,kernel_size // 2 + 1)]
#             for y in range(-kernel_size //2+1,kernel_size // 2 + 1)
#         ]).unsqueeze(0).unsqueeze(0)

#     def forward(self,x):
#         "前向传播:生成注意力权重并与输入特征相乘"
#         filtered = self.gaussian_filter(x)
#         attention= self.activation(self.norm(filtered))
#         return x*attention

# 2要跑的，递归门控注意力机制
# class SimplifiedRecursiveAttention(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.channels = channels
        
#         # 门控生成器
#         self.gate_gen = nn.Sequential(
#             nn.Conv2d(channels // 2, channels // 8, 1),
#             nn.ReLU(),
#             nn.Conv2d(channels // 8, 1, 1),
#             nn.Sigmoid()
#         )
        
#         # 最终融合
#         self.final_conv = nn.Conv2d(channels, channels, 1)

#     def forward(self, x):
#         # 拆分特征
#         base, gate_input = torch.split(x, self.channels // 2, dim=1)
        
#         # 生成门控图
#         gate = self.gate_gen(gate_input)
        
#         # 应用门控
#         enhanced_base = base * gate
        
#         # 融合结果
#         out = torch.cat([enhanced_base, gate_input], dim=1)
#         return self.final_conv(out)


# 第二个模块 76.2,到时候跑消融实验再改改看，能不能不要和原本那么像，大部分都一样，虽然确实创新了，但感觉光这样创新度有点低。保持性能在改改样子。
class RecursiveGatedAttention(nn.Module):
    def __init__(self, dim, order=2, reduction=8, kernel_size=3):
        """
        Recursive Gated Attention (RGA) Module - Fixed Version
        
        Args:
            dim (int): 输入通道数
            order (int): 递归阶数 (默认2阶)
            reduction (int): 通道压缩比例 (默认8)
            kernel_size (int): 深度卷积核大小 (默认3)
        """
        super().__init__()
        self.order = order
        self.dim = dim
        
        # 计算通道拆分大小
        self.split_sizes = [dim // (2 ** i) for i in range(1, order)]
        self.split_sizes.append(dim // (2 ** (order - 1)))
        self.split_sizes.reverse()
        
        # 确保总和不超过dim
        total = sum(self.split_sizes)
        if total > dim:
            self.split_sizes[-1] = dim - sum(self.split_sizes[:-1])
        
        # 输入投影
        self.proj_in = nn.Conv2d(dim, self.split_sizes[0] + sum(self.split_sizes), 1)
        
        # 递归门控生成器
        self.gate_convs = nn.ModuleList()
        for i in range(order):
            in_ch = self.split_sizes[i]
            gate_conv = nn.Sequential(
                nn.Conv2d(in_ch, max(in_ch // reduction, 8), 1),
                nn.GELU(),
                nn.Conv2d(max(in_ch // reduction, 8), 1, 1),
                nn.Sigmoid()
            )
            self.gate_convs.append(gate_conv)
        
        # 特征转换卷积
        self.transform_convs = nn.ModuleList()
        for i in range(order - 1):
            self.transform_convs.append(
                nn.Conv2d(self.split_sizes[i], self.split_sizes[i+1], 1)
            )
        
        # 深度卷积增强局部交互
        self.dwconv = nn.Conv2d(sum(self.split_sizes), sum(self.split_sizes), 
                               kernel_size, padding=kernel_size//2, 
                               groups=sum(self.split_sizes))
        
        # 输出投影
        self.proj_out = nn.Conv2d(self.split_sizes[-1], dim, 1)
        
        # 初始化缩放因子
        self.scale = nn.Parameter(torch.tensor(1.0))
        
        print(f'[RGA] order={order}, split_sizes={self.split_sizes}')

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 特征投影
        fused_x = self.proj_in(x)
        base, gates = torch.split(
            fused_x, [self.split_sizes[0], sum(self.split_sizes)], dim=1
        )
        
        # 应用深度卷积增强空间交互
        gates = self.dwconv(gates) * self.scale
        
        # 拆分门控信号
        gate_list = torch.split(gates, self.split_sizes, dim=1)
        
        # 递归门控处理
        out = base
        for i in range(self.order):
            # 生成门控图
            gate_map = self.gate_convs[i](gate_list[i])
            
            # 应用门控
            out = out * gate_map
            
            # 特征转换 (除了最后一次迭代)
            if i < self.order - 1:
                out = self.transform_convs[i](out)
        
        # 最终投影
        return self.proj_out(out)
# 通道拆分逻辑重构：

# 使用与原始gnconv相同的通道拆分策略

# split_sizes = [dim//2^(order-1), dim//2^(order-2), ..., dim//2^0]

# 确保最后一个拆分大小包含剩余通道

# 投影层修复：

# proj_in 输出通道改为 split_sizes[0] + sum(split_sizes)

# 正确拆分基础特征和门控特征

# 深度卷积修复：

# 输入/输出通道数设为 sum(split_sizes)

# 分组数设为 sum(split_sizes) 实现真正的深度可分离卷积

# 特征转换修复：

# 转换卷积的输入/输出通道严格匹配 split_sizes[i] 和 split_sizes[i+1]

# 确保通道数在转换过程中一致

# 输出投影修复：

# 输入通道设为 split_sizes[-1]

# 输出通道恢复为原始输入维度 dim  
# 递归门控机制：

# 采用多阶递归处理（默认2阶）

# 每阶生成独立的门控图，逐步细化注意力

# 通过特征转换卷积连接不同阶的特征

# 动态通道分配：

# 根据递归阶数自动计算通道拆分比例

# 高阶特征分配更多通道，增强表达能力

# 确保通道总数与输入一致

# 局部-全局特征融合：

# 深度卷积（DWConv）增强局部空间交互

# 门控机制捕获跨通道依赖

# 1×1卷积实现跨通道信息融合

# 自适应缩放：

# 可学习的缩放因子平衡门控强度

# 增强模块的适应性
from torch.nn import init
from torch.nn.parameter import Parameter
class SoftPooling2D(torch.nn.Module):
    def __init__(self,kernel_size,stride=None,padding=0):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size,stride,padding, count_include_pad=False)
    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp*x)
        return x/x_exp_pool 

# class StdPool(nn.Module):
#     def __init__(self):
#         super(StdPool, self).__init__()
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         std = x.view(b, c, -1).std(dim=2, keepdim=True)
#         std = std.reshape(b, c, 1, 1)
#         return std
# class MCAGate(nn.Module):
#     def __init__(self, k_size, pool_types=['avg', 'std']):
#         """Constructs a MCAGate module.
#         Args:
#             k_size: kernel size
#             pool_types: pooling type. 'avg': average pooling, 'max': max pooling, 'std': standard deviation pooling.
#         """
#         super(MCAGate, self).__init__()
#         self.pools = nn.ModuleList([])
#         for pool_type in pool_types:
#             if pool_type == 'avg':
#                 self.pools.append(nn.AdaptiveAvgPool2d(1))
#             elif pool_type == 'max':
#                 self.pools.append(nn.AdaptiveMaxPool2d(1))
#             elif pool_type == 'std':
#                 self.pools.append(StdPool())
#             else:
#                 raise NotImplementedError
#         self.conv = nn.Conv2d(1, 1, kernel_size=(1, k_size), stride=1, padding=(0, (k_size - 1) // 2), bias=False)
#         self.sigmoid = nn.Sigmoid()
#         self.weight = nn.Parameter(torch.rand(2))
#     def forward(self, x):
#         feats = [pool(x) for pool in self.pools]
#         if len(feats) == 1:
#             out = feats[0]
#         elif len(feats) == 2:
#             weight = torch.sigmoid(self.weight)
#             out = 1 / 2 * (feats[0] + feats[1]) + weight[0] * feats[0] + weight[1] * feats[1]
#         else:
#             assert False, "Feature Extraction Exception!"
#         out = out.permute(0, 3, 2, 1).contiguous()
#         out = self.conv(out)
#         out = out.permute(0, 3, 2, 1).contiguous()
#         out = self.sigmoid(out)
#         out = out.expand_as(x)
#         return x * out
# class MCALayer(nn.Module):
#     def __init__(self, inp, no_spatial=False):
#         """Constructs a MCA module.
#         Args:
#             inp: Number of channels of the input feature maps
#             no_spatial: whether to build channel dimension interactions
#         """
#         super(MCALayer, self).__init__()
#         lambd = 1.5
#         gamma = 1
#         temp = round(abs((math.log2(inp) - gamma) / lambd))
#         kernel = temp if temp % 2 else temp - 1
#         self.h_cw = MCAGate(3)
#         self.w_hc = MCAGate(3)
#         self.no_spatial = no_spatial
#         if not no_spatial:
#             self.c_hw = MCAGate(kernel)
#     def forward(self, x):
#         x_h = x.permute(0, 2, 1, 3).contiguous()
#         x_h = self.h_cw(x_h)
#         x_h = x_h.permute(0, 2, 1, 3).contiguous()
#         x_w = x.permute(0, 3, 2, 1).contiguous()
#         x_w = self.w_hc(x_w)
#         x_w = x_w.permute(0, 3, 2, 1).contiguous()
#         if not self.no_spatial:
#             x_c = self.c_hw(x)
#             x_out = 1 / 3 * (x_c + x_h + x_w)
#         else:
#             x_out = 1 / 2 * (x_h + x_w)
#         return x_out

class MCALayer(nn.Module):
    """增强的多维坐标注意力层 - 零参数增加设计"""
    def __init__(self, inp, no_spatial=False):
        super(MCALayer, self).__init__()
        self.no_spatial = no_spatial
        self.inp = inp
        
        # 保持原有的MCA结构
        lambd = 1.5
        gamma = 1
        temp = round(abs((math.log2(inp) - gamma) / lambd))
        kernel = temp if temp % 2 else temp - 1
        
        self.h_cw = MCAGate(3)
        self.w_hc = MCAGate(3)
        
        if not no_spatial:
            self.c_hw = MCAGate(kernel)
    
    def compute_local_range(self, x, kernel_size=3):
        """计算局部动态范围（最大值-最小值）- 无参数操作"""
        padding = kernel_size // 2
        max_pool = F.max_pool2d(x, kernel_size, 1, padding)
        min_pool = -F.max_pool2d(-x, kernel_size, 1, padding)
        return max_pool - min_pool
    
    def compute_local_variance(self, x, kernel_size=3):
        """计算局部方差 - 无参数操作"""
        padding = kernel_size // 2
        mean = F.avg_pool2d(x, kernel_size, 1, padding)
        variance = F.avg_pool2d((x - mean) ** 2, kernel_size, 1, padding)
        return variance
    
    def frequency_enhancement(self, x):
        """频率域增强 - 无参数操作"""
        # 傅里叶变换
        fft = torch.fft.fft2(x, norm='ortho')
        
        # 提取幅度谱
        magnitude = torch.abs(fft)
        
        # 对幅度谱进行轻微增强
        enhanced_magnitude = magnitude * 1.1  # 轻微增强高频成分
        
        # 重建傅里叶变换结果
        phase = torch.angle(fft)
        enhanced_fft = enhanced_magnitude * torch.exp(1j * phase)
        
        # 逆傅里叶变换
        enhanced = torch.fft.ifft2(enhanced_fft, norm='ortho').real
        
        return enhanced
    
    def channel_shuffle(self, x, groups=4):
        """通道混洗 - 无参数操作"""
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        
        # 重塑
        x = x.view(batch_size, groups, channels_per_group, height, width)
        
        # 转置
        x = torch.transpose(x, 1, 2).contiguous()
        
        # 重塑回原形状
        x = x.view(batch_size, -1, height, width)
        
        return x
    
    def forward(self, x):
        # 原有的MCA处理流程
        x_h = x.permute(0, 2, 1, 3).contiguous()
        x_h = self.h_cw(x_h)
        x_h = x_h.permute(0, 2, 1, 3).contiguous()
        
        x_w = x.permute(0, 3, 2, 1).contiguous()
        x_w = self.w_hc(x_w)
        x_w = x_w.permute(0, 3, 2, 1).contiguous()
        
        if not self.no_spatial:
            x_c = self.c_hw(x)
            x_out = 1 / 3 * (x_c + x_h + x_w)
        else:
            x_out = 1 / 2 * (x_h + x_w)
        
        # 新增的无参数增强操作
        # 1. 计算局部动态范围
        local_range = self.compute_local_range(x_out)
        
        # 2. 计算局部方差
        local_variance = self.compute_local_variance(x_out)
        
        # 3. 频率域增强
        freq_enhanced = self.frequency_enhancement(x_out)
        
        # 4. 通道混洗
        shuffled = self.channel_shuffle(x_out)
        
        # 自适应融合 - 使用简单的加权平均
        # 这些权重是固定的，不增加参数
        enhanced = 0.4 * x_out + 0.2 * local_range + 0.2 * local_variance + 0.1 * freq_enhanced + 0.1 * shuffled
        
        return enhanced

# 保持原有的StdPool和MCAGate不变
class StdPool(nn.Module):
    def __init__(self):
        super(StdPool, self).__init__()
    def forward(self, x):
        b, c, _, _ = x.size()
        std = x.view(b, c, -1).std(dim=2, keepdim=True)
        std = std.reshape(b, c, 1, 1)
        return std

class MCAGate(nn.Module):
    def __init__(self, k_size, pool_types=['avg', 'std']):
        super(MCAGate, self).__init__()
        self.pools = nn.ModuleList([])
        for pool_type in pool_types:
            if pool_type == 'avg':
                self.pools.append(nn.AdaptiveAvgPool2d(1))
            elif pool_type == 'max':
                self.pools.append(nn.AdaptiveMaxPool2d(1))
            elif pool_type == 'std':
                self.pools.append(StdPool())
            else:
                raise NotImplementedError
        self.conv = nn.Conv2d(1, 1, kernel_size=(1, k_size), stride=1, padding=(0, (k_size - 1) // 2), bias=False)
        self.sigmoid = nn.Sigmoid()
        self.weight = nn.Parameter(torch.rand(2))
    
    def forward(self, x):
        feats = [pool(x) for pool in self.pools]
        if len(feats) == 1:
            out = feats[0]
        elif len(feats) == 2:
            weight = torch.sigmoid(self.weight)
            out = 1 / 2 * (feats[0] + feats[1]) + weight[0] * feats[0] + weight[1] * feats[1]
        else:
            assert False, "Feature Extraction Exception!"
        out = out.permute(0, 3, 2, 1).contiguous()
        out = self.conv(out)
        out = out.permute(0, 3, 2, 1).contiguous()
        out = self.sigmoid(out)
        out = out.expand_as(x)
        return x * out


class EdgeAwareFeatureEnhancer(nn.Module):
    def __init__ (self, in_channels):
        super(EdgeAwareFeatureEnhancer,self).__init__()
        self.edge_extractor = nn.AvgPool2d(kernel_size=3, stride=1,padding=1)
        self.weight_generator = nn.Sequential(
             nn.Conv2d(in_channels,in_channels, kernel_size=1),
             nn.BatchNorm2d(in_channels),
             nn.Sigmoid()
        )
    def forward(self, x):
    #边缘特征提取
        edge_features=x-self.edge_extractor(x)
        edge_weights=self.weight_generator(edge_features)
        enhanced_features= edge_weights*x+x
        return enhanced_features

class DoubleConv1(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv1, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # MCALayer(
            #     mid_channels
            # ),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # EdgeEnhancedGRFB(in_channels, mid_channels, stride=1, scale=0.1, visual=12),
            EdgeEnhancedGRFB(mid_channels, out_channels, stride=1, scale=0.1, visual=12)
        )
class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
#             GRFBUNet原版代码就是这里使用DoubleConv1，其他地方都是DoubleConv，是我自以为是，在GRFBUNet里加了个1,下采样在特征提取的使用这个多尺度特征模块。
            DoubleConv1(in_channels, out_channels)
    
        )
# class HalfConv(nn.Module):
#     def __init__(self, dim, n_div=2):
#         super().__init__()
#         self.dim_conv3 = dim // n_div
#         self.dim_untouched = dim - self.dim_conv3
#         self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
#     def forward(self, x: Tensor) -> Tensor:
#         # for training/inference
#         x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
#         x1 = self.partial_conv3(x1)
#         x = torch.cat((x1, x2), 1)
#         return x


class Up(nn.Module):#上采样，进行特征恢复还原
    def __init__(self, in_channels, out_channels, bilinear=True,use_attention=False):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2,use_attention)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels,use_attention)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):#根据类别数，输出通道数
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

class BasicConv(nn.Module):

        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                     bn=True, bias=False):
            super(BasicConv, self).__init__()
            self.out_channels = out_channels
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=bias)
            self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.01, affine=True) if bn else None
            self.relu = nn.ReLU(inplace=True) if relu else None

        def forward(self, x):
            x = self.conv(x)
            if self.bn is not None:
                x = self.bn(x)
            if self.relu is not None:
                x = self.relu(x)
            return x

# class GRFB(nn.Module):
#         def __init__(self, in_channels, out_channels, stride=1, scale=0.1, visual=12):
#             super(GRFB, self).__init__()
#             self.scale = scale
#             self.out_channels = out_channels
#             inter_planes = in_channels // 8
#             self.branch0 = nn.Sequential(
#                 BasicConv(in_channels, 2 * inter_planes, kernel_size=1, stride=stride),
#                 BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual,
#                           relu=False),
#                 BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=1, stride=stride)
#             )
#             self.branch1 = nn.Sequential(
#                 BasicConv(in_channels, inter_planes, kernel_size=1, stride=1),
#                 BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=inter_planes),
#                 BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=1, stride=stride),
#                 BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=2 * visual,
#                           dilation=2 * visual, relu=False),
#                 BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=1, stride=1)
#             )
#             self.branch2 = nn.Sequential(
#                 BasicConv(in_channels, inter_planes, kernel_size=1, stride=1),
#                 BasicConv(inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, groups=inter_planes),
#                 BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=1, stride=stride),
#                 BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=stride, padding=1, groups=2 * inter_planes),
#                 BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=1, stride=stride),
#                 BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=3 * visual,
#                           dilation=3 * visual, relu=False),
#                 BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=1, stride=stride)
#             )

#             self.ConvLinear = BasicConv(14 * inter_planes, out_channels, kernel_size=1, stride=1, relu=False)
#             self.shortcut = BasicConv(in_channels, out_channels, kernel_size=1, stride=stride, relu=False)
#             self.relu = nn.ReLU(inplace=False)

#         def forward(self, x):
#             x0 = self.branch0(x)
#             x1 = self.branch1(x)
#             x2 = self.branch2(x)

#             out = torch.cat((x, x0, x1, x2), 1)
#             out = self.ConvLinear(out)
#             short = self.shortcut(x)
#             out = out * self.scale + short
#             out = self.relu(out)

#             return out
# 这个暂时先跑，估计也不行，下一个基于欧拉公式的特征融合模块，当前是sobel算子(不行)、余弦相似度
#换欧拉公式，但这个不是直接用的，引入基于欧拉公式的特征融合机制。改进后的模块会计算每个分支的相位和振幅，然后通过欧拉公式进行特征加权融合，最后使用注意力机制动态融合各分支结果。多尺度边缘增强注意力融合模块
# class ChannelAttentionModule(nn.Module):
#     def __init__(self, in_channels, reduction=4):
#         super(ChannelAttentionModule, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return self.sigmoid(out)


# class SpatialAttentionModule(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttentionModule, self).__init__()
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)
# class EdgeEnhancedGRFB(nn.Module):
#     """边缘增强的特征提取模块"""
#     def __init__(self, in_channels, out_channels, stride=1, scale=0.1, visual=12, fusion_factor=4.0):
#         super(EdgeEnhancedGRFB, self).__init__()
#         self.scale = scale
#         self.inter_planes = max(in_channels // 8, 4)
        
#         self.edge_enhancer = EdgeAwareFeatureEnhancer(in_channels)
        
#         # 多分支特征提取
#         self.branch_dir = nn.Sequential(
#             BasicConv(in_channels, 2*self.inter_planes, 1),
#             BasicConv(2*self.inter_planes, 2*self.inter_planes, 3, 
#                       padding=visual, dilation=visual, relu=False),
#             BasicConv(2*self.inter_planes, 2*self.inter_planes, 1)
#         )
        
#         self.branch_edge = nn.Sequential(
#             BasicConv(in_channels, self.inter_planes, 1),
#             EdgeAwareFeatureEnhancer(self.inter_planes),  # 内置边缘增强
#             BasicConv(self.inter_planes, 2*self.inter_planes, (3, 3), 
#                       stride, padding=1, groups=self.inter_planes),
#             BasicConv(2*self.inter_planes, 2*self.inter_planes, 3, 
#                       padding=2*visual, dilation=2*visual, relu=False),
#             BasicConv(2*self.inter_planes, 2*self.inter_planes, 1)
#         )
        
#         self.branch_ctx = nn.Sequential(
#             BasicConv(in_channels, self.inter_planes, 3, padding=1),
#             BasicConv(self.inter_planes, 2*self.inter_planes, 3,
#                       stride=stride, padding=1, groups=2),
#             BasicConv(2*self.inter_planes, 2*self.inter_planes, 3,
#                       padding=3*visual, dilation=3*visual, relu=False),
#             BasicConv(2*self.inter_planes, 2*self.inter_planes, 1)
#         )
        
#         # 特征融合核心
#         self.concat_channels = in_channels + 6 * self.inter_planes
#         fusion_dim = int(out_channels // fusion_factor)
        
#         # 特征融合层
#         self.fusion_down = nn.Conv2d(2*self.concat_channels, fusion_dim, 1)
#         self.conv3 = nn.Conv2d(fusion_dim, fusion_dim, 3, padding=1)
#         self.conv5 = nn.Conv2d(fusion_dim, fusion_dim, 5, padding=2)
#         self.conv7 = nn.Conv2d(fusion_dim, fusion_dim, 7, padding=3)
        
#         # 注意力机制
#         self.channel_att = ChannelAttentionModule(fusion_dim)
#         self.spatial_att = SpatialAttentionModule()
        
#         self.fusion_up = nn.Conv2d(fusion_dim, out_channels, 1)
        
#         # 残差连接
#         self.shortcut = BasicConv(in_channels, out_channels, 1, stride, relu=False)
#         self.relu = nn.ReLU(inplace=False)
        
#         # 目标特定增强
#         self.target_enhancer = nn.Sequential(
#             nn.Conv2d(out_channels, 3, 3, padding=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         identity = x
        
#         # 边缘增强输入
#         x_enhanced = self.edge_enhancer(x)
        
#         # 多分支特征提取
#         dir_feat = self.branch_dir(x_enhanced)
#         edge_feat = self.branch_edge(x_enhanced)
#         ctx_feat = self.branch_ctx(x_enhanced)
        
#         # 特征拼接
#         branch_concat = torch.cat((dir_feat, edge_feat, ctx_feat), 1)
#         concat_feat = torch.cat((x, branch_concat), 1)
        
#         # 特征融合
#         x_cat = torch.cat([concat_feat, concat_feat], dim=1)
#         x_down = self.fusion_down(x_cat)
#         residual = x_down
        
#         # 多尺度特征提取
#         x3 = self.conv3(x_down)
#         x5 = self.conv5(x_down)
#         x7 = self.conv7(x_down)
#         scale_features = x3 + x5 + x7
        
#         # 空间注意力增强
#         spatial_weights = self.spatial_att(scale_features)
#         spatial_enhanced = scale_features * spatial_weights
        
#         # 通道注意力增强
#         channel_weights = self.channel_att(x_down)
        
#         # 特征融合与升维
#         fused = residual + spatial_enhanced * channel_weights
#         out = self.fusion_up(fused)
        
#         # 残差连接
#         short = self.shortcut(identity)
#         out = out * self.scale + short
#         out = self.relu(out)
        
#         # 目标特定增强
#         target_weights = self.target_enhancer(out)
#         out = out * (1 + target_weights.mean(dim=1, keepdim=True))
#         return out





# 第一个模块
class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class FusionConv(nn.Module):
    def __init__(self, in_channels, out_channels, factor=4.0):
        super(FusionConv, self).__init__()  # 必须先调用父类初始化
        
        # 中间通道数，用于降维
        dim = int(out_channels // factor)
        
        # 所有层定义必须在super().__init__()之后
        # 降维卷积(1x1)
        self.down = nn.Conv2d(2*in_channels, dim, kernel_size=1, stride=1)
        # 三种不同尺度的卷积(感受野不同)
        self.conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2)
        self.conv_7x7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3)
        # 空间注意力模块
        self.spatial_attention = SpatialAttentionModule()
        self.channel_attention = ChannelAttentionModule(dim)  # 注意：这里需要传入通道数
        self.up = nn.Conv2d(dim, out_channels, kernel_size=1, stride=1)
    
    def forward(self, x1, x2):
        x_fused = torch.cat([x1, x2], dim=1)  # 修正：移除tensors=关键字
        x_fused = self.down(x_fused)
        res = x_fused
        x_3x3 = self.conv_3x3(x_fused)
        x_5x5 = self.conv_5x5(x_fused)
        x_7x7 = self.conv_7x7(x_fused)
        x_fused_s = x_3x3 + x_5x5 + x_7x7

        # 乘以空间注意力权重(空间维度增强)
        x_fused_s = x_fused_s * self.spatial_attention(x_fused_s)
        # 通道注意力提取
        x_fused_c = self.channel_attention(x_fused)
        # 融合空间增强和通道增强后的特征，并升维输出
        x_out = self.up(res + x_fused_s * x_fused_c)
        return x_out

class EdgeEnhancedGRFB(nn.Module):
    """边缘增强的 GRFB 模块（集成 FusionConv 双注意力 + 多尺度融合）"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 scale=0.1,
                 visual=12,
                 fusion_factor=4.0):          # FusionConv 的通道压缩倍率
        super().__init__()
        self.scale = scale
        self.out_channels = out_channels
        self.inter_planes = max(in_channels // 8, 4)

        # 1. 输入边缘增强
        self.edge_enhancer = EdgeAwareFeatureEnhancer(in_channels)

        # 2. 多分支特征提取（与原实现相同）
        self.branch_dir = nn.Sequential(
            BasicConv(in_channels, 2 * self.inter_planes, 1),
            BasicConv(2 * self.inter_planes, 2 * self.inter_planes, 3,
                      padding=visual, dilation=visual, relu=False),
            BasicConv(2 * self.inter_planes, 2 * self.inter_planes, 1)
        )
        self.branch_edge = nn.Sequential(
            BasicConv(in_channels, self.inter_planes, 1),
            EdgeAwareFeatureEnhancer(self.inter_planes),
            BasicConv(self.inter_planes, 2 * self.inter_planes, (3, 3), stride,
                      padding=1, groups=self.inter_planes),
            BasicConv(2 * self.inter_planes, 2 * self.inter_planes, 3,
                      padding=2 * visual, dilation=2 * visual, relu=False),
            BasicConv(2 * self.inter_planes, 2 * self.inter_planes, 1)
        )
        self.branch_ctx = nn.Sequential(
            BasicConv(in_channels, self.inter_planes, 3, padding=1),
            BasicConv(self.inter_planes, 2 * self.inter_planes, 3,
                      stride=stride, padding=1, groups=2),
            BasicConv(2 * self.inter_planes, 2 * self.inter_planes, 3,
                      padding=3 * visual, dilation=3 * visual, relu=False),
            BasicConv(2 * self.inter_planes, 2 * self.inter_planes, 1)
        )

        # 3. 用 FusionConv 替换原 ConvLinear + 前后 EdgeEnhancer
        self.concat_channels = in_channels + 6 * self.inter_planes
        self.fusion_conv = FusionConv(self.concat_channels,
                                      out_channels,
                                      factor=fusion_factor)

        # 4. 残差连接
        self.shortcut = BasicConv(in_channels, out_channels, 1, stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

        # 5. 目标特定增强
        self.target_enhancer = nn.Sequential(
            nn.Conv2d(out_channels, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x

        # 1. 输入边缘增强
        x_enhanced = self.edge_enhancer(x)

        # 2. 多分支提取
        dir_feat   = self.branch_dir(x_enhanced)
        edge_feat  = self.branch_edge(x_enhanced)
        ctx_feat   = self.branch_ctx(x_enhanced)

        # 3. 拼接分支特征 + 残差输入
        branch_concat = torch.cat((dir_feat, edge_feat, ctx_feat), 1)
        concat_feat   = torch.cat((x, branch_concat), 1)

        # 4. FusionConv 融合 + 通道-空间-多尺度增强
        out = self.fusion_conv(concat_feat, concat_feat)   # 第二个参数可复用，也可传入其他辅助特征

        # 5. 残差连接
        short = self.shortcut(identity)
        out   = out * self.scale + short
        out   = self.relu(out)

        # 6. 目标特定增强
        target_weights = self.target_enhancer(out)
        out = out * (1 + target_weights.mean(dim=1, keepdim=True))

        return out
# def gram_schmidt(input):
#     def projection(u, v):
#         return (torch.dot(u.view(-1), v.view(-1)) / torch.dot(u.view(-1), u.view(-1))) * u
#     output = []
#     for x in input:
#         for y in output:
#             x = x - projection(y, x)
#         x = x / x.norm(p=2)
#         output.append(x)
#     return torch.stack(output)
# def initialize_orthogonal_filters(c, h, w):
#     if h * w < c:
#         n = c // (h * w)
#         gram = []
#         for i in range(n):
#             gram.append(gram_schmidt(torch.rand([h * w, 1, h, w])))
#         return torch.cat(gram, dim=0)
#     else:
#         return gram_schmidt(torch.rand([c, 1, h, w]))
# class OrthogonalFusionGRFB(nn.Module):
#     """正交变换增强的多尺度GRFB模块"""
#     def __init__(self, in_channels, out_channels, stride=1, scale=0.1, visual=12, fusion_factor=4.0):
#         super().__init__()
#         self.scale = scale
#         self.out_channels = out_channels
#         self.inter_planes = max(in_channels // 8, 4)
        
#         # 1. 多分支特征提取（保持原结构）
#         self.branch_dir = nn.Sequential(
#             BasicConv(in_channels, 2 * self.inter_planes, 1),
#             BasicConv(2 * self.inter_planes, 2 * self.inter_planes, 3,
#                       padding=visual, dilation=visual, relu=False),
#             BasicConv(2 * self.inter_planes, 2 * self.inter_planes, 1)
#         )
#         self.branch_edge = nn.Sequential(
#             BasicConv(in_channels, self.inter_planes, 1),
#             BasicConv(self.inter_planes, 2 * self.inter_planes, (3, 3), stride,
#                       padding=1, groups=self.inter_planes),
#             BasicConv(2 * self.inter_planes, 2 * self.inter_planes, 3,
#                       padding=2 * visual, dilation=2 * visual, relu=False),
#             BasicConv(2 * self.inter_planes, 2 * self.inter_planes, 1)
#         )
#         self.branch_ctx = nn.Sequential(
#             BasicConv(in_channels, self.inter_planes, 3, padding=1),
#             BasicConv(self.inter_planes, 2 * self.inter_planes, 3,
#                       stride=stride, padding=1, groups=2),
#             BasicConv(2 * self.inter_planes, 2 * self.inter_planes, 3,
#                       padding=3 * visual, dilation=3 * visual, relu=False),
#             BasicConv(2 * self.inter_planes, 2 * self.inter_planes, 1)
#         )

#         # 2. 正交变换融合模块（创新点）
#         self.ortho_fusion = OrthogonalFusionModule(
#             in_channels + 6 * self.inter_planes, 
#             out_channels,
#             factor=fusion_factor
#         )
        
#         # 3. 残差连接
#         self.shortcut = BasicConv(in_channels, out_channels, 1, stride, relu=False)
#         self.relu = nn.ReLU(inplace=False)

#     def forward(self, x):
#         identity = x
        
#         # 多分支特征提取
#         dir_feat = self.branch_dir(x)
#         edge_feat = self.branch_edge(x)
#         ctx_feat = self.branch_ctx(x)
        
#         # 拼接特征
#         branch_concat = torch.cat((dir_feat, edge_feat, ctx_feat), 1)
#         concat_feat = torch.cat((x, branch_concat), 1)
        
#         # 正交变换融合（创新点）
#         out = self.ortho_fusion(concat_feat)
        
#         # 残差连接
#         short = self.shortcut(identity)
#         out = out * self.scale + short
#         out = self.relu(out)
        
#         return out


# class OrthogonalFusionModule(nn.Module):
#     """正交变换融合模块 - 核心创新点"""
#     def __init__(self, in_channels, out_channels, factor=4.0):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.mid_channels = int(out_channels // factor)
        
#         # 降维卷积
#         self.down = nn.Conv2d(in_channels, self.mid_channels, 1)
        
#         # 多尺度特征提取
#         self.conv_3x3 = nn.Conv2d(self.mid_channels, self.mid_channels, 3, padding=1)
#         self.conv_5x5 = nn.Conv2d(self.mid_channels, self.mid_channels, 5, padding=2)
#         self.conv_7x7 = nn.Conv2d(self.mid_channels, self.mid_channels, 7, padding=3)
        
#         # 正交变换注意力（创新点）
#         self.ortho_attention = OrthogonalSpatialAttention(self.mid_channels)
        
#         # 通道注意力
#         self.channel_attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(self.mid_channels, self.mid_channels // 4, 1),
#             nn.ReLU(),
#             nn.Conv2d(self.mid_channels // 4, self.mid_channels, 1),
#             nn.Sigmoid()
#         )
        
#         # 上采样卷积
#         self.up = nn.Conv2d(self.mid_channels, out_channels, 1)
        
#     def forward(self, x):
#         residual = self.down(x)
        
#         # 多尺度特征提取
#         f3 = self.conv_3x3(residual)
#         f5 = self.conv_5x5(residual)
#         f7 = self.conv_7x7(residual)
        
#         # 多尺度特征融合
#         fused = f3 + f5 + f7
        
#         # 正交空间注意力（创新点）
#         ortho_att = self.ortho_attention(fused)
        
#         # 通道注意力
#         channel_att = self.channel_attention(fused)
        
#         # 特征增强
#         enhanced = fused * ortho_att * channel_att
        
#         # 残差连接与上采样
#         output = self.up(residual + enhanced)
        
#         return output


# class OrthogonalSpatialAttention(nn.Module):
#     """正交空间注意力机制 - 核心创新点"""
#     def __init__(self, channels, ortho_size=7):
#         super().__init__()
#         self.channels = channels
#         self.ortho_size = ortho_size
        
#         # 创建正交滤波器组
#         self.register_buffer("ortho_filters", 
#                            initialize_orthogonal_filters(channels, ortho_size, ortho_size))
        
#         # 自适应池化确保输入尺寸匹配
#         self.pool = nn.AdaptiveAvgPool2d((ortho_size, ortho_size))
        
#         # 卷积层用于特征变换
#         self.conv = nn.Conv2d(channels, channels, 1)
        
#     def forward(self, x):
#         b, c, h, w = x.shape
        
#         # 自适应池化到正交尺寸
#         x_pooled = self.pool(x)
        
#         # 应用正交变换
#         ortho_feat = (self.ortho_filters * x_pooled).sum(dim=(-1, -2), keepdim=True)
        
#         # 特征变换
#         ortho_att = self.conv(ortho_feat)
        
#         # 上采样回原始尺寸
#         ortho_att = F.interpolate(ortho_att, size=(h, w), mode='bilinear', align_corners=False)
        
#         # 生成空间注意力图
#         spatial_att = torch.sigmoid(ortho_att)
        
#         return spatial_att

class GRFBUNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64,
                 use_attention: bool = False):
        super(GRFBUNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.attn1 = RecursiveGatedAttention(base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        #最后两个特征层进行上采样
        x5_attn = self.attn1(x5)
        x = self.up1(x5_attn, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return {"out": logits}

    
    # 要不我所幸不该了，和原来有变化还长点，就行了，因为我不是改进的Unet，我是改进的GRFN-Unet
    
    
# if __name__ == '__main__':
#     model = GRFBUNet(in_channels=3, num_classes=3)

#     # 输入张量的形状
#     input_tensor = torch.randn(1, 3, 256, 256)
    
#     # 计算 FLOPs 和参数量
#     flops, params = profile(model, inputs=(input_tensor,))
#     print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
#     print(f"Parameters: {params / 1e6:.2f} M")

#     output = model(input_tensor)
    
#     # 修改后的输出处理代码
#     if isinstance(output, dict):
#         # 尝试查找可能的输出键名
#         possible_keys = ['out', 'output', 'logits', 'pred', 'seg']
        
#         # 查找包含张量的键
#         tensor_keys = [k for k, v in output.items() if torch.is_tensor(v)]
        
#         if tensor_keys:
#             print(f"模型输出是字典，包含以下张量键: {tensor_keys}")
            
#             # 尝试使用常见键名
#             for key in possible_keys:
#                 if key in output:
#                     print(f"使用键 '{key}': 输出形状为 {output[key].shape}")
#                     break
#             else:
#                 # 如果没有常见键名，使用第一个张量
#                 first_key = tensor_keys[0]
#                 print(f"使用键 '{first_key}': 输出形状为 {output[first_key].shape}")
#         else:
#             print("错误: 输出字典中没有找到张量")
#     else:
#         print(f"输出形状: {output.shape}")