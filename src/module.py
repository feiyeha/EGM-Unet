class InfoGatedReconstructUnit(nn.Module):
# 基于信息量的门控掩码重构单元
    def __init__(self,in_channels: int,gate_threshold: float = 0.5):
        super().__init__()
        num_groups = in_channels
        self.group_norm = GroupBatchNorm2d(in_channels, num_groups=num_groups)
        self.gate threshold = gate_threshold # 信息量门控阈值
        self.sigmoid = nn.Sigmoid() # 门控激活函数        
    def forward(self,x:torch.Tensor)-> torch.Tensor:
        norm_feat = self.group_norm(x)
        channel_weights = self.group_norm.scale / self.group_norm.scale.sum()
        info_scores = self.sigmoid(horm_feat * channel_weights)
        
        high_info_mask= info_scores >= self.gate_threshold
        Low_info_mask= info_scores < self.gate_threshold
        high_info_feat = high info mask *x
        low_info_feat = low_info_mask *x
        return high_info_feat,low_info_feat#这里也可以余弦相似度融合一下
    
# 通过信息量门控机制，实现特征通道的细粒度解耦，抑制噪声干扰，保留关键视觉信息
# 语仪分割住务:实际问题:不同类别的特征信息量差异显著，传统方法依赖全局特征聚合，易导致小
# 目标的特征被背景冗余信息淹没。解决方案:门控掩码筛选出高信息量特征(目标区域关键语义信息)，低信
# 息量特征(背景区域)被抑制。

class SpatialVarianceModulation(nn.Module):
    def __init__(self, eps: float = 1e-4):
        super(SpatialVarianceModulation,self).__init_()
        self.activation = nn.Sigmoid()
        #激活函数:将权重映射压缩到(0,1)范围
        self.eps=eps #数值稳定性常数，避免方差为零的情况

    def forward(self,feature_map: torch.Tensor)-> torch.Tensor:
        batch_size,channels, height, width = feature_map.size()
        #===== 该模块通过计算输入特征图的局部方差来动态调整每个位置的响应强度 =====
        #计算空间维度(HxW)上的均值
        spatial_mean = feature_map.mean(dim=[2,3], keepdim=True)
        # 计算特征值与均值的偏差(中心化处理)【绝对距离 ====> 相对距离】
        # 然后求平方
        squared_deviation=(feature_map-spatial_mean).pow(2)
        # ===== 自适应权重生成 =====
        # 计算空间方差的无偏估计:Σ(偏差2)/(HxW-1)
        spatial_variance = squared_deviation.sum(dim=[2,3], keepdim=True) / (height * width - 1)
        # 核心调制公式:对权重进行Siqmoid激活，权重映射到(0,1)区间
        # 数学形式:权重 =(偏差2)/(4*方差)+0.5
        modulation_coeff = squared_deviation/(4*(spatial_variance + self.eps))+ 0.5
        weight = self.activation(modulation_coeff)
        #===== 特征调制 =====
        #原始特征与权重图逐元素相乘
        # weight s 0.5 ，小幅增强
        # weight s1，保留原始值
        # 效果:增强高方差区域(边缘/纹理)，抑制低方差区域(平滑背景
        modulated_feature =featuré_map * weight
        return modulated_feature
    
# 简单实现特征图的对比度自适应增强:
# 作用位置:任何单一输出特征后，或者任何即插即用模块中。
# 主要功能:其核心功能是对输入特征图进行自适应对比度增强
# 通过计算每个通道方差调整特征响应
# 增强局部细节同时保留全局信息，解决图像对比度低、细节糊特点。
# 1、对噪声主导的低对比度区域进行抑
# 制。2、对重要信息区域进行保留。(注意力都可以这么写)
# Multi-Kernel Perception Unit (MKP)

class FeatureAdjuster(nn.Module):
    def __init__(self):
        super(FeatureAdjuster, self).__init__()
        self.relu_activation = nn.ReLU()
    # 余弦相似度特征图融合:我是否可以chuck一下。不同尺度的特征融合一下
    def forward(self,feature_a,feature_b):
        # 获取 feature_a 和 feature_b 的形状信息
        shape_a,shape_b =feature_a.size(),feature_b.size()
        # 断言 feature_a 和 feature_b 在通道数必须相同
        assert shape_a[1]== shape_b[1]
        # 计算 feature_a 和 feature_b 在通道特征上的余弦相似度
        cosine similarity =F.cosine_similarity(feature_a,feature_b,dim=1)
        # 在余弦相似度结果上增加一个维度，方便后续的乘法运算
        cosine_similarity = cosine_similarity.unsqueeze(1)

        feature_a = feature_a + feature _b * cosine_similarity
        # 对更新后的 feature_a 应用 ReLU 激活函数
        feature_a = self.relu_activation(feature_a)
        return feature_a
    

class SpatialAttentionModule(nn.Module):
    def __init__ (self,kernel size=7):
        super(SpatialAttentionModule, self).__init__()
        # 卷积用于融合通道后的空间特征(输入通道为2:平均图+最大图)
        self.conv1 = nn.Conv2d(2,1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        #对通道维进行平均池化，结果为 Nx1xHxW
        avg_out = torch.mean(x,dim=1,keepdim=True)
        #对通道维进行最大泄化
        max_out,_= torch.max(x,dim=1, keepdim=True)    
        x = torch.cat([avg_out,max_out],dim=1)
        x= self.conv1(x)
        return self.sigmoid(x)
    
class ChannelAttentionModule(nn.Module):
    def __init__ (self,kernel size=7):
        super(SpatialAttentionModule, self).__init__()
        # 卷积用于融合通道后的空间特征(输入通道为2:平均图+最大图)
        self.conv1 = nn.Conv2d(2,1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        #对通道维进行平均池化，结果为 Nx1xHxW
        avg_out = torch.mean(x,dim=1,keepdim=True)
        #对通道维进行最大泄化
        max_out= self.fc(self.max_pool(x))    
        out = avg_out+ max_out
        return self.sigmoid(out)  
    
class FusionConv(nn.Module):
    def __init__(self,in_channels, out_channels, factor=4.0):
        # 中间通道数，用于降维
        dim =int(out_channels // factor)
        # 降维卷积(1x1)
        self.down = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)
        # 三种不同尺度的卷积(感受野不同)
        self.conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3,stride=1, padding=1)
        self.conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5,stride=1, padding=2)
        self.conv_7x7 = nn.Conv2d(dim, dim, kernel size=7,stride=1, padding=3)
        # 空间注意力模块
        self.spatial_attention = SpatialAttentionModule()
        self.channel_attention = ChannelAttentionModule()
        self.up = nn.Conv2d(dim, out_channels, kernel_size=1, stride=1)
    def forward(self,x1,x2):
        x_fused =torch.cat(tensors:[x1,x2],dim=1)
        x_fused = self.down(x_fused)
        res = x_fused
        x_3x3 = self.conv_3x3(x_fused)
        x_5x5 = self.conv_5x5(x_fused)
        x_7x7 = self.conv_7x7(x_fused)
        x_fused_s = x_3x3 + x_5x5 + x_7x7

        #乘以空间注意力权重(空间维度增强)
        x_fused_s =x_fused_s *self.spatial_attention(x_fused_s)
        # ---------
        # 通道注意力提取---
        x_fused_c = self.channel_attention(x_fused)
        # 融合空间增强和通道增强后的特征，并升维输出
        x_out = self.up(res +x_fused_s * x_fused_c)
        return x_out

class EulerFeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels, phase_channels=32, hidden_channels=64):
        """
        基于欧拉公式的特征融合模块
        参数:
            in_channels: 输入张量的通道数
            out_channels: 输出张量的通道数
            phase_channels: 相位/振幅计算的通道数
            hidden_channels: 中间变换层的通道数
        """
        super().__init__()
        
        # 相位计算器 (输出相位图)
        self.horizontal_phase_calculator = nn.Conv2d(in_channels, phase_channels, kernel_size=3, padding=1)
        self.vertical_phase_calculator = nn.Conv2d(in_channels, phase_channels, kernel_size=3, padding=1)
        
        # 振幅计算器 (输出振幅图)
        self.horizontal_feature_conv = nn.Conv2d(in_channels, phase_channels, kernel_size=3, padding=1)
        self.vertical_feature_conv = nn.Conv2d(in_channels, phase_channels, kernel_size=3, padding=1)
        
        # 特征变换层
        self.vertical_transform_conv = nn.Sequential(
            nn.Conv2d(2 * phase_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.horizontal_transform_conv = nn.Sequential(
            nn.Conv2d(2 * phase_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.channel_feature_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.ReLU()
        )
        
        # 特征融合层 (处理拼接后的特征)
        self.feature_fusion_layer = nn.Sequential(
            nn.Conv2d(in_channels + 3 * hidden_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
    def forward(self, input_tensor): # 前向传播兩数
        horizontal_phase = self.horizontal_phase_calculator(input_tensor) # 计算水平方向的相们
        vertical_phase = self.vertical_phase_calculator(input_tensor) # 计算垂直方向的相位
        horizontal_amplitude = self.horizontal_feature_conv(input_tensor) # 提取水平方向的振幅
        vertical_amplitude = self.vertical_feature_conv(input_tensor) # 提取垂直方向的振幅
        #【创新点 欧拉公式的特征加权/融合】
        #欧拉公式推导:https://blog.csdn.net/m0 66890670/article/details/144183868
        horizontal_euler = torch.cat([horizontal_amplitude * torch.cos(horizontal_phase),
                                      horizontal_amplitude * torch.sin(horizontal_phase)], dim=1)
        #欧拉公式展开水平方向
        vertical_euler = torch.cat([vertical_amplitude * torch.cos(vertical_phase),
                                    vertical_amplitude * torch.sin(vertical_phase)],dim=1)
        original_input = input_tensor
        transformed_w = self.vertical_transform_conv(vertical_euler) # 处理垂直方向的特征 W
        transformed_c = self.channel_feature_conv(input_tensor) # 提取通道方向的特征 C
        transformed_h = self.horizontal_transform_conv(horizontal_euler)#处理水平方向的特征 H torch.size([10,64,32,32])
        merged_features = torch.cat([original_input,transformed_h,transformed_w,transformed_cl],dim=1)
        output_tensor = self.feature_fusion_layer(merged_features)
        #返回处理后的张量
        return output_tensor

    
    
    
class HeightWidthFeatureDepthwiseconv(nn.Module):
    def __init__(self,group_channels,square_kernel=3, band_kernel=11):
        super().__init__()
        #深度可分离卷积层，处理方形区域，使用group_channels个分组进行卷积
        self.square_depthwise_conv = nn.Conv2d(group_channels, group_channels, kernel_size=square_kernel,
                                               padding=square_kernel //2,groups=group_channels)
        # 深度可分离卷积层，处理水平 W 方向上的条形区域，使用group_channels个分组进行卷积
        self.horizontal_band_depthwise_conv = nn.Conv2d(group_channels, group_channels,kernel_size=(1, band_kernel),
                                                        padding=(0,band_kernel //2),groups=group_channels)
        # 深度可分离卷积层，处理竖直 H 方向上的条形区域，使用group_channels个分组进行卷积
        self.vertical_band_depthwise_conv = nn.Conv2d(group_channels, group_channels,kernel_size=(band_kernel,1),
                                                      padding=(band_kernel //2,0),groups=group_channels)
        self.convolution_batchnorm_activation=nn.Sequential(
            nn.Conv2d(group_channels *4,group_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(group_channels),
            # ReLU激活函数，inplace=True表示直接在输入数据上进行修改以节省内存
            nn.ReLU(inplace=True),
        )
        
    def forward(self, input_tensor):
        #保存原始输入张量
        original_input = input_tensor
        #对输入张量进行方形卷积操作
        square_convolved = self.square_depthwise_conv(input_tensor)
        #对输入张量进行水平条形卷枳操作
        horizontal_convolved = self.horizontal_band_depthwise_conv(input_tensor)
        #对输入张量进行垂直条形卷积操作
        vertical_convolved = self.vertical_band_depthwise_conv(input_tensor)
        #沿着通道维度(dim=1)将原始输入、方形卷积结果、水平条形卷积结果和垂直条形卷积结果拼接起来
        merged_features = torch.cat((original_input, square_convolved, horizontal_convolved,vertical_convolved),dim=1)
        output_tensor=self.convolution_batchnorm_activation(merged_features)
        return output_tensor

    
class VarianceAttentionModule(nn.Module):
    def __init__(self,eps: float = 1e-4):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.eps = eps
    def forward(self,x:torch.Tensor)-> torch.Tensor:
        B,C,H,W=x.shape
        #计算空间维度的方差【空间维度的方差平方】
        spatial_var =torch.var(x,dim=(-2,-1),keepdim=True).pow(2)
        #全局方差归一化因子【全局方差统计量】
        global_var_norm = spatial_var.sum(dim=[2,3],keepdim=True)/(H*W- 1)
        # 计算往意力系数
        attention_coef =(spatial_var /(4*(global_var_norm + self.eps)))+ 0.5
        #生成注意力权重
        attention_weight =self.sigmoid(attention_coef)
        # 应用注意力机制
        return x*attention weight
    
# 还用的sobel算子

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
    
    
    
    
    
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class RecursiveGatedAttention(nn.Module):
    """
    支持两种输入：
        - 3-D: [B, N, C]
        - 4-D: [B, C, H, W]
    输出与输入形状保持一致。
    """
    def __init__(self,
                 dim,
                 num_heads=8,
                 recurrent_steps=3,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = sqrt(self.head_dim)
        self.steps     = recurrent_steps

        # 共享的 QKV 映射
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 门控
        self.gate_z = nn.Linear(dim, dim)
        self.gate_r = nn.Linear(dim, dim)

    def forward(self, x):
        # ---------- 维度检查 ----------
        if x.dim() == 4:                    # [B, C, H, W]
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)   # -> [B, N, C]
            need_reshape = True
        elif x.dim() == 3:                  # [B, N, C]
            B, N, C = x.shape
            need_reshape = False
        else:
            raise ValueError("Input must be 3-D [B, N, C] or 4-D [B, C, H, W]")

        # ---------- 核心计算 ----------
        h = x
        for _ in range(self.steps):
            # 1. QKV
            qkv = self.qkv(h).reshape(B, -1, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, d_k]
            q, k, v = qkv[0], qkv[1], qkv[2]

            # 2. 注意力
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            out = (attn @ v).transpose(1, 2).reshape(B, -1, C)  # [B, N, C]

            # 3. 门控更新
            z = torch.sigmoid(self.gate_z(out))
            r = torch.sigmoid(self.gate_r(out))
            h = (1 - z) * h + z * torch.tanh(r * out)

        h = self.proj(h)
        h = self.proj_drop(h)

        # ---------- 如果需要还原 4-D ----------
        if need_reshape:
            h = h.transpose(1, 2).view(B, C, H, W)

        return h
    
    
    
# 这个居然涨点还达到76
class SimplifiedRecursiveAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # 门控生成器
        self.gate_gen = nn.Sequential(
            nn.Conv2d(channels // 2, channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, 1, 1),
            nn.Sigmoid()
        )
        
        # 最终融合
        self.final_conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        # 拆分特征
        base, gate_input = torch.split(x, self.channels // 2, dim=1)
        
        # 生成门控图
        gate = self.gate_gen(gate_input)
        
        # 应用门控
        enhanced_base = base * gate
        
        # 融合结果
        out = torch.cat([enhanced_base, gate_input], dim=1)
        return self.final_conv(out)
    
# 递归门控卷积    
class gnconv(nn.Module):
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2*dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)
        
        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i+1], 1) for i in range(order-1)]
        )

        self.scale = s

        print('[gconv]', order, 'order with dims=', self.dims, 'scale=%.4f'%self.scale)


    def forward(self, x, mask=None, dummy=False):
        B, C, H, W = x.shape

        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc) * self.scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]

        for i in range(self.order -1):
            x = self.pws[i](x) * dw_list[i+1]

        x = self.proj_out(x)

        return x

# Poly-Scale Convolution (PSConv)
# class PSConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, parts=4, bias=False):
#         super(PSConv2d, self).__init__()
#         self.gwconv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation, dilation, groups=parts, bias=bias)
#         self.gwconv_shift = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 2 * dilation, 2 * dilation, groups=parts, bias=bias)
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

#         def backward_hook(grad):
#             out = grad.clone()
#             out[self.mask] = 0
#             return out

#         self.mask = torch.zeros(self.conv.weight.shape).byte().cuda()
#         _in_channels = in_channels // parts
#         _out_channels = out_channels // parts
#         for i in range(parts):
#             self.mask[i * _out_channels: (i + 1) * _out_channels, i * _in_channels: (i + 1) * _in_channels, : , :] = 1
#             self.mask[(i + parts//2)%parts * _out_channels: ((i + parts//2)%parts + 1) * _out_channels, i * _in_channels: (i + 1) * _in_channels, :, :] = 1
#         self.conv.weight.data[self.mask] = 0
#         self.conv.weight.register_hook(backward_hook)

#     def forward(self, x):
#         x1, x2 = x.chunk(2, dim=1)
#         x_shift = self.gwconv_shift(torch.cat((x2, x1), dim=1))
#         return self.gwconv(x) + self.conv(x) + x_shift


# # PSConv-based Group Convolution
# class PSGConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, parts=4, bias=False):
#         super(PSGConv2d, self).__init__()
#         self.gwconv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation, dilation, groups=groups * parts, bias=bias)
#         self.gwconv_shift = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 2 * dilation, 2 * dilation, groups=groups * parts, bias=bias)
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)

#         def backward_hook(grad):
#             out = grad.clone()
#             out[self.mask] = 0
#             return out

#         self.mask = torch.zeros(self.conv.weight.shape).byte().cuda()
#         _in_channels = in_channels // (groups * parts)
#         _out_channels = out_channels // (groups * parts)
#         for i in range(parts):
#             for j in range(groups):
#                 self.mask[(i + j * groups) * _out_channels: (i + j * groups + 1) * _out_channels, i * _in_channels: (i + 1) * _in_channels, : , :] = 1
#                 self.mask[((i + parts // 2) % parts + j * groups) * _out_channels: ((i + parts // 2) % parts + j * groups + 1) * _out_channels, i * _in_channels: (i + 1) * _in_channels, :, :] = 1
#         self.conv.weight.data[self.mask] = 0
#         self.conv.weight.register_hook(backward_hook)
#         self.groups = groups

#     def forward(self, x):
#         x_split = (z.chunk(2, dim=1) for z in x.chunk(self.groups, dim=1))
#         x_merge = torch.cat(tuple(torch.cat((x2, x1), dim=1) for (x1, x2) in x_split), dim=1)
#         x_shift = self.gwconv_shift(x_merge)
#         return self.gwconv(x) + self.conv(x) + x_shift
    
    
    
# 风车卷积
# class PSConv(nn.Module):  
#     ''' Pinwheel-shaped Convolution using the Asymmetric Padding method. '''
    
#     def __init__(self, c1, c2, k, s):
#         super().__init__()

#         # self.k = k
#         p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
#         self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
#         self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
#         self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
#         self.cat = Conv(c2, c2, 2, s=1, p=0)

#     def forward(self, x):
#         yw0 = self.cw(self.pad[0](x))
#         yw1 = self.cw(self.pad[1](x))
#         yh0 = self.ch(self.pad[2](x))
#         yh1 = self.ch(self.pad[3](x))
#         return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))



# 小波，sobel算子，
class HalfConv(nn.Module):
    def __init__(self, dim, n_div=2):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
    def forward(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x
    
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
class GaussianAttention(nn.Module):
    def __init__(self, channels, kernel_size=5, sigma=1.0):
        super().__init__()
        # 配置归一化层参数
        norm_cfg = dict(type='BN', requires_grad=True)
        #生成高斯核并初始化卷积层
        gaussian_kernel = self.create_gaussian_kernel(kernel_size, sigma)
        gaussian_kernel = nn.Parameter(gaussian_kernel, reguires_grad=False).clone()
        #初始化分组卷积实现高斯滤波
        self.gaussian_filter = nn.Conv2d(
        channels, channels,
        kernel_size=kernel_size,
        padding=kernel_size //2,
        groups=channels,
        bias=False
        )
        self.gaussian_filter.weight.data = gaussian_kernel.repeat(channels, 1, 1, 1)

        self.norm = build_normalization(norm_cfg, channels)[1]
        self.activation = nn.GELU()
    def create_gaussian_kernel(self,kernel size, sigma):
        return torch.FloatTensor([
            [(1/(2 *math.pi * sigma ** 2))* math.exp(-(x** 2 +y** 2)/(2 * sigma ** 2))
             for x in range(-kernel_size //2 +1,kernel_size // 2 + 1)]
            for y in range(-kernel_size //2+1,kernel_size // 2 + 1)
        ]).unsqueeze(0).unsqueeze(0)

    def forward(self,x):
    """前向传播:生成注意力权重并与输入特征相乘"""
        filtered = self.gaussian_filter(x)
        attention= self.activation(self.norm(filtered))
        return x*attention




class ScharrConv(nn.Module):
    def __init__(self, channel):
        super(ScharrConv, self).__init__()
        
        # 定义Scharr算子的水平和垂直卷积核
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

        # 定义卷积层，但不学习卷积核，直接使用Scharr核
        self.scharr_kernel_x_conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.scharr_kernel_y_conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        
        # 将卷积核的权重设置为Scharr算子的核，权重初始化,就是自定义的卷积层的权重初始化为扩展后的卷积核，实例化为Scharr核
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
        # 计算边缘轻度
        edge_strength = torch.sqrt( grad_x ** 2 + grad_y ** 2)
        #生成边缘注意力权重
        edge_attention = self.activation(self.norm(edge_strength)
        # 应用注意力机制增强特征
        enhanced_feature =x*edge_attention
        # 计算梯度幅值
        edge_magnitude = grad_x * 0.5 + grad_y * 0.5
        
        return edge_magnitude
    
    
    
# 在L2/L1范数基础上添加动态归一化：结合归一化的方案，归一化后通过伽马校正（gamma=0.5）增强暗部细节。  适合低对比度图像。
edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2)  # L2范数
edge_magnitude = (edge_magnitude - edge_magnitude.min()) / (edge_magnitude.max() - edge_magnitude.min() + 1e-8)
edge_magnitude = torch.pow(edge_magnitude, 0.5)  # 伽马校正增强对比度




# 根据梯度方向对水平和垂直梯度加权，方向加权融合
weight_x = torch.abs(torch.cos(grad_dir))  # 水平方向权重
weight_y = torch.abs(torch.sin(grad_dir))  # 垂直方向权重
edge_magnitude = weight_x * grad_x + weight_y * grad_y


# 自适应阈值处理
threshold = torch.mean(edge_magnitude) * 0.5  # 动态阈值
edge_magnitude = (edge_magnitude > threshold).float()

# L1范数（绝对值之和）
edge_magnitude = torch.abs(grad_x) + torch.abs(grad_y)  # 计算L1范数