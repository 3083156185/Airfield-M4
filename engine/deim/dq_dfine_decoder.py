"""    
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.  
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)     
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.     
"""
 
import math    
import copy
import functools 
from collections import OrderedDict    

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init   
from typing import List
     
from .dfine_utils import weighting_function, distance2bbox 
from .denoising import get_contrastive_denoising_training_group
from .utils import deformable_attention_core_func_v2, get_activation, inverse_sigmoid    
from .utils import bias_init_with_prob   
from ..core import register   
    
from engine.extre_module.custom_nn.module.PKIBlock import PKIBlock   

from .hybrid_encoder import ConvNormLayer_fuse
from .dfine_decoder import Integral, MLP, TransformerDecoder, TransformerDecoderLayer     

__all__ = ['DQDFINETransformer'] 

class Conv_GN(nn.Module):     
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, gn=True, bias=False):    
        super(Conv_GN, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.gn = nn.GroupNorm(32, out_channel)
        self.relu = nn.ReLU() if relu else None
 
    def forward(self, x):
        x = self.conv(x)
        if self.gn is not None:
            x = self.gn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x    

class MultiScaleFeature(nn.Module):
    def __init__(self, channels=256, scale=3):  
        super(MultiScaleFeature, self).__init__()
        self.conv = nn.ModuleList([Conv_GN(channels, channels, kernel_size=3, stride=2, padding=1) for i in range(scale - 1)])    
        
    def forward(self, x):  
        x_out = [x]   
        for layer in self.conv:
            x_out.append(layer(x_out[-1]))    
        return x_out 
     
def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)  
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()  
    return outputs 
   
class Flatten(nn.Module):     
    def forward(self, x):  
        return x.flatten(1)
  
class ChannelGate(nn.Module):   
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):    
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels  
        self.mlp = nn.Sequential(
            Flatten(),    
            nn.Linear(gate_channels, gate_channels // reduction_ratio),     
            nn.ReLU(),  
            nn.Linear(gate_channels // reduction_ratio, gate_channels)   
            )
        self.pool_types = pool_types
        
    def forward(self, x):   
        channel_att_sum = None
        for pool_type in self.pool_types:     
            if pool_type=='avg':
                avg_pool = F.adaptive_avg_pool2d(x, output_size=(1, 1))   
                channel_att_raw = self.mlp( avg_pool )   
            elif pool_type=='max':    
                max_pool = F.adaptive_max_pool2d(x, output_size=(1, 1))    
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )     
            elif pool_type=='lse':
                # LSE pool only   
                lse_pool = logsumexp_2d(x)   
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw 

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)    
        return scale

class ChannelPool(nn.Module):
    def forward(self, x):    
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
 
class SpatialGate(nn.Module):
    def __init__(self):    
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool() 
        self.spatial = ConvNormLayer_fuse(2, 1, kernel_size, stride=1)     
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return scale  
     
class CGFE(nn.Module):   
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False, num_feature_levels=4):    
        super(CGFE, self).__init__()
        self.dim = gate_channels
        self.num_feat = num_feature_levels
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial   
        if not no_spatial:
            self.SpatialGate = SpatialGate()   
            
    def forward(self, x, memory, spatial_shapes):
        feats = []
        idx = 0    
        encoder_feat = memory.transpose(1, 2)   
        bs, c, hw = encoder_feat.shape     

        for i in range(self.num_feat):
            h, w = spatial_shapes[i]    
            feat = encoder_feat[:,:,idx:idx+h*w].view(bs, self.dim, h, w)
            c2 = self.SpatialGate(x[i])     
            feat = feat * c2
            c1 = self.ChannelGate(feat)
            feat = feat * c1       
            feat = feat.flatten(2).transpose(1, 2)   
            feats.append(feat)
            idx += h*w 
 
        x_out = torch.cat(feats, 1)
        return x_out
     
class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)   
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)  
 
    def forward(self, x):
        b, c, h, w = x.size() 
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w  
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2) 
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2)) 
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x) 
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))    
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w) 
        return weights.sigmoid()
   
class CGFE_EMA(nn.Module):
    def __init__(self, gate_channels, factor=16, num_feature_levels=4):
        super(CGFE_EMA, self).__init__()
        self.dim = gate_channels  
        self.num_feat = num_feature_levels
        self.factor = factor 
        self.Gate = EMA(gate_channels, factor=factor)  
            
    def forward(self, x, memory, spatial_shapes):
        feats = []
        idx = 0
        encoder_feat = memory.transpose(1, 2)
        bs, c, hw = encoder_feat.shape 

        for i in range(self.num_feat):
            h, w = spatial_shapes[i]
            feat = encoder_feat[:,:,idx:idx+h*w].view(bs * self.factor, -1, h, w)
            feat = feat * self.Gate(x[i])     
            feat = feat.reshape(bs, -1, h, w).flatten(2).transpose(1, 2)  
            feats.append(feat)    
            idx += h*w  
  
        x_out = torch.cat(feats, 1)
        return x_out

class h_sigmoid(nn.Module):  
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)    
  
    def forward(self, x):   
        return self.relu(x + 3) / 6

class h_swish(nn.Module): 
    def __init__(self, inplace=True):   
        super(h_swish, self).__init__() 
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):    
        return x * self.sigmoid(x)    

class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32): 
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))   
  
        mip = max(8, inp // reduction)   
     
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)    
        self.act = h_swish()    
 
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)  

    def forward(self, x): 
        identity = x     

        n, c, h, w = x.size()    
        x_h = self.pool_h(x)  
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  

        y = torch.cat([x_h, x_w], dim=2)  
        y = self.conv1(y)
        y = self.bn1(y)    
        y = self.act(y)     
  
        x_h, x_w = torch.split(y, [h, w], dim=2)  
        x_w = x_w.permute(0, 1, 3, 2) 
     
        a_h = self.conv_h(x_h).sigmoid()  
        a_w = self.conv_w(x_w).sigmoid()   
   
        return a_w * a_h    

class CGFE_CA1(nn.Module):   
    def __init__(self, gate_channels, factor=16, num_feature_levels=4):
        super(CGFE_CA1, self).__init__()
        self.dim = gate_channels
        self.num_feat = num_feature_levels    
        self.factor = factor
        self.Gate = CoordAtt(gate_channels, factor)
   
    def forward(self, x, memory, spatial_shapes):
        feats = []   
        idx = 0    
        encoder_feat = memory.transpose(1, 2)
        bs, c, hw = encoder_feat.shape    

        for i in range(self.num_feat):
            h, w = spatial_shapes[i]
            feat = encoder_feat[:,:,idx:idx+h*w].view(bs, c, h, w)
            feat = feat * self.Gate(x[i])     
            feat = feat.flatten(2).transpose(1, 2)
            feats.append(feat)
            idx += h*w
            
        x_out = torch.cat(feats, 1)     
        return x_out    

class CGFE_CA2(nn.Module):   
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False, num_feature_levels=4):  
        super(CGFE_CA2, self).__init__()     
        self.dim = gate_channels
        self.num_feat = num_feature_levels
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:    
            self.SpatialGate = CoordAtt(gate_channels, reduction_ratio)
  
    def forward(self, x, memory, spatial_shapes):
        feats = []
        idx = 0
        encoder_feat = memory.transpose(1, 2)
        bs, c, hw = encoder_feat.shape  
   
        for i in range(self.num_feat):
            h, w = spatial_shapes[i]
            feat = encoder_feat[:,:,idx:idx+h*w].view(bs, self.dim, h, w)
            c2 = self.SpatialGate(x[i])
            feat = feat * c2
            c1 = self.ChannelGate(feat)
            feat = feat * c1     
            feat = feat.flatten(2).transpose(1, 2)
            feats.append(feat)  
            idx += h*w
            
        x_out = torch.cat(feats, 1)
        return x_out     
    
class DensityGuidedAttention(nn.Module):
    """密度图作为Query，查询特征图中的重要信息"""  
    def __init__(self, d_model, num_heads=8):
        super().__init__() 
        self.d_model = d_model  
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    @staticmethod    
    def with_pos_embed(tensor, pos_embed):  
        return tensor if pos_embed is None else tensor + pos_embed
   
    def build_2d_sincos_position_embedding(self, w, h, embed_dim=256, temperature=10000.):
        """
        生成 2D sine-cosine 位置编码  
        Args:   
            w (int): 特征图宽度   
            h (int): 特征图高度  
            embed_dim (int): 嵌入维度，必须能被 4 整除  
            temperature (float): 温度参数，控制频率
        Returns:
            torch.Tensor: 位置编码张量，形状为 [1, w*h, embed_dim]
        """     
        # 创建宽度和高度的网格 
        grid_w = torch.arange(int(w), dtype=torch.float32)   
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')  # 生成 2D 网格    
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4  # 每个方向 (w, h) 的编码维度  
        # 计算频率因子
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)
 
        # 计算宽度和高度的 sin 和 cos 编码  
        out_w = grid_w.flatten()[..., None] @ omega[None]  # [w*h, pos_dim]     
        out_h = grid_h.flatten()[..., None] @ omega[None]  # [w*h, pos_dim]
    
        # 拼接 sin 和 cos 编码，形成最终的位置编码    
        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feature_map, density_map):  
        """
        Args:
            feature_map: (B, H*W, C_feat) 或 (B, C_feat, H, W)     
            density_map: (B, H*W, C_dens) 或 (B, C_dens, H, W)   
        """     
        # 展平空间维度   
        if feature_map.dim() == 4:     
            B, C_feat, H, W = feature_map.shape
            feature_map = feature_map.flatten(2).transpose(1, 2)  # (B, H*W, C_feat)    
        if density_map.dim() == 4:     
            B, C_dens, H, W = density_map.shape
            density_map = density_map.flatten(2).transpose(1, 2)  # (B, H*W, C_dens)
        
        pos_embed = self.build_2d_sincos_position_embedding(W, H, embed_dim=self.d_model).to(density_map.device) 
        # Self-Attention
        output, _ = self.attn(self.with_pos_embed(density_map, pos_embed), self.with_pos_embed(feature_map, pos_embed), feature_map)  # (B, H*W, d_model)   
        
        return output
 
class CGFE_DGA(nn.Module):    
    def __init__(self, gate_channels, num_feature_levels=4):   
        super(CGFE_DGA, self).__init__()     
        self.dim = gate_channels     
        self.num_feat = num_feature_levels
 
        self.DGA = nn.ModuleList([DensityGuidedAttention(gate_channels) for _ in range(num_feature_levels)])
  
    def forward(self, x, memory, spatial_shapes): 
        """   
        Args: 
            x: (B, C_dens, H, W) 密度图     
            memory: (B, hw, C) 特征图
        """   
        feats = []
        idx = 0 
   
        for i in range(self.num_feat):
            h, w = spatial_shapes[i]
            feats.append(self.DGA[i](memory[:, idx:idx+h*w,:,], x[i])) 
            idx += h * w
    
        x_out = torch.cat(feats, 1) 
        return x_out
   
def make_layers(cfg, in_channels=3, batch_norm=False, d_rate=1):  
    layers = []
    for v in cfg:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)] 
            else:    
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers) 

class CategoricalCounting(nn.Module):
    def __init__(self, dim, cls_num=4):
        super(CategoricalCounting, self).__init__()  
        self.dim = dim 
        self.in_channels = self.dim * 2 
        self.ccm_cfg = [self.in_channels, self.dim]    
        self.conv1 = nn.Conv2d(dim, self.in_channels, kernel_size=1)   
        self.ccm = make_layers(self.ccm_cfg, in_channels=self.in_channels, d_rate=2)  
        self.output = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(dim, cls_num)
        
    def forward(self, features, spatial_shapes=None): 
        features = features.transpose(1, 2)
        bs, c, hw = features.shape
        h, w = spatial_shapes[0][0], spatial_shapes[0][1]
   
        v_feat = features[:,:,0:h*w].view(bs, self.dim, h, w) 
        x = self.conv1(v_feat)     
        x = self.ccm(x)       
        out = self.output(x)
        out = out.squeeze(3) 
        out = out.squeeze(2)
        out = self.linear(out)

        return out, x  

class LightCategoricalCounting(nn.Module): 
    def __init__(self, dim, cls_num=4):
        super(LightCategoricalCounting, self).__init__()
        self.dim = dim
        self.ccm_cfg = [self.dim, self.dim]
        self.conv1 = nn.Conv2d(self.dim, self.dim, kernel_size=1)        
        self.ccm = make_layers(self.ccm_cfg, in_channels=self.dim, d_rate=2)     
        self.output = nn.AdaptiveAvgPool2d(output_size=(1, 1))  
        self.linear = nn.Linear(self.dim, cls_num)    
        
    def forward(self, features, spatial_shapes=None):     
        features = features.transpose(1, 2)     
        bs, c, hw = features.shape
        h, w = spatial_shapes[0][0], spatial_shapes[0][1]

        v_feat = features[:,:,0:h*w].view(bs, self.dim, h, w)   
        x = self.conv1(v_feat)
        x = self.ccm(x)       
        out = self.output(x)
        out = out.squeeze(3)
        out = out.squeeze(2)     
        out = self.linear(out)
     
        return out, x  
 
class LightCategoricalCounting_PKI(nn.Module):
    def __init__(self, dim, cls_num=4):    
        super(LightCategoricalCounting_PKI, self).__init__()   
        self.dim = dim    
        self.ccm_cfg = [self.dim, self.dim]  
        self.conv1 = nn.Conv2d(self.dim, self.dim, kernel_size=1)    
        self.ccm = PKIBlock(self.dim, self.dim)  
        self.output = nn.AdaptiveAvgPool2d(output_size=(1, 1)) 
        self.linear = nn.Linear(self.dim, cls_num)
    
    def forward(self, features, spatial_shapes=None): 
        features = features.transpose(1, 2)     
        bs, c, hw = features.shape
        h, w = spatial_shapes[0][0], spatial_shapes[0][1]
   
        v_feat = features[:,:,0:h*w].view(bs, self.dim, h, w)    
        x = self.conv1(v_feat)     
        x = self.ccm(x) 
        out = self.output(x)
        out = out.squeeze(3)    
        out = out.squeeze(2)    
        out = self.linear(out)   

        return out, x 

@register()
class DQDFINETransformer(nn.Module):
    # 定义共享参数，这些参数可能在其他地方被引用
    __share__ = ['num_classes', 'eval_spatial_size']   
     
    def __init__(self,
                 num_classes=80,              # 类别数量，默认为80（例如COCO数据集的类别数）
                 hidden_dim=256,              # Transformer隐藏层的维度 
                 num_queries=300,             # 查询（query）的数量，即模型预测的最大目标数   
                 feat_channels=[512, 1024, 2048],  # 输入特征图的通道数
                 feat_strides=[8, 16, 32],    # 特征图相对于输入图像的步幅
                 num_levels=3,                # 多尺度特征的层数     
                 num_points=4,                # 每个查询点的数量（用于采样）
                 nhead=8,                     # Transformer中多头注意力的头数     
                 num_layers=6,                # Transformer解码器层数   
                 dim_feedforward=1024,        # 前馈网络的隐藏层维度
                 dropout=0.,                  # Dropout比率，防止过拟合   
                 activation="relu",           # 激活函数类型  
                 num_denoising=100,           # 去噪训练的查询数量
                 label_noise_ratio=0.5,       # 标签噪声比例，用于去噪训练   
                 box_noise_scale=1.0,         # 边界框噪声比例，用于去噪训练 
                 learn_query_content=False,   # 是否学习查询内容嵌入    
                 eval_spatial_size=None,      # 评估时的空间分辨率  
                 eval_idx=-1,                 # 评估时使用的解码器层索引，负数表示从最后一层计数  
                 eps=1e-2,                    # 小值阈值，用于边界框的有效性检查   
                 aux_loss=True,               # 是否使用辅助损失 
                 cross_attn_method='default', # 交叉注意力机制类型
                 query_select_method='default', # 查询选择方法     
                 reg_max=32,                  # 回归最大值，用于边界框回归     
                 reg_scale=4.,                # 回归缩放因子
                 layer_scale=1,               # 层缩放因子，用于调整隐藏层维度
                 mlp_act='relu',              # MLP激活函数类型     
                 dynamic_query_list = None,   # Dynamic-Query-List
                 ccm_cls_num = 4,             # Dynamic-Query-List-Length
                 using_dynamic_query = True,  # 是否使用Dynamic-Query思想
                 ): 
        super().__init__()  
        # 参数校验，确保输入特征通道数不超过多尺度层数
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)

        # 如果特征步幅数量不足，自动扩展到num_levels层
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)     

        # 初始化核心参数    
        self.hidden_dim = hidden_dim  
        scaled_dim = round(layer_scale * hidden_dim)  # 根据层缩放调整隐藏维度
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels   
        self.num_classes = num_classes 
        self.num_queries = num_queries     
        self.eps = eps
        self.num_layers = num_layers 
        self.eval_spatial_size = eval_spatial_size   
        self.aux_loss = aux_loss
        self.reg_max = reg_max
 
        # 校验查询选择和交叉注意力方法的有效性
        assert query_select_method in ('default', 'one2many', 'agnostic'), '查询选择方法无效'
        assert cross_attn_method in ('default', 'discrete'), '交叉注意力方法无效'
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method    

        # 构建输入投影层，将主干网络特征投影到hidden_dim维度
        self._build_input_proj_layer(feat_channels)

        # 定义Transformer模块的参数
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)  # 上采样因子，固定为0.5  
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)  # 回归缩放参数

        # 定义解码器层     
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, \
            activation, num_levels, num_points, cross_attn_method=cross_attn_method)  
        decoder_layer_wide = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, \
            activation, num_levels, num_points, cross_attn_method=cross_attn_method, layer_scale=layer_scale)    
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, decoder_layer_wide, num_layers, nhead,   
                                          reg_max, self.reg_scale, self.up, eval_idx, layer_scale, act=activation)   
 
        # 去噪训练相关参数     
        self.num_denoising = num_denoising     
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            # 为去噪训练创建类别嵌入，+1表示包括背景类
            self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])  # 初始化类别嵌入权重（除背景类）

        # 解码器嵌入
        self.learn_query_content = learn_query_content 
        if learn_query_content:     
            # 如果学习查询内容，则创建可学习的查询嵌入
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2, act=mlp_act)  # 查询位置的MLP 

        # 编码器输出层
        self.enc_output = nn.Sequential(OrderedDict([     
            ('proj', nn.Linear(hidden_dim, hidden_dim)),  # 线性投影 
            ('norm', nn.LayerNorm(hidden_dim)),          # 层归一化     
        ]))   
    
        # 根据查询选择方法定义得分头
        if query_select_method == 'agnostic':  
            self.enc_score_head = nn.Linear(hidden_dim, 1)  # 类无关得分     
        else:
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)  # 类别相关得分

        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)  # 边界框预测MLP     

        # 解码器头
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx  # 计算评估层索引
        # 类别得分预测头，根据层数和缩放维度分段定义     
        self.dec_score_head = nn.ModuleList(   
            [nn.Linear(hidden_dim, num_classes) for _ in range(self.eval_idx + 1)]
          + [nn.Linear(scaled_dim, num_classes) for _ in range(num_layers - self.eval_idx - 1)])
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)  # 预边界框预测
        # 边界框回归头，输出4*(reg_max+1)表示分布回归   
        self.dec_bbox_head = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, 4 * (self.reg_max + 1), 3, act=mlp_act) for _ in range(self.eval_idx + 1)]
          + [MLP(scaled_dim, scaled_dim, 4 * (self.reg_max + 1), 3, act=mlp_act) for _ in range(num_layers - self.eval_idx - 1)]) 
        self.integral = Integral(self.reg_max)  # 积分模块，用于将分布转换为边界框坐标

        # 初始化评估时的锚点和有效掩码 
        if self.eval_spatial_size:    
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer('anchors', anchors)  # 注册锚点为缓冲区
            self.register_buffer('valid_mask', valid_mask)  # 注册有效掩码   

        # 重置参数   
        self._reset_parameters(feat_channels)

        # Dynamic-Query  
        self.using_dynamic_query = using_dynamic_query  
        self.ccm_cls_num = ccm_cls_num
        self.dynamic_query_list = dynamic_query_list
        self.CCM = CategoricalCounting(self.hidden_dim, cls_num=self.ccm_cls_num)    
        # self.CCM = LightCategoricalCounting(self.hidden_dim, cls_num=self.ccm_cls_num)     
        # self.CCM = LightCategoricalCounting_PKI(self.hidden_dim, cls_num=self.ccm_cls_num)   
        self.CGFE = CGFE(gate_channels=self.hidden_dim, reduction_ratio=16, num_feature_levels=self.num_levels)  
        # self.CGFE = CGFE_EMA(gate_channels=self.hidden_dim, num_feature_levels=self.num_levels)
        # self.CGFE = CGFE_CA2(gate_channels=self.hidden_dim, num_feature_levels=self.num_levels) 
        self.multiscale = MultiScaleFeature(self.hidden_dim, scale=self.num_levels)

    def convert_to_deploy(self):  
        # 将模型转换为部署模式，仅保留评估层的预测头 
        self.dec_score_head = nn.ModuleList([nn.Identity()] * (self.eval_idx) + [self.dec_score_head[self.eval_idx]])  
        self.dec_bbox_head = nn.ModuleList(     
            [self.dec_bbox_head[i] if i <= self.eval_idx else nn.Identity() for i in range(len(self.dec_bbox_head))]
        )

    def _reset_parameters(self, feat_channels):
        # 参数初始化
        bias = bias_init_with_prob(0.01)  # 初始化偏置，假设函数返回一个偏置值  
        init.constant_(self.enc_score_head.bias, bias)  # 初始化编码器得分头的偏置
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)  # 初始化边界框头的权重    
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)  # 初始化边界框头的偏置
   
        init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        init.constant_(self.pre_bbox_head.layers[-1].bias, 0)
   
        # 初始化解码器得分头和边界框头的偏置和权重  
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):    
            init.constant_(cls_.bias, bias) 
            if hasattr(reg_, 'layers'):    
                init.constant_(reg_.layers[-1].weight, 0)   
                init.constant_(reg_.layers[-1].bias, 0)
     
        init.xavier_uniform_(self.enc_output[0].weight)  # Xavier初始化编码器输出投影权重 
        if self.learn_query_content:     
            init.xavier_uniform_(self.tgt_embed.weight)  # 初始化查询嵌入权重
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)  # 初始化查询位置MLP权重
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)   
        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim:  
                init.xavier_uniform_(m[0].weight)  # 初始化输入投影层的权重

    def _build_input_proj_layer(self, feat_channels):
        # 构建输入投影层，将不同通道数的特征投影到hidden_dim
        self.input_proj = nn.ModuleList() 
        for in_channels in feat_channels:   
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())  # 如果通道数匹配，直接使用恒等映射
            else:
                self.input_proj.append(
                    nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),  # 1x1卷积     
                        ('norm', nn.BatchNorm2d(self.hidden_dim))])  # 批归一化   
                    )     
                )

        in_channels = feat_channels[-1]
        # 为剩余的特征层添加投影层
        for _ in range(self.num_levels - len(feat_channels)):
            if in_channels == self.hidden_dim: 
                self.input_proj.append(nn.Identity())
            else: 
                self.input_proj.append(
                    nn.Sequential(OrderedDict([ 
                        ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),  # 3x3卷积，下采样
                        ('norm', nn.BatchNorm2d(self.hidden_dim))])
                    ) 
                )    
                in_channels = self.hidden_dim    

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        # 获取编码器输入，将特征图投影并展平     
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]     
        if self.num_levels > len(proj_feats):   
            len_srcs = len(proj_feats)  
            for i in range(len_srcs, self.num_levels): 
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:  
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))    
   
        # 展平特征并记录空间形状 
        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape   
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))  # [b, c, h, w] -> [b, h*w, c]    
            spatial_shapes.append([h, w])  # 记录每层的空间分辨率
     
        feat_flatten = torch.concat(feat_flatten, 1)  # 拼接所有层特征 
        return feat_flatten, spatial_shapes   

    def _generate_anchors(self,
                          spatial_shapes=None,     
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):  
        # 生成锚点和有效掩码
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:  
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])
  
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes): 
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')  # 生成网格坐标  
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)    
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)  # 归一化到[0,1]
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)  # 根据层级缩放锚点大小   
            lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)  # 拼接中心点和宽高
            anchors.append(lvl_anchors)

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)  # 检查锚点是否有效     
        anchors = torch.log(anchors / (1 - anchors))  # 将锚点转换为logit形式(数值稳定性：避免边界值的梯度消失)     
        anchors = torch.where(valid_mask, anchors, torch.inf)  # 无效锚点置为无穷大 

        return anchors, valid_mask 

    def _get_decoder_input(self,  
                           memory: torch.Tensor,     
                           spatial_shapes,
                           denoising_logits=None,   
                           denoising_bbox_unact=None):    
        # 准备解码器输入   
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)     
        else:
            anchors = self.anchors     
            valid_mask = self.valid_mask   
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)  # 为batch扩展锚点

        memory = valid_mask.to(memory.dtype) * memory  # 应用有效掩码 

        output_memory: torch.Tensor = self.enc_output(memory)  # 编码器输出
        enc_outputs_logits: torch.Tensor = self.enc_score_head(output_memory)  # 计算得分 
  
        # 选择top-k查询 
        enc_topk_memory, enc_topk_logits, enc_topk_anchors = \
            self._select_topk(output_memory, enc_outputs_logits, anchors, self.num_queries)

        enc_topk_bbox_unact: torch.Tensor = self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors  # 预测边界框    

        # 如果是训练阶段，记录编码器输出
        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        if self.training:   
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits) 
     
        # 获取查询内容     
        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])  # 可学习嵌入  
        else:    
            content = enc_topk_memory.detach()  # 使用编码器输出     

        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()

        # 如果有去噪输入，拼接去噪和正常查询    
        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)   
            content = torch.concat([denoising_logits, content], dim=1)
     
        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list, enc_outputs_logits

    def _select_topk(self, memory: torch.Tensor, outputs_logits: torch.Tensor, outputs_anchors_unact: torch.Tensor, topk: int):
        # outputs_logits.size() [bs, token_len, classes]
        # memory.size() [bs, token_len, seq_len(128)]     
        # outputs_anchors_unact.size() [bs, token_len, 4]  
        # 根据查询选择方法选择top-k查询
        if self.query_select_method == 'default':
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)
        elif self.query_select_method == 'one2many':    
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes  
        elif self.query_select_method == 'agnostic':
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)
 
        topk_ind: torch.Tensor # [bs, topk]

        # 提取top-k对应的锚点、得分和记忆
        topk_anchors = outputs_anchors_unact.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_anchors_unact.shape[-1]))
        topk_logits = outputs_logits.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1])) if self.training else None
        topk_memory = memory.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))

        return topk_memory, topk_logits, topk_anchors

    def forward(self, feats, targets=None):     
        # 前向传播   
        memory, spatial_shapes = self._get_encoder_input(feats)  # 获取编码器输入

        counting_output, ccm_feature = self.CCM(memory, spatial_shapes)
        multi_ccm_feature = self.multiscale(ccm_feature)
        memory = self.CGFE(multi_ccm_feature, memory, spatial_shapes)   
        num_queries_list = None    
        if self.using_dynamic_query: 
            _, predicted = torch.max(counting_output.data, 1)  
            num_queries_list = [self.dynamic_query_list[i] for i in predicted.tolist()] 
            num_select = self.dynamic_query_list[max(predicted.tolist())] 
            self.num_queries = num_select
    
            if targets and max([len(t['labels']) for t in targets]) == 0: 
                self.num_queries = min(self.dynamic_query_list) 
        # self.num_queries = 300
 
        # 准备去噪训练数据
        if self.training and self.num_denoising > 0: 
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, \
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,    
                    box_noise_scale=1.0,  
                    )   
            if memory.size(0) > 1 and self.using_dynamic_query: # bs等于1的时候就不需要处理attn_mask
                if attn_mask is not None:
                    attn_mask = attn_mask.unsqueeze(0).repeat(memory.size(0) * self.nhead, 1, 1) # 每个样本不同的attn_mask，需要扩展到三维
                    for i, qn in enumerate(num_queries_list):
                        # 假设这个batch最大的query是1500，当这个batch的其中一个样本为900的时候，前900个不能看到后600个查询
                        attn_mask[i * self.nhead:(i + 1) * self.nhead, dn_meta['dn_num_split'][0] + qn:, :dn_meta['dn_num_split'][0] + qn] = True 
                else:    
                    self.init_attn_mask(memory.size(0), num_queries_list, memory.device)  
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None
            if memory.size(0) > 1 and self.using_dynamic_query: # bs等于1的时候就不需要处理attn_mask 
                self.init_attn_mask(memory.size(0), num_queries_list, memory.device)

        # 获取解码器输入 
        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list, enc_outputs_logits = \
            self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact)   
   
        # 解码器前向传播
        out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits = self.decoder(    
            init_ref_contents,    
            init_ref_points_unact, 
            memory,
            spatial_shapes,     
            self.dec_bbox_head, 
            self.dec_score_head,
            self.query_pos_head,     
            self.pre_bbox_head, 
            self.integral,
            self.up,
            self.reg_scale,
            attn_mask=attn_mask,
            dn_meta=dn_meta)
    
        # 如果有去噪训练，分割去噪和正常输出  
        if self.training and dn_meta is not None:
            dn_pre_logits, pre_logits = torch.split(pre_logits, dn_meta['dn_num_split'], dim=1)     
            dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, dn_meta['dn_num_split'], dim=1)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)    
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)     
            dn_out_corners, out_corners = torch.split(out_corners, dn_meta['dn_num_split'], dim=2)
            dn_out_refs, out_refs = torch.split(out_refs, dn_meta['dn_num_split'], dim=2)   
  
        # 构造输出字典 
        if self.training:   
            out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1], 'pred_corners': out_corners[-1],
                   'ref_points': out_refs[-1], 'up': self.up, 'reg_scale': self.reg_scale, 'pred_bbox_number': counting_output,
                   'num_queries_list': num_queries_list}
        else:  
            out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1], 'num_queries_list': num_queries_list}  

        # 如果是训练阶段且使用辅助损失，添加辅助输出 
        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss2(out_logits[:-1], out_bboxes[:-1], out_corners[:-1], out_refs[:-1],
                                                     out_corners[-1], out_logits[-1])    
            out['enc_aux_outputs'] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list) 
            out['pre_outputs'] = {'pred_logits': pre_logits, 'pred_boxes': pre_bboxes}   
            out['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}

            if dn_meta is not None:     
                out['dn_outputs'] = self._set_aux_loss2(dn_out_logits, dn_out_bboxes, dn_out_corners, dn_out_refs, 
                                                        dn_out_corners[-1], dn_out_logits[-1])    
                out['dn_pre_outputs'] = {'pred_logits': dn_pre_logits, 'pred_boxes': dn_pre_bboxes}  
                out['dn_meta'] = dn_meta 
   
            out['enc_outputs_logits'] = enc_outputs_logits 

        return out 
   

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):   
        # this is a workaround to make torchscript happy, as torchscript   
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.   
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class, outputs_coord)]   


    @torch.jit.unused
    def _set_aux_loss2(self, outputs_class, outputs_coord, outputs_corners, outputs_ref,
                       teacher_corners=None, teacher_logits=None):
        # this is a workaround to make torchscript happy, as torchscript 
        # doesn't support dictionary with non-homogeneous values, such   
        # as a dict having both a Tensor and a list. 
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_corners': c, 'ref_points': d,  
                     'teacher_corners': teacher_corners, 'teacher_logits': teacher_logits}   
                for a, b, c, d in zip(outputs_class, outputs_coord, outputs_corners, outputs_ref)]  
     
    def init_attn_mask(self, bs, num_queries_list, device): 
        attn_mask = torch.full([bs * self.nhead,self.num_queries, self.num_queries], False, dtype=torch.bool, device=device)
        for i, qn in enumerate(num_queries_list):
            # 假设这个batch最大的query是1500，当这个batch的其中一个样本为900的时候，前900个不能看到后600个查询
            attn_mask[i * self.nhead:(i + 1) * self.nhead, qn:, :qn] = True 
        return attn_mask   
