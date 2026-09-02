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

__all__ = ['DFINETransformer']     

 
class MLP(nn.Module):   
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):   
        super().__init__()     
        self.num_layers = num_layers     
        h = [hidden_dim] * (num_layers - 1)  
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))    
        self.act = get_activation(act)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x) 
        return x
  

class MSDeformableAttention(nn.Module):     
    def __init__(
        self,     
        embed_dim=256, 
        num_heads=8,    
        num_levels=4,
        num_points=4,
        method='default', 
        offset_scale=0.5,
    ):
        """Multi-Scale Deformable Attention
        """ 
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads  
        self.num_levels = num_levels
        self.offset_scale = offset_scale    
    
        if isinstance(num_points, list):
            assert len(num_points) == num_levels, ''     
            num_points_list = num_points     
        else: 
            num_points_list = [num_points for _ in range(num_levels)]  

        self.num_points_list = num_points_list
     
        num_points_scale = [1/n for n in num_points_list for _ in range(n)]    
        self.register_buffer('num_points_scale', torch.tensor(num_points_scale, dtype=torch.float32))
     
        self.total_points = num_heads * sum(num_points_list)    
        self.method = method  
     
        self.head_dim = embed_dim // num_heads 
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)  
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
 
        self.ms_deformable_attn_core = functools.partial(deformable_attention_core_func_v2, method=self.method)     
 
        self._reset_parameters()    
   
        if method == 'discrete':
            for p in self.sampling_offsets.parameters():   
                p.requires_grad = False     

    def _reset_parameters(self):  
        # sampling_offsets   
        init.constant_(self.sampling_offsets.weight, 0)   
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)  
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])     
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()
    
        # attention_weights
        init.constant_(self.attention_weights.weight, 0)    
        init.constant_(self.attention_weights.bias, 0)


    def forward(self,
                query: torch.Tensor,
                reference_points: torch.Tensor,   
                value: torch.Tensor,
                value_spatial_shapes: List[int]):  
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area  
            value (Tensor): [bs, value_length, C] 编码器的输出  
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]

        Returns:
            output (Tensor): [bs, Length_{query}, C]   
        """    
        bs, Len_q = query.shape[:2]     
 
        sampling_offsets: torch.Tensor = self.sampling_offsets(query) # 通过query得到采样点的offsets   
        sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.num_heads, sum(self.num_points_list), 2) # reshape至reference_points的shape

        attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list)) # 通过query生成注意力权重
        attention_weights = F.softmax(attention_weights, dim=-1) # 在采样点的数量维度做softmax

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)  
            sampling_locations = reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            # reference_points [8, 480, None, 1,  4]    
            # sampling_offsets [8, 480, 8,    12, 2] 
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)  
            offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset    
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))    

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,           # 模型的维度，默认256
                 n_head=8,             # 多头注意力机制的头数，默认8
                 dim_feedforward=1024, # 前馈网络的隐藏层维度，默认1024 
                 dropout=0.,          # dropout比例，默认0（无dropout）
                 activation='relu',   # 激活函数类型，默认ReLU
                 n_levels=4,          # 多尺度特征的层级数
                 n_points=4,          # 每个查询的参考点数
                 cross_attn_method='default', # 交叉注意力机制的方法
                 layer_scale=None):   # 层缩放因子，可选 
        super(TransformerDecoderLayer, self).__init__()
        
        # 如果指定了layer_scale，则调整模型维度和前馈网络维度
        if layer_scale is not None:     
            dim_feedforward = round(layer_scale * dim_feedforward)
            d_model = round(layer_scale * d_model)  
     
        # 自注意力机制部分
        # 使用PyTorch内置的多头注意力模块，batch_first=True表示输入格式为(batch, seq, feature) 
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)  
        self.dropout1 = nn.Dropout(dropout)  # 自注意力后的dropout层
        self.norm1 = nn.LayerNorm(d_model)   # 自注意力后的层归一化   

        # 交叉注意力机制部分
        # 使用自定义的MSDeformableAttention（多尺度可变形注意力） 
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, \
                                              method=cross_attn_method)   
        self.dropout2 = nn.Dropout(dropout)  # 交叉注意力后的dropout层 

        # 门控机制    
        # 用于控制信息流的Gate模块
        self.gateway = Gate(d_model)
    
        # 前馈神经网络（FFN）部分
        self.linear1 = nn.Linear(d_model, dim_feedforward)  # 第一层线性变换
        self.activation = get_activation(activation)       # 获取指定的激活函数
        self.dropout3 = nn.Dropout(dropout)                # 激活后的dropout
        self.linear2 = nn.Linear(dim_feedforward, d_model) # 第二层线性变换 
        self.dropout4 = nn.Dropout(dropout)                # FFN输出后的dropout   
        self.norm3 = nn.LayerNorm(d_model)                 # FFN后的层归一化
    
        # 初始化参数    
        self._reset_parameters()  

    def _reset_parameters(self):
        # 使用Xavier均匀分布初始化线性层的权重     
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)
 
    def with_pos_embed(self, tensor, pos):     
        # 如果提供了位置嵌入，则将其加到输入张量上，否则返回原张量
        return tensor if pos is None else tensor + pos
    
    def forward_ffn(self, tgt):
        # 前馈网络的前向传播
        # 线性变换 -> 激活函数 -> dropout -> 线性变换 
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
    
    def forward(self,     
                target,              # 目标查询（decoder的输入）    
                reference_points,    # 参考点，用于交叉注意力   
                value,              # 编码器的输出，作为交叉注意力的值  
                spatial_shapes,     # 空间形状信息，用于多尺度处理
                attn_mask=None,     # 自注意力掩码，可选
                query_pos_embed=None): # 查询的位置嵌入，可选

        # 自注意力部分     
        # q和k都加入位置嵌入（如果有），value直接使用target
        q = k = self.with_pos_embed(target, query_pos_embed) 
        # 执行多头自注意力计算，返回结果和注意力权重（权重在此处未使用，用_表示）
        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        # 残差连接并应用dropout     
        target = target + self.dropout1(target2)
        # 层归一化  
        target = self.norm1(target)
  
        # 交叉注意力部分
        # 使用多尺度可变形注意力机制
        target2 = self.cross_attn( 
            self.with_pos_embed(target, query_pos_embed), # 查询（带位置嵌入）
            reference_points,                             # 参考点   
            value,                                        # 编码器输出 
            spatial_shapes)                               # 空间形状 
     
        # 通过门控机制融合交叉注意力结果
        target = self.gateway(target, self.dropout2(target2))

        # 前馈网络部分 
        target2 = self.forward_ffn(target)
        # 残差连接并应用dropout    
        target = target + self.dropout4(target2)     
        # 层归一化，并限制输出范围（防止数值溢出）
        target = self.norm3(target.clamp(min=-65504, max=65504))
     
        return target  # 返回处理后的目标查询     
     
 
class Gate(nn.Module):   
    def __init__(self, d_model):
        super(Gate, self).__init__()     
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = bias_init_with_prob(0.5)    
        init.constant_(self.gate.bias, bias) 
        init.constant_(self.gate.weight, 0)
        self.norm = nn.LayerNorm(d_model)    

    def forward(self, x1, x2):
        gate_input = torch.cat([x1, x2], dim=-1)
        gates = torch.sigmoid(self.gate(gate_input))   
        gate1, gate2 = gates.chunk(2, dim=-1)     
        return self.norm(gate1 * x1 + gate2 * x2)
  
     
class Integral(nn.Module):
    """
    A static layer that calculates integral results from a distribution.
    
    This layer computes the target location using the formula: `sum{Pr(n) * W(n)}`,
    where Pr(n) is the softmax probability vector representing the discrete    
    distribution, and W(n) is the non-uniform Weighting Function.
    
    Args:
        reg_max (int): Max number of the discrete bins. Default is 32.   
                       It can be adjusted based on the dataset or task requirements.  
    """ 
  
    def __init__(self, reg_max=32):    
        super(Integral, self).__init__()
        self.reg_max = reg_max     
   
    def forward(self, x, project):
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)     
        x = F.linear(x, project.to(x.device)).reshape(-1, 4)  
        return x.reshape(list(shape[:-1]) + [-1])


class LQE(nn.Module):
    """    
    位置质量估计器 (Location Quality Estimator, LQE)
    用于评估和调整边界框预测的质量分数，结合分布统计信息提升精度
    """  
    def __init__(self, k, hidden_dim, num_layers, reg_max, act='relu'):
        """
        初始化 LQE 模块    
        参数:
            k: 前 k 个最高概率值的数量，用于统计分析
            hidden_dim: MLP隐藏层维度
            num_layers: MLP层数 
            reg_max: 回归的最大值（边界框分布的最大范围）
            act: 激活函数类型，默认为 'relu'     
        """   
        super(LQE, self).__init__()
        self.k = k 
        self.reg_max = reg_max
        # 定义一个多层感知机（MLP），输入维度为 4*(k+1)，输出为 1
        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers, act=act)
        # 初始化最后一层的偏置和权重为 0
        init.constant_(self.reg_conf.layers[-1].bias, 0)  
        init.constant_(self.reg_conf.layers[-1].weight, 0)     

    def forward(self, scores, pred_corners): 
        """
        前向传播    
        参数:
            scores: 初始分类得分 [B, L, num_classes]
            pred_corners: 预测的边界框角点分布 [B, L, 4*(reg_max+1)]    
        返回:
            调整后的质量分数 
        """
        B, L, _ = pred_corners.size()
        # 将预测的角点分布重塑为 [B, L, 4, reg_max+1]，并计算 softmax 概率
        prob = F.softmax(pred_corners.reshape(B, L, 4, self.reg_max+1), dim=-1)
        # 提取前 k 个最高概率值及其索引  
        prob_topk, _ = prob.topk(self.k, dim=-1)    
        # 将 top-k 概率及其均值拼接，作为统计特征     
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        # 通过 MLP 计算质量分数调整值
        quality_score = self.reg_conf(stat.reshape(B, L, -1)) 
        # 将初始得分与质量调整值相加
        return scores + quality_score     
     
     
class TransformerDecoder(nn.Module): 
    """
    Transformer 解码器，实现细粒度分布精炼 (Fine-grained Distribution Refinement, FDR)
    
    该解码器通过多层迭代更新，利用注意力机制、位置质量估计器和分布精炼技术， 
    提升目标检测中边界框的精度和鲁棒性。 
    """     
  
    def __init__(self, hidden_dim, decoder_layer, decoder_layer_wide, num_layers, num_head, reg_max, reg_scale, up, 
                 eval_idx=-1, layer_scale=2, act='relu'):    
        """  
        初始化 Transformer 解码器
        参数:     
            hidden_dim: 隐藏层维度
            decoder_layer: 标准解码器层对象
            decoder_layer_wide: 更宽的解码器层对象（用于后期层）   
            num_layers: 总层数
            num_head: 注意力头数
            reg_max: 回归最大值
            reg_scale: 回归缩放因子  
            up: 上采样因子
            eval_idx: 评估时使用的层索引，负值表示从后往前数
            layer_scale: 层缩放因子，用于调整宽层维度
            act: 激活函数类型，默认为 'relu'
        """
        super(TransformerDecoder, self).__init__()   
        self.hidden_dim = hidden_dim   
        self.num_layers = num_layers
        self.layer_scale = layer_scale 
        self.num_head = num_head
        # 如果 eval_idx < 0，则从末尾计算实际索引
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx     
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max    
        # 创建解码器层列表：前 eval_idx+1 层使用标准层，之后使用宽层  
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)] \
                    + [copy.deepcopy(decoder_layer_wide) for _ in range(num_layers - self.eval_idx - 1)])    
        # 为每层创建 LQE 模块 
        self.lqe_layers = nn.ModuleList([copy.deepcopy(LQE(4, 64, 2, reg_max, act=act)) for _ in range(num_layers)])

    def value_op(self, memory, value_proj, value_scale, memory_mask, memory_spatial_shapes):   
        """    
        预处理 MSDeformableAttention 的值（value）
        参数:   
            memory: 编码器输出
            value_proj: 值投影函数（可选）
            value_scale: 值缩放因子（可选）  
            memory_mask: 内存掩码（可选）
            memory_spatial_shapes: 空间形状信息     
        返回:
            处理后的值，按空间形状分割   
        """
        value = value_proj(memory) if value_proj is not None else memory
        # 如果指定了缩放因子，则进行插值调整     
        value = F.interpolate(memory, size=value_scale) if value_scale is not None else value  
        # 应用掩码（如果有）   
        if memory_mask is not None:    
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        # 重塑为 [B, C, num_head, -1]
        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)    
        # 按空间形状分割值 
        split_shape = [h * w for h, w in memory_spatial_shapes]     
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)  

    def convert_to_deploy(self):
        """    
        将模型转换为部署模式
        - 固定权重函数
        - 裁剪层至 eval_idx+1  
        - 调整 LQE 层，仅保留 eval_idx 层的实例    
        """
        self.project = weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True)
        self.layers = self.layers[:self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList([nn.Identity()] * (self.eval_idx) + [self.lqe_layers[self.eval_idx]])

    def forward(self,  
                target,             # 初始目标查询
                ref_points_unact,   # 未激活的参考点   
                memory,            # 编码器输出
                spatial_shapes,    # 空间形状信息    
                bbox_head,         # 边界框预测头列表 
                score_head,        # 分数预测头列表
                query_pos_head,    # 查询位置嵌入头
                pre_bbox_head,     # 预边界框预测头   
                integral,          # 积分函数
                up,                # 上采样因子    
                reg_scale,         # 回归缩放因子    
                attn_mask=None,    # 注意力掩码（可选）
                memory_mask=None,  # 内存掩码（可选）
                dn_meta=None):     # 去噪元数据（可选）  
        """     
        前向传播
        返回:
            边界框列表、分类得分列表、预测角点列表、参考点列表、预边界框、预得分
        """
        output = target
        output_detach = pred_corners_undetach = 0
        # 预处理值   
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)

        dec_out_bboxes = []       # 存储每层的边界框预测    
        dec_out_logits = []       # 存储每层的分类得分
        dec_out_pred_corners = [] # 存储每层的角点预测    
        dec_out_refs = []         # 存储每层的参考点   
        # 如果没有预定义 project，则动态生成权重函数   
        if not hasattr(self, 'project'):
            project = weighting_function(self.reg_max, up, reg_scale)
        else: 
            project = self.project 
  
        ref_points_detach = F.sigmoid(ref_points_unact)  # 激活初始参考点
 
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)  # 为参考点增加维度     
            query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)  # 计算查询位置嵌入
  
            # 如果进入宽层且需要缩放，则调整维度
            if i >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(query_pos_embed, scale_factor=self.layer_scale)  
                value = self.value_op(memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes)
                output = F.interpolate(output, size=query_pos_embed.shape[-1])    
                output_detach = output.detach()     
 
            # 通过解码器层更新 output   
            output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed)    
    
            if i == 0:  
                # 第一层：使用逆sigmoid精炼初始边界框预测
                pre_bboxes = F.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
                pre_scores = score_head[0](output)   
                ref_points_initial = pre_bboxes.detach()

            # 使用 FDR 精炼边界框角点，融合前一层的校正
            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach   
            inter_ref_bbox = distance2bbox(ref_points_initial, integral(pred_corners, project), reg_scale)  
   
            # 在训练模式或评估层时，计算得分和输出
            if self.training or i == self.eval_idx:
                scores = score_head[i](output)
                scores = self.lqe_layers[i](scores, pred_corners)  # 使用 LQE 调整得分    
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)     
 
                if not self.training:  # 推理模式下，仅计算到 eval_idx 层   
                    break

            pred_corners_undetach = pred_corners  
            ref_points_detach = inter_ref_bbox.detach()  
            output_detach = output.detach() 

        # 将结果堆叠为张量返回    
        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), \
               torch.stack(dec_out_pred_corners), torch.stack(dec_out_refs), pre_bboxes, pre_scores     
    
    
@register()
class DFINETransformer(nn.Module):   
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
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

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
                   'ref_points': out_refs[-1], 'up': self.up, 'reg_scale': self.reg_scale}  
        else:
            out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}

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
