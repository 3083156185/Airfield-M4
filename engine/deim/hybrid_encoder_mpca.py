"""  
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.   
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
""" 

import copy  
from collections import OrderedDict  

import torch 
import torch.nn as nn 
import torch.nn.functional as F
     
from .utils import get_activation 
     
from ..core import register    
from .hybrid_encoder import HybridEncoder
from engine.extre_module.custom_nn.featurefusion.mpca import MultiScalePCA, MultiScalePCA_Down

__all__ = ['HybridEncoder_MPCA']     

@register() 
class HybridEncoder_MPCA(HybridEncoder):
    def __init__(self, in_channels=..., feat_strides=..., hidden_dim=256, nhead=8, dim_feedforward=1024, dropout=0, enc_act='gelu', use_encoder_idx=..., num_encoder_layers=1, pe_temperature=10000, expansion=1, depth_mult=1, act='silu', eval_spatial_size=None, version='dfine'):
        super().__init__(in_channels, feat_strides, hidden_dim, nhead, dim_feedforward, dropout, enc_act, use_encoder_idx, num_encoder_layers, pe_temperature, expansion, depth_mult, act, eval_spatial_size, version)
        
        self.fpn_fusion_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):     
            self.fpn_fusion_blocks.append(MultiScalePCA([hidden_dim, hidden_dim], output_channel=hidden_dim * 2))

        self.pan_fusion_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.pan_fusion_blocks.append(MultiScalePCA_Down([hidden_dim, hidden_dim], output_channel=hidden_dim * 2))
   
        del self.downsample_convs   
   
    def forward(self, feats):
        """     
        前向传播函数
        Args:    
            feats (list[torch.Tensor]): 输入特征图列表，形状为 [B, C, H, W]，长度需与 in_channels 一致    
        Returns:
            list[torch.Tensor]: 融合后的多尺度特征图列表
        """
        # 检查输入特征图数量是否与预期一致
        assert len(feats) == len(self.in_channels) 
 
        # 输入投影：将所有特征图投影到 hidden_dim 通道    
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # Transformer 编码器：对指定层进行特征增强  
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):     
                h, w = proj_feats[enc_ind].shape[2:]  # 获取当前特征图的高度和宽度
                # 将特征图展平并调整维度：[B, C, H, W] -> [B, H*W, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)    
                # 根据训练或评估模式选择位置编码
                if self.training or self.eval_spatial_size is None:     
                    # 训练时动态生成位置编码
                    pos_embed = self.build_2d_sincos_position_embedding( 
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)  
                else:
                    # 评估时使用预计算的位置编码 
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)    

                # Transformer 编码器处理
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                # 将输出重塑回特征图形状：[B, H*W, C] -> [B, C, H, W]  
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()     

        # FPN 融合：自顶向下融合高层特征到低层特征 
        inner_outs = [proj_feats[-1]]  # 从最顶层特征图开始
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]         # 当前高层特征    
            feat_low = proj_feats[idx - 1]     # 当前低层特征
            # 横向卷积处理高层特征     
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)   
            inner_outs[0] = feat_heigh
            # 上采样高层特征
            # 拼接上采样后的高层特征和低层特征，并通过 FPN 块处理   
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                self.fpn_fusion_blocks[len(self.in_channels) - 1 - idx]([feat_low, feat_heigh]))
            inner_outs.insert(0, inner_out)  # 将融合结果插入列表开头

            # feat_heigh->P5 feat_low->P4
            # feat_heigh->Conv1x1->Upsample->P4
            # Concat(feat_heigh(P4), feat_low)->FPN_Blocks->Insert to inner_out 
    
        # PAN 融合：自底向上融合低层特征到高层特征   
        outs = [inner_outs[0]]  # 从最底层特征图开始
        for idx in range(len(self.in_channels) - 1):  
            feat_low = outs[-1]             # 当前低层特征
            feat_height = inner_outs[idx + 1]  # 当前高层特征   
            # 拼接下采样后的低层特征和高层特征，并通过 PAN 块处理     
            out = self.pan_blocks[idx](self.pan_fusion_blocks[idx]([feat_height, feat_low]))
            outs.append(out)  # 将融合结果添加到列表    
 
            # feat_low->P3 feat_height->P4
            # feat_low->DownSample->P4
            # Concat(feat_heigh, feat_low(P4))->PAN_Blocks->append to outs     
  
        # 返回融合后的多尺度特征图
        return outs   
