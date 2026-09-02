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
from engine.extre_module.custom_nn.transformer.AdaptiveSparseSA import AdaptiveSparseSA  
   
__all__ = ['HybridEncoder_ASSA']  
    
@register()     
class HybridEncoder_ASSA(HybridEncoder):
    def __init__(self, in_channels=..., feat_strides=..., hidden_dim=256, nhead=8, dim_feedforward=1024, dropout=0, enc_act='gelu', use_encoder_idx=..., num_encoder_layers=1, pe_temperature=10000, expansion=1, depth_mult=1, act='silu', eval_spatial_size=None, version='dfine'):    
        super().__init__(in_channels, feat_strides, hidden_dim, nhead, dim_feedforward, dropout, enc_act, use_encoder_idx, num_encoder_layers, pe_temperature, expansion, depth_mult, act, eval_spatial_size, version)
  
        self.encoder = nn.ModuleList([ 
            AdaptiveSparseSA(hidden_dim, num_heads=nhead, sparseAtt=True) for _ in range(len(use_encoder_idx))
        ])    
 
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
                proj_feats[enc_ind] = self.encoder[i](proj_feats[enc_ind]) 
  
        # FPN 融合：自顶向下融合高层特征到低层特征
        inner_outs = [proj_feats[-1]]  # 从最顶层特征图开始 
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]         # 当前高层特征
            feat_low = proj_feats[idx - 1]     # 当前低层特征
            # 横向卷积处理高层特征     
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)    
            inner_outs[0] = feat_heigh    
            # 上采样高层特征
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            # 拼接上采样后的高层特征和低层特征，并通过 FPN 块处理
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)  # 将融合结果插入列表开头

            # feat_heigh->P5 feat_low->P4 
            # feat_heigh->Conv1x1->Upsample->P4
            # Concat(feat_heigh(P4), feat_low)->FPN_Blocks->Insert to inner_out
    
        # PAN 融合：自底向上融合低层特征到高层特征 
        outs = [inner_outs[0]]  # 从最底层特征图开始     
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]             # 当前低层特征     
            feat_height = inner_outs[idx + 1]  # 当前高层特征
            # 下采样低层特征
            downsample_feat = self.downsample_convs[idx](feat_low)    
            # 拼接下采样后的低层特征和高层特征，并通过 PAN 块处理  
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            outs.append(out)  # 将融合结果添加到列表
 
            # feat_low->P3 feat_height->P4
            # feat_low->DownSample->P4
            # Concat(feat_heigh, feat_low(P4))->PAN_Blocks->append to outs
    
        # 返回融合后的多尺度特征图     
        return outs