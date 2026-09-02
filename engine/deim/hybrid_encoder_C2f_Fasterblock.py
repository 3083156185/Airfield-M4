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
   
from functools import partial
from ..extre_module.custom_nn.module.fasterblock import Faster_Block
from ..extre_module.ultralytics_nn.block import C2f_Block

__all__ = ['HybridEncoder_C2f_FasterBlock']
 
@register()   
class HybridEncoder_C2f_FasterBlock(HybridEncoder):
    def __init__(self, in_channels=..., feat_strides=..., hidden_dim=256, nhead=8, dim_feedforward=1024, dropout=0, enc_act='gelu', use_encoder_idx=..., num_encoder_layers=1, pe_temperature=10000, expansion=1, depth_mult=1, act='silu', eval_spatial_size=None, version='dfine'): 
        super().__init__(in_channels, feat_strides, hidden_dim, nhead, dim_feedforward, dropout, enc_act, use_encoder_idx, num_encoder_layers, pe_temperature, expansion, depth_mult, act, eval_spatial_size, version)

        self.fpn_blocks = nn.ModuleList()     # FPN 融合块   
        for _ in range(len(in_channels) - 1, 0, -1):  # 从高层到低层遍历    
            # FPN 块，融合上采样后的高层特征和低层特征
            self.fpn_blocks.append(  
                C2f_Block(hidden_dim * 2, hidden_dim, partial(Faster_Block), n=round(3 * depth_mult))
            )
        
        self.pan_blocks = nn.ModuleList()        # PAN 融合块
        for _ in range(len(in_channels) - 1):    # 从低层到高层遍历     
            # PAN 块，融合下采样后的低层特征和高层特征    
            self.pan_blocks.append(
                C2f_Block(hidden_dim * 2, hidden_dim, partial(Faster_Block), n=round(3 * depth_mult))    
            )     
