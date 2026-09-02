"""
reference     
- https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py  
    
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved. 
"""
 
import torch
import torch.nn as nn 
import torch.nn.functional as F
import os, re
from .common import FrozenBatchNorm2d  
from ..core import register     
from ..extre_module.custom_nn.attention.ema import EMA
from ..extre_module.custom_nn.attention.simam import SimAM   
from ..misc.dist_utils import Multiprocess_sync    
import logging     
   
from .hgnetv2 import HGNetv2, HG_Stage
   
from functools import partial 
from ..extre_module.custom_nn.module.starblock import Star_Block   

# Constants for initialization     
kaiming_normal_ = nn.init.kaiming_normal_
zeros_ = nn.init.zeros_     
ones_ = nn.init.ones_     
   
__all__ = ['HGNetv2_Star']    
    
class HG_Stage_Star(HG_Stage): 
    def __init__(self, in_chs, mid_chs, out_chs, block_num, layer_num, downsample=True, light_block=False, kernel_size=3, use_lab=False, agg='se', drop_path=0):    
        super().__init__(in_chs, mid_chs, out_chs, block_num, layer_num, downsample, light_block, kernel_size, use_lab, agg, drop_path)  

        # 2. 构建多个 HG_Block
        blocks_list = []
        for i in range(block_num):     
            blocks_list.append( 
                Star_Block(
                    in_chs if i == 0 else out_chs,  # 第一个 HG_Block 需要匹配输入通道
                    out_chs,   # 输出通道数   
                )
            ) 
        
        # 3. 将所有 HG_Block 组成一个顺序执行的模块
        self.blocks = nn.Sequential(*blocks_list)
  
@register()   
class HGNetv2_Star(HGNetv2):
    def __init__(self, name, use_lab=False, return_idx=..., freeze_stem_only=True, freeze_at=0, freeze_norm=True, pretrained=True, agg='se', local_model_dir='weight/hgnetv2/'):   
        super().__init__(name, use_lab, return_idx, freeze_stem_only, freeze_at, freeze_norm, pretrained, agg, local_model_dir)
    
        stage_config = self.arch_configs[name]['stage_config'] 

        # stages  
        self.stages = nn.ModuleList()    
        for i, k in enumerate(stage_config):
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num = stage_config[
                k] 
            self.stages.append(
                HG_Stage_Star( 
                    in_channels,
                    mid_channels,
                    out_channels,
                    block_num,
                    layer_num,   
                    downsample,   
                    light_block,
                    kernel_size,  
                    use_lab,  
                    agg))