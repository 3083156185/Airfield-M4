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
from .hgnetv2 import HG_Block, HG_Stage, HGNetv2    

from ..extre_module.custom_nn.conv_module.DilatedReparamConv import DilatedReparamConv
     
# Constants for initialization  
kaiming_normal_ = nn.init.kaiming_normal_
zeros_ = nn.init.zeros_
ones_ = nn.init.ones_

__all__ = ['HGNetv2_DRC']
 
class HG_Block_DRC(HG_Block):   
    def __init__(self, in_chs, mid_chs, out_chs, layer_num, kernel_size=3, residual=False, light_block=False, use_lab=False, agg='ese', drop_path=0, drc_k=11):
        super().__init__(in_chs, mid_chs, out_chs, layer_num, kernel_size, residual, light_block, use_lab, agg, drop_path)  
  
        self.layers = nn.ModuleList() 
        for i in range(layer_num):
            self.layers.append(DilatedReparamConv(in_chs if i == 0 else mid_chs,
                                                  mid_chs, 
                                                  kernel_size=drc_k))  

class HG_Stage_DRC(HG_Stage): 
    def __init__(self, in_chs, mid_chs, out_chs, block_num, layer_num, downsample=True, light_block=False, kernel_size=3, use_lab=False, agg='se', drop_path=0, drc_k=11): 
        super().__init__(in_chs, mid_chs, out_chs, block_num, layer_num, downsample, light_block, kernel_size, use_lab, agg, drop_path) 
  
        # 2. 构建多个 HG_Block 
        blocks_list = []  
        for i in range(block_num):
            blocks_list.append(
                HG_Block_DRC(  
                    in_chs if i == 0 else out_chs,  # 第一个 HG_Block 需要匹配输入通道    
                    mid_chs,   # 中间层通道数   
                    out_chs,   # 输出通道数
                    layer_num, # 该 HG_Block 内的层数   
                    residual=False if i == 0 else True, # 只有第一个块不使用残差连接
                    kernel_size=kernel_size, # 卷积核大小 
                    light_block=light_block, # 是否使用轻量级卷积块（深度可分离卷积）
                    use_lab=use_lab, # 是否使用可学习的仿射变换模块
                    agg=agg, # 特征聚合方式 ('se' 或 'ese' 或 'ema' 或 'simam')     
                    drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path, # DropPath 比例（用于正则化）   
                    drc_k=drc_k   
                )     
            )   
        
        # 3. 将所有 HG_Block 组成一个顺序执行的模块    
        self.blocks = nn.Sequential(*blocks_list)
    
@register()   
class HGNetv2_DRC(HGNetv2):     
    def __init__(self, name, use_lab=False, return_idx=..., freeze_stem_only=True, freeze_at=0, freeze_norm=True, pretrained=True, agg='se', local_model_dir='weight/hgnetv2/'):
        super().__init__(name, use_lab, return_idx, freeze_stem_only, freeze_at, freeze_norm, pretrained, agg, local_model_dir)
 
        stage_config = { 
                        # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                        "stage1": [16, 16, 64, 1, False, False, 3, 3, 5],
                        "stage2": [64, 32, 256, 1, True, False, 3, 3, 7],
                        "stage3": [256, 64, 512, 2, True, True, 5, 3, 9], 
                        "stage4": [512, 128, 1024, 1, True, True, 5, 3, 11],   
                        }
    
        # stages 
        self.stages = nn.ModuleList()     
        for i, k in enumerate(stage_config):  
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num, drc_k = stage_config[ 
                k]    
            self.stages.append(
                HG_Stage_DRC(
                    in_channels,   
                    mid_channels,  
                    out_channels,
                    block_num,    
                    layer_num,
                    downsample,     
                    light_block,
                    kernel_size,     
                    use_lab,
                    agg,
                    drc_k=drc_k))     
