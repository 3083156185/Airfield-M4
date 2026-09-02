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
from ..extre_module.custom_nn.module.iRMB import iRMB 
from ..extre_module.custom_nn.module.starblock import Star_Block   
from ..extre_module.custom_nn.module.fasterblock import Faster_Block
from ..extre_module.custom_nn.block.MANet import MANet
from ..extre_module.ultralytics_nn.block import C3_Block

# Constants for initialization
kaiming_normal_ = nn.init.kaiming_normal_     
zeros_ = nn.init.zeros_     
ones_ = nn.init.ones_
     
__all__ = ['HGNetv2_MANet_iRMB']    

class HG_Stage_MANet_iRMB(HG_Stage):     
    def __init__(self, in_chs, mid_chs, out_chs, block_num, layer_num, downsample=True, light_block=False, kernel_size=3, use_lab=False, agg='se', drop_path=0):   
        super().__init__(in_chs, mid_chs, out_chs, block_num, layer_num, downsample, light_block, kernel_size, use_lab, agg, drop_path)
     
        # 2. 构建多个 HG_Block  
        blocks_list = []   
        for i in range(block_num):  
            blocks_list.append(  
                MANet(   
                    in_chs if i == 0 else out_chs,  # 第一个 HG_Block 需要匹配输入通道
                    out_chs,   # 输出通道数  
                    partial(iRMB),    
                    n=layer_num, # 该 HG_Block 内的层数    
                )
            )  
        
        # 3. 将所有 HG_Block 组成一个顺序执行的模块
        self.blocks = nn.Sequential(*blocks_list)

@register()
class HGNetv2_MANet_iRMB(HGNetv2):     
    arch_configs = {
        'B0': {
            'stem_channels': [3, 16, 16],     
            'stage_config': {     
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num   
                "stage1": [16, 16, 64, 1, False, False, 3, 1],  
                "stage2": [64, 32, 128, 1, True, False, 3, 1],
                "stage3": [128, 64, 256, 2, True, True, 5, 1],
                "stage4": [256, 128, 512, 1, True, True, 5, 1],
            },     
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'
        },
        'B1': {   
            'stem_channels': [3, 24, 32],    
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num    
                "stage1": [32, 32, 64, 1, False, False, 3, 3],   
                "stage2": [64, 48, 256, 1, True, False, 3, 3],
                "stage3": [256, 96, 512, 2, True, True, 5, 3],
                "stage4": [512, 192, 1024, 1, True, True, 5, 3],
            }, 
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B1_stage1.pth'    
        },
        'B2': {   
            'stem_channels': [3, 24, 32],
            'stage_config': {  
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 96, 1, False, False, 3, 4],
                "stage2": [96, 64, 384, 1, True, False, 3, 4],
                "stage3": [384, 128, 768, 3, True, True, 5, 4],
                "stage4": [768, 256, 1536, 1, True, True, 5, 4],     
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B2_stage1.pth'
        },     
        'B3': {  
            'stem_channels': [3, 24, 32],
            'stage_config': {    
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num 
                "stage1": [32, 32, 128, 1, False, False, 3, 5], 
                "stage2": [128, 64, 512, 1, True, False, 3, 5],
                "stage3": [512, 128, 1024, 3, True, True, 5, 5],
                "stage4": [1024, 256, 2048, 1, True, True, 5, 5],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B3_stage1.pth'
        },
        'B4': {
            'stem_channels': [3, 32, 48],
            'stage_config': {  
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num     
                "stage1": [48, 48, 128, 1, False, False, 3, 6],   
                "stage2": [128, 96, 512, 1, True, False, 3, 6],
                "stage3": [512, 192, 1024, 3, True, True, 5, 6],   
                "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
            },    
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B4_stage1.pth'
        },
        'B5': {
            'stem_channels': [3, 32, 64],     
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num   
                "stage1": [64, 64, 128, 1, False, False, 3, 6],
                "stage2": [128, 128, 512, 2, True, False, 3, 6], 
                "stage3": [512, 256, 1024, 5, True, True, 5, 6],
                "stage4": [1024, 512, 2048, 2, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B5_stage1.pth'  
        }, 
        'B6': {
            'stem_channels': [3, 48, 96],   
            'stage_config': {   
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [96, 96, 192, 2, False, False, 3, 6],  
                "stage2": [192, 192, 512, 3, True, False, 3, 6],
                "stage3": [512, 384, 1024, 6, True, True, 5, 6], 
                "stage4": [1024, 768, 2048, 3, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B6_stage1.pth' 
        }, 
    }
    def __init__(self, name, use_lab=False, return_idx=..., freeze_stem_only=True, freeze_at=0, freeze_norm=True, pretrained=True, agg='se', local_model_dir='weight/hgnetv2/'):
        super().__init__(name, use_lab, return_idx, freeze_stem_only, freeze_at, freeze_norm, pretrained, agg, local_model_dir)   

        stage_config = self.arch_configs[name]['stage_config']

        # stages
        self.stages = nn.ModuleList()
        for i, k in enumerate(stage_config):
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num = stage_config[
                k]
            self.stages.append(
                HG_Stage_MANet_iRMB(
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
