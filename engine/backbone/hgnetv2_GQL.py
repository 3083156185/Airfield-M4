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
  
from ..extre_module.custom_nn.attention.GQL import GQL, GQL_Weight  
from ..extre_module.custom_nn.module.MSBlock import MSBlock_GQL  

__all__ = ['HGNetv2_GQL']  
     
class HG_Stage_MSBlock(HG_Stage):  
    def __init__(self, in_chs, mid_chs, out_chs, block_num, layer_num, downsample=True, light_block=False, kernel_size=3, use_lab=False, agg='se', drop_path=0, msblock_kernel=[1, 3, 3], gql_param=[3, 4]):    
        super().__init__(in_chs, mid_chs, out_chs, block_num, layer_num, downsample, light_block, kernel_size, use_lab, agg, drop_path)
 
        # 2. 构建多个 HG_Block
        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                MSBlock_GQL(
                    in_chs if i == 0 else out_chs,  # 第一个 HG_Block 需要匹配输入通道
                    out_chs,   # 输出通道数 
                    kernel_sizes=msblock_kernel, 
                    down_ratio=1,
                    mid_expand_ratio=2,
                    layers_num=layer_num, 
                    gql_param=gql_param   
                )
            )
 
        # 3. 将所有 HG_Block 组成一个顺序执行的模块    
        self.blocks = nn.ModuleList(blocks_list)
     
    def forward(self, input):   
        """
        前向传播过程：
        1. 先进行可选的下采样  
        2. 通过多个 HG_Block 进行特征提取
        """
        x, query = input
        x = self.downsample(x)  # 进行下采样（如果启用）   
        for layer in self.blocks:
            x = layer((x, query))  # 通过多个 HG_Block 进行特征提取   
        return x 
   
@register()
class HGNetv2_GQL(HGNetv2):    
    def __init__(self, name, use_lab=False, return_idx=..., freeze_stem_only=True, freeze_at=0, freeze_norm=True, pretrained=True, agg='se', local_model_dir='weight/hgnetv2/', gql_param=[3, 4]):
        super().__init__(name, use_lab, return_idx, freeze_stem_only, freeze_at, freeze_norm, pretrained, agg, local_model_dir)

        stage_config = {     
                        # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                        "stage1": [16, 16, 64, 1, False, False, 3, 1, [1, 3, 3]],
                        "stage2": [64, 32, 256, 1, True, False, 3, 1, [1, 5, 5]],
                        "stage3": [256, 64, 512, 1, True, True, 5, 1, [1, 7, 7]],
                        "stage4": [512, 128, 1024, 1, True, True, 5, 1, [1, 9, 9]],   
                        }     
   
        # stages
        self.stages = nn.ModuleList()
        for i, k in enumerate(stage_config):   
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num, msblock_kernel = stage_config[
                k]
            self.stages.append(     
                HG_Stage_MSBlock(    
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
                    msblock_kernel=msblock_kernel,
                    gql_param=gql_param))
        
        self.query = nn.Parameter(
                torch.randn((gql_param[0], gql_param[1] * gql_param[1])),
                requires_grad=True   
            )

    def forward(self, x):  
        x = self.stem(x)    
        outs = []  
        for idx, stage in enumerate(self.stages):
            x = stage((x, self.query))
            if idx in self.return_idx:     
                outs.append(x)     
        return outs