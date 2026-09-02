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
from ..misc.dist_utils import Multiprocess_sync, is_dist_available_and_initialized
from ..logger_module import get_logger     
    
from ..extre_module.custom_nn.mamba.ASSM import ASSM  
from ..extre_module.ultralytics_nn.block import C2f_Block
from functools import partial     
     
from .hgnetv2 import HGNetv2, HG_Stage

logger = get_logger(__name__)
   
__all__ = ['HGNetv2_Mamba'] 
 
class HG_Stage_Mamba(HG_Stage):
    def __init__(self, in_chs, mid_chs, out_chs, block_num, layer_num, downsample=True, light_block=False, kernel_size=3, use_lab=False, agg='se', drop_path=0): 
        super().__init__(in_chs, mid_chs, out_chs, block_num, layer_num, downsample, light_block, kernel_size, use_lab, agg, drop_path)     

        # 2. 构建多个 HG_Block
        blocks_list = []   
        for i in range(block_num):
            blocks_list.append( 
                # MetaFormer_Block(in_chs if i == 0 else out_chs, out_chs, token_mixer=partial(ASSM), selfatt=True)     
                C2f_Block(in_chs if i == 0 else out_chs, out_chs, module=partial(ASSM), selfatt=True)     
            )    
        
        # 3. 将所有 HG_Block 组成一个顺序执行的模块
        self.blocks = nn.Sequential(*blocks_list)
    
@register()
class HGNetv2_Mamba(HGNetv2):
    def __init__(self, name, use_lab=False, return_idx=..., freeze_stem_only=True, freeze_at=0, freeze_norm=True, pretrained=True, agg='se', local_model_dir='weight/hgnetv2/'):
        super().__init__(name, use_lab, return_idx, freeze_stem_only, freeze_at, freeze_norm, pretrained, agg, local_model_dir)
    
        stage_config = {   
                        # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num 
                        "stage1": [16, 16, 64, 1, False, False, 3, 3, 5, 'HG'],   
                        "stage2": [64, 32, 256, 1, True, False, 3, 3, 7, 'HG'],     
                        "stage3": [256, 64, 512, 2, True, True, 5, 3, 9, 'Mamba'],
                        "stage4": [512, 128, 1024, 1, True, True, 5, 3, 11, 'Mamba'],
                        }     
        download_url = self.arch_configs[name]['url']  
    
        # stages
        self.stages = nn.ModuleList()
        for i, k in enumerate(stage_config):
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num, drc_k, stage_type = stage_config[
                k]
            if stage_type == 'HG': 
                self.stages.append(
                    HG_Stage(
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
            elif stage_type == 'Mamba':  
                self.stages.append(
                    HG_Stage_Mamba(
                        in_channels,
                        mid_channels,
                        out_channels,   
                        block_num,
                        layer_num,    
                        downsample,
                        light_block,
                        kernel_size,    
                        use_lab,   
                        agg)  
                )
        
        if pretrained: 
            RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m" 
            try:    
                model_path = local_model_dir + 'PPHGNetV2_' + name + '_stage1.pth'
                if os.path.exists(model_path): 
                    state = torch.load(model_path, map_location='cpu') 
                    logger.info(f"Loaded stage1 {name} HGNetV2 from local file.")
                else: 
                    # If the file doesn't exist locally, download from the URL
                    if is_dist_available_and_initialized() and torch.distributed.get_rank() == 0:    
                        logger.warning(GREEN + "If the pretrained HGNetV2 can't be downloaded automatically. Please check your network connection." + RESET)   
                        logger.warning(GREEN + "Please check your network connection. Or download the model manually from " + RESET + f"{download_url}" + GREEN + " to " + RESET + f"{local_model_dir}." + RESET)
                        state = torch.hub.load_state_dict_from_url(download_url, map_location='cpu', model_dir=local_model_dir)   
                        Multiprocess_sync()
                    else:   
                        Multiprocess_sync()
                        state = torch.load(local_model_dir)     
  
                    logger.info(f"Loaded stage1 {name} HGNetV2 from URL.")

                # need_keep_key_prefix_list = ['stem.*', 'stages.0.*']  
                # need_pop_key_list = []
                # for key in state.keys():
                #     need_pop = True    
                #     for keep in need_keep_key_prefix_list:     
                #         if re.match(keep, key):  
                #             need_pop = False  
                #             break 
                #     if need_pop:
                #         need_pop_key_list.append(key)
                # for key in need_pop_key_list:  
                #     state.pop(key)    
   
                logger.info(RED + f'Loading Pretrained State Dict Key Names:{state.keys()}' + RESET) 
 
                self.load_state_dict(state, strict=False)

            except (Exception, KeyboardInterrupt) as e:     
                if (is_dist_available_and_initialized() and torch.distributed.get_rank() == 0) or (not is_dist_available_and_initialized()):
                    logger.error(f"{str(e)}")
                    logger.error(RED + "CRITICAL WARNING: Failed to load pretrained HGNetV2 model" + RESET)     
                    logger.error(GREEN + "Please check your network connection. Or download the model manually from " \
                                + RESET + f"{download_url}" + GREEN + " to " + RESET + f"{local_model_dir}." + RESET)
                exit()