import torch    
import torch.nn as nn
import torch.nn.functional as F   
import os, re
from .common import FrozenBatchNorm2d     
from ..core import register

from .hgnetv2 import HGNetv2 
    
from ..extre_module.custom_nn.stem.LoG import LoGStem   
    
__all__ = ['HGNetv2_LoGStem']

@register() 
class HGNetv2_LoGStem(HGNetv2):
    def __init__(self, name, use_lab=False, return_idx=..., freeze_stem_only=True, freeze_at=0, freeze_norm=True, pretrained=True, agg='se', local_model_dir='weight/hgnetv2/'):
        super().__init__(name, use_lab, return_idx, freeze_stem_only, freeze_at, freeze_norm, pretrained, agg, local_model_dir)
     
        stem_channels = self.arch_configs[name]['stem_channels']   
        self.stem = LoGStem(stem_channels[0], stem_channels[2]) 
