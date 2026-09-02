"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)    
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from .common import (
    get_activation,
    FrozenBatchNorm2d,
    freeze_batch_norm2d,
)
from .presnet import PResNet
from .test_resnet import MResNet

from .timm_model import TimmModel
from .torchvision_model import TorchVisionModel  

from .csp_resnet import CSPResNet
from .csp_darknet import CSPDarkNet, CSPPAN
 
from .hgnetv2 import HGNetv2    
from .hgnetv2_pconv import HGNetv2_PConv
from .hgnetv2_MANet_iRMB import HGNetv2_MANet_iRMB 
from .hgnetv2_star import HGNetv2_Star     
from .hgnetv2_LoGStem import HGNetv2_LoGStem    
from .hgnetv2_DRC import HGNetv2_DRC    
from .hgnetv2_DRC_Mamba import HGNetv2_DRC_Mamba   
from .hgnetv2_Mamba import HGNetv2_Mamba 
from .hgnetv2_LWGA import HGNetv2_LWGA  
from .hgnetv2_GQL import HGNetv2_GQL
     
from .dinov3 import ConvNeXt
from .vision_transformer import DinoVisionTransformer

from .dinov3_adapter import DINOv3STAs