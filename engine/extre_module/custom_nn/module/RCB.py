'''
本文件由BiliBili：魔傀面具整理  
engine/extre_module/module_images/CVPR2025-RepConvBlock.png
论文链接：https://arxiv.org/pdf/2502.20087  
'''
   
import os, sys     
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops
 
import torch
import torch.nn as nn    
import torch.nn.functional as F  
from einops import rearrange
from timm.layers import DropPath
from torch.utils.checkpoint import checkpoint
   
from engine.extre_module.ultralytics_nn.conv import Conv
from engine.extre_module.custom_nn.module.UniRepLKBlock import DilatedReparamBlock
   
class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    Originally proposed in ConvNeXt V2 (https://arxiv.org/abs/2301.00808)   
    This implementation is more efficient than the original (https://github.com/facebookresearch/ConvNeXt-V2)
    We assume the inputs to this layer are (N, C, H, W)
    """    
    def __init__(self, dim, use_bias=True):   
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))     
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):    
        Gx = torch.norm(x, p=2, dim=(-1, -2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x  

class SEModule(nn.Module):     
    def __init__(self, dim, red=8, inner_act=nn.GELU, out_act=nn.Sigmoid):     
        super().__init__()
        inner_dim = max(16, dim // red)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, inner_dim, kernel_size=1),     
            inner_act(),   
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            out_act(),     
        )
        
    def forward(self, x):
        x = x * self.proj(x)     
        return x     

class LayerScale(nn.Module): 
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1)*init_value,     
                                   requires_grad=True)    
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)
    
    def forward(self, x):   
        x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])
        return x
        
class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):   
        super().__init__(normalized_shape=dim, eps=1e-6)  
    
    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = super().forward(x)   
        x = rearrange(x, 'b h w c -> b c h w')  
        return x.contiguous()

class ResDWConv(nn.Conv2d):
    '''
    Depthwise convolution with residual connection
    '''
    def __init__(self, dim, kernel_size=3): 
        super().__init__(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
    
    def forward(self, x):    
        x = x + super().forward(x)   
        return x

class RepConvBlock(nn.Module):   

    def __init__(self,     
                 in_dim,    
                 dim, 
                 kernel_size=7,
                 mlp_ratio=4,  
                 ls_init_value=None,   
                 res_scale=False,
                 drop_path=0,   
                 norm_layer=LayerNorm2d,  
                 use_gemm=False,     
                 deploy=False,
                 use_checkpoint=False):
        super().__init__()
        
        self.res_scale = res_scale
        self.use_checkpoint = use_checkpoint  
        
        mlp_dim = int(dim*mlp_ratio)     
        
        self.dwconv = ResDWConv(dim, kernel_size=3)
    
        self.proj = nn.Sequential(
            norm_layer(dim),
            DilatedReparamBlock(dim, kernel_size=kernel_size, deploy=deploy, use_sync_bn=False, attempt_use_lk_impl=use_gemm),    
            nn.BatchNorm2d(dim),
            SEModule(dim),
            nn.Conv2d(dim, mlp_dim, kernel_size=1), 
            nn.GELU(), 
            ResDWConv(mlp_dim, kernel_size=3),
            GRN(mlp_dim),
            nn.Conv2d(mlp_dim, dim, kernel_size=1),    
            DropPath(drop_path) if drop_path > 0 else nn.Identity(),    
        )
 
        self.ls = LayerScale(dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity() 
    
        self.conv1x1 = Conv(in_dim, dim, k=1) if in_dim != dim else nn.Identity()     
  
    def forward_features(self, x):    
        x = self.conv1x1(x)   
        x = self.dwconv(x) 
        
        if self.res_scale:   
            x = self.ls(x) + self.proj(x)
        else:  
            drop_path = self.proj[-1]
            x = x + drop_path(self.ls(self.proj[:-1](x)))

        return x     
    
    def forward(self, x):  
   
        if self.use_checkpoint and x.requires_grad:  
            x = checkpoint(self.forward_features, x, use_reentrant=False)
        else:
            x = self.forward_features(x)   
   
        return x     

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"  
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)   
 
    module = RepConvBlock(in_channel, out_channel, mlp_ratio=3).to(device)
  
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET) 

    print(ORANGE)  
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),  
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)     
