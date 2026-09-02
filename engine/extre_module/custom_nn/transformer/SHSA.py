''' 
本文件由BiliBili：魔傀面具整理 
engine/extre_module/module_images/CVPR2024-SHSA.png     
论文链接：https://arxiv.org/pdf/2401.16456
'''   
   
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')   

import warnings     
warnings.filterwarnings('ignore')
from calflops import calculate_flops  

import torch

from engine.extre_module.ultralytics_nn.conv import Conv
    
class SHSA_GroupNorm(torch.nn.GroupNorm):
    """    
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]  
    """    
    def __init__(self, num_channels, **kwargs):    
        super().__init__(1, num_channels, **kwargs)
  
class SHSA(torch.nn.Module): 
    """Single-Head Self-Attention"""     
    def __init__(self, dim, qk_dim=16):
        super().__init__()
        self.scale = qk_dim ** -0.5   
        self.qk_dim = qk_dim
        self.dim = dim  
        pdim = dim // 2
        self.pdim = pdim 
  
        self.pre_norm = SHSA_GroupNorm(pdim)

        self.qkv = Conv(pdim, qk_dim * 2 + pdim, act=False)     
        self.proj = torch.nn.Sequential(torch.nn.SiLU(), Conv(     
            dim, dim, act=False))  
        
        torch.nn.init.constant_(self.proj[1].bn.weight, 0.0)    
        torch.nn.init.constant_(self.proj[1].bn.bias, 0)     

    def forward(self, x):
        B, C, H, W = x.shape
        x1, x2 = torch.split(x, [self.pdim, self.dim - self.pdim], dim = 1)
        x1 = self.pre_norm(x1)    
        qkv = self.qkv(x1)  
        q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.pdim], dim = 1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)    
    
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim = -1)  
        x1 = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)
        x = self.proj(torch.cat([x1, x2], dim = 1))     

        return x   
    
if __name__ == '__main__':   
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    batch_size, channel, height, width = 1, 64, 20, 20     
    inputs = torch.randn((batch_size, channel, height, width)).to(device) 

    module = SHSA(channel).to(device)
 
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,    
                                     input_shape=(batch_size, channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,     
                                     print_detailed=True)    
    print(RESET)     
