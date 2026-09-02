'''   
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/ICCV2025-ConvAttn.md
engine/extre_module/module_images/ICCV2025-ConvAttn.png   
论文链接：https://arxiv.org/abs/2503.06671
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
   
from engine.extre_module.ultralytics_nn.conv import Conv

def _geo_ensemble(k):    
    k = k.detach()

    k_hflip = k.flip([3])  
    k_vflip = k.flip([2])   
    k_hvflip = k.flip([2, 3])   
    k_rot90 = torch.rot90(k, -1, [2, 3])
    k_rot90_hflip = k_rot90.flip([3]) 
    k_rot90_vflip = k_rot90.flip([2])     
    k_rot90_hvflip = k_rot90.flip([2, 3])
    k = (k + k_hflip + k_vflip + k_hvflip + k_rot90 + k_rot90_hflip + k_rot90_vflip + k_rot90_hvflip) / 8     
    return k 
     
class ConvolutionalAttention(nn.Module):    
    def __init__(self, pdim: int, kernel_size: int = 13):
        super().__init__()  
        self.pdim = pdim
        self.lk_size = kernel_size
        self.sk_size = 3
        self.dwc_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(pdim, pdim // 2, 1, 1, 0),
            nn.GELU(),     
            nn.Conv2d(pdim // 2, pdim * self.sk_size * self.sk_size, 1, 1, 0)
        )
        nn.init.zeros_(self.dwc_proj[-1].weight)
        nn.init.zeros_(self.dwc_proj[-1].bias)
   
    def forward(self, x: torch.Tensor, lk_filter: torch.Tensor) -> torch.Tensor:  
        x1, x2 = torch.split(x, [self.pdim, x.shape[1]-self.pdim], dim=1)  
    
        # Dynamic Conv
        bs = x1.shape[0] 
        dynamic_kernel = self.dwc_proj(x[:, :self.pdim]).reshape(-1, 1, self.sk_size, self.sk_size)    
        x1_ = rearrange(x1, 'b c h w -> 1 (b c) h w')     
        x1_ = F.conv2d(x1_, dynamic_kernel, stride=1, padding=self.sk_size//2, groups=bs * self.pdim)
        x1_ = rearrange(x1_, '1 (b c) h w -> b c h w', b=bs, c=self.pdim)    
        
        # Static LK Conv + Dynamic Conv
        x1 = F.conv2d(x1, lk_filter.to(x1.dtype), stride=1, padding=self.lk_size // 2) + x1_
        
        x = torch.cat([x1, x2], dim=1)   
        return x    
    
    def extra_repr(self):  
        return f'pdim={self.pdim}'   

class ConvAttn(nn.Module):
    def __init__(self, inc_dim, dim: int, kernel_size: int = 13):   
        super().__init__()   
        pdim = dim // 4    
        self.plk_filter = nn.Parameter(torch.randn(pdim, pdim, kernel_size, kernel_size)) 
        self.plk = ConvolutionalAttention(pdim, kernel_size)
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)    

        self.conv1x1 = Conv(inc_dim, dim) if inc_dim != dim else nn.Identity()  

    def forward(self, x: torch.Tensor) -> torch.Tensor:    
        x = self.conv1x1(x)   
        x = self.plk(x, _geo_ensemble(self.plk_filter))
        x = self.aggr(x)  
        return x 
     
if __name__ == '__main__':    
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32     
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = ConvAttn(in_channel, out_channel, kernel_size=13).to(device)

    outputs = module(inputs)    
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width), 
                                     output_as_string=True, 
                                     output_precision=4, 
                                     print_detailed=True) 
    print(RESET)   
