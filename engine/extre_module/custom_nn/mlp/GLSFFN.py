'''
本文件由BiliBili：魔傀面具整理   
engine/extre_module/module_images/自研模块-GLSFFN.png
engine/extre_module/module_images/自研模块-GLSFFN.md
'''    

import warnings
warnings.filterwarnings('ignore')
import os, sys     
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..') 
from calflops import calculate_flops
import torch
import torch.nn as nn
import torch.nn.functional as F    
from einops import rearrange 


class GLSFFN(nn.Module): 
    def __init__(self, in_features, hidden_features, out_features, bias=False): 
        super(GLSFFN, self).__init__() 
        self.patch_size = 4   
        C = hidden_features * 2  
 
        self.project_in = nn.Conv2d(in_features, C, kernel_size=1, bias=bias)     
 
        # Learnable local patch frequency filter
        self.fft_local = nn.Parameter(torch.ones((C, 1, 1, self.patch_size, self.patch_size // 2 + 1))) 
        self.alpha = nn.Parameter(torch.tensor(0.5))    
        self.beta  = nn.Parameter(torch.tensor(0.5))
  
        # Global spectral branch: rfft2 statistics → per-channel gate for fft_local 
        self.global_spec_fc = nn.Sequential(  
            nn.Linear(C, C // 4),     
            nn.GELU(),
            nn.Linear(C // 4, C), 
            nn.Sigmoid(),
        )
   
        self.dwconv = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1, groups=C, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias)
    
    def forward(self, x):
        x = self.project_in(x)    
        B, C, H, W = x.shape

        # Global spectral descriptor: frequency energy per channel [B, C]  
        global_spec = torch.fft.rfft2(x.float(), norm='ortho').abs().mean(dim=(-2, -1)).to(x.dtype)     
        # Per-channel gate derived from global spectrum
        global_gate = self.global_spec_fc(global_spec)          # [B, C], sigmoid in [0,1]
        global_gate = global_gate.view(B, C, 1, 1, 1, 1)        # broadcast over patches & freq bins   

        # Local patch FFT (all complex arithmetic in float32)
        x_patch = rearrange(x, 'b c (h ph) (w pw) -> b c h w ph pw', ph=self.patch_size, pw=self.patch_size)
        Xf = torch.fft.rfft2(x_patch.float())                   # [B,C,h,w,ph,pw_r] complex64
   
        # Global-conditioned adaptive filter: fft_local scaled per sample per channel    
        # fft_local [C,1,1,ph,pw_r] × gate [B,C,1,1,1,1] → [B,C,1,1,ph,pw_r]    
        adaptive_fft = (self.fft_local * global_gate).float()
        Xf = Xf * adaptive_fft
 
        # Autocorrelation power spectrum and spatial residual
        power = Xf * torch.conj(Xf)
        R = torch.fft.irfft2(power, s=(self.patch_size, self.patch_size))

        Xf_new = Xf + self.alpha.float() * power
        x_patch_new = torch.fft.irfft2(Xf_new, s=(self.patch_size, self.patch_size)) 
        x_patch_new = x_patch_new + self.beta.float() * R     
 
        x = rearrange(x_patch_new, 'b c h w ph pw -> b c (h ph) (w pw)',
                      ph=self.patch_size, pw=self.patch_size).to(x.dtype)
   
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2 
        x = self.project_out(x)    
        return x
     
  
if __name__ == '__main__': 
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, hidden_channel, out_channel, height, width = 1, 16, 64, 16, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)    

    module = GLSFFN(in_features=in_channel, hidden_features=hidden_channel, out_features=out_channel).to(device)     
    
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)  

    print(ORANGE) 
    flops, macs, _ = calculate_flops(model=module,     
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,     
                                     print_detailed=True) 
    print(RESET)
