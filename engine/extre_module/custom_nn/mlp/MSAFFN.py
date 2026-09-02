'''     
本文件由BiliBili：魔傀面具整理    
engine/extre_module/module_images/自研模块-MSAFFN.png
engine/extre_module/module_images/自研模块-MSAFFN.md
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
  
  
class _MultiScaleSpectralRouter(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 2) 
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fuse = nn.Sequential(    
            nn.Conv2d(channels * 2, hidden, 1),   
            nn.GELU(),    
            nn.Conv2d(hidden, 2, 1),
        )

    def forward(self, x):
        spatial = self.pool(x)
        spectral = torch.fft.rfft2(x.float(), norm='ortho').abs().mean(dim=(-2, -1), keepdim=True).to(x.dtype)
        feat = torch.cat([spatial, spectral], dim=1)     
        weights = self.fuse(feat).flatten(1) 
        return torch.softmax(weights, dim=1)  
     

class MSAFFN(nn.Module):   
    def __init__(self, in_features, hidden_features, out_features, bias=False):   
        super(MSAFFN, self).__init__()     
        self.patch_size_fine = 4   
        self.patch_size_coarse = 8
        C = hidden_features * 2    
 
        self.project_in = nn.Conv2d(in_features, C, kernel_size=1, bias=bias)   
  
        # Fine-scale (4×4) frequency filter and autocorrelation parameters   
        self.fft_fine = nn.Parameter(torch.ones((C, 1, 1, 4, 3))) 
        self.alpha_fine = nn.Parameter(torch.tensor(0.5))    
        self.beta_fine = nn.Parameter(torch.tensor(0.5))   

        # Coarse-scale (8×8) frequency filter and autocorrelation parameters     
        self.fft_coarse = nn.Parameter(torch.ones((C, 1, 1, 8, 5)))
        self.alpha_coarse = nn.Parameter(torch.tensor(0.5))  
        self.beta_coarse = nn.Parameter(torch.tensor(0.5))
  
        # Spectral router: determines per-sample mixing ratio between scales
        self.router = _MultiScaleSpectralRouter(C)

        self.dwconv = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1, groups=C, bias=bias) 
        self.project_out = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias)    

    def _autocorr_branch(self, x, patch_size, fft_weight, alpha, beta):  
        dtype = x.dtype
        B, C, H, W = x.shape
   
        # Pad to be divisible by patch_size     
        pad_h = (-H) % patch_size     
        pad_w = (-W) % patch_size
        if pad_h > 0 or pad_w > 0:   
            x = F.pad(x, (0, pad_w, 0, pad_h)) 

        x_patch = rearrange(x, 'b c (h ph) (w pw) -> b c h w ph pw', ph=patch_size, pw=patch_size) 

        # FFT with learnable filter (all ops in float32 for numerical stability)     
        Xf = torch.fft.rfft2(x_patch.float())
        Xf = Xf * fft_weight.float()     

        # Autocorrelation power spectrum: |X|^2 = X * conj(X)
        power = Xf * torch.conj(Xf)    
        R = torch.fft.irfft2(power, s=(patch_size, patch_size))    
   
        # Frequency-domain enhancement + spatial autocorrelation residual     
        Xf_new = Xf + alpha.float() * power
        out = torch.fft.irfft2(Xf_new, s=(patch_size, patch_size)) 
        out = out + beta.float() * R  

        out = rearrange(out, 'b c h w ph pw -> b c (h ph) (w pw)', ph=patch_size, pw=patch_size) 
        return out[:, :, :H, :W].to(dtype)     

    def forward(self, x):  
        x = self.project_in(x)

        # Adaptive routing weights from spectral statistics
        weights = self.router(x)                         # [B, 2]    
        w_fine   = weights[:, 0].view(-1, 1, 1, 1)
        w_coarse = weights[:, 1].view(-1, 1, 1, 1)

        # Dual-scale autocorrelation branches
        x_fine   = self._autocorr_branch(x, self.patch_size_fine,   self.fft_fine,   self.alpha_fine,   self.beta_fine)
        x_coarse = self._autocorr_branch(x, self.patch_size_coarse, self.fft_coarse, self.alpha_coarse, self.beta_coarse) 

        # Softmax-weighted fusion 
        x = w_fine * x_fine + w_coarse * x_coarse 

        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
     
  
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m" 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, hidden_channel, out_channel, height, width = 1, 16, 64, 16, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device) 

    module = MSAFFN(in_features=in_channel, hidden_features=hidden_channel, out_features=out_channel).to(device)
    
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET) 
  
    print(ORANGE)  
    flops, macs, _ = calculate_flops(model=module, 
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,    
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)
