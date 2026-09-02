'''     
本文件由BiliBili：魔傀面具整理  
engine/extre_module/module_images/AAAI2026-CirculantAttention.png
engine/extre_module/module_images/AAAI2026-CirculantAttention.md
论文链接：https://arxiv.org/pdf/2512.21542    
'''
    
import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops  

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange     

class Linear(nn.Linear):
    r""" Linear layer for complex number inputs.
    """
    def __init__(self, in_features, out_features, device=None, dtype=None):     
        super(Linear, self).__init__(in_features, out_features, False, device, dtype)  

    def forward(self, x):
        x = torch.view_as_real(x).transpose(-2, -1)  
        x = torch.nn.functional.linear(x, self.weight).transpose(-2, -1)    
        if x.dtype != torch.float32:     
            x = x.to(torch.float32)
        x = torch.view_as_complex(x.contiguous())
        return x
     
class CirculantAttention(nn.Module):
    r""" Circulant Attention
    https://arxiv.org/abs/2512.21542    
    """  
    def __init__(self, dim, proj_drop=0.):   
        super().__init__()
        self.qkv = Linear(dim, dim * 3)
        self.gate = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.SiLU())
        self.proj = nn.Conv2d(dim, dim, 1) 
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w

        # Prepare Q, K, V, T
        #    (1) qkv=fc(x), qkv=fft(qkv) is mathematically equivalent to x=fft(x), qkv=fc(x)
        #    (2) The latter requires fewer FFT computations, delivering higher throughput
        t = self.gate(x)   
        x = x.permute(0, 2, 3, 1) # b c h w -> b h w c    
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)

        # Equation 15 of the paper   
        #    (1) We use d=1 in practice, as discussed in Table 5     
        #    (2) The 1/N factor is implicitly achieved by norm='ortho' in calculating Q, K    
        attn = torch.conj(q) * k
        attn = torch.fft.irfft2(attn, s=(h, w), dim=(1, 2), norm='ortho')
   
        # Equation 16 of the paper    
        attn = attn.reshape(b, n, c).softmax(dim=1).reshape(b, h, w, c)
        attn = torch.fft.rfft2(attn, dim=(1, 2))     
        x = torch.conj(attn) * v 
        x = torch.fft.irfft2(x, s=(h, w), dim=(1, 2), norm='ortho')

        # Output     
        x = x.permute(0, 3, 1, 2) * t
        x = self.proj(x)
        return x
     
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"     
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    batch_size, channel, height, width = 1, 64, 20, 20
    inputs = torch.randn((batch_size, channel, height, width)).to(device)     

    module = CirculantAttention(channel).to(device).eval()
   
    outputs = module(inputs) 
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)   
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width),    
                                     output_as_string=True,     
                                     output_precision=4,
                                     print_detailed=True)     
    print(RESET)   
