'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/自研模块-FreqLAWDS.png
engine/extre_module/module_images/自研模块-FreqLAWDS.md
自研模块：Frequency-guided Light Adaptive-weight Downsampling
'''

import os, sys     
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')     

import warnings
warnings.filterwarnings('ignore')     
from calflops import calculate_flops
    
import warnings    

warnings.filterwarnings('ignore')     
from calflops import calculate_flops  

import torch, math
import torch.nn as nn
import torch.nn.functional as F
     
from engine.extre_module.ultralytics_nn.conv import Conv

   
def _safe_groups(channels, group):
    base = max(1, channels // max(1, group)) 
    return max(1, math.gcd(channels, base))
     

def _pad_to_even(x):  
    _, _, h, w = x.shape    
    pad_h = h % 2     
    pad_w = w % 2     
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
    return x

  
def _gather_subpixels(x):
    return torch.stack(  
        (     
            x[:, :, 0::2, 0::2],
            x[:, :, 1::2, 0::2],
            x[:, :, 0::2, 1::2],  
            x[:, :, 1::2, 1::2],     
        ),
        dim=-1,
    )   
    

class FreqLAWDS(nn.Module):
    # Frequency-guided Light Adaptive-weight Downsampling
    def __init__(self, in_ch, out_ch, group=16, freq_ratio=0.5) -> None:   
        super().__init__()
        groups = _safe_groups(in_ch, group)
 
        self.in_ch = in_ch
        self.softmax = nn.Softmax(dim=-1)  
        self.freq_scale = nn.Parameter(torch.tensor(float(freq_ratio)))
        self.res_scale = nn.Parameter(torch.tensor(float(freq_ratio) * 0.5))  

        self.local_attention = nn.Sequential(   
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv(in_ch, in_ch, k=1),
        )
        self.ds_conv = Conv(in_ch, in_ch * 4, k=3, s=2, g=groups)

        self.low_proj = Conv(in_ch, in_ch, 1, g=groups)     
        self.high_proj = Conv(in_ch, in_ch, 1, g=groups)
        self.freq_gate = nn.Sequential(
            Conv(in_ch * 3, in_ch, 1),    
            nn.Conv2d(in_ch, in_ch * 4, kernel_size=1, groups=groups, bias=True),
        )
        self.freq_residual = nn.Sequential(  
            Conv(in_ch * 2, in_ch, 1),
            Conv(in_ch, in_ch, 3, g=groups),
        )
        self.output = Conv(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()    
   
        haar = torch.tensor( 
            [  
                [[0.5, 0.5], [0.5, 0.5]],
                [[-0.5, -0.5], [0.5, 0.5]],
                [[-0.5, 0.5], [-0.5, 0.5]], 
                [[0.5, -0.5], [-0.5, 0.5]],
            ],
            dtype=torch.float32,
        ).unsqueeze(1)
        self.register_buffer('haar_weight', haar, persistent=False)

    def _wavelet_decompose(self, x):
        filters = self.haar_weight.repeat(self.in_ch, 1, 1, 1) 
        coeffs = F.conv2d(x, filters, stride=2, groups=self.in_ch)
        b, _, h, w = coeffs.shape 
        coeffs = coeffs.view(b, self.in_ch, 4, h, w)
        ll = coeffs[:, :, 0] 
        hf = coeffs[:, :, 1:].abs().sum(dim=2)    
        return ll, hf
  
    def forward(self, x):     
        x = _pad_to_even(x)  

        local_logits = _gather_subpixels(self.local_attention(x))

        candidates = self.ds_conv(x)
        b, _, h, w = candidates.shape  
        candidates = candidates.view(b, 4, self.in_ch, h, w).permute(0, 2, 3, 4, 1).contiguous()     

        ll, hf = self._wavelet_decompose(x)  
        coarse = F.avg_pool2d(x, kernel_size=2, stride=2) 

        freq_context = torch.cat((self.low_proj(ll), self.high_proj(hf), coarse), dim=1)  
        freq_logits = self.freq_gate(freq_context).view(b, 4, self.in_ch, h, w).permute(0, 2, 3, 4, 1).contiguous()

        att = self.softmax(local_logits + self.freq_scale * freq_logits)     
        fused = torch.sum(candidates * att, dim=-1)

        residual = self.freq_residual(torch.cat((ll, hf), dim=1))
        return self.output(fused + self.res_scale * residual) 
  

if __name__ == '__main__':  
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = (   
        "\033[91m",
        "\033[92m",
        "\033[94m",   
        "\033[93m",    
        "\033[38;5;208m",   
        "\033[0m",  
    )     
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32     
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device) 

    module = FreqLAWDS(in_channel, out_channel, group=16).to(device)  
   
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
     
    print(ORANGE)
    flops, macs, _ = calculate_flops(  
        model=module,
        input_shape=(batch_size, in_channel, height, width),
        output_as_string=True,
        output_precision=4,
        print_detailed=True,
    )  
    print(RESET)
