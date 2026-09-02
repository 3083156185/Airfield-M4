'''  
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/自研模块-WDAF.png
engine/extre_module/module_images/自研模块-WDAF.md
'''  

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops
   
import torch
import torch.nn as nn
import torch.nn.functional as F     
   
from engine.extre_module.ultralytics_nn.conv import Conv, DSConv


class HaarWaveletDecomposition(nn.Module):    
    def __init__(self, channels):     
        super().__init__()
        self.channels = channels    
 
        weights = torch.ones(4, 1, 2, 2)     
        weights[1, 0, 0, 1] = -1 
        weights[1, 0, 1, 1] = -1
        weights[2, 0, 1, 0] = -1    
        weights[2, 0, 1, 1] = -1
        weights[3, 0, 1, 0] = -1
        weights[3, 0, 0, 1] = -1 
        weights = torch.cat([weights] * channels, dim=0)    
        self.register_buffer("weights", weights)
   
    def forward(self, x):   
        pad_h = x.shape[-2] % 2  
        pad_w = x.shape[-1] % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
 
        out = F.conv2d(x, self.weights, bias=None, stride=2, groups=self.channels) / 4.0
        batch_size, _, height, width = out.shape
        out = out.view(batch_size, self.channels, 4, height, width)
        out = out.transpose(1, 2).reshape(batch_size, self.channels * 4, height, width)    
        low, lh, hl, hh = out.chunk(4, dim=1)    
        high = lh + hl + hh    
        return low, high 
 
     
class WDAF(nn.Module): 
    def __init__(self, inc, ouc) -> None:
        super().__init__()
        if len(inc) != 2: 
            raise ValueError(f"WDAF expects exactly two input channels, got {len(inc)}")   
  
        self.conv_align1 = Conv(inc[0], ouc, 1)
        self.conv_align2 = Conv(inc[1], ouc, 1)  
    
        self.spatial_gate = DSConv(ouc * 2, ouc * 2, 3)   
        self.low_gate = DSConv(ouc * 2, ouc * 2, 3)   
        self.high_gate = DSConv(ouc * 2, ouc * 2, 3)
        self.sigmoid = nn.Sigmoid()
 
        self.wavelet = HaarWaveletDecomposition(ouc)
        self.branch_balance = nn.Parameter(torch.zeros(2)) 
        self.frequency_balance = nn.Parameter(torch.zeros(2))
        self.wavelet_scale = nn.Parameter(torch.tensor(0.1))

        self.wavelet_proj = DSConv(ouc, ouc, 3)
        self.conv_final = Conv(ouc, ouc, 1)
     
    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:     
            raise ValueError("WDAF expects a list or tuple with two feature maps")     
 
        x1, x2 = x
        if x1.shape[-2:] != x2.shape[-2:]: 
            raise ValueError("WDAF expects both inputs to have the same spatial shape")
 
        height, width = x1.shape[-2:] 
        x1 = self.conv_align1(x1)
        x2 = self.conv_align2(x2)

        spatial_logits = self.sigmoid(self.spatial_gate(torch.cat([x1, x2], dim=1)))
        x1_weight, x2_weight = torch.chunk(spatial_logits, 2, dim=1)    
        branch_weight = torch.softmax(self.branch_balance, dim=0)
        spatial_fused = branch_weight[0] * (x1 * x1_weight) + branch_weight[1] * (x2 * x2_weight)
 
        low1, high1 = self.wavelet(x1)     
        low2, high2 = self.wavelet(x2)     

        low_logits = self.sigmoid(self.low_gate(torch.cat([low1, low2], dim=1)))
        high_logits = self.sigmoid(self.high_gate(torch.cat([high1, high2], dim=1)))     
        low1_weight, low2_weight = torch.chunk(low_logits, 2, dim=1)    
        high1_weight, high2_weight = torch.chunk(high_logits, 2, dim=1)  

        freq_weight = torch.softmax(self.frequency_balance, dim=0)   
        low_fused = low1 * low1_weight + low2 * low2_weight
        high_fused = high1 * high1_weight + high2 * high2_weight
        wavelet_guidance = freq_weight[0] * low_fused + freq_weight[1] * high_fused  
        wavelet_guidance = self.wavelet_proj(wavelet_guidance)  
        wavelet_guidance = F.interpolate(wavelet_guidance, size=(height, width), mode='bilinear', align_corners=False)    
  
        return self.conv_final(spatial_fused + self.wavelet_scale * wavelet_guidance)   
  

if __name__ == '__main__': 
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  
    batch_size, channel_1, channel_2, height, width = 1, 32, 16, 21, 19 
    ouc_channel = 32
    inputs_1 = torch.randn((batch_size, channel_1, height, width)).to(device)
    inputs_2 = torch.randn((batch_size, channel_2, height, width)).to(device)

    module = WDAF([channel_1, channel_2], ouc_channel).to(device)

    outputs = module([inputs_1, inputs_2])
    print(GREEN + f'inputs1.size:{inputs_1.size()} inputs2.size:{inputs_2.size()} outputs.size:{outputs.size()}' + RESET)
  
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     args=[[inputs_1, inputs_2]],     
                                     output_as_string=True,  
                                     output_precision=4,  
                                     print_detailed=True)    
    print(RESET)
