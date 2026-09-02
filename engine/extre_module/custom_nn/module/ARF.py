'''
本文件由BiliBili：魔傀面具整理 
engine/extre_module/module_images/TGRS2025-ARF.md
engine/extre_module/module_images/TGRS2025-ARF.png   
论文链接：https://arxiv.org/abs/2402.19289 
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..') 

import warnings   
warnings.filterwarnings('ignore')     
from calflops import calculate_flops    
    
import torch
import torch.nn as nn    
from engine.extre_module.ultralytics_nn.conv import Conv
   
class ARF(nn.Module):
    def __init__(self, c1, c2) -> None:     
        super().__init__()     

        self.conv3x3 = Conv(c1, c1, k=3, g=c1)  
        self.conv5x5 = Conv(c1, c1, k=5, g=c1)
        self.conv7x7 = Conv(c1, c1, k=7, g=c1)  
  
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)   
        self.spatial_conv = Conv(2, c1 * 3, 3, act=nn.Sigmoid)   

        self.conv_final = Conv(c1, c2, k=1) if c1 != c2 else nn.Identity()
 
    def forward(self, x):
        conv3x3_f = self.conv3x3(x)
        conv5x5_f = self.conv5x5(conv3x3_f) 
        conv7x7_f = self.conv7x7(conv5x5_f)   
        mutil_f = torch.cat([conv3x3_f, conv5x5_f, conv7x7_f], dim=1)  

        avg_spatial = torch.mean(mutil_f, dim=1, keepdim=True)  # [B, 1, H, W]
        max_spatial, _ = torch.max(mutil_f, dim=1, keepdim=True)  # [B, 1, H, W] 
        spatial_pooled = torch.cat([avg_spatial, max_spatial], dim=1) # [B, 2, H, W]   
        spatial_attention = self.spatial_conv(spatial_pooled)   
        conv3x3_att, conv5x5_att, conv7x7_att = torch.chunk(spatial_attention, chunks=3, dim=1)
        final_attention = conv3x3_f * conv3x3_att + conv5x5_f * conv5x5_att + conv7x7_f * conv7x7_att 
        x = x * final_attention  
 
        return self.conv_final(x)

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32 
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)
   
    module = ARF(in_channel, out_channel).to(device)
     
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)    

    print(ORANGE) 
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),    
                                     output_as_string=True, 
                                     output_precision=4,    
                                     print_detailed=True)
    print(RESET)    
