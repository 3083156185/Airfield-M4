''' 
本文件由BiliBili：魔傀面具整理    
engine/extre_module/module_images/自研模块-DPAS.png   
engine/extre_module/module_images/自研模块-DPAS.md
'''
     
import warnings     
warnings.filterwarnings('ignore') 
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')
from calflops import calculate_flops    

import torch    
import torch.nn as nn
  

class AdaptiveCut(nn.Module):
    """CutD with learnable positional bias across 4 sub-pixel positions."""     
    def __init__(self, in_channels, out_channels):  
        super().__init__() 
        self.pos_weight = nn.Parameter(torch.zeros(4))
        self.conv_fusion = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, stride=1)
        self.batch_norm = nn.BatchNorm2d(out_channels) 
    
    def forward(self, x):
        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 1::2, 0::2]     
        x2 = x[:, :, 0::2, 1::2]     
        x3 = x[:, :, 1::2, 1::2]   
        w = torch.softmax(self.pos_weight, dim=0)
        x = torch.cat([x0 * w[0], x1 * w[1], x2 * w[2], x3 * w[3]], dim=1)  
        return self.batch_norm(self.conv_fusion(x))     
  

class PathGate(nn.Module):
    """Content-adaptive soft weighting for multi-path feature aggregation."""     
    def __init__(self, channel, num_paths, reduction=4):  
        super().__init__() 
        hidden = max(channel // reduction, num_paths)    
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential( 
            nn.Conv2d(channel, hidden, 1, bias=False),
            nn.SiLU(),  
            nn.Conv2d(hidden, num_paths, 1, bias=False)
        )
     
    def forward(self, paths):
        ref = sum(paths) / len(paths) 
        w = torch.softmax(self.fc(self.pool(ref)), dim=1)  # [B, num_paths, 1, 1]
        return sum(paths[i] * w[:, i:i+1] for i in range(len(paths)))   

   
class DPAS(nn.Module):     
    def __init__(self, in_channels=3, out_channels=96):    
        super().__init__()  
        out_c14 = int(out_channels / 4)     
        out_c12 = int(out_channels / 2)    

        self.conv_init = nn.Conv2d(in_channels, out_c14, kernel_size=7, stride=1, padding=3)
  
        # Stage 1: original → 2x downsampling
        self.conv_1 = nn.Conv2d(out_c14, out_c12, kernel_size=3, stride=1, padding=1, groups=out_c14)  
        self.conv_x1 = nn.Conv2d(out_c12, out_c12, kernel_size=3, stride=2, padding=1, groups=out_c12)   
        self.batch_norm_x1 = nn.BatchNorm2d(out_c12)     
        self.adaptive_cut_c = AdaptiveCut(out_c14, out_c12)  
        self.path_gate1 = PathGate(out_c12, num_paths=2)

        # Stage 2: 2x → 4x downsampling     
        self.conv_2 = nn.Conv2d(out_c12, out_channels, kernel_size=3, stride=1, padding=1, groups=out_c12)
        self.conv_x2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels)
        self.batch_norm_x2 = nn.BatchNorm2d(out_channels)
        self.max_m = nn.MaxPool2d(kernel_size=2, stride=2)  
        self.batch_norm_m = nn.BatchNorm2d(out_channels)     
        self.adaptive_cut_r = AdaptiveCut(out_c12, out_channels)
        self.path_gate2 = PathGate(out_channels, num_paths=3)    
   
    def forward(self, x):
        x = self.conv_init(x)  # [B, C/4, H, W]     
     
        # Stage 1  
        c = self.adaptive_cut_c(x)
        xc = self.batch_norm_x1(self.conv_x1(self.conv_1(x)))
        x = self.path_gate1([xc, c])    # [B, C/2, H/2, W/2]
     
        # Stage 2  
        r = x     
        x_exp = self.conv_2(r)          # [B, C, H/2, W/2]
        conv_d = self.batch_norm_x2(self.conv_x2(x_exp))
        max_d  = self.batch_norm_m(self.max_m(x_exp))     
        cut_d  = self.adaptive_cut_r(r)
        x = self.path_gate2([conv_d, max_d, cut_d])  # [B, C, H/4, W/4]   
        return x
 
     
if __name__ == '__main__':   
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device) 
     
    module = DPAS(in_channel, out_channel).to(device) 

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)  
    flops, macs, _ = calculate_flops(model=module,    
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True, 
                                     output_precision=4,     
                                     print_detailed=True)
    print(RESET)    
