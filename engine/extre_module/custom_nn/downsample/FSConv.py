'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/TGRS2025-FSConv.png
engine/extre_module/module_images/TGRS2025-FSConv.md
论文链接：https://ieeexplore.ieee.org/document/11175146  
''' 

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')
 
import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops
    
import torch, math
import torch.nn as nn
from pytorch_wavelets import DWTForward

from engine.extre_module.ultralytics_nn.conv import Conv
  
class SEBlock(nn.Module):     
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()   
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),   
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()  
        )  
  
    def forward(self, x):
        b, c, _, _ = x.size() 
        y = self.avg_pool(x).view(b, c)     
        y = self.fc(y).view(b, c, 1, 1)   
        return x * y

class FSConv(nn.Module):
    def __init__(self, c1, c2, k=3):
        super().__init__()
        self.c1 = c1 
    
        self.conv1 = Conv(c1, 2 * c1, 1, g=c1)
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv2 = Conv(c1, c2, k, 2, g=math.gcd(c1, c2))  
        self.conv3 = Conv(c1 * 3, c2, 3, d=1, g=math.gcd(c1 * 3, c2))
        self.se = SEBlock(c2)  
        self.conv4 = Conv(c1, c2, 3, g=math.gcd(c1, c2))    
        self.conv5 = Conv(2 * c2, c2, 1)
    
    def forward(self, x):   
        x0 = self.conv1(x)
        x1, x2 = torch.split(x0, self.c1, dim=1)     
        conv_spatial = self.conv2(x1)    

        yL, yH = self.wt(x2)

        # Extract the high-frequency subbands  
        y_HL = yH[0][:, :, 0, :]
        y_LH = yH[0][:, :, 1, :]    
        y_HH = yH[0][:, :, 2, :]
   
        high_frequency_fused = torch.cat([y_HL, y_LH, y_HH], dim=1)    
        high_frequency_fused_output = self.conv3(high_frequency_fused)    

        # Apply SE attention
        high_frequency_fused_output = self.se(high_frequency_fused_output)   

        low_frequency_fused_output = self.conv4(yL)     
   
        spatial_output = conv_spatial * high_frequency_fused_output

        fused = torch.cat([spatial_output, low_frequency_fused_output], dim=1) 
        out = self.conv5(fused)   

        return out
    
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"  
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 2, 64, 128, 32, 32  
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = FSConv(in_channel, out_channel, 3).to(device) 
   
    outputs = module(inputs)    
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
    
    print(ORANGE)     
    flops, macs, _ = calculate_flops(model=module,    
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,     
                                     output_precision=4, 
                                     print_detailed=True)   
    print(RESET)