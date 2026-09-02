'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/TGRS2025-GLGM.png
engine/extre_module/module_images/TGRS2025-GLGM.md
论文链接：https://ieeexplore.ieee.org/document/11232501     
'''
   
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')  
 
import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops    

import math 
import torch
import torch.nn as nn     

from engine.extre_module.ultralytics_nn.conv import Conv    

class CA(nn.Module):   
    def __init__(self, channel, b=1, gamma=2):
        super(CA, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
   
        self.avg_pool = nn.AdaptiveAvgPool2d(1)   
        self.max_pool = nn.AdaptiveMaxPool2d(1)   
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)    
        self.sigmoid = nn.Sigmoid()    
     
    def forward(self, x):    
        y1 = self.avg_pool(x)     
        y1 = self.conv(y1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y2 = self.max_pool(x)
        y2 = self.conv(y2.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)  
        y = self.sigmoid(y1 + y2)
     
        return y.expand_as(x) 

class SA(nn.Module):   
    def __init__(self, kernel_size=7):
        super(SA, self).__init__()    
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1 
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  
        return self.sigmoid(x)     

class GLGM(nn.Module):
    def __init__(self, in_channels, out_channels):  
        super(GLGM, self).__init__()     
        self.conv1 = Conv(in_channels, out_channels, 3, act=nn.ReLU)    
        self.dirate1_conv = Conv(out_channels, out_channels, k=3, d=1, act=nn.ReLU)
        self.dirate3_conv = Conv(out_channels, out_channels, k=3, d=3, act=nn.ReLU)
        self.conv2 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if out_channels != in_channels:  
            self.shortcut = nn.Sequential(Conv(in_channels, out_channels, 1, act=False))     
        else:   
            self.shortcut = None
  
        self.ca = CA(out_channels)
        self.sa = SA() 
        self.act = nn.ReLU(inplace=True)     

    def forward(self, x): 
        residual = x
        if self.shortcut is not None:   
            residual = self.shortcut(x)
        out = self.conv1(x)  
        out1 = self.dirate1_conv(out)
        out2 = self.dirate3_conv(out)
        out = self.conv2(torch.cat((out1, out2), 1))
        out = self.bn2(out)
        out = self.ca(out) * out    
        out = self.sa(out) * out     
        out += residual
        out = self.act(out)  
        return out   

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = GLGM(in_channel, out_channel).to(device)
     
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
    
    print(ORANGE) 
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),    
                                     output_as_string=True,     
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)