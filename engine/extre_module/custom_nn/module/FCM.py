'''     
本文件由BiliBili：魔傀面具整理     
engine/extre_module/module_images/AAAI2025-FCM.png     
论文链接：https://arxiv.org/abs/2504.20670  
'''
 
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')     

import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops 
  
import torch     
import torch.nn as nn  

from engine.extre_module.ultralytics_nn.conv import Conv
 
class Channel(nn.Module):   
    def __init__(self, dim):   
        super().__init__()     
        self.dwconv = self.dconv = nn.Conv2d(    
            dim, dim, 3, 
            1, 1, groups=dim
        )     
        self.Apt = nn.AdaptiveAvgPool2d(1) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):     
        x2 = self.dwconv(x)     
        x5 = self.Apt(x2)
        x6 = self.sigmoid(x5)  
    
        return x6


class Spatial(nn.Module):    
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, 1, 1, 1) 
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()
  
    def forward(self, x):
        x1 = self.conv1(x)
        x5 = self.bn(x1)  
        x6 = self.sigmoid(x5)

        return x6
     
class FCM(nn.Module):
    def __init__(self, dim,dim_out):
        super().__init__()  
        self.one = dim // 4    
        self.two = dim - dim // 4  
        self.conv1 = Conv(dim // 4, dim // 4, 3, 1, 1)
        self.conv12 = Conv(dim // 4, dim // 4, 3, 1, 1)
        self.conv123 = Conv(dim // 4, dim, 1, 1)
 
        self.conv2 = Conv(dim - dim // 4, dim, 1, 1)
        self.conv3 = Conv(dim, dim_out, 1, 1)
        self.spatial = Spatial(dim)     
        self.channel = Channel(dim)
   
    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)     
        x4 = self.conv2(x2)     
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        x5 = x33 + x44
        x5 = self.conv3(x5)    
        return x5

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device) 
 
    module = FCM(in_channel, out_channel).to(device)    

    outputs = module(inputs)     
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
     
    print(ORANGE)  
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,  
                                     print_detailed=True)     
    print(RESET)