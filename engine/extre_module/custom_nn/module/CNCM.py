'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/TGRS2025-CNCM.png
engine/extre_module/module_images/TGRS2025-CNCM.md 
论文链接：https://ieeexplore.ieee.org/document/10855453  
'''     
  
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..') 
     
import warnings    
warnings.filterwarnings('ignore')     
from calflops import calculate_flops 
   
import torch     
import torch.nn as nn
     
from engine.extre_module.custom_nn.module.CSSC import CSSC
from engine.extre_module.ultralytics_nn.conv import Conv     

class CNCM(nn.Module):
    def __init__(self, channel_in, channel_out, reduction=16):
        super(CNCM, self).__init__()

        # RCSSC
        self.unit_1 = CSSC(int(channel_in / 2.), int(channel_in / 2.), reduction)  
        self.unit_2 = CSSC(int(channel_in / 2.), int(channel_in / 2.), reduction)
   
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channel_in, out_channels=int(channel_in / 2.), kernel_size=3, padding=1),
            nn.LeakyReLU()   
        )  
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=int(channel_in * 3 / 2.), out_channels=int(channel_in / 2.), kernel_size=3,  
                      padding=1),   
            nn.LeakyReLU()   
        )    
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=channel_in * 2, out_channels=channel_in, kernel_size=1, padding=0,
                      stride=1),  # 做压缩
            nn.Conv2d(in_channels=channel_in, out_channels=channel_in, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )

        self.conv1x1 = Conv(channel_in, channel_out, 1, act=nn.LeakyReLU) if channel_in != channel_out else nn.Identity()
     
    def forward(self, x):
        residual = x  
        c1 = self.unit_1(self.conv1(x))
        x = torch.cat([residual, c1], 1)     
        c2 = self.unit_2(self.conv2(x))   
        x = torch.cat([c2, x], 1)  
        x = self.conv3(x)
        x = torch.add(x, residual)
        return self.conv1x1(x)

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')     
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32  
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)  

    module = CNCM(in_channel, out_channel).to(device)  

    outputs = module(inputs)     
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)   

    print(ORANGE)   
    flops, macs, _ = calculate_flops(model=module, 
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,  
                                     output_precision=4,     
                                     print_detailed=True)    
    print(RESET)