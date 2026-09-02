''' 
本文件由BiliBili：魔傀面具整理  
论文链接：https://arxiv.org/pdf/2303.03667
''' 
  
import os, sys   
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../..')     

import warnings  
warnings.filterwarnings('ignore')
from calflops import calculate_flops

import torch   
import torch.nn as nn   
import torch.nn.functional as F

from engine.extre_module.ultralytics_nn.conv import Conv   
from engine.extre_module.custom_nn.conv_module.FDConv import FDConv 
from engine.extre_module.custom_nn.conv_module.ConvAttn import ConvAttn     

class PartialConv(nn.Module):   
    def __init__(self, inc, ouc, n_div=4): 
        super().__init__()

        self.partial_channels = inc // n_div   
        self.identity_channels = inc - self.partial_channels

        self.partial_conv = Conv(self.partial_channels, self.partial_channels, 3)
        # self.partial_conv = FDConv(self.partial_channels, self.partial_channels, kernel_size=3) 
        # self.partial_conv = ConvAttn(self.partial_channels, self.partial_channels, kernel_size=13)
  
        self.conv_adjust = Conv(inc, ouc, 1) if inc != ouc else nn.Identity()  

    def forward(self, x):     
        x1, x2 = torch.split(x, (self.partial_channels, self.identity_channels), 1)
        x1 = self.partial_conv(x1)
        y = torch.cat([x1, x2], 1)   
        y = self.conv_adjust(y)
        return y  
     
if __name__ == '__main__':     
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"     
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 2, 64, 128, 32, 32   
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)
 
    module = PartialConv(in_channel, out_channel).to(device)

    outputs = module(inputs)   
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
 
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,   
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,    
                                     print_detailed=True)
    print(RESET)   
