'''  
本文件由BiliBili：魔傀面具整理  
论文链接：https://arxiv.org/abs/1911.11907
'''  
   
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../..') 

import warnings 
warnings.filterwarnings('ignore')
from calflops import calculate_flops

import torch
import torch.nn as nn
import torch.nn.functional as F
  
from engine.extre_module.ultralytics_nn.conv import Conv, DWConv
from engine.extre_module.custom_nn.conv_module.FDConv import FDConv  
from engine.extre_module.custom_nn.conv_module.ConvAttn import ConvAttn  
  
class GhostConv(nn.Module):
    def __init__(self, inc, ouc, dw_conv=5):
        super().__init__()
        c_ = ouc // 2   

        self.primary_conv = Conv(inc, c_, 3)   
        # self.primary_conv = FDConv(inc, c_, kernel_size=3)
        # self.primary_conv = ConvAttn(inc, c_, kernel_size=13)  

        self.cheap_conv = DWConv(c_, c_, dw_conv)
    
    def forward(self, x):   
        y = self.primary_conv(x)
        y = torch.cat((y, self.cheap_conv(y)), 1)
        return y
 
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   
    batch_size, in_channel, out_channel, height, width = 2, 64, 64, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)  
     
    module = GhostConv(in_channel, out_channel).to(device)    
  
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)   

    print(ORANGE)  
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,   
                                     print_detailed=True)
    print(RESET)