'''     
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/INFFUS2025-DPCF.md     
engine/extre_module/module_images/INFFUS2025-DPCF.png    
论文链接：https://arxiv.org/pdf/2505.23214
'''    

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')    
from calflops import calculate_flops

import math
import torch    
import torch.nn as nn     
import torch.nn.functional as F
try:    
    from mamba_ssm import Mamba  
except Exception as e:     
    pass
  
from engine.extre_module.ultralytics_nn.conv import Conv 

class AdaptiveCombiner(nn.Module):    
    def __init__(self):    
        super(AdaptiveCombiner, self).__init__()
        # 定义可学习参数d，形状与p和i相同，这里假设p和i的形状为(batch_size, channel, w, h)
        self.d = nn.Parameter(torch.randn(1, 1, 1, 1))     

    def forward(self, p, i):
        batch_size, channel, w, h = p.shape  
        # 将self.d扩展为与p和i相同的形状    
        d = self.d.expand(batch_size, channel, w, h)   
        edge_att = torch.sigmoid(d)
        return edge_att * p + (1 - edge_att) * i   


class DPCF(nn.Module):    
    def __init__(self, in_features, out_features) -> None:    
         super().__init__()     
         self.ac = AdaptiveCombiner()     
         self.tail_conv = Conv(in_features[1], out_features)  
         self.conv1x1 = Conv(in_features[0], in_features[1], 1) if in_features[0] != in_features[1] else nn.Identity()
    def forward(self, input): 
        x_low, x_high = input     
        x_low = self.conv1x1(x_low) 
        image_size = x_high.size(2) 
 
        if x_high != None:
            x_high = torch.chunk(x_high, 4, dim=1)
        if x_low != None:   
            x_low = F.interpolate(x_low, size=[image_size, image_size], mode='bilinear', align_corners=True)
            x_low = torch.chunk(x_low, 4, dim=1) 

        x0 = self.ac(x_low[0], x_high[0])  
        x1 = self.ac(x_low[1], x_high[1])    
        x2 = self.ac(x_low[2], x_high[2])
        x3 = self.ac(x_low[3], x_high[3])   

        x = torch.cat((x0, x1, x2, x3), dim=1)    
        x = self.tail_conv(x)    
  
        return x  
     
if __name__ == '__main__':   
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"  
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel_1, channel_2, height, width = 1, 32, 16, 32, 32
    ouc_channel = 64
    inputs_1 = torch.randn((batch_size, channel_1, height * 2, width * 2)).to(device)
    inputs_2 = torch.randn((batch_size, channel_2, height, width)).to(device)    
   
    module = DPCF([channel_1, channel_2], ouc_channel).to(device)     
     
    outputs = module([inputs_1, inputs_2])
    print(GREEN + f'inputs1.size:{inputs_1.size()} inputs2.size:{inputs_2.size()} outputs.size:{outputs.size()}' + RESET)     

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     args=[[inputs_1, inputs_2]],     
                                     output_as_string=True,  
                                     output_precision=4,
                                     print_detailed=True)    
    print(RESET)