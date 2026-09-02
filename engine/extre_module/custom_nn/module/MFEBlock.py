'''  
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/MICCAI2023-MFEblock.png
engine/extre_module/module_images/MICCAI2023-MFEblock.md
论文链接：https://arxiv.org/abs/2306.14119  
'''     
 
import os, sys  
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')  
from calflops import calculate_flops
     
import torch     
import torch.nn as nn
    
from engine.extre_module.ultralytics_nn.conv import Conv    
 
class MFEblock(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=[2, 4, 8]):
        super(MFEblock, self).__init__()
        act = nn.ReLU 
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.layer1 = Conv(in_channels, in_channels, 3, act=act)
        self.layer2 = Conv(in_channels, in_channels, 3, d=rate1, act=act) 
        self.layer3 = Conv(in_channels, in_channels, 3, d=rate2, act=act)    
        self.layer4 = Conv(in_channels, in_channels, 3, d=rate3, act=act)
        self.project = Conv(in_channels, out_channels, 1, act=act)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim = 2)
        self.softmax_1 = nn.Sigmoid()    
        self.SE1 = nn.Conv2d(in_channels,in_channels,1)  
        self.SE2 = nn.Conv2d(in_channels,in_channels,1)
        self.SE3 = nn.Conv2d(in_channels,in_channels,1) 
        self.SE4 = nn.Conv2d(in_channels,in_channels,1)
    def forward(self, x):   
        y0 = self.layer1(x)    
        y1 = self.layer2(y0+x)     
        y2 = self.layer3(y1+x)
        y3 = self.layer4(y2+x)
        y0_weight = self.SE1(self.gap(y0))
        y1_weight = self.SE2(self.gap(y1))
        y2_weight = self.SE3(self.gap(y2))
        y3_weight = self.SE4(self.gap(y3))
        weight = torch.cat([y0_weight,y1_weight,y2_weight,y3_weight],2)   
        weight = self.softmax(self.softmax_1(weight))    
        y0_weight = torch.unsqueeze(weight[:,:,0],2)     
        y1_weight = torch.unsqueeze(weight[:,:,1],2)
        y2_weight = torch.unsqueeze(weight[:,:,2],2)
        y3_weight = torch.unsqueeze(weight[:,:,3],2)    
        x_att = y0_weight*y0+y1_weight*y1+y2_weight*y2+y3_weight*y3
        return self.project(x_att+x) 
     
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"     
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = MFEblock(in_channel, out_channel).to(device)
  
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)   
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,    
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)