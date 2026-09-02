'''
本文件由BiliBili：魔傀面具整理   
engine/extre_module/module_images/TIP2025-CAFM.png     
engine/extre_module/module_images/TIP2025-CAFM.md
论文链接：https://ieeexplore.ieee.org/document/10955125  
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings  
warnings.filterwarnings('ignore')   
from calflops import calculate_flops  

import torch   
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from engine.extre_module.ultralytics_nn.conv import Conv 

class CAFM(nn.Module):
    def __init__(self, in_channels, out_channel=112):
        super(CAFM, self).__init__()

        self.conv_adjust = nn.ModuleList([])   
        for i in in_channels:
            if i != out_channel:
                self.conv_adjust.append(Conv(i, out_channel, 1))
            else:
                self.conv_adjust.append(nn.Identity())
    
        self.conv1_spatial = nn.Conv2d(2, 1, 3, stride=1, padding=1, groups=1)  
        self.conv2_spatial = nn.Conv2d(1, 1, 3, stride=1, padding=1, groups=1) 

        self.avg1 = nn.Conv2d(out_channel, 64, 1, stride=1, padding=0)
        self.avg2 = nn.Conv2d(out_channel, 64, 1, stride=1, padding=0)  
        self.max1 = nn.Conv2d(out_channel, 64, 1, stride=1, padding=0)
        self.max2 = nn.Conv2d(out_channel, 64, 1, stride=1, padding=0)

        self.avg11 = nn.Conv2d(64, out_channel, 1, stride=1, padding=0)
        self.avg22 = nn.Conv2d(64, out_channel, 1, stride=1, padding=0)
        self.max11 = nn.Conv2d(64, out_channel, 1, stride=1, padding=0)  
        self.max22 = nn.Conv2d(64, out_channel, 1, stride=1, padding=0)
        self.out_channel = out_channel 
 
        self.fusion = nn.Conv2d(out_channel * 2, out_channel, 1, 1, 0)
  
    def forward(self, x):
        f1, f2 = x 

        f1 = self.conv_adjust[0](f1)  
        f2 = self.conv_adjust[1](f2)   

        b, c, h, w = f1.size()
    
        f1 = f1.reshape([b, c, -1])     
        f2 = f2.reshape([b, c, -1])  

        avg_1 = torch.mean(f1, dim=-1, keepdim=True).unsqueeze(-1) 
        max_1, _ = torch.max(f1, dim=-1, keepdim=True)    
        max_1 = max_1.unsqueeze(-1)
  
        avg_1 = F.relu(self.avg1(avg_1))
        max_1 = F.relu(self.max1(max_1))     
        avg_1 = self.avg11(avg_1).squeeze(-1) 
        max_1 = self.max11(max_1).squeeze(-1)
        a1 = avg_1 + max_1     
  
        avg_2 = torch.mean(f2, dim=-1, keepdim=True).unsqueeze(-1)
        max_2, _ = torch.max(f2, dim=-1, keepdim=True)
        max_2 = max_2.unsqueeze(-1)
   
        avg_2 = F.relu(self.avg2(avg_2))
        max_2 = F.relu(self.max2(max_2))
        avg_2 = self.avg22(avg_2).squeeze(-1)    
        max_2 = self.max22(max_2).squeeze(-1)
        a2 = avg_2 + max_2   

        cross = torch.matmul(a1, a2.transpose(1, 2))  
  
        a1 = torch.matmul(F.softmax(cross, dim=-1), f1)
        a2 = torch.matmul(F.softmax(cross.transpose(1, 2), dim=-1), f2)  
     
        a1 = a1.reshape([b, c, h, w])
        avg_out = torch.mean(a1, dim=1, keepdim=True)  
        max_out, _ = torch.max(a1, dim=1, keepdim=True)
        a1 = torch.cat([avg_out, max_out], dim=1) 
        a1 = F.relu(self.conv1_spatial(a1))    
        a1 = self.conv2_spatial(a1)   
        a1 = a1.reshape([b, 1, -1])   
        a1 = F.softmax(a1, dim=-1)

        a2 = a2.reshape([b, c, h, w])    
        avg_out = torch.mean(a2, dim=1, keepdim=True)  
        max_out, _ = torch.max(a2, dim=1, keepdim=True)  
        a2 = torch.cat([avg_out, max_out], dim=1)
        a2 = F.relu(self.conv1_spatial(a2)) 
        a2 = self.conv2_spatial(a2)
        a2 = a2.reshape([b, 1, -1])  
        a2 = F.softmax(a2, dim=-1) 

        f1 = f1 * a1 + f1
        f2 = f2 * a2 + f2  
  
        f1 = f1.reshape([b, c, h, w])
        f2 = f2.reshape([b, c, h, w])

        out = self.fusion(torch.cat((f1, f2), dim=1))
        return out

if __name__ == '__main__':   
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel_1, channel_2, height, width = 1, 32, 16, 32, 32
    ouc_channel = 32
    inputs_1 = torch.randn((batch_size, channel_1, height, width)).to(device)
    inputs_2 = torch.randn((batch_size, channel_2, height, width)).to(device)  
   
    module = CAFM([channel_1, channel_2], ouc_channel).to(device)   
   
    outputs = module([inputs_1, inputs_2])
    print(GREEN + f'inputs1.size:{inputs_1.size()} inputs2.size:{inputs_2.size()} outputs.size:{outputs.size()}' + RESET)
  
    print(ORANGE) 
    flops, macs, _ = calculate_flops(model=module,  
                                     args=[[inputs_1, inputs_2]],
                                     output_as_string=True,
                                     output_precision=4, 
                                     print_detailed=True) 
    print(RESET)