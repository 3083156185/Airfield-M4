'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/TGRS2026-CKConv.png     
engine/extre_module/module_images/TGRS2026-CKConv.md  
论文链接：https://ieeexplore.ieee.org/document/11320952  
'''    
 
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')
   
import warnings
warnings.filterwarnings('ignore')    
from calflops import calculate_flops

import torch  
import torch.nn as nn
from engine.extre_module.ultralytics_nn.conv import Conv     
     
class CKConv(nn.Module):
    def __init__(self, c1, c2, kk=[3, 5, 7], s=1):     
        super().__init__()

        if not isinstance(kk, list) or not all(ki in [3, 5, 7, 9] for ki in kk):
            raise ValueError("k must be a list containing 3, 5, and/or 7")

        self.kk = kk    
        self.c1 = c1   
        self.c2 = c2    
        self.s = s

        self.conv_1x1 = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()    
   
        self.branches = nn.ModuleDict()
     
        for ki in kk:    

            self.branches[f'k{ki}_body'] = Conv(c2, c2//2, (3, 3), s=1, g=c2//2)
            self.branches[f'k{ki}_head_h'] = Conv(c2, c2//2, (1, ki), s=s, p=(0, (ki - 1) // 2), g=c2//2)
            self.branches[f'k{ki}_head_v'] = Conv(c2//2, c2//2, (ki, 1), s=s, p=((ki - 1) // 2, 0), g=c2//2)
            self.branches[f'k{ki}_conv2'] = nn.Conv2d(c2//2, c2, 1, groups=c2//2)
     
        self.conv_fuse = nn.Conv2d(len(kk) * c2, c2, 1, groups=16)   # note 1

    def forward(self, x):    

        outputs = []  

        x = self.conv_1x1(x)

        for ki in self.kk:  
            y = self.branches[f'k{ki}_head_h'](x)   
            y = self.branches[f'k{ki}_head_v'](y)  
            ys = self.branches[f'k{ki}_body'](x)   
            out = ys + y
            out = self.branches[f'k{ki}_conv2'](out)
            outputs.append(out)

        out = torch.cat(outputs, dim=1)
        out = self.conv_fuse(out)    
    
        return out

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32  
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = CKConv(in_channel, out_channel).to(device)

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)    
 
    print(ORANGE)    
    flops, macs, _ = calculate_flops(model=module,   
                                     input_shape=(batch_size, in_channel, height, width),  
                                     output_as_string=True,    
                                     output_precision=4,     
                                     print_detailed=True)    
    print(RESET) 
