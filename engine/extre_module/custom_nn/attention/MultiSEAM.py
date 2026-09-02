'''  
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/MultiSEAM.png
论文链接：https://arxiv.org/pdf/2208.02019v2  
'''     

import warnings 
warnings.filterwarnings('ignore') 
from calflops import calculate_flops
    
import torch, math
import torch.nn as nn 
import torch.nn.functional as F     
   
class Residual(nn.Module): 
    def __init__(self, fn): 
        super(Residual, self).__init__()   
        self.fn = fn
 
    def forward(self, x):   
        return self.fn(x) + x

def DcovN(c1, c2, depth, kernel_size=3, patch_size=3):
    dcovn = nn.Sequential(
        nn.Conv2d(c1, c2, kernel_size=patch_size, stride=patch_size),   
        nn.SiLU(), 
        nn.BatchNorm2d(c2), 
        *[nn.Sequential(     
            Residual(nn.Sequential( 
                nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=kernel_size, stride=1, padding=1, groups=c2),
                nn.SiLU(),
                nn.BatchNorm2d(c2)    
            )),
            nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=1, stride=1, padding=0, groups=1),
            nn.SiLU(),   
            nn.BatchNorm2d(c2)
        ) for i in range(depth)]
    )
    return dcovn   

class MultiSEAM(nn.Module):
    def __init__(self, c1, depth=1, kernel_size=3, patch_size=[3, 5, 7], reduction=16):    
        super(MultiSEAM, self).__init__()    
        self.DCovN0 = DcovN(c1, c1, depth, kernel_size=kernel_size, patch_size=patch_size[0])     
        self.DCovN1 = DcovN(c1, c1, depth, kernel_size=kernel_size, patch_size=patch_size[1])    
        self.DCovN2 = DcovN(c1, c1, depth, kernel_size=kernel_size, patch_size=patch_size[2])   
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1) 
        self.fc = nn.Sequential(
            nn.Linear(c1, c1 // reduction, bias=False),     
            nn.ReLU(inplace=True),  
            nn.Linear(c1 // reduction, c1, bias=False),
            nn.Sigmoid()
        )
     
    def forward(self, x):    
        b, c, _, _ = x.size() 
        y0 = self.DCovN0(x)
        y1 = self.DCovN1(x) 
        y2 = self.DCovN2(x)
        y0 = self.avg_pool(y0).view(b, c)  
        y1 = self.avg_pool(y1).view(b, c)
        y2 = self.avg_pool(y2).view(b, c)  
        y4 = self.avg_pool(x).view(b, c)   
        y = (y0 + y1 + y2 + y4) / 4 
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.exp(y)     
        return x * y.expand_as(x)

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"   
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel, height, width = 1, 16, 32, 32
    inputs = torch.randn((batch_size, channel, height, width)).to(device)
  
    module = MultiSEAM(channel).to(device)  
 
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
   
    print(ORANGE)     
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width), 
                                     output_as_string=True,    
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)