'''    
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/TGRS2025-ACAB.png   
engine/extre_module/module_images/TGRS2025-ACAB.md    
论文链接：https://ieeexplore.ieee.org/document/11232501  
'''
import warnings  
warnings.filterwarnings('ignore')    
from calflops import calculate_flops    
   
import math
import torch
import torch.nn as nn  

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
   
class ACAB(nn.Module):  
    def __init__(self, channel, kernel_size=7, b=1, gamma=2):   
        super().__init__()

        self.ca = CA(channel, b, gamma)   
        self.sa = SA(kernel_size=kernel_size)   
    
    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x   

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"     
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')     
    batch_size, channel, height, width = 1, 16, 32, 32     
    inputs = torch.randn((batch_size, channel, height, width)).to(device)     
 
    module = ACAB(channel).to(device)    

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)     

    print(ORANGE)     
    flops, macs, _ = calculate_flops(model=module, 
                                     input_shape=(batch_size, channel, height, width),     
                                     output_as_string=True,  
                                     output_precision=4, 
                                     print_detailed=True)
    print(RESET)   
