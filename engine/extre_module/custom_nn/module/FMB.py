'''     
本文件由BiliBili：魔傀面具整理     
engine/extre_module/module_images/ECCV2024-FMB.png
论文链接：https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06713.pdf
'''

import os, sys   
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops    
    
import torch
import torch.nn as nn
import torch.nn.functional as F    

from engine.extre_module.ultralytics_nn.conv import Conv 
 
class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)  
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim,hidden_dim,3,1,1,groups=dim),
            nn.Conv2d(hidden_dim,hidden_dim,1,1,0)
        )
        self.act =nn.GELU()  
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)     

    def forward(self, x): 
        x = self.conv_0(x)     
        x = self.act(x)
        x = self.conv_1(x)     
        return x 

class PCFN(nn.Module):     
    def __init__(self, dim, growth_rate=2.0, p_rate=0.25):
        super().__init__()    
        hidden_dim = int(dim * growth_rate)
        p_dim = int(hidden_dim * p_rate)    
        self.conv_0 = nn.Conv2d(dim,hidden_dim,1,1,0)   
        self.conv_1 = nn.Conv2d(p_dim, p_dim ,3,1,1)

        self.act =nn.GELU()
        self.conv_2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

        self.p_dim = p_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        if self.training:    
            x = self.act(self.conv_0(x))
            x1, x2 = torch.split(x,[self.p_dim,self.hidden_dim-self.p_dim],dim=1)
            x1 = self.act(self.conv_1(x1))   
            x = self.conv_2(torch.cat([x1,x2], dim=1))  
        else:  
            x = self.act(self.conv_0(x))     
            x[:,:self.p_dim,:,:] = self.act(self.conv_1(x[:,:self.p_dim,:,:]))
            x = self.conv_2(x)
        return x
     
class SMFA(nn.Module):
    def __init__(self, dim=36):   
        super(SMFA, self).__init__()
        self.linear_0 = nn.Conv2d(dim,dim*2,1,1,0)   
        self.linear_1 = nn.Conv2d(dim,dim,1,1,0)
        self.linear_2 = nn.Conv2d(dim,dim,1,1,0)     
    
        self.lde = DMlp(dim,2)     
 
        self.dw_conv = nn.Conv2d(dim,dim,3,1,1,groups=dim)     
   
        self.gelu = nn.GELU()
        self.down_scale = 8   

        self.alpha = nn.Parameter(torch.ones((1,dim,1,1)))
        self.belt = nn.Parameter(torch.zeros((1,dim,1,1))) 
  
    def forward(self, f):   
        _,_,h,w = f.shape
        y, x = self.linear_0(f).chunk(2, dim=1)
        x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))  
        x_v = torch.var(x, dim=(-2,-1), keepdim=True)
        x_l = x * F.interpolate(self.gelu(self.linear_1(x_s * self.alpha + x_v * self.belt)), size=(h,w), mode='nearest')
        y_d = self.lde(y)
        return self.linear_2(x_l + y_d)  
  
class FMB(nn.Module):  
    def __init__(self, in_dim, dim, ffn_scale=2.0):
        super().__init__()  

        self.smfa = SMFA(dim)     
        self.pcfn = PCFN(dim, ffn_scale)
        self.conv1x1 = Conv(in_dim, dim, 1) if in_dim != dim else nn.Identity()    
 
    def forward(self, x):
        x = self.conv1x1(x)
        x = self.smfa(F.normalize(x)) + x
        x = self.pcfn(F.normalize(x)) + x
        return x

if __name__ == '__main__': 
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32   
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)
     
    module = FMB(in_channel, out_channel).to(device)  
     
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
 
    print(ORANGE) 
    flops, macs, _ = calculate_flops(model=module,   
                                     input_shape=(batch_size, in_channel, height, width),   
                                     output_as_string=True, 
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)   
