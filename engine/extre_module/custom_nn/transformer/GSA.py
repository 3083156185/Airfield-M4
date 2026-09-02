'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/ACMMM2025-GSA.png    
engine/extre_module/module_images/ACMMM2025-GSA.md
论文链接：https://arxiv.org/pdf/2407.19768
'''  

import warnings   
warnings.filterwarnings('ignore')    
from calflops import calculate_flops
 
import torch
import torch.nn as nn  
from einops import rearrange
   
class GSA(nn.Module):
    def __init__(self, channels, num_heads=8, bias=False):    
        super(GSA, self).__init__()    
        self.channels = channels     
        self.num_heads = num_heads     

        self.temperature = nn.Parameter(torch.ones(1, 1, 1))   
        self.act = nn.ReLU()     
  
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, stride=1, padding=1, groups=channels * 3, bias=bias) 
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)
    
    def forward(self, x):
        b, c, h, w = x.shape 
        qkv = self.qkv_dwconv(self.qkv(x))     
        q, k, v = qkv.chunk(3, dim=1)   
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)   
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)    
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)    
 
        attn = (q @ k.transpose(-2, -1)) * self.temperature 
        attn = self.act(attn)
        out = (attn @ v) 
        y = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        y = rearrange(y, 'b (head c) h w -> b (c head) h w', head=self.num_heads, h=h, w=w)     
        y = self.project_out(y)
        return y

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel, height, width = 1, 16, 20, 20 
    inputs = torch.randn((batch_size, channel, height, width)).to(device)  
  
    module = GSA(channel, num_heads=8).to(device) 
    
    outputs = module(inputs)     
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)     
   
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width),     
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)    
    print(RESET)