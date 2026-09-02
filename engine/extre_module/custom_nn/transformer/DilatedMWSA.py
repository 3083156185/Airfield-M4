'''     
本文件由BiliBili：魔傀面具整理
论文链接：https://arxiv.org/abs/2404.07846
'''    

import warnings
warnings.filterwarnings('ignore')  
from calflops import calculate_flops 
    
import torch
import torch.nn as nn   
from einops import rearrange
  
class DilatedMWSA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(DilatedMWSA, self).__init__()     
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)  
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, dilation=2, padding=2, groups=dim*3, bias=bias) 
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape 

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1) 
     
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads) 

        q = torch.nn.functional.normalize(q, dim=-1) 
        k = torch.nn.functional.normalize(k, dim=-1)
    
        attn = (q @ k.transpose(-2, -1)) * self.temperature    
        attn = attn.softmax(dim=-1)

        out = (attn @ v)    

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
  
        out = self.project_out(out)  
        return out     

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"  
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  
    batch_size, channel, height, width = 1, 16, 20, 20  
    inputs = torch.randn((batch_size, channel, height, width)).to(device)
  
    module = DilatedMWSA(channel, num_heads=8).to(device)  
  
    outputs = module(inputs)  
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
  
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,     
                                     input_shape=(batch_size, channel, height, width),     
                                     output_as_string=True,     
                                     output_precision=4,
                                     print_detailed=True)  
    print(RESET)