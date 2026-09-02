'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/ACMMM2025-RSA.png   
engine/extre_module/module_images/ACMMM2025-RSA.md  
论文链接：https://arxiv.org/pdf/2407.19768     
'''   

import warnings     
warnings.filterwarnings('ignore')
from calflops import calculate_flops     
    
import torch   
import torch.nn as nn
from einops import rearrange     
  
class RSA(nn.Module):   
    def __init__(self, channels, shifts=1, window_sizes=4, bias=False):
        super(RSA, self).__init__()    
        self.channels = channels 
        self.shifts   = shifts
        self.window_sizes = window_sizes

        self.temperature = nn.Parameter(torch.ones(1, 1, 1))   
        self.act = nn.ReLU()

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=bias)     
        self.qkv_dwconv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, stride=1, padding=1, groups=channels * 3, bias=bias)  
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

    def forward(self, x):     
        b, c, h, w = x.shape
        wsize = self.window_sizes
        x_ = x
        if self.shifts > 0:
            x_ = torch.roll(x_, shifts=(-wsize//2, -wsize//2), dims=(2,3))
        qkv = self.qkv_dwconv(self.qkv(x_))   
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b c (h dh) (w dw) -> b (h w) (dh dw) c', dh=wsize, dw=wsize)
        k = rearrange(k, 'b c (h dh) (w dw) -> b (h w) (dh dw) c', dh=wsize, dw=wsize)    
        v = rearrange(v, 'b c (h dh) (w dw) -> b (h w) (dh dw) c', dh=wsize, dw=wsize)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
    
        attn = (q.transpose(-2, -1) @ k) * self.temperature # b (h w) (dh dw) (dh dw)
        attn = self.act(attn)
        out = (v @ attn)   
        out = rearrange(out, 'b (h w) (dh dw) c-> b (c) (h dh) (w dw)', h=h//wsize, w=w//wsize, dh=wsize, dw=wsize)
        if self.shifts > 0:
            out = torch.roll(out, shifts=(wsize//2, wsize//2), dims=(2, 3)) 
        y = self.project_out(out)
        return y   
     
if __name__ == '__main__': 
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"     
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel, height, width = 1, 16, 20, 20   
    inputs = torch.randn((batch_size, channel, height, width)).to(device)

    module = RSA(channel).to(device) 
    
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE) 
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width),   
                                     output_as_string=True,     
                                     output_precision=4,
                                     print_detailed=True)    
    print(RESET)    
