'''  
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/ACMMM2025-EGSA.png
engine/extre_module/module_images/ACMMM2025-EGSA.md  
论文链接：https://dl.acm.org/doi/epdf/10.1145/3746027.3755657   
'''   

import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops
     
import torch     
import torch.nn as nn  
  
class EfficientGlobalSA(nn.Module):
    def __init__(self,dim, heads=8, reduction_ratio=8):
        super().__init__()
        self.scale = (dim // heads) ** -0.5   
        self.heads = heads
        self.rd = reduction_ratio
    
        self.to_q = nn.Conv2d(dim, dim, 1, bias=True, groups=dim)   
        self.to_k = nn.Conv2d(dim, dim, reduction_ratio, stride=reduction_ratio, bias=True, groups=dim)
        self.to_v = nn.Conv2d(dim, dim, reduction_ratio, stride=reduction_ratio, bias=True, groups=dim)    
        self.to_out = nn.Conv2d(dim, dim, 1, bias=True)
        self.softmax = nn.Softmax(dim=-1) 
 
    def forward(self, x):    
        B, C, H, W = x.shape  
        q, k, v = (self.to_q(x), self.to_k(x), self.to_v(x))
        hk, wk = k.shape[-2:]   
        _q = q.reshape(B, self.heads, -1, H * W).transpose(-2, -1)  
        _k = k.reshape(B, self.heads, -1, hk * wk)  
        _v = v.reshape(B, self.heads, -1, hk * wk).transpose(-2, -1)   
        attn = (_q @ _k) * self.scale

        attn = self.softmax(attn)    
        res = (attn @ _v)
        res = res.transpose(-2, -1).reshape(B, -1, H, W)
        return self.to_out(res)

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   
    batch_size, channel, height, width = 1, 16, 64, 64 
    inputs = torch.randn((batch_size, channel, height, width)).to(device)
   
    module = EfficientGlobalSA(channel, heads=8, reduction_ratio=8).to(device)   

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
  
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,  
                                     print_detailed=True) 
    print(RESET)   
