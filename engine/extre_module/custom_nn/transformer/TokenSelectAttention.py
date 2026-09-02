'''  
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/NN2025-TokenSelectiveAttention.md  
engine/extre_module/module_images/NN2025-TokenSelectiveAttention.png
论文链接：https://arxiv.org/pdf/2410.03171v3
''' 

import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops     

import torch  
import torch.nn as nn  
from einops import rearrange  

# Token Selective Attention     
class Token_Selective_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False, k=0.8, group_num=4):   
        super(Token_Selective_Attention, self).__init__()
        self.num_heads = num_heads
        self.k = k   
        self.group_num = group_num
        self.dim_group = dim // group_num    
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv3d(self.group_num, self.group_num * 3, kernel_size=(1, 1, 1), bias=False)
        self.qkv_conv = nn.Conv3d(self.group_num * 3, self.group_num * 3, kernel_size=(1, 3, 3), padding=(0, 1, 1),
                                  groups=self.group_num * 3, bias=bias)  # 331   
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.w = nn.Parameter(torch.ones(2))
  
    def forward(self, x):    
        b, c, h, w = x.shape
        x = x.reshape(b,self.group_num,c//self.group_num,h,w)  
        b, t, c, h, w = x.shape  # 2,4,32,8,8     

        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)   
    
        q = rearrange(q, 'b t (head c) h w -> b head c (h w t)', head=self.num_heads)     
        k = rearrange(k, 'b t (head c) h w -> b head c (h w t)', head=self.num_heads)
        v = rearrange(v, 'b t (head c) h w -> b head c (h w t)', head=self.num_heads)   

        q = torch.nn.functional.normalize(q, dim=-1)     
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, _, N = q.shape  # N=hw    
    
        mask = torch.zeros(b, self.num_heads, N, N, device=x.device, requires_grad=False)

        attn = (q.transpose(-2, -1) @ k) * self.temperature  # [b, hw, hw]     
   
        index = torch.topk(attn, k=int(N * self.k), dim=-1, largest=True)[1]
        mask.scatter_(-1, index, 1.)
        attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))  
        attn = attn.softmax(dim=-1)
    
        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)  # [b, c, hw]

        out = rearrange(out, 'b head c (h w t) -> b t (head c) h w', head=self.num_heads, h=h, w=w)
    
        out = out.reshape(b, -1, h, w)     
        out = self.project_out(out)
    
 
        return out   

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel, height, width = 1, 64, 20, 20   
    inputs = torch.randn((batch_size, channel, height, width)).to(device)
   
    module = Token_Selective_Attention(channel, num_heads=8).to(device)

    outputs = module(inputs) 
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
    
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width),     
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)  
