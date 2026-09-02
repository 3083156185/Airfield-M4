'''
本文件由BiliBili：魔傀面具整理    
engine/extre_module/module_images/MALA.md   
论文链接：https://arxiv.org/pdf/2507.00698
''' 
   
import warnings     
warnings.filterwarnings('ignore')    
from calflops import calculate_flops

import torch
import torch.nn as nn
from einops import rearrange    
from typing import Tuple    
   
def rotate_every_two(x):
    x1 = x[:, :, :, ::2]    
    x2 = x[:, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)
     
def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)  
     
class RoPE(nn.Module):  
  
    def __init__(self, embed_dim, num_heads): 
        '''
        recurrent_chunk_size: (clh clw)     
        num_chunks: (nch ncw)
        clh * clw == cl 
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented  
        '''
        super().__init__()     
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 4))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()   
        self.register_buffer('angle', angle)
     
    def forward(self, slen: Tuple[int]):    
        '''
        slen: (h, w)  
        h * w == l
        recurrent is not implemented
        '''
        # index = torch.arange(slen[0]*slen[1]).to(self.angle)   
        index_h = torch.arange(slen[0]).to(self.angle)  
        index_w = torch.arange(slen[1]).to(self.angle)
        # sin = torch.sin(index[:, None] * self.angle[None, :]) #(l d1)   
        # sin = sin.reshape(slen[0], slen[1], -1).transpose(0, 1) #(w h d1)
        sin_h = torch.sin(index_h[:, None] * self.angle[None, :]) #(h d1//2)
        sin_w = torch.sin(index_w[:, None] * self.angle[None, :]) #(w d1//2)  
        sin_h = sin_h.unsqueeze(1).repeat(1, slen[1], 1) #(h w d1//2)
        sin_w = sin_w.unsqueeze(0).repeat(slen[0], 1, 1) #(h w d1//2)     
        sin = torch.cat([sin_h, sin_w], -1) #(h w d1)
        # cos = torch.cos(index[:, None] * self.angle[None, :]) #(l d1)
        # cos = cos.reshape(slen[0], slen[1], -1).transpose(0, 1) #(w h d1) 
        cos_h = torch.cos(index_h[:, None] * self.angle[None, :]) #(h d1//2)   
        cos_w = torch.cos(index_w[:, None] * self.angle[None, :]) #(w d1//2)     
        cos_h = cos_h.unsqueeze(1).repeat(1, slen[1], 1) #(h w d1//2)
        cos_w = cos_w.unsqueeze(0).repeat(slen[0], 1, 1) #(h w d1//2)
        cos = torch.cat([cos_h, cos_w], -1) #(h w d1)  

        retention_rel_pos = (sin.flatten(0, 1), cos.flatten(0, 1))  
     
        return retention_rel_pos
     
class MALA(nn.Module):

    def __init__(self, dim, num_heads=8):
        super().__init__()  
        self.dim = dim     
        self.num_heads = num_heads
        self.head_dim = dim // num_heads 
        self.qkvo = nn.Conv2d(dim, dim * 4, 1)
        self.lepe = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)  
        self.proj = nn.Conv2d(dim, dim, 1)
        self.scale = self.head_dim ** -0.5  
        self.elu = nn.ELU()   

        self.repo = RoPE(dim, num_heads) 
   
    def forward(self, x: torch.Tensor):
        '''    
        x: (b c h w)
        sin: ((h w) d1)
        cos: ((h w) d1)    
        '''   
        B, C, H, W = x.shape
        sin, cos = self.repo((H, W))    
        qkvo = self.qkvo(x) #(b 3*c h w)
        qkv = qkvo[:, :3*self.dim, :, :]     
        o = qkvo[:, 3*self.dim:, :, :]  
        lepe = self.lepe(qkv[:, 2*self.dim:, :, :]) # (b c h w)    

        q, k, v = rearrange(qkv, 'b (m n d) h w -> m b n (h w) d', m=3, n=self.num_heads) # (b n (h w) d)

        q = self.elu(q) + 1
        k = self.elu(k) + 1

        z = q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) * self.scale

        q = theta_shift(q, sin, cos)   
        k = theta_shift(k, sin, cos)

        kv = (k.transpose(-2, -1) * (self.scale / (H*W)) ** 0.5) @ (v * (self.scale / (H*W)) ** 0.5)
 
        res = q @ kv * (1 + 1/(z + 1e-6)) - z * v.mean(dim=2, keepdim=True)    
    
        res = rearrange(res, 'b n (h w) d -> b (n d) h w', h=H, w=W)
        res = res + lepe   
        return self.proj(res * o)

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"   
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel, height, width = 1, 64, 128, 128
    inputs = torch.randn((batch_size, channel, height, width)).to(device)   
    
    module = MALA(channel).to(device)
    
    outputs = module(inputs)     
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)     

    print(ORANGE)  
    flops, macs, _ = calculate_flops(model=module,    
                                     input_shape=(batch_size, channel, height, width),
                                     output_as_string=True,    
                                     output_precision=4,
                                     print_detailed=True)   
    print(RESET) 
