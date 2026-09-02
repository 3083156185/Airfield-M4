'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/TGRS2026-CGTA.png
engine/extre_module/module_images/TGRS2026-CGTA.md   
论文链接：https://ieeexplore.ieee.org/document/11373098    
'''

import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops

import math

import torch
import torch.nn as nn 
import torch.nn.functional as F 
    

class RetentionGate(nn.Module):  
    def __init__(self, dim, hidden_dim=64):
        super().__init__()   
        self.ret_gate = nn.Sequential(    
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):   
        return self.ret_gate(x).squeeze(-1)
   
   
def get_dynamic_topk_tokens(H, W, training):
    if training:
        time_level = max(int(math.log(H // 4, 4)), int(math.log(W // 4, 4)))  
    else:
        time_level = max(2, max(int(math.log(H // 16, 4)), int(math.log(W // 16, 4))))
    scale = 4 ** time_level
    k_tokens = (H // scale) * (W // scale)
    return k_tokens, scale   
   

class CGTA(nn.Module):  
    """
    Curvature-Guided Token Attention (CGTA).   

    This file is a standalone extraction of the original project class with all  
    project-local dependencies inlined so it can be imported directly as:     
    
        from CGTA import CGTA
    """

    def __init__(
        self,  
        dim,   
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,   
        attn_drop=0.0,   
        proj_drop=0.0,    
        c_ratio=0.5,
    ):     
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."    
        self.num_heads = num_heads
        head_dim = dim // num_heads 

        self.cr = int(dim * c_ratio)
        self.scale = qk_scale or (head_dim * c_ratio) ** -0.5  
 
        self.curvature_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)  
        self.gate = RetentionGate(dim, hidden_dim=dim // 2)

        self.q = nn.Linear(dim, self.cr, bias=qkv_bias)
        self.kv_reduce = nn.Conv1d(dim, self.cr, 1)     
        self.k = nn.Linear(self.cr, self.cr, bias=qkv_bias)   
        self.v = nn.Linear(self.cr, dim, bias=qkv_bias)

        self.norm_act = nn.Sequential(
            nn.LayerNorm(self.cr),    
            nn.GELU(),
        )     
     
        self.cpe = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
     
        self.proj = nn.Linear(dim, dim) 
        self.attn_drop = nn.Dropout(attn_drop)  
        self.proj_drop = nn.Dropout(proj_drop)  

        self.alpha = nn.Parameter(torch.zeros(1))  
        self.beta = nn.Parameter(torch.tensor(0.5))     

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        x_tokens = x.flatten(2).transpose(1, 2).contiguous()
    
        curvature = self.curvature_conv(x).mean(dim=1, keepdim=True).view(B, -1)
        curvature = F.layer_norm(curvature, curvature.shape[1:])  
        gate_score = self.gate(x_tokens)  
        score = (curvature.abs() + gate_score) / 2
   
        k_tokens, scale = get_dynamic_topk_tokens(H, W, self.training)  

        _, topk_indices = score.topk(k_tokens, dim=-1, largest=True, sorted=False)
        x_topk = torch.gather(   
            x_tokens,   
            dim=1,    
            index=topk_indices.unsqueeze(-1).expand(-1, -1, C), 
        )   
        score_topk = torch.gather(score, 1, topk_indices)

        q = self.q(x_tokens).reshape(B, N, self.num_heads, self.cr // self.num_heads).permute(0, 2, 1, 3) 
        kv_input = x_topk.transpose(1, 2)
        kv_compressed = self.kv_reduce(kv_input).transpose(1, 2)
        kv_compressed = self.norm_act(kv_compressed)
     
        k = self.k(kv_compressed).reshape(B, k_tokens, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v(kv_compressed).reshape(B, k_tokens, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 
        v = v * score_topk.unsqueeze(1).unsqueeze(-1)     

        curvature_topk = torch.gather(curvature, 1, topk_indices).unsqueeze(1) 
   
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale   
        attn_mod = attn_logits * (1 + self.alpha * curvature_topk.unsqueeze(1))
        attn_std = F.softmax(attn_logits, dim=-1)
        attn_cga = F.softmax(attn_mod, dim=-1)   
        attn = self.beta * attn_std + (1 - self.beta) * attn_cga
        attn = self.attn_drop(attn) 
    
        cpe = self.cpe(
            v.transpose(1, 2)
            .reshape(B, -1, C) 
            .transpose(1, 2)     
            .contiguous()
            .view(B, C, H // scale, W // scale) 
        ) 
        v = v + cpe.view(B, C, -1).view(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)    

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)  
        x = self.proj_drop(x) 
        return x.transpose(1, 2).reshape(B, C, H, W).contiguous()

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel, height, width = 1, 64, 20, 20
    inputs = torch.randn((batch_size, channel, height, width)).to(device)  

    module = CGTA(channel, num_heads=8).to(device).eval()   

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET) 

    print(ORANGE) 
    flops, macs, _ = calculate_flops(model=module,   
                                     input_shape=(batch_size, channel, height, width), 
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)   
