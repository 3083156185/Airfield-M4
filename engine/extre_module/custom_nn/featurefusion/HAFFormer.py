'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/PR2026-HAFFormer.png
engine/extre_module/module_images/PR2026-HAFFormer.md
论文链接：https://www.sciencedirect.com/science/article/pii/S0031320326003158 
'''
 
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')     
  
import warnings   
warnings.filterwarnings('ignore')
from calflops import calculate_flops   
   
import torch     
import torch.nn as nn   
from einops import rearrange     
 
from engine.extre_module.ultralytics_nn.conv import Conv

class Concat(nn.Module): 
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):   
        super(Concat, self).__init__()   
        self.d = dimension

    def forward(self, x): 
        # print(x.shape)  
        return torch.cat(x, self.d)    
  
class CrossAttention_S(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention_S, self).__init__()  
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1)) 

        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias) 
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,   
                                  bias=bias)
  
        self.qk = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)

        self.qk_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2,     
                                   bias=bias)  
    
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)  

    def forward(self, x):  
        fea_0 = x[0]  # 2024/11/1 added by wwc
        fea_1 = x[1]  # 2024/11/1 added by wwc
        b, c, h, w = fea_0.shape   
    
        qk = self.qk_dwconv(self.qk(fea_0))
        q, k = qk.chunk(2, dim=1)
   
        v = self.v_dwconv(self.v(fea_1))

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

class HAFFormer(nn.Module):   
    def __init__(self, in_dim, out_dim):
        super(HAFFormer, self).__init__() 
        bias = False
        num_heads = 8
        self.dim = out_dim

        self.mhca_rgb = CrossAttention_S(out_dim, num_heads, bias)  
        self.mhca_ir = CrossAttention_S(out_dim, num_heads, bias)
   
        # Concat    
        self.concat = Concat(dimension=1)
        self.conv = nn.Sequential(nn.Conv2d(2 * out_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=bias), nn.GELU()) 
        self.dwconv = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, groups=out_dim, bias=bias)    
    
        self.conv1x1_1 = Conv(in_dim[0], out_dim, 1) if in_dim[0] != out_dim else nn.Identity()   
        self.conv1x1_2 = Conv(in_dim[1], out_dim, 1) if in_dim[1] != out_dim else nn.Identity()

    def forward(self, x):     
        rgb_fea = self.conv1x1_1(x[0]) 
        ir_fea = self.conv1x1_2(x[1])

        # Cross Attention
        out_fea = self.mhca_rgb([rgb_fea, ir_fea])     
        out_fea_rgb = out_fea + rgb_fea
     
        out_fea = self.mhca_ir([ir_fea, rgb_fea])
        out_fea_ir = out_fea + ir_fea
 
        # Gated Fusion 
        fea_cat = self.concat([out_fea_rgb, out_fea_ir])    
        fea_conv = self.conv(fea_cat)
        w = self.dwconv(fea_conv).sigmoid()     
        new_fea = w * out_fea_rgb + (1 - w) * out_fea_ir

        return new_fea  

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"  
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')     
    batch_size, channel_1, channel_2, height, width = 1, 32, 16, 32, 32
    ouc_channel = 32
    inputs_1 = torch.randn((batch_size, channel_1, height, width)).to(device)
    inputs_2 = torch.randn((batch_size, channel_2, height, width)).to(device)

    module = HAFFormer([channel_1, channel_2], ouc_channel).to(device)    

    outputs = module([inputs_1, inputs_2])
    print(GREEN + f'inputs1.size:{inputs_1.size()} inputs2.size:{inputs_2.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,   
                                     args=[[inputs_1, inputs_2]],
                                     output_as_string=True, 
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)   
