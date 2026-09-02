'''   
本文件由BiliBili：魔傀面具整理     
engine/extre_module/module_images/CVPR2025-GDSAFusion.png
论文链接：https://arxiv.org/pdf/2502.20087
'''   
     
import os, sys  
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')
    
import warnings  
warnings.filterwarnings('ignore')
from calflops import calculate_flops

import torch     
import torch.nn as nn 
import torch.nn.functional as F
from einops import rearrange, einsum  
from timm.layers import DropPath   
from torch.utils.checkpoint import checkpoint   
try:
    from natten.functional import na2d_av
except ImportError:     
    na2d_av = None 
   
from engine.extre_module.ultralytics_nn.conv import Conv
from engine.extre_module.custom_nn.module.UniRepLKBlock import DilatedReparamBlock  
   
class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    Originally proposed in ConvNeXt V2 (https://arxiv.org/abs/2301.00808)  
    This implementation is more efficient than the original (https://github.com/facebookresearch/ConvNeXt-V2)
    We assume the inputs to this layer are (N, C, H, W)
    """
    def __init__(self, dim, use_bias=True):
        super().__init__()     
        self.use_bias = use_bias 
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))  
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))     

    def forward(self, x):     
        Gx = torch.norm(x, p=2, dim=(-1, -2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta 
        else:
            return (self.gamma * Nx + 1) * x   
 
class ResDWConv(nn.Conv2d):   
    '''
    Depthwise convolution with residual connection
    '''
    def __init__(self, dim, kernel_size=3):
        super().__init__(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
   
    def forward(self, x):  
        x = x + super().forward(x)   
        return x 

class SEModule(nn.Module):
    def __init__(self, dim, red=8, inner_act=nn.GELU, out_act=nn.Sigmoid): 
        super().__init__() 
        inner_dim = max(16, dim // red)   
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),    
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            inner_act(),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            out_act(),
        )  
        
    def forward(self, x):    
        x = x * self.proj(x)
        return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__() 
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1)*init_value, 
                                   requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)     

    def forward(self, x):     
        x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])  
        return x
   
class LayerNorm2d(nn.LayerNorm):    
    def __init__(self, dim):   
        super().__init__(normalized_shape=dim, eps=1e-6)    
     
    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c') 
        x = super().forward(x)    
        x = rearrange(x, 'b h w c -> b c h w')
        return x.contiguous() 
   
class GDSAFusion(nn.Module):
    def __init__(self, in_dim, out_channel, kernel_size=7, smk_size=5, num_heads=2, mlp_ratio=1,
                 deploy=False, use_gemm=True, norm_layer=LayerNorm2d, res_scale=True,     
                 ls_init_value=1.0, drop_path=0):
        super().__init__()
        dim, ctx_dim = in_dim  

        self.kernel_size = kernel_size   
        self.smk_size = smk_size    
        self.num_heads = num_heads 
        self.scale = (dim // self.num_heads) ** -0.5  
        self.res_scale = res_scale    
    
        out_dim = dim + ctx_dim
        mlp_dim = int(dim * mlp_ratio)
  
        self.dwconv1 = ResDWConv(out_dim, kernel_size=3) 
        self.norm1 = norm_layer(out_dim)

        self.fusion = nn.Sequential( 
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim),     
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, dim, kernel_size=1),  
            GRN(dim),
        )
        
        self.weight_query = nn.Sequential( 
            nn.Conv2d(dim, dim//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim//2),
        )    
 
        self.weight_key = nn.Sequential(  
            nn.AdaptiveAvgPool2d(7),   
            nn.Conv2d(ctx_dim, dim//2, kernel_size=1, bias=False),  
            nn.BatchNorm2d(dim//2),     
        )
        
        self.weight_proj = nn.Conv2d(49, kernel_size**2 + smk_size**2, kernel_size=1)  
        
        self.dyconv_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),    
        )
     
        self.lepe = nn.Sequential( 
            DilatedReparamBlock(dim, kernel_size=kernel_size, deploy=deploy, use_sync_bn=False, attempt_use_lk_impl=use_gemm), 
            nn.BatchNorm2d(dim),     
        )
        
        self.se_layer = SEModule(dim)
        
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),     
            nn.BatchNorm2d(dim),    
            nn.SiLU(),  
        )    

        self.proj = nn.Sequential( 
            nn.BatchNorm2d(dim),  
            nn.Conv2d(dim, out_dim, kernel_size=1),
        )     
    
        self.dwconv2 = ResDWConv(out_dim, kernel_size=3)  
        self.norm2 = norm_layer(out_dim)
     
        self.mlp = nn.Sequential(
            nn.Conv2d(out_dim, mlp_dim, kernel_size=1),
            nn.GELU(),
            ResDWConv(mlp_dim, kernel_size=3),
            GRN(mlp_dim), 
            nn.Conv2d(mlp_dim, out_dim, kernel_size=1),    
        ) 

        self.ls1 = LayerScale(out_dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()
        self.ls2 = LayerScale(out_dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()    
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()    
        self.get_rpb()
 
        self.conv1x1 = Conv(out_dim, out_channel) if out_dim != out_channel else nn.Identity()
    
    def get_rpb(self):
        self.rpb_size1 = 2 * self.smk_size - 1   
        self.rpb1 = nn.Parameter(torch.empty(self.num_heads, self.rpb_size1, self.rpb_size1)) 
        self.rpb_size2 = 2 * self.kernel_size - 1
        self.rpb2 = nn.Parameter(torch.empty(self.num_heads, self.rpb_size2, self.rpb_size2))
        nn.init.zeros_(self.rpb1) 
        nn.init.zeros_(self.rpb2)
    
  
    @torch.no_grad() 
    def generate_idx(self, kernel_size):
        rpb_size = 2 * kernel_size - 1
        idx_h = torch.arange(0, kernel_size)
        idx_w = torch.arange(0, kernel_size)
        idx_k = ((idx_h.unsqueeze(-1) * rpb_size) + idx_w).view(-1) 
        return (idx_h, idx_w, idx_k)
    
    
    def apply_rpb(self, attn, rpb, height, width, kernel_size, idx_h, idx_w, idx_k):
        """  
        RPB implementation directly borrowed from https://tinyurl.com/mrbub4t3 
        """     
        num_repeat_h = torch.ones(kernel_size, dtype=torch.long)   
        num_repeat_w = torch.ones(kernel_size, dtype=torch.long)   
        num_repeat_h[kernel_size//2] = height - (kernel_size-1) 
        num_repeat_w[kernel_size//2] = width - (kernel_size-1)  
        bias_hw = (idx_h.repeat_interleave(num_repeat_h).unsqueeze(-1) * (2*kernel_size-1)) + idx_w.repeat_interleave(num_repeat_w)
        bias_idx = bias_hw.unsqueeze(-1) + idx_k     
        bias_idx = bias_idx.reshape(-1, int(kernel_size**2)) 
        bias_idx = torch.flip(bias_idx, [0])   
        rpb = torch.flatten(rpb, 1, 2)[:, bias_idx]
        rpb = rpb.reshape(1, int(self.num_heads), int(height), int(width), int(kernel_size**2))
        return attn + rpb

    def forward(self, x):
        x, x_f = x  

        B, C, H, W = x.shape
        B, C_h, H_h, W_h = x_f.shape
   
        x_f = torch.cat([x, x_f], dim=1)
        x_f = self.dwconv1(x_f)
        identity = x_f    
        x_f = self.norm1(x_f)
        x = self.fusion(x_f) 
        gate = self.gate(x)
        lepe = self.lepe(x)     
    
        query, key = torch.split(x_f, split_size_or_sections=[C, C_h], dim=1)
        query = self.weight_query(query) * self.scale
        key = self.weight_key(key)
        query = rearrange(query, 'b (g c) h w -> b g c (h w)', g=self.num_heads)    
        key = rearrange(key, 'b (g c) h w -> b g c (h w)', g=self.num_heads)  
        weight = einsum(query, key, 'b g c n, b g c l -> b g n l')   
        weight = rearrange(weight, 'b g n l -> b l g n').contiguous()
        weight = self.weight_proj(weight)
        weight = rearrange(weight, 'b l g (h w) -> b g h w l', h=H, w=W)
   
        attn1, attn2 = torch.split(weight, split_size_or_sections=[self.smk_size**2, self.kernel_size**2], dim=-1)     
        rpb1_idx = self.generate_idx(self.smk_size)   
        rpb2_idx = self.generate_idx(self.kernel_size)    
        attn1 = self.apply_rpb(attn1, self.rpb1, H, W, self.smk_size, *rpb1_idx)     
        attn2 = self.apply_rpb(attn2, self.rpb2, H, W, self.kernel_size, *rpb2_idx)  
        attn1 = torch.softmax(attn1, dim=-1)
        attn2 = torch.softmax(attn2, dim=-1)
        value = rearrange(x, 'b (m g c) h w -> m b g h w c', m=2, g=self.num_heads)
 
        x1 = na2d_av(attn1, value[0], kernel_size=self.smk_size)
        x2 = na2d_av(attn2, value[1], kernel_size=self.kernel_size)
     
        x = torch.cat([x1, x2], dim=1)  
        x = rearrange(x, 'b g h w c -> b (g c) h w', h=H, w=W)
        x = self.dyconv_proj(x)

        x = x + lepe   
        x = self.se_layer(x)   
     
        x = gate * x
        x = self.proj(x)   

        if self.res_scale: 
            x = self.ls2(x) + self.drop_path(self.mlp(self.norm2(x)))
        else: 
            x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))  

        x = self.dwconv2(x)   
     
        if self.res_scale:
            x = self.ls2(x) + self.drop_path(self.mlp(self.norm2(x)))
        else:   
            x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))

        x  = self.conv1x1(x)
        return x   
    
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel_1, channel_2, height, width = 1, 32, 16, 32, 32
    ouc_channel = 32
    inputs_1 = torch.randn((batch_size, channel_1, height, width)).to(device)  
    inputs_2 = torch.randn((batch_size, channel_2, height, width)).to(device)

    # 需要安装一个na2d_av库  
    # 百度云链接：https://pan.baidu.com/s/1NMVqUC5V7BvvMi_1txjyuQ?pwd=jr29
    module = GDSAFusion([channel_1, channel_2], ouc_channel).to(device)   

    outputs = module([inputs_1, inputs_2])
    print(GREEN + f'inputs1.size:{inputs_1.size()} inputs2.size:{inputs_2.size()} outputs.size:{outputs.size()}' + RESET)
  
    print(ORANGE)     
    flops, macs, _ = calculate_flops(model=module,
                                     args=[[inputs_1, inputs_2]],
                                     output_as_string=True,   
                                     output_precision=4,    
                                     print_detailed=True)
    print(RESET)    
