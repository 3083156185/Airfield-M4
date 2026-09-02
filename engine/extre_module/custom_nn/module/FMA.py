'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/IJCV2024-FMA.png    
engine/extre_module/module_images/IJCV2024-FMA.md
论文链接：https://link.springer.com/article/10.1007/s11263-024-02147-y  
'''

import os, sys 
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings    
warnings.filterwarnings('ignore')   
from calflops import calculate_flops
    
import math    
import torch 
import torch.nn as nn  
import torch.nn.functional as F
from einops import rearrange 
  
from engine.extre_module.ultralytics_nn.conv import Conv  

class MeanShift(nn.Conv2d):
    def __init__(   
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)  
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)     
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():  
            p.requires_grad = False
     

class LayerNorm(nn.Module):  
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)     
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):   
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))    
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:     
            raise NotImplementedError    
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":   
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)  
            s = (x - u).pow(2).mean(1, keepdim=True) 
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None] 
            return x   
  
     
class FourierUnit(nn.Module):
    def __init__(self, dim, groups=1, fft_norm='ortho'):   
        super().__init__()
        self.groups = groups     
        self.fft_norm = fft_norm   
   
        self.conv_layer = nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=1, stride=1,
                                    padding=0, groups=self.groups, bias=False)   
        self.act = nn.GELU()
 
    def forward(self, x):   
        batch, c, h, w = x.size()
        r_size = x.size()  
        dtype = x.dtype
        
        # 使用新的 FFT API   
        # torch.fft.rfft2 替代 torch.rfft   
        ffted = torch.fft.rfft2(x.float(), norm='ortho')  # (batch, c, h, w//2+1)  
        
        # 将复数转换为实数表示 (batch, c, h, w//2+1, 2)
        ffted = torch.stack([ffted.real, ffted.imag], dim=-1)    
        
        # (batch, c, 2, h, w//2+1)    
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        ffted = self.conv_layer(ffted.to(dtype))  # (batch, c*2, h, w//2+1)    
        ffted = self.act(ffted).float()

        # (batch, c, 2, h, w//2+1)
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:])  
        # (batch, c, h, w//2+1, 2)  
        ffted = ffted.permute(0, 1, 3, 4, 2).contiguous()  
     
        # 将实数表示转换回复数
        ffted_complex = torch.complex(ffted[..., 0].float(), ffted[..., 1].float())   
        
        # 使用 torch.fft.irfft2 替代 torch.irfft
        output = torch.fft.irfft2(ffted_complex, s=(h, w), norm='ortho')    
   
        return output.to(dtype)     

   
class FMA(nn.Module):  
    def __init__(self, indim, outdim, num_heads=8):     
        super().__init__()
        layer_scale_init_value = 1e-6  
        self.num_heads = num_heads
        self.norm = LayerNorm(indim, eps=1e-6, data_format="channels_first")   
        self.a = FourierUnit(indim)
        self.v = nn.Conv2d(indim, indim, 1)   
        self.act = nn.GELU()
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones(num_heads), requires_grad=True)
        self.CPE = nn.Conv2d(indim, indim, kernel_size=3, stride=1, padding=1, groups=indim)    
        self.proj = nn.Conv2d(indim, indim, 1)
    
        self.conv1x1 = Conv(indim, outdim, 1) if indim != outdim else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape   
        N = H * W
        shortcut = x
        pos_embed = self.CPE(x)
        x = self.norm(x)     
        a = self.a(x)
        v = self.v(x)   
        a = rearrange(a, 'b (head c) h w -> b head c (h w)', head=self.num_heads)   
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)   
        a_all = torch.split(a, math.ceil(N // 4), dim=-1)   
        v_all = torch.split(v, math.ceil(N // 4), dim=-1)
        attns = []    
        for a, v in zip(a_all, v_all):
            attn = a * v   
            attn = self.layer_scale.unsqueeze(-1).unsqueeze(-1) * attn
            attns.append(attn)
        x = torch.cat(attns, dim=-1)
        x = F.softmax(x, dim=-1)    
        x = rearrange(x, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)  
        x = x + pos_embed   
        x = self.proj(x)     
        out = x + shortcut
        
        return self.conv1x1(out)   
  
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"   
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32   
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)
  
    module = FMA(in_channel, out_channel).to(device)    
     
    outputs = module(inputs)   
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET) 

    print(ORANGE)     
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True, 
                                     output_precision=4,
                                     print_detailed=True)   
    print(RESET)  
