'''   
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/自研模块-EdgeLAWDS.png
engine/extre_module/module_images/自研模块-EdgeLAWDS.md
自研模块：Edge-aware Light Adaptive-weight downsampling 
'''

import math
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')    

import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops     

import torch    
import torch.nn as nn   
import torch.nn.functional as F    
from einops import rearrange 

from engine.extre_module.ultralytics_nn.conv import Conv  


def _safe_groups(channels, group):   
    base = max(1, channels // max(1, group))   
    return max(1, math.gcd(channels, base))
    
  
class EdgeLAWDS(nn.Module):    
    # Edge-aware Light Adaptive-weight downsampling     
    def __init__(self, in_ch, out_ch, group=16, edge_ratio=0.5) -> None:     
        super().__init__()

        hidden_ch = max(8, int(in_ch * edge_ratio))
        groups = _safe_groups(in_ch, group)

        self.softmax = nn.Softmax(dim=-1)
        self.local_attention = nn.Sequential(  
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv(in_ch, in_ch, k=1)    
        )
        self.ds_conv = Conv(in_ch, in_ch * 4, k=3, s=2, g=groups)
   
        # Encode directional edge priors into an auxiliary 2x2 attention bias.
        self.edge_context = nn.Sequential(
            Conv(in_ch + 5, hidden_ch, k=3),
            Conv(hidden_ch, in_ch, k=1, act=False),
        )
        self.edge_bias = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True)  
        self.edge_residual = Conv(in_ch, in_ch, k=1, act=False)   

        self.edge_scale = nn.Parameter(torch.tensor(1.0))
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.proj = Conv(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()     
    
        sobel_x = torch.tensor([     
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],    
            [-1.0, 0.0, 1.0],
        ])
        sobel_y = torch.tensor([
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0], 
            [1.0, 2.0, 1.0],
        ])   
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3), persistent=False) 
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3), persistent=False) 

    @staticmethod  
    def _pad_to_even(x):
        _, _, h, w = x.shape   
        pad_h = h % 2     
        pad_w = w % 2 
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        return x    
  
    def _edge_features(self, x):
        gray = x.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        mag = torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)
        orientation = torch.cat( 
            (grad_x.abs(), grad_y.abs(), (grad_x - grad_y).abs(), (grad_x + grad_y).abs()),
            dim=1,
        )    
        return mag, orientation
   
    def forward(self, x):
        x = self._pad_to_even(x)
        mag, orientation = self._edge_features(x)

        local_att = rearrange(    
            self.local_attention(x),     
            'bs ch (s1 h) (s2 w) -> bs ch h w (s1 s2)',
            s1=2,
            s2=2,
        )     

        edge_hidden = self.edge_context(torch.cat((x, mag, orientation), dim=1))    
        edge_att = rearrange(
            self.edge_bias(edge_hidden),
            'bs ch (s1 h) (s2 w) -> bs ch h w (s1 s2)',  
            s1=2,
            s2=2,   
        ) 
        att = self.softmax(local_att + self.edge_scale * edge_att)
 
        x_local = rearrange(self.ds_conv(x), 'bs (s ch) h w -> bs ch h w s', s=4)
        x_local = torch.sum(x_local * att, dim=-1)    

        edge_residual = self.edge_residual(F.avg_pool2d(edge_hidden, kernel_size=2, stride=2)) 
        return self.proj(x_local + self.residual_scale * edge_residual) 
 
   
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32    
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)   
  
    module = EdgeLAWDS(in_channel, out_channel, group=16).to(device)

    outputs = module(inputs)   
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
    
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),   
                                     output_as_string=True, 
                                     output_precision=4,   
                                     print_detailed=True)
    print(RESET)  
