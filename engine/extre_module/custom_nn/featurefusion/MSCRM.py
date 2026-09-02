'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/自研模块-MSCRM.png     
engine/extre_module/module_images/自研模块-MSCRM.md     
'''

import warnings
warnings.filterwarnings('ignore')     
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')
from calflops import calculate_flops
import torch   
import torch.nn as nn   
import torch.nn.functional as F
  
from engine.extre_module.ultralytics_nn.conv import Conv, DSConv
  

class _MultiScaleStripPool(nn.Module):
    """Aggregate horizontal and vertical strip contexts at two spatial scales."""
    def __init__(self, n_strips: int = 4):
        super().__init__()
        self.h_pool1 = nn.AdaptiveAvgPool2d((n_strips,     1))
        self.w_pool1 = nn.AdaptiveAvgPool2d((1,     n_strips))
        self.h_pool2 = nn.AdaptiveAvgPool2d((n_strips * 2, 1))
        self.w_pool2 = nn.AdaptiveAvgPool2d((1, n_strips * 2))    
        # total context tokens: n_strips + n_strips + 2*n_strips + 2*n_strips = 6*n_strips   
        self.n_ctx = n_strips * 6   

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]    
        h1 = self.h_pool1(x).view(b, c, -1)   # (B, C, n_strips)
        w1 = self.w_pool1(x).view(b, c, -1)   # (B, C, n_strips)
        h2 = self.h_pool2(x).view(b, c, -1)   # (B, C, n_strips*2)
        w2 = self.w_pool2(x).view(b, c, -1)   # (B, C, n_strips*2)
        return torch.cat([h1, w1, h2, w2], dim=2)  # (B, C, n_ctx)    
 

class MSCRM(nn.Module):    
    def __init__(self, in_channel, out_channel, n_strips: int = 4):
        super().__init__()  
        self.channel = out_channel
     
        self.conv1x1 = nn.ModuleList([
            Conv(c, out_channel, 1) if c != out_channel else nn.Identity()
            for c in in_channel  
        ])   

        # Initial additive fusion    
        self.conv_fuse = DSConv(out_channel, out_channel, 3, act=nn.ReLU)    
  
        # Multi-scale horizontal + vertical strip pooling (replaces ExternalAttention)  
        self.strip_pool = _MultiScaleStripPool(n_strips)     
        # Context projection applied on (B, C, n_ctx, 1)
        self.ctx_proj = nn.Conv2d(out_channel, out_channel, 1)     

        # Q from pixel-level spatial features; K/V from multi-scale strip context
        self.conv_query = DSConv(out_channel, out_channel, 3, act=nn.ReLU)
        self.conv_key   = nn.Conv2d(out_channel, out_channel, 1)
        self.conv_value = nn.Conv2d(out_channel, out_channel, 1)

        self.conv_scale = DSConv(out_channel, out_channel, 3, act=nn.ReLU)    
        self.sigmoid = nn.Sigmoid()
        self.relu    = nn.ReLU(inplace=True)
   
    def forward(self, inputs):     
        g, x = inputs
        g = self.conv1x1[0](g) 
        x = self.conv1x1[1](x)

        # Additive fusion     
        fusion = self.conv_fuse(g + x)   # (B, C, H, W)
   
        # Multi-scale strip context: encodes directional shape priors at two granularities     
        ctx    = self.strip_pool(fusion)             # (B, C, n_ctx) 
        ctx_4d = self.ctx_proj(ctx.unsqueeze(3))     # (B, C, n_ctx, 1)    

        # Cross-attention: each pixel queries all strip contexts   
        b, c, h, w = x.shape    
        query   = self.conv_query(x).view(b, c, -1).permute(0, 2, 1)       # (B, HW, C)
        key     = self.conv_key(ctx_4d).view(b, c, -1)                     # (B, C, n_ctx)
        value   = self.conv_value(ctx_4d).view(b, c, -1).permute(0, 2, 1)  # (B, n_ctx, C)
 
        sim     = torch.bmm(query, key) * (c ** -0.5)   # (B, HW, n_ctx)
        sim     = F.softmax(sim, dim=-1)   
 
        refined = torch.bmm(sim, value)                                      # (B, HW, C)    
        refined = refined.permute(0, 2, 1).contiguous().view(b, c, h, w)    # (B, C, H, W)
  
        scale = self.sigmoid(self.conv_scale(refined))    
        return self.relu(x * scale)   


if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel_1, channel_2, height, width = 1, 32, 16, 32, 32     
    ouc_channel = 32
    inputs_1 = torch.randn((batch_size, channel_1, height, width)).to(device)
    inputs_2 = torch.randn((batch_size, channel_2, height, width)).to(device)   
   
    module = MSCRM([channel_1, channel_2], ouc_channel).to(device)  
   
    outputs = module([inputs_1, inputs_2])    
    print(GREEN + f'inputs1.size:{inputs_1.size()} inputs2.size:{inputs_2.size()} outputs.size:{outputs.size()}' + RESET)    
  
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     args=[[inputs_1, inputs_2]],
                                     output_as_string=True,  
                                     output_precision=4,    
                                     print_detailed=True)
    print(RESET)
