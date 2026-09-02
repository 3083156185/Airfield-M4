'''    
本文件由BiliBili：魔傀面具整理     
engine/extre_module/module_images/自研模块-DCGRM.png
engine/extre_module/module_images/自研模块-DCGRM.md
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

  
class DCGRM(nn.Module):  
    def __init__(self, in_channel, out_channel):
        super().__init__()    
        self.channel = out_channel     
     
        self.conv1x1 = nn.ModuleList([   
            Conv(c, out_channel, 1) if c != out_channel else nn.Identity()   
            for c in in_channel
        ])    
  
        # Discrepancy branch: 3×3 conv to encode cross-branch mismatch (edge-like features)  
        self.disc_branch = DSConv(out_channel, out_channel, 3, act=nn.ReLU)  
        # Consistency branch: 1×1 conv to encode shared semantics
        self.cons_branch = Conv(out_channel, out_channel, 1, act=nn.ReLU)  
        # Fuse disc + cons into a single complementarity-aware feature stream   
        self.fusion_proj = Conv(out_channel * 2, out_channel, 1, act=nn.ReLU)
     
        # Adaptive tri-region soft mask — replaces fixed Sobel + hardcoded fg/confusion split 
        self.mask_gen = nn.Sequential( 
            Conv(out_channel, out_channel // 2, 3, act=nn.ReLU),
            nn.Conv2d(out_channel // 2, 3, 1),
        )
    
        # Q/K/V cross-attention (Q from fusion, K/V from tri-region context vectors)
        self.conv_query = Conv(out_channel, out_channel, 3, g=out_channel, act=nn.ReLU)   
        self.conv_key   = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.ReLU(inplace=True))
        self.conv_value = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.ReLU(inplace=True))  

        self.conv_scale = Conv(out_channel, out_channel, 3, g=out_channel, act=nn.ReLU)  
        self.sigmoid = nn.Sigmoid()   
        self.relu    = nn.ReLU(inplace=True)
     
    def forward(self, inputs):
        g, x = inputs    
        g = self.conv1x1[0](g) 
        x = self.conv1x1[1](x)  

        # Explicit discrepancy-complementarity decomposition
        disc = torch.abs(g - x)   # cross-branch mismatch → boundary / confusion regions   
        cons = g * x              # element-wise product  → shared semantic regions
 
        disc_feat = self.disc_branch(disc)   
        cons_feat = self.cons_branch(cons)
        fusion    = self.fusion_proj(torch.cat([disc_feat, cons_feat], dim=1))

        # Adaptive tri-region soft probability map (no fixed Sobel prior needed) 
        prob = F.softmax(self.mask_gen(disc_feat), dim=1)   # (B, 3, H, W)  
    
        # Region-weighted context vector extraction
        b, c, h, w = x.shape    
        f         = x.view(b, h * w, c)                               # (B, HW, C)   
        prob_flat = prob.view(b, 3, h * w)                            # (B, 3, HW)
        context   = torch.bmm(prob_flat, f)                           # (B, 3, C)   
        context   = context.permute(0, 2, 1).unsqueeze(3)            # (B, C, 3, 1) 
 
        # Cross-attention: Q from complementarity-aware fusion, K/V from context 
        query   = self.conv_query(fusion).view(b, c, -1).permute(0, 2, 1)  # (B, HW, C)
        key     = self.conv_key(context).view(b, c, -1)                    # (B, C, 3)
        value   = self.conv_value(context).view(b, c, -1).permute(0, 2, 1) # (B, 3, C) 

        sim     = torch.bmm(query, key) * (c ** -0.5)  # (B, HW, 3)
        sim     = F.softmax(sim, dim=-1)
 
        refined = torch.bmm(sim, value)                                     # (B, HW, C)
        refined = refined.permute(0, 2, 1).contiguous().view(b, c, h, w)   # (B, C, H, W) 
  
        scale = self.sigmoid(self.conv_scale(refined))
        return self.relu(fusion * scale)   
     

if __name__ == '__main__': 
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel_1, channel_2, height, width = 1, 32, 16, 32, 32     
    ouc_channel = 32
    inputs_1 = torch.randn((batch_size, channel_1, height, width)).to(device) 
    inputs_2 = torch.randn((batch_size, channel_2, height, width)).to(device)     

    module = DCGRM([channel_1, channel_2], ouc_channel).to(device)

    outputs = module([inputs_1, inputs_2])  
    print(GREEN + f'inputs1.size:{inputs_1.size()} inputs2.size:{inputs_2.size()} outputs.size:{outputs.size()}' + RESET)     
     
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,     
                                     args=[[inputs_1, inputs_2]],
                                     output_as_string=True,     
                                     output_precision=4,
                                     print_detailed=True)   
    print(RESET)
