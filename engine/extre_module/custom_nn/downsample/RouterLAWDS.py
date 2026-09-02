'''   
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/自研模块-RouterLAWDS.png
engine/extre_module/module_images/自研模块-RouterLAWDS.md
自研模块：Frequency-guided Light Adaptive-weight Downsampling  
'''

import os, sys 
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..') 

import warnings
warnings.filterwarnings('ignore') 
from calflops import calculate_flops
     
import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops    
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange 
  
from engine.extre_module.ultralytics_nn.conv import Conv
   

     
def _select_groups(c1, c2, group):
    target = max(1, c1 // group) if group > 0 else 1
    while target > 1 and (c1 % target != 0 or c2 % target != 0):
        target -= 1 
    return target
     

class _LocalLAWDSBranch(nn.Module):
    def __init__(self, in_ch, out_ch, group):   
        super().__init__()
        conv_groups = _select_groups(in_ch, out_ch * 4, group)
        self.attention = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv(in_ch, out_ch, k=1),
        )
        self.ds_conv = Conv(in_ch, out_ch * 4, k=3, s=2, g=conv_groups)  
        self.softmax = nn.Softmax(dim=-1)   
  
    def forward(self, x): 
        att = rearrange(self.attention(x), 'b c (s1 h) (s2 w) -> b c h w (s1 s2)', s1=2, s2=2)
        att = self.softmax(att)    
        feat = rearrange(self.ds_conv(x), 'b (s c) h w -> b c h w s', s=4)
        return torch.sum(feat * att, dim=-1)    


class _PoolBranch(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.project = Conv(in_ch * 2, out_ch, k=1)
     
    def forward(self, x): 
        avg = F.avg_pool2d(x, kernel_size=2, stride=2) 
        max_ = F.max_pool2d(x, kernel_size=2, stride=2)     
        return self.project(torch.cat([avg, max_], dim=1))
    

class _DepthwiseBranch(nn.Module): 
    def __init__(self, in_ch, out_ch):
        super().__init__() 
        self.branch = nn.Sequential(
            Conv(in_ch, in_ch, k=3, s=2, g=in_ch),
            Conv(in_ch, out_ch, k=1),
        )
  
    def forward(self, x): 
        return self.branch(x)    
     

class RouterLAWDS(nn.Module):     
    def __init__(self, in_ch, out_ch, group=16, branch_ratio=0.25) -> None:     
        super().__init__()
        hidden = max(8, int(in_ch * branch_ratio))
        router_groups = _select_groups(in_ch, hidden, group)    

        self.local_branch = _LocalLAWDSBranch(in_ch, out_ch, group)
        self.pool_branch = _PoolBranch(in_ch, out_ch)     
        self.depthwise_branch = _DepthwiseBranch(in_ch, out_ch)     
   
        self.global_router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, kernel_size=1, bias=True),
        )   
        self.local_router = nn.Sequential( 
            Conv(in_ch, hidden, k=3, s=2, g=router_groups),   
            nn.Conv2d(hidden, 3, kernel_size=1, bias=True),  
        ) 
        self.output = Conv(out_ch, out_ch, k=1)    
   
    @staticmethod
    def _pad_to_even(x):  
        _, _, h, w = x.shape 
        pad_h = h % 2
        pad_w = w % 2    
        if pad_h == 0 and pad_w == 0: 
            return x
        return F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
  
    def forward(self, x):
        x = self._pad_to_even(x)    

        local_feat = self.local_branch(x)
        pool_feat = self.pool_branch(x)
        depth_feat = self.depthwise_branch(x)

        router_logits = self.local_router(x) + self.global_router(x) 
        router_weight = torch.softmax(router_logits, dim=1)     

        fused = (
            router_weight[:, 0:1] * local_feat +
            router_weight[:, 1:2] * pool_feat +    
            router_weight[:, 2:3] * depth_feat   
        )
        return self.output(fused)    
   

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"     
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')     
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)
  
    module = RouterLAWDS(in_channel, out_channel, group=16).to(device)

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
  
    print(ORANGE)
    flops, macs, _ = calculate_flops(
        model=module,
        input_shape=(batch_size, in_channel, height, width),  
        output_as_string=True,  
        output_precision=4, 
        print_detailed=True,  
    )
    print(RESET)    
