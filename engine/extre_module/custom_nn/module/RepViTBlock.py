'''
本文件由BiliBili：魔傀面具整理   
engine/extre_module/module_images/CVPR2024-RepViTBlock.png
engine/extre_module/module_images/CVPR2024-RepViTBlock.md
论文链接：https://arxiv.org/pdf/2307.09283 
'''    
 
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')
  
import warnings  
warnings.filterwarnings('ignore')
from calflops import calculate_flops

import torch
import torch.nn as nn
from timm.layers import SqueezeExcite     

from engine.extre_module.ultralytics_nn.conv import Conv     
from engine.extre_module.ultralytics_nn.block import RepVGGDW

class Conv2d_BN(torch.nn.Sequential):     
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,    
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__() 
        self.add_module('c', torch.nn.Conv2d(    
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b)) 
        torch.nn.init.constant_(self.bn.weight, bn_weight_init) 
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()  
    def convert_to_deploy(self):     
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size( 
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device) 
        m.weight.data.copy_(w)   
        m.bias.data.copy_(b)   
        return m
  
class Residual(nn.Module):   
    def __init__(self, fn):   
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x):     
        return self.fn(x) + x     

class RepViTBlock(nn.Module):
    def __init__(self, inp, oup, use_se=True):
        super(RepViTBlock, self).__init__()
 
        self.identity = inp == oup 
        hidden_dim = 2 * inp     

        self.token_mixer = nn.Sequential( 
            RepVGGDW(inp),     
            SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
        )
        self.channel_mixer = nn.Sequential(
                # pw
                Conv(inp, hidden_dim, 1, act=False),
                nn.GELU(),    
                # pw-linear
                Conv(hidden_dim, oup, 1, act=False),
            )

    def forward(self, x):     
        return x + self.channel_mixer(self.token_mixer(x)) if self.identity else self.channel_mixer(self.token_mixer(x))
     
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)
 
    module = RepViTBlock(in_channel, out_channel).to(device)

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,  
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,   
                                     output_precision=4,
                                     print_detailed=True)  
    print(RESET)