'''
本文件由BiliBili：魔傀面具整理    
engine/extre_module/module_images/自研模块-FasterCGABlock.png   
engine/extre_module/module_images/自研模块-FasterCGABlock.md    
'''
  
import os, sys    
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')   

import warnings  
warnings.filterwarnings('ignore')
from calflops import calculate_flops  
     
import torch     
import torch.nn as nn
from timm.layers import DropPath
    
from engine.extre_module.ultralytics_nn.conv import Conv
    

def _check_kernel_size(kernel_size, name):
    if kernel_size % 2 == 0:
        raise ValueError(f"{name} should be odd")

     
class PartialConv3(nn.Module):
    def __init__(self, dim, n_div=4, forward='split_cat'):
        super().__init__()
        if dim < n_div: 
            raise ValueError("dim should be greater than or equal to n_div")   
        self.dim_conv3 = dim // n_div  
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False) 
    
        if forward == 'slicing':
            self.forward = self.forward_slicing   
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat     
        else:
            raise NotImplementedError   
    
    def forward_slicing(self, x):
        x = x.clone()   
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x
  
    def forward_split_cat(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        return torch.cat((x1, x2), 1)     

   
class CrossGatedAggregation(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0, context_kernel=5, gate_kernel=3):  
        super().__init__()
        _check_kernel_size(context_kernel, "context_kernel")
        _check_kernel_size(gate_kernel, "gate_kernel")
     
        hidden_dim = max(dim, int(dim * mlp_ratio))
        self.context_seed = Conv(dim, hidden_dim, 1)
        self.gate_seed = Conv(dim, hidden_dim, 1)
        self.context_branch = nn.Sequential(  
            nn.Conv2d(hidden_dim, hidden_dim, context_kernel, 1, context_kernel // 2, groups=hidden_dim, bias=False), 
            nn.BatchNorm2d(hidden_dim), 
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),    
            nn.GELU(),     
        )     
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, gate_kernel, 1, gate_kernel // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Sigmoid(),   
        )
        self.channel_gate = nn.Sequential(   
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=True),
            nn.Sigmoid(),  
        )
        self.selection_proj = nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False)     
        self.project = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),     
        )

    def forward(self, x):     
        context_seed = self.context_seed(x)
        gate_seed = self.gate_seed(x)   

        context = self.context_branch(context_seed)
        context_out = context * self.spatial_gate(gate_seed)   
        gate_out = self.selection_proj(gate_seed * self.channel_gate(context))  
        return self.project(context_out + gate_out)

    
class Faster_CGA_Block(nn.Module): 
    def __init__(
        self,
        inc,
        ouc,   
        n_div=4,   
        mlp_ratio=2, 
        drop_path=0.1,
        layer_scale_init_value=0.0, 
        pconv_fw_type='split_cat',
        context_kernel=5,     
        gate_kernel=3,     
    ): 
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.adjust_channel = Conv(inc, ouc, 1) if inc != ouc else None
        self.spatial_mixing = PartialConv3(ouc, n_div, pconv_fw_type)     
        self.cross_gated_aggregation = CrossGatedAggregation( 
            ouc,
            mlp_ratio=mlp_ratio,   
            context_kernel=context_kernel,  
            gate_kernel=gate_kernel,
        )
        self.layer_scale = None
        if layer_scale_init_value > 0:     
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((ouc)), requires_grad=True) 

    def forward(self, x):     
        if self.adjust_channel is not None:    
            x = self.adjust_channel(x)

        shortcut = x   
        x = self.spatial_mixing(x)
        x = self.cross_gated_aggregation(x) 
        if self.layer_scale is not None:
            x = self.layer_scale.view(1, -1, 1, 1) * x
        return shortcut + self.drop_path(x)   

  
if __name__ == '__main__':  
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)
     
    print(RED + '-' * 20 + " Faster_CGA_Block " + '-' * 20 + RESET)

    module = Faster_CGA_Block(in_channel, out_channel).to(device)

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
    
    print(ORANGE)    
    flops, macs, _ = calculate_flops(model=module,    
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)     
    print(RESET)   
