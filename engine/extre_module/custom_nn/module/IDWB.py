'''
本文件由BiliBili：魔傀面具整理    
engine/extre_module/module_images/CVPR2024-InceptionDWBlock.png     
论文链接：https://arxiv.org/pdf/2303.16900 
'''    
   
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')     

import warnings
warnings.filterwarnings('ignore')  
from calflops import calculate_flops
  
import torch
import torch.nn as nn    
from timm.layers import to_2tuple, DropPath     
     
from engine.extre_module.ultralytics_nn.conv import Conv    
  
class InceptionDWConv2d(nn.Module):    
    """ Inception depthweise convolution  
    """     
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
    
        gc = int(in_channels * branch_ratio) # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size//2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)
        
    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat( 
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), 
            dim=1, 
        )    

class ConvMlp(nn.Module):     
    """ MLP using 1x1 convs that keeps spatial dims
    copied from timm: https://github.com/huggingface/pytorch-image-models/blob/v0.6.11/timm/models/layers/mlp.py 
    """   
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,    
            norm_layer=None, bias=True, drop=0.):
        super().__init__()    
        out_features = out_features or in_features   
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])     
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])    

    def forward(self, x):    
        x = self.fc1(x)   
        x = self.norm(x)  
        x = self.act(x)
        x = self.drop(x)     
        x = self.fc2(x)  
        return x    
 
class InceptionDWBlock(nn.Module):    
    """ MetaNeXtBlock Block  
    Args:    
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.     
    """   

    def __init__(   
            self,
            in_dim,
            dim,  
            token_mixer=InceptionDWConv2d, 
            norm_layer=nn.BatchNorm2d, 
            mlp_layer=ConvMlp,   
            mlp_ratio=4,    
            act_layer=nn.GELU,
            ls_init_value=1e-6,
            drop_path=0.,   
            
    ):  
        super().__init__()    
        self.token_mixer = token_mixer(dim)  
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)     
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity() 
   
        self.conv1x1 = Conv(in_dim, dim) if in_dim != dim else nn.Identity() 
    
    def forward(self, x):  
        x = self.conv1x1(x)
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)   
        x = self.mlp(x)    
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))   
        x = self.drop_path(x) + shortcut  
        return x    
   
if __name__ == '__main__': 
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = InceptionDWBlock(in_channel, out_channel).to(device)     

    outputs = module(inputs)     
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
    
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,    
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,
                                     output_precision=4, 
                                     print_detailed=True)
    print(RESET)   
