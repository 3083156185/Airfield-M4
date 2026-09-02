''' 
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/CVPR2025-nnWNet.png
engine/extre_module/module_images/CVPR2025-nnWNet.md   
论文链接：https://openaccess.thecvf.com/content/CVPR2025/papers/Zhou_nnWNet_Rethinking_the_Use_of_Transformers_in_Biomedical_Image_Segmentation_CVPR_2025_paper.pdf
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

BNNorm2d = nn.BatchNorm2d  
LNNorm = nn.LayerNorm
Activation = nn.GELU     
     
__all__ = ['GlobalBlock', 'LocalBlock', 'OPE'] 

class Pooling(nn.Module):
    def __init__(self, pool_size=3):
        super().__init__()  
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x   

class up_conv(nn.Module):     
    def __init__(self, ch_in, ch_out):     
        super(up_conv, self).__init__()  
        self.up = nn.Sequential(     
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            Conv(ch_in, ch_out, k=3, act=Activation)   
        )    

    def forward(self, x):    
        x = self.up(x)
        return x
     
class down_conv(nn.Module):   
    def __init__(self, ch_in, ch_out):    
        super(down_conv, self).__init__()    
        self.down = Conv(ch_in, ch_out, k=3, s=2, act=Activation)
   
    def forward(self, x):
        x = self.down(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, groups=1):  
        super(ResBlock, self).__init__()     
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = Conv(inplanes, planes, k=3, act=Activation)   
        self.conv2 = Conv(planes, planes, k=3, g=groups, act=False) 
        self.act = Activation()
    
        if self.inplanes != self.planes:  
            self.down = Conv(inplanes, planes, act=False)
    
    def forward(self, x):

        identity = x 

        out = self.conv1(x)

        out = self.conv2(out) 

        if self.inplanes != self.planes:   
            identity = self.down(x)   

        out = out + identity
        out = self.act(out)   

        return out
    
class OPE(nn.Module):  
    def __init__(self, inplanes, planes):
        super(OPE, self).__init__()
        self.conv1 = Conv(inplanes, inplanes, 3, act=Activation)
        self.down = down_conv(inplanes, planes)

    def forward(self, x):  
        out = self.conv1(x)
        out = self.down(out) 

        return out 
  
class LocalBlock(nn.Module):
    def __init__(self, inplanes, planes, down_or_up=None):     
        super(LocalBlock, self).__init__()
        hidden_planes = planes // 2    
        if down_or_up is None:
            self.BasicBlock = nn.Sequential(
                ResBlock(inplanes=inplanes, planes=hidden_planes, groups=4),    
                Conv(hidden_planes, planes, k=1, act=Activation)  
            )    
        elif down_or_up == 'down':   
            self.BasicBlock = nn.Sequential(
                ResBlock(inplanes=inplanes, planes=hidden_planes, groups=4),   
                down_conv(hidden_planes, planes)  
            )
        elif down_or_up == 'up':     
            self.BasicBlock = nn.Sequential(    
                ResBlock(inplanes=inplanes, planes=hidden_planes, groups=4),
                up_conv(hidden_planes, planes),   
            )

    def forward(self, x):
        out = self.BasicBlock(x)
        return out   
  
class GroupNorm(nn.GroupNorm):     
    def __init__(self, num_channels, **kwargs):   
        super().__init__(1, num_channels, **kwargs)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super().__init__()

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)     
        self.act = Activation()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)  
        self.drop = nn.Dropout(drop)

    def forward(self, x):  
        x = self.fc1(x)  
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)   
        x = self.drop(x)     
        return x
   
class GlobalBlock(nn.Module): 
    def __init__(self, in_dim, dim, pool_size=3, mlp_ratio=4., drop=0., drop_path=0.):  
        super(GlobalBlock, self).__init__()   
  
        self.in_dim = in_dim     
        self.dim = dim    
    
        self.proj = nn.Conv2d(in_dim, dim, kernel_size=3, padding=1, groups=4)
        self.norm1 = GroupNorm(dim)
        self.attn = Pooling(pool_size=pool_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()    
        self.norm2 = GroupNorm(dim) 
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, drop=drop)   
  
    def forward(self, x):  
        x = self.proj(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))    
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x    

if __name__ == '__main__':     
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"     
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)     

    print(RED + '-'*20 + " GlobalBlock " + '-'*20 + RESET)

    module = GlobalBlock(in_channel, out_channel).to(device)

    outputs = module(inputs) 
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)    
  
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,  
                                     input_shape=(batch_size, in_channel, height, width),   
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)

    print(RED + '-'*20 + " LocalBlock " + '-'*20 + RESET)  
  
    module = LocalBlock(in_channel, out_channel).to(device)
 
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),   
                                     output_as_string=True,     
                                     output_precision=4,     
                                     print_detailed=True)
    print(RESET)

    print(RED + '-'*20 + " OPE " + '-'*20 + RESET)  

    module = OPE(in_channel, out_channel).to(device)    
   
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET) 
     
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,     
                                     input_shape=(batch_size, in_channel, height, width),   
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)     
    print(RESET)