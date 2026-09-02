'''   
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/CVPR2026-DEGConv.png   
engine/extre_module/module_images/CVPR2026-DEGConv.md 
论文链接：https://arxiv.org/pdf/2603.01361 
'''
  
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')
 
import warnings
warnings.filterwarnings('ignore')     
from calflops import calculate_flops 

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
   
from engine.extre_module.ultralytics_nn.conv import Conv     
     
def image2patches(x):     
    x = einops.rearrange(x, "b c (hg h) (wg w) -> (hg wg b) c h w", hg=2, wg=2)
    return x     
    

def patches2image(x):
    x = einops.rearrange(x, "(hg wg b) c h w -> b c (hg h) (wg w)", hg=2, wg=2)  
    return x

class EdgeConv(nn.Module):
    def __init__(
        self,    
        in_channels,
        mid_channels,
        out_channels,
        kernel_size=3,
        bias=True,
    ):     
        super().__init__()

        self.in_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,    
            bias=bias,
        )
        self.w_conv = nn.Conv2d(     
            mid_channels,  
            mid_channels,  
            kernel_size=(1, kernel_size),
            stride=1,
            padding=(0, kernel_size // 2),
            groups=mid_channels, 
        ) 

        self.h_conv = nn.Conv2d(
            mid_channels,
            mid_channels,  
            kernel_size=(kernel_size, 1),
            stride=1,
            padding=(kernel_size // 2, 0),    
            groups=mid_channels,   
        )

        self.out_proj = nn.Conv2d(
            in_channels=mid_channels * 2,
            out_channels=out_channels,
            kernel_size=1,
            bias=True,
        )    

    def forward(self, x):    
        x = self.in_proj(x)
        x_w = self.w_conv(x)
        x_h = self.h_conv(x)
        x = torch.cat([x_w, x_h], dim=1)
        x = self.out_proj(x)
        return x
   

class DEGConv(nn.Module):
    def __init__(self, in_dim, out_dim, nbins=36, cell_size=(4, 4)):
        super().__init__()     

        self.nbins = nbins  
        self.cell_size = cell_size

        self.hog_feat = nn.Sequential(    
            nn.Conv2d(nbins, in_dim, kernel_size=1), 
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim, bias=False),
            nn.GroupNorm(in_dim // 8, in_dim),   
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),   
        )  

        self.weight = nn.Sequential(
            EdgeConv(in_channels=in_dim, mid_channels=in_dim // 2, out_channels=in_dim),
            nn.GroupNorm(in_dim // 8, in_dim),   
        )   
    
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, stride=1),
            nn.GroupNorm(in_dim // 8, in_dim),
        )
 
        self.fuse_block = nn.Sequential(   
            EdgeConv(in_channels=in_dim, mid_channels=in_dim // 2, out_channels=in_dim, kernel_size=3),
            nn.GroupNorm(in_dim // 8, in_dim),    
        )
    
        self.sigmoid = nn.Sigmoid()   

        self.conv_1x1 = Conv(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x):     
        residual = x

        x = image2patches(x)  
     
        x_hog = self.get_hog_feature(x)     
        x_hog = self.hog_feat(x_hog)

        x1 = self.sigmoid(self.weight(x + x_hog))
        x2 = self.conv(x) 
        x = x1 * x2
  
        x = patches2image(x)
  
        x = x + residual
        x = self.fuse_block(x)
  
        return self.conv_1x1(x)

    def get_hog_feature(self, x):
        x_mean = x.mean(dim=1, keepdim=True)
        b, _, h, w = x_mean.shape     
        device = x_mean.device 
  
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device) 
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device) 
        dx = F.conv2d(x_mean.float(), sobel_x, padding=1)    
        dy = F.conv2d(x_mean.float(), sobel_y, padding=1)

        gradient_dir = torch.atan2(dy, dx)    
        gradient_dir = torch.abs(gradient_dir) 
   
        cell_h, cell_w = self.cell_size    
        h_cells = int(h / cell_h)   
        w_cells = int(w / cell_w)     

        dirs_crop = gradient_dir[:, :, : h_cells * cell_h, : w_cells * cell_w]    

        dirs = dirs_crop.reshape(b, h_cells, w_cells, -1)  
     
        bin_with = torch.pi / self.nbins
        bin_indices = (dirs / bin_with).floor().long()    
        bin_indices = torch.clamp(bin_indices, 0, self.nbins - 1) 

        bin_indices_flat = bin_indices.reshape(b * h_cells * w_cells, dirs.shape[-1])
        weight = []
        for i in range(bin_indices_flat.shape[0]):
            bins = bin_indices_flat[i]    
            count = torch.bincount(bins, minlength=self.nbins)    
            weight.append(count)
   
        weight = torch.stack(weight, dim=0).reshape(b, h_cells, w_cells, -1) / 64

        start = torch.pi / (2 * self.nbins)
        hog_feature = torch.linspace(start, torch.pi - start, self.nbins).to(device).repeat(b, h_cells, w_cells, 1) * weight     

        return hog_feature.permute(0, 3, 1, 2)
  
if __name__ == '__main__': 
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"     
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 20, 20
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = DEGConv(in_channel, out_channel).to(device) 

    outputs = module(inputs)   
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)   
     
    print(ORANGE)   
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,  
                                     print_detailed=True)
    print(RESET)   
