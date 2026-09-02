'''
本文件由BiliBili：魔傀面具整理
论文链接：https://arxiv.org/abs/2503.14012     
'''  

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')
    
import warnings 
warnings.filterwarnings('ignore')
from calflops import calculate_flops    
    
import torch, math  
import torch.nn as nn    
 
from engine.extre_module.ultralytics_nn.conv import Conv
 
class Conv_Extra(nn.Module):    
    def __init__(self, channel):
        super(Conv_Extra, self).__init__()
        self.block = nn.Sequential(Conv(channel, 64, 1),
                                   Conv(64, 64, 3),
                                   Conv(64, channel, 1, act=False))
    def forward(self, x):
        out = self.block(x)     
        return out
  
class Gaussian(nn.Module):
    def __init__(self, dim, size, sigma, feature_extra=True):
        super().__init__()
        self.feature_extra = feature_extra 
        gaussian = self.gaussian_kernel(size, sigma)
        gaussian = nn.Parameter(data=gaussian, requires_grad=False).clone()     
        self.gaussian = nn.Conv2d(dim, dim, kernel_size=size, stride=1, padding=int(size // 2), groups=dim, bias=False)
        self.gaussian.weight.data = gaussian.repeat(dim, 1, 1, 1)     
        self.norm = nn.BatchNorm2d(dim)  
        self.act = nn.SiLU()
        if feature_extra == True: 
            self.conv_extra = Conv_Extra(dim)  

    def forward(self, x):
        edges_o = self.gaussian(x)
        gaussian = self.act(self.norm(edges_o))
        if self.feature_extra == True:  
            out = self.conv_extra(x + gaussian)   
        else:     
            out = gaussian
        return out 
 
    def gaussian_kernel(self, size: int, sigma: float):
        kernel = torch.FloatTensor([    
            [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))   
             for x in range(-size // 2 + 1, size // 2 + 1)]  
             for y in range(-size // 2 + 1, size // 2 + 1)
             ]).unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()

class DRFD_LoG(nn.Module):
    def __init__(self, dim):
        super().__init__()  
        self.dim = dim
        self.outdim = dim * 2
        self.conv = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim)    
        self.conv_c = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=2, padding=1, groups=dim * 2)
        self.act_c = nn.SiLU()
        self.norm_c = nn.BatchNorm2d(dim * 2)
        self.max_m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm_m = nn.BatchNorm2d(dim * 2)     
        self.fusion = nn.Conv2d(dim * 4, self.outdim, kernel_size=1, stride=1)   
        # gaussian
        self.gaussian = Gaussian(self.outdim, 5, 0.5, feature_extra=False)    
        self.norm_g = nn.BatchNorm2d(self.outdim)     
    
    def forward(self, x):  # x = [B, C, H, W]
 
        x = self.conv(x)  # x = [B, 2C, H, W]
        gaussian = self.gaussian(x)
        x = self.norm_g(x + gaussian)   
        max = self.norm_m(self.max_m(x))  # m = [B, 2C, H/2, W/2]  
        conv = self.norm_c(self.act_c(self.conv_c(x)))  # c = [B, 2C, H/2, W/2]
        x = torch.cat([conv, max], dim=1)  # x = [B, 2C+2C, H/2, W/2]  -->  [B, 4C, H/2, W/2]
        x = self.fusion(x)  # x = [B, 4C, H/2, W/2]     -->  [B, 2C, H/2, W/2]     
  
        return x     
   
class LoGFilter(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, sigma):
        super(LoGFilter, self).__init__()
        # 7x7 convolution with stride 1 for feature reinforcement, Channels from 3 to 1/4C.
        self.conv_init = nn.Conv2d(in_c, out_c, kernel_size=7, stride=1, padding=3)
        """创建高斯-拉普拉斯核"""
        # 初始化二维坐标  
        ax = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=torch.float32) 
        xx, yy = torch.meshgrid(ax, ax)  
        # 计算高斯-拉普拉斯核
        kernel = (xx**2 + yy**2 - 2 * sigma**2) / (2 * math.pi * sigma**4) * torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        # 归一化  
        kernel = kernel - kernel.mean()    
        kernel = kernel / kernel.sum()
        log_kernel = kernel.unsqueeze(0).unsqueeze(0) # 添加 batch 和 channel 维度  
        self.LoG = nn.Conv2d(out_c, out_c, kernel_size=kernel_size, stride=1, padding=int(kernel_size // 2), groups=out_c, bias=False)
        self.LoG.weight.data = log_kernel.repeat(out_c, 1, 1, 1)
        self.act = nn.SiLU()   
        self.norm1 = nn.BatchNorm2d(out_c)    
        self.norm2 = nn.BatchNorm2d(out_c)   
   
    def forward(self, x):     
        # 7x7 convolution with stride 1 for feature reinforcement, Channels from 3 to 1/4C.    
        x = self.conv_init(x)  # x = [B, C/4, H, W]     
        LoG = self.LoG(x)
        LoG_edge = self.act(self.norm1(LoG))
        x = self.norm2(x + LoG_edge)
        return x
     
class LoGStem(nn.Module): 

    def __init__(self, in_chans, stem_dim):     
        super().__init__()
        out_c14 = int(stem_dim / 4)  # stem_dim / 2
        out_c12 = int(stem_dim / 2)  # stem_dim / 2
        # original size to 2x downsampling layer
        self.Conv_D = nn.Sequential(   
            nn.Conv2d(out_c14, out_c12, kernel_size=3, stride=1, padding=1, groups=out_c14),   
            Conv(out_c12, out_c12, 3, 2, g=out_c12)
            )     
        # 定义LoG滤波器
        self.LoG = LoGFilter(in_chans, out_c14, 7, 1.0) 
        # gaussian
        self.gaussian = Gaussian(out_c12, 9, 0.5)
        self.norm = nn.BatchNorm2d(out_c12)    
        self.drfd = DRFD_LoG(out_c12)

    def forward(self, x):
        x = self.LoG(x)   
        # original size to 2x downsampling layer     
        x = self.Conv_D(x)
        x = self.norm(x + self.gaussian(x))
        x = self.drfd(x)

        return x  # x = [B, C, H/4, W/4] 

if __name__ == '__main__':  
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')     
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)
 
    module = LoGStem(in_channel, out_channel).to(device)     
     
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,     
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,   
                                     output_precision=4,    
                                     print_detailed=True)    
    print(RESET)