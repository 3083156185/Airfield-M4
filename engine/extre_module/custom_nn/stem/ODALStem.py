'''  
本文件由BiliBili：魔傀面具整理  
engine/extre_module/module_images/自研模块-ODALStem.png    
engine/extre_module/module_images/自研模块-ODALStem.md
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

class FixedLoGConv(nn.Module):     
    def __init__(self, dim, kernel_size, sigma):
        super().__init__()    
        kernel = self.log_kernel(kernel_size, sigma)  
        self.filter = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=int(kernel_size // 2), groups=dim, bias=False)    
        self.filter.weight.data = kernel.repeat(dim, 1, 1, 1)  
        self.filter.weight.requires_grad_(False)    

    def forward(self, x):
        return self.filter(x)

    @staticmethod
    def log_kernel(kernel_size: int, sigma: float):
        ax = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = (xx**2 + yy**2 - 2 * sigma**2) / (2 * math.pi * sigma**4) * torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))  
        kernel = kernel - kernel.mean()    
        kernel = kernel / kernel.abs().sum().clamp_min(1e-6)
        return kernel.unsqueeze(0).unsqueeze(0)   

   
class DirectionalLoGFilter(nn.Module):     
    def __init__(self, dim, kernel_size=5, sigma=0.8):     
        super().__init__()
        self.filters = nn.ModuleList(
            [
                nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=int(kernel_size // 2), groups=dim, bias=False)
                for _ in range(4)
            ]
        )  
        kernels = self.directional_kernels(kernel_size, sigma)
        for conv, kernel in zip(self.filters, kernels):
            conv.weight.data = kernel.repeat(dim, 1, 1, 1)
            conv.weight.requires_grad_(False)
   
    def forward(self, x):
        return torch.stack([conv(x) for conv in self.filters], dim=1)
 
    @staticmethod
    def directional_kernels(kernel_size: int, sigma: float):     
        if kernel_size < 3 or kernel_size % 2 == 0: 
            raise ValueError('kernel_size must be an odd integer greater than or equal to 3')

        ax = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')     
        gaussian = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))     
        center = kernel_size // 2
        second = torch.zeros(kernel_size, dtype=torch.float32)     
        second[center - 1], second[center], second[center + 1] = 1.0, -2.0, 1.0

        horizontal = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)
        horizontal[center, :] = second

        vertical = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)     
        vertical[:, center] = second
 
        diag_main = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)   
        diag_anti = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)
        for idx, value in enumerate(second):    
            diag_main[idx, idx] = value    
            diag_anti[idx, kernel_size - 1 - idx] = value
  
        kernels = [] 
        for base in [horizontal, vertical, diag_main, diag_anti]:  
            kernel = base * gaussian 
            kernel = kernel - kernel.mean()     
            kernel = kernel / kernel.abs().sum().clamp_min(1e-6)
            kernels.append(kernel.unsqueeze(0).unsqueeze(0))    
        return kernels    
   
    
class OrientationGate(nn.Module): 
    def __init__(self, dim, num_paths=4, reduction=4):   
        super().__init__() 
        hidden = max(dim // reduction, num_paths)    
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(dim, hidden, kernel_size=1, stride=1) 
        self.act = nn.SiLU()
        self.fc2 = nn.Conv2d(hidden, num_paths, kernel_size=1, stride=1)  
  
    def forward(self, x):  
        weights = self.fc2(self.act(self.fc1(self.pool(x)))).flatten(1)   
        return torch.softmax(weights, dim=1)
 

class ODALBlock(nn.Module):
    def __init__(self, in_c, out_c, iso_kernel_size=7, iso_sigma=1.0, dir_kernel_size=5, dir_sigma=0.8):   
        super().__init__()    
        self.conv_init = nn.Conv2d(in_c, out_c, kernel_size=7, stride=1, padding=3)   
        self.iso_filter = FixedLoGConv(out_c, iso_kernel_size, iso_sigma)
        self.dir_filter = DirectionalLoGFilter(out_c, dir_kernel_size, dir_sigma)
        self.orientation_gate = OrientationGate(out_c)
        self.iso_norm = nn.BatchNorm2d(out_c)     
        self.dir_norm = nn.BatchNorm2d(out_c)
        self.out_norm = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU()
        self.direction_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        x = self.conv_init(x)
        iso = self.act(self.iso_norm(self.iso_filter(x)))
        direction_weights = self.orientation_gate(x)
        directional = self.dir_filter(x)  
        directional = torch.sum(directional * direction_weights[:, :, None, None, None], dim=1) 
        directional = self.act(self.dir_norm(directional))
        scale = torch.sigmoid(self.direction_scale)
        out = self.act(self.out_norm(x + iso + scale * directional))
        return out, direction_weights
    

class ODALStem(nn.Module):

    def __init__(self, in_chans, stem_dim):  
        super().__init__()   
        out_c14 = int(stem_dim / 4)     
        out_c12 = int(stem_dim / 2)
        self.Conv_D = nn.Sequential(
            nn.Conv2d(out_c14, out_c12, kernel_size=3, stride=1, padding=1, groups=out_c14),
            Conv(out_c12, out_c12, 3, 2, g=out_c12) 
        )
        self.odal = ODALBlock(in_chans, out_c14, 7, 1.0, 5, 0.8)
        self.gaussian = Gaussian(out_c12, 9, 0.5)  
        self.norm = nn.BatchNorm2d(out_c12)    
        self.drfd = DRFD_LoG(out_c12)  
        self.last_orientation_weights = None

    def forward(self, x):
        x, direction_weights = self.odal(x)
        self.last_orientation_weights = direction_weights.detach()     
        x = self.Conv_D(x)
        x = self.norm(x + self.gaussian(x))
        x = self.drfd(x)
        return x


if __name__ == '__main__': 
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"     
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32   
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = ODALStem(in_channel, out_channel).to(device)    
    
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)    
     
    print(ORANGE)   
    flops, macs, _ = calculate_flops(model=module,     
                                     input_shape=(batch_size, in_channel, height, width),     
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET) 
