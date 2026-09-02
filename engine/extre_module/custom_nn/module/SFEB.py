'''
本文件由BiliBili：魔傀面具整理    
engine/extre_module/module_images/CVPR2026-SFEB.png
engine/extre_module/module_images/CVPR2026-SFEB.md
论文链接：https://arxiv.org/pdf/2603.18834
'''

import warnings
warnings.filterwarnings('ignore')     
from calflops import calculate_flops     
     
import torch     
import torch.nn as nn   
import torch.nn.functional as F   
    
def _require_even_channels(channels, name):   
    if channels <= 0:
        raise ValueError(f"{name} must be positive, got {channels}.") 
    if channels % 2 != 0:   
        raise ValueError(
            f"{name} must be even because the block splits channels into "  
            f"local/global halves, got {channels}."     
        )     


class ChannelAttention(nn.Module): 
    def __init__(self, in_channels, reduction_ratio=16):  
        super().__init__()
        hidden_channels = max(1, in_channels // reduction_ratio)  
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)    
        self.max_pool = nn.AdaptiveMaxPool2d(1)    
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),  
            nn.ReLU(inplace=True),     
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),    
        )
   
    def forward(self, x):
        x = self.conv(x)
        avg_out = self.fc(self.avg_pool(x))    
        max_out = self.fc(self.max_pool(x))
        return avg_out + max_out
   
 
class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels):    
        super().__init__()
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels * 2 + 2,
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,   
            padding=0,  
            bias=False,
        ) 
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)     
        self.channel_attn = ChannelAttention(out_channels * 2)
        self.conv_layer2 = nn.Conv2d(
            in_channels=out_channels * 2,
            out_channels=out_channels * 2,
            kernel_size=1,    
            stride=1,
            padding=0,
            bias=False,
        )
   
    def forward(self, x):  
        batch = x.shape[0]     
        fft_dim = (-2, -1)
    
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm="ortho")
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)   
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1) + ffted.size()[3:])  
    
        height, width = ffted.shape[-2:]   
        coords_vert = torch.linspace(0, 1, height, device=x.device, dtype=x.dtype) 
        coords_hor = torch.linspace(0, 1, width, device=x.device, dtype=x.dtype)
        coords_vert = coords_vert[None, None, :, None].expand(batch, 1, height, width)
        coords_hor = coords_hor[None, None, None, :].expand(batch, 1, height, width)
        ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)
  
        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))
        ffted = ffted * self.channel_attn(ffted)
        ffted = self.conv_layer2(ffted)   
        ffted = ffted.view((batch, -1, 2) + ffted.size()[2:])     
        ffted = ffted.permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        return torch.fft.irfftn(ffted, s=x.shape[-2:], dim=fft_dim, norm="ortho")
     
   
class SpectralTransform(nn.Module):    
    def __init__(self, in_channels, out_channels):
        super().__init__() 
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.fu = FourierUnit(out_channels, out_channels) 
        self.conv2 = nn.Conv2d(
            in_channels + out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,     
        )
  
    def forward(self, x):   
        x1 = self.conv1(x)
        x2 = self.fu(x1)
        return self.conv2(torch.cat([x, x2], dim=1)) 
   

class WindowStd(nn.Module):    
    def __init__(self, kernel_size=3, channels=None, eps=1e-5):    
        super().__init__()
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:   
            if len(kernel_size) != 2:    
                raise ValueError("kernel_size must be an int or a tuple of length 2.")     
            self.kernel_size = kernel_size

        self.channels = channels 
        self.eps = eps
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        self.register_buffer("mean_kernel", torch.empty(0), persistent=False)  

        if self.channels is not None:    
            self._init_weight()    
    
    def _init_weight(self, device=None, dtype=None):  
        kernel_h, kernel_w = self.kernel_size
        kernel_area = kernel_h * kernel_w
        kernel = torch.ones(1, 1, kernel_h, kernel_w, device=device, dtype=dtype) / kernel_area   
        self.mean_kernel = kernel.repeat(self.channels, 1, 1, 1)

    def forward(self, x):    
        if x.ndim != 4:
            raise ValueError(f"Expected a 4D tensor (B, C, H, W), got shape {tuple(x.shape)}.")
 
        channels = x.shape[1]   
        if self.channels is None:
            self.channels = channels
            self._init_weight(device=x.device, dtype=x.dtype)  
        elif self.channels != channels:    
            raise ValueError(
                f"Input channels ({channels}) do not match initialized channels ({self.channels})."
            )  
        elif self.mean_kernel.device != x.device or self.mean_kernel.dtype != x.dtype:    
            self._init_weight(device=x.device, dtype=x.dtype)    

        x_padded = F.pad(
            x,     
            pad=(self.padding[1], self.padding[1], self.padding[0], self.padding[0]),
            mode="reflect",
        )     
        mean = F.conv2d(x_padded, self.mean_kernel, stride=1, padding=0, groups=channels)

        x_squared = x * x
        x_squared_padded = F.pad(
            x_squared,  
            pad=(self.padding[1], self.padding[1], self.padding[0], self.padding[0]),
            mode="reflect",  
        )
        mean_squared = F.conv2d(     
            x_squared_padded,
            self.mean_kernel, 
            stride=1,
            padding=0,   
            groups=channels,
        )
    
        return torch.sqrt(torch.clamp(mean_squared - mean * mean, min=self.eps))     

    
class FFC(nn.Module):    
    def __init__(self, in_channels, out_channels):   
        super().__init__()
        _require_even_channels(in_channels, "in_channels")
        _require_even_channels(out_channels, "out_channels")
     
        self.local_in_channels = in_channels // 2 
        self.global_in_channels = in_channels // 2  
        self.local_out_channels = out_channels // 2
        self.global_out_channels = out_channels // 2

        self.convl2l = nn.Conv2d(
            self.local_in_channels,
            self.local_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )  
        self.conv1 = nn.Conv2d(    
            self.local_in_channels,  
            self.local_out_channels,
            kernel_size=1,
            bias=False,  
        )
        self.std = WindowStd(kernel_size=3, channels=self.local_in_channels)
        self.sigmoid = nn.Sigmoid()
        self.convg2g = SpectralTransform(self.global_in_channels, self.global_out_channels)  
   
    def forward(self, x):
        if isinstance(x, tuple):
            x_l, x_g = x
        else:  
            x_l, x_g = torch.split(    
                x, 
                [self.local_in_channels, self.global_in_channels],     
                dim=1,
            )   
     
        feature = self.convl2l(x_l)   
        std = self.std(x_l)     
        weight = self.sigmoid(self.conv1(std))   
        out_xl = feature * weight     
        out_xg = self.convg2g(x_g)   
        return out_xl, out_xg
     
     
class FFC_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()     
        _require_even_channels(out_channels, "out_channels")
        self.ffc = FFC(in_channels, out_channels)
        self.bn_l = nn.BatchNorm2d(out_channels // 2)  
        self.bn_g = nn.BatchNorm2d(out_channels // 2) 
        self.act_l = nn.ReLU(inplace=True)   
        self.act_g = nn.ReLU(inplace=True)    
  
    def forward(self, x):   
        x_l, x_g = self.ffc(x)  
        x_l = self.act_l(self.bn_l(x_l))  
        x_g = self.act_g(self.bn_g(x_g))    
        return x_l, x_g


class SFEB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        _require_even_channels(in_channels, "in_channels")  
        _require_even_channels(out_channels, "out_channels")
    
        self.in_channels = in_channels    
        self.out_channels = out_channels     
        self.conv1 = FFC_BN_ACT(in_channels, out_channels)
        self.conv2 = FFC_BN_ACT(out_channels, out_channels) 
        self.shortcut = (     
            nn.Identity()  
            if in_channels == out_channels  
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )
        self.fuse = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
  
    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected a 4D tensor (B, C, H, W), got shape {tuple(x.shape)}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected input with {self.in_channels} channels, got {x.shape[1]}."  
            )   

        identity = self.shortcut(x)
        x_l, x_g = torch.split(x, [self.in_channels // 2, self.in_channels // 2], dim=1)   
        x_l, x_g = self.conv1((x_l, x_g))  
        x_l, x_g = self.conv2((x_l, x_g))    

        id_l, id_g = torch.split(   
            identity,
            [self.out_channels // 2, self.out_channels // 2],    
            dim=1, 
        )     
        out = torch.cat((id_l + x_l, id_g + x_g), dim=1)
        return self.fuse(out)
  
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32     
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)    
   
    module = SFEB(in_channel, out_channel).to(device) 
  
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
    
    print(ORANGE)  
    flops, macs, _ = calculate_flops(model=module, 
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,  
                                     print_detailed=True)     
    print(RESET)