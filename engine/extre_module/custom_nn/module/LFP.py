'''     
本文件由BiliBili：魔傀面具整理    
engine/extre_module/module_images/CVPR2026-LFP.png    
engine/extre_module/module_images/CVPR2026-LFP.md  
论文链接：https://arxiv.org/pdf/2508.06878  
'''
   
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings  
warnings.filterwarnings('ignore')    
from calflops import calculate_flops

from contextlib import nullcontext
import torch
import torch.nn as nn     
import torch.nn.functional as F 
from pytorch_wavelets import DWTForward, DWTInverse
     
from engine.extre_module.ultralytics_nn.conv import Conv   

def _disabled_autocast(): 
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):     
        return torch.amp.autocast(device_type="cuda", enabled=False)    
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):     
        return torch.cuda.amp.autocast(enabled=False)    
    return nullcontext()
     
     
class ConvDWT(nn.Module):
    """Wavelet downsampling: (B, C, H, W) -> (B, 4C, H/2, W/2)."""    
   
    def __init__(self, wave="haar", mode="zero"):    
        super().__init__()
        self.dwt_forward = DWTForward(J=1, wave=wave, mode=mode)
     
    def forward(self, x):
        with _disabled_autocast():
            if x.dtype != torch.float32: 
                x = x.float()
            low_freqs, high_freqs = self.dwt_forward(x)
    
        _, _, h, w = x.shape 
        high_freqs = high_freqs[0].transpose(1, 2).reshape(     
            high_freqs[0].shape[0], -1, high_freqs[0].shape[3], high_freqs[0].shape[4]
        ) 
        output = torch.cat((low_freqs, high_freqs), dim=1)
        return F.interpolate(output, size=(h // 2, w // 2), mode="bilinear", align_corners=False)    
    
   
class ConvIDWT(nn.Module):
    """Wavelet reconstruction from low- and high-frequency tensors."""

    def __init__(self, wave="haar", mode="zero"):    
        super().__init__()    
        self.dwt_inverse = DWTInverse(wave=wave, mode=mode) 
 
    def forward(self, low_freqs, high_freqs): 
        batch, channels, height, width = low_freqs.shape
        high_freqs = high_freqs.reshape(batch, channels, 3, height, width)
 
        with _disabled_autocast():   
            reconstruction = self.dwt_inverse((low_freqs, [high_freqs.float()]))
   
        return F.interpolate(  
            reconstruction, size=(2 * height, 2 * width), mode="bilinear", align_corners=False
        ) 
 
 
class SpatialAttention(nn.Module): 
    def __init__(self, kernel_size=7, bn_before_sigmoid=False):    
        super().__init__()
        if kernel_size not in (3, 7): 
            raise ValueError("kernel_size must be 3 or 7")  
   
        padding = 3 if kernel_size == 7 else 1 
        self.bn_before_sigmoid = bn_before_sigmoid
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)    
        if bn_before_sigmoid: 
            self.bn = nn.BatchNorm2d(1) 
            self.bn.bias.data.fill_(0)     
            self.bn.bias.requires_grad = False
        self.sigmoid = nn.Sigmoid()    
    
    def forward(self, x): 
        avg_out = torch.mean(x, dim=1, keepdim=True)  
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        if self.bn_before_sigmoid:
            x = self.bn(x) 
        return self.sigmoid(x)


class LearnableGaussianFilterBank(nn.Module):   
    def __init__(self, kernel_size, num_filters, num_channels):
        super().__init__()     
        self.kernel_size = kernel_size
        self.num_filters = num_filters    
        self.num_channels = num_channels
        self.padding = kernel_size // 2
        self.sigmas = nn.ParameterList(
            [nn.Parameter(torch.tensor([1.0])) for _ in range(num_filters)]
        )     
     
    def forward(self, x):
        weights = [
            self._gaussian_kernel(self.kernel_size, sigma).repeat(self.num_channels, 1, 1, 1)  
            for sigma in self.sigmas  
        ]
        filtered_outputs = [
            F.conv2d(   
                F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode="replicate"),
                weight.to(device=x.device, dtype=x.dtype),     
                groups=self.num_channels,
            )
            for weight in weights
        ]
        return torch.cat(filtered_outputs, dim=1)
 
    def _gaussian_kernel(self, kernel_size, sigma):   
        kernel = torch.zeros(1, 1, kernel_size, kernel_size, dtype=sigma.dtype, device=sigma.device)     
        center = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):    
                kernel[:, :, i, j] = torch.exp( 
                    -((i - center) ** 2 + (j - center) ** 2) / (2 * sigma ** 2)
                )  
        return kernel / kernel.sum()  
     
   
class LFP(nn.Module):
    """Low-frequency guided feature purification with DWT/IDWT."""  
    
    def __init__(self, in_channels, out_channels, wave="haar", mode="symmetric", with_gauss=True, gauss_gate=0.5):    
        super().__init__()  
        self.dwt = ConvDWT(wave=wave, mode=mode)
        self.idwt = ConvIDWT(wave=wave, mode=mode)    
        self.with_gauss = with_gauss  
        self.gauss_gate = gauss_gate    
        self.attention = SpatialAttention() 

        if self.with_gauss:
            self.gaussian_filter = LearnableGaussianFilterBank(  
                kernel_size=3, num_filters=1, num_channels=3 * in_channels
            )
        
        self.conv_1x1 = Conv(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
  
    def forward(self, x): 
        _, channels, _, _ = x.shape    
        dwt_out = self.dwt(x)

        low_freqs = dwt_out[:, :channels, :, :]     
        high_freqs = dwt_out[:, channels:, :, :]

        attention = self.attention(low_freqs)
        high_freqs = high_freqs * attention
    
        if self.with_gauss:   
            blurred_high_freqs = self.gaussian_filter(high_freqs)
            mask = (high_freqs.abs() < self.gauss_gate).to(high_freqs.dtype) 
            high_freqs = high_freqs * (1 - mask) + blurred_high_freqs * mask
  
        return self.conv_1x1(self.idwt(low_freqs, high_freqs))  
    

if __name__ == '__main__':   
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"  
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32 
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = LFP(in_channel, out_channel).to(device)

    outputs = module(inputs) 
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
     
    print(ORANGE)   
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,     
                                     output_precision=4,
                                     print_detailed=True)   
    print(RESET)
