'''
本文件由BiliBili：魔傀面具整理   
engine/extre_module/module_images/CVPR2026-FAAFusion.png   
engine/extre_module/module_images/CVPR2026-FAAFusion.md    
论文链接：https://arxiv.org/pdf/2602.23790  
'''     

import os, sys 
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..') 

import warnings  
warnings.filterwarnings('ignore')
from calflops import calculate_flops

import math    
import torch
import torch.nn as nn   
import torch.nn.functional as F

from engine.extre_module.ultralytics_nn.conv import Conv
     
class FAAFusion(nn.Module):
    """Fourier Angle Alignment fusion block.     

    The module aligns the dominant orientation of a high-level feature map to a    
    lower-level feature map in local windows, then fuses the aligned result back     
    into the low-level feature.
   
    Args:   
        in_channels: Number of channels in ``x_high`` and ``x_low``.
        m: Local window size used for orientation estimation. Must be odd.   
        c_mid: Intermediate projection channel count. 
        eps: Small value for numerical stability. 
        layer_scale_init_value: Initial value for the learnable fusion scale. 
 
    Inputs:    
        x_high: Tensor of shape ``[B, C, H_h, W_h]``. 
        x_low: Tensor of shape ``[B, C, H_l, W_l]``. 

    Returns:    
        Tensor of shape ``[B, C, H_l, W_l]``. 
    """

    def __init__(
        self,
        in_channels,
        out_channels,   
        m: int = 7,  
        eps: float = 1e-8,     
        layer_scale_init_value: float = 1e-5,
    ) -> None:
        super().__init__()    
        if m % 2 == 0:
            raise ValueError(f"m must be odd, got {m}")
   
        self.conv1x1_1 = Conv(in_channels[0], out_channels, 1) if in_channels[0] != out_channels else nn.Identity()     
        self.conv1x1_2 = Conv(in_channels[1], out_channels, 1) if in_channels[1] != out_channels else nn.Identity()
 
        c_mid = out_channels // 4    

        self.in_channels = out_channels
        self.m = m 
        self.c_mid = c_mid
        self.eps = eps    
    
        self.layer_scale = nn.Parameter(    
            torch.full((1, 1, 1, 1), layer_scale_init_value),
            requires_grad=True,
        ) 
 
        self.proj_low = nn.Conv2d(     
            in_channels=out_channels, out_channels=c_mid, kernel_size=1, bias=False
        )
        self.proj_high = nn.Conv2d( 
            in_channels=out_channels, out_channels=c_mid, kernel_size=1, bias=False
        )     
        self.recon = nn.Conv2d( 
            in_channels=c_mid, out_channels=out_channels, kernel_size=1, bias=False
        )

        self._init_freq_grids(m)    
   
    def _init_freq_grids(self, m: int) -> None:
        h_freq = torch.fft.fftfreq(m, d=1.0) * m     
        w_freq = torch.fft.fftfreq(m, d=1.0) * m
        try:  
            h_grid, w_grid = torch.meshgrid(h_freq, w_freq, indexing="ij")
        except TypeError:   
            h_grid, w_grid = torch.meshgrid(h_freq, w_freq)   

        rho = torch.sqrt(h_grid**2 + w_grid**2)    
        theta = torch.atan2(h_grid, w_grid)     
        theta = (theta + 2 * math.pi) % (2 * math.pi) 
 
        mask = rho > self.eps
        self.register_buffer("valid_thetas", theta[mask]) 
        self.register_buffer("valid_rhos", rho[mask])
        self.register_buffer("mask_flat", mask.reshape(-1))  
  
    def _estimate_main_direction(self, x_local: torch.Tensor) -> torch.Tensor:    
        """Estimate the dominant orientation from the local magnitude spectrum."""
        batch_windows = x_local.shape[0]
        device = x_local.device   

        x_fft = torch.fft.fft2(x_local.squeeze(1), norm="ortho")   
        x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-2, -1))    
        magnitude = x_fft_shifted.abs() + self.eps
   
        magnitude_flat = magnitude.reshape(batch_windows, -1)     
        magnitude_valid = magnitude_flat[:, self.mask_flat]
        rho_valid = self.valid_rhos.to(device)     
 
        weighted_energy = magnitude_valid * rho_valid.unsqueeze(0)
        max_idx = torch.argmax(weighted_energy, dim=1) 
        theta_est = self.valid_thetas.to(device)[max_idx]
        return theta_est 
   
    def _rotate_spatial_patch(     
        self, patch: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:   
        batch_windows, _, patch_h, _ = patch.shape   
        device = patch.device     

        cos_t = torch.cos(theta).reshape(batch_windows)  
        sin_t = torch.sin(theta).reshape(batch_windows)

        center = (patch_h - 1) / 2.0
        rot_mat = torch.zeros(batch_windows, 2, 3, device=device, dtype=patch.dtype)
        rot_mat[:, 0, 0] = cos_t
        rot_mat[:, 0, 1] = -sin_t
        rot_mat[:, 1, 0] = sin_t    
        rot_mat[:, 1, 1] = cos_t
        rot_mat[:, 0, 2] = (     
            center  
            - cos_t * center
            + sin_t * center  
        )
        rot_mat[:, 1, 2] = (
            center
            - sin_t * center
            - cos_t * center
        )
     
        grid = F.affine_grid(rot_mat, patch.size(), align_corners=False)  
        return F.grid_sample(    
            patch,
            grid,
            mode="bilinear",  
            padding_mode="zeros",
            align_corners=False,    
        )     

    def _build_overlap_norm(   
        self, height: int, width: int, device: torch.device, dtype: torch.dtype 
    ) -> torch.Tensor:
        ones = torch.ones(1, 1, height, width, device=device, dtype=dtype)
        ones_unfold = F.unfold(ones, kernel_size=self.m, stride=1, padding=0)
        return F.fold(
            ones_unfold,     
            output_size=(height, width),
            kernel_size=self.m,
            stride=1,     
            padding=0,
        )     

    def forward(self, inputs) -> torch.Tensor:     
        x_high, x_low = inputs    
    
        x_high = self.conv1x1_1(x_high)
        x_low = self.conv1x1_2(x_low)     
    
        if x_high.ndim != 4 or x_low.ndim != 4:  
            raise ValueError(
                "x_high and x_low must both be 4D tensors shaped [B, C, H, W]"
            )

        batch_size, channels, height_low, width_low = x_low.shape
        _, high_channels, height_high, width_high = x_high.shape    

        if height_low < self.m or width_low < self.m:
            raise ValueError(  
                "x_low spatial size must be at least m x m. "
                f"Got {(height_low, width_low)} with m={self.m}."   
            )
    
        if (height_high, width_high) != (height_low, width_low): 
            x_high_up = F.interpolate(
                x_high,    
                size=(height_low, width_low),    
                mode="bilinear",
                align_corners=False,
            )    
        else:
            x_high_up = x_high
 
        xl_proj = self.proj_low(x_low)   
        xh_proj = self.proj_high(x_high_up) 
   
        num_windows = (height_low - self.m + 1) * (width_low - self.m + 1) 
        xh_aligned_cmid = torch.zeros_like(xh_proj)    
        overlap_norm = self._build_overlap_norm(    
            height_low, width_low, x_low.device, x_low.dtype     
        )
   
        for channel_idx in range(self.c_mid):  
            xl_channel = xl_proj[:, channel_idx : channel_idx + 1]   
            xh_channel = xh_proj[:, channel_idx : channel_idx + 1]    
  
            xl_unfold = F.unfold(xl_channel, kernel_size=self.m, stride=1, padding=0)   
            xh_unfold = F.unfold(xh_channel, kernel_size=self.m, stride=1, padding=0)

            xl_patches = xl_unfold.transpose(1, 2).reshape(
                batch_size * num_windows, 1, self.m, self.m
            )    
            xh_patches = xh_unfold.transpose(1, 2).reshape(  
                batch_size * num_windows, 1, self.m, self.m  
            )
    
            theta_low = self._estimate_main_direction(xl_patches)
            theta_high = self._estimate_main_direction(xh_patches)
     
            theta_delta = torch.remainder(theta_low, math.pi) - torch.remainder(   
                theta_high, math.pi  
            )

            xh_rotated = self._rotate_spatial_patch(xh_patches, theta_delta)   
            xh_rotated_flat = xh_rotated.reshape(batch_size, num_windows, -1).transpose(   
                1, 2  
            )
            xh_aligned_map = F.fold(     
                xh_rotated_flat,     
                output_size=(height_low, width_low),
                kernel_size=self.m,
                stride=1,
                padding=0,   
            )    
            xh_aligned_cmid[:, channel_idx : channel_idx + 1] = ( 
                xh_aligned_map / (overlap_norm + self.eps)    
            )     

        xh_recon = self.recon(xh_aligned_cmid)
        x_high_modulated = self.layer_scale * xh_recon + x_high_up   
        return x_low + x_high_modulated  
    

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"     
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')     
    batch_size, channel_1, channel_2, height, width = 1, 32, 16, 32, 32     
    ouc_channel = 32
    inputs_1 = torch.randn((batch_size, channel_1, height, width)).to(device)   
    inputs_2 = torch.randn((batch_size, channel_2, height * 2, width * 2)).to(device)
  
    module = FAAFusion([channel_1, channel_2], ouc_channel).to(device)

    # inputs_1 为深层特征图, inputs_2 为浅层特征图, 举个例子：inputs_1为P4,inputs_2为P3  
    # 可以参考engine/extre_module/custom_nn/featurefusion/mpca.py的教程  
    outputs = module([inputs_1, inputs_2]) 
    print(GREEN + f'inputs1.size:{inputs_1.size()} inputs2.size:{inputs_2.size()} outputs.size:{outputs.size()}' + RESET)     
 
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     args=[[inputs_1, inputs_2]], 
                                     output_as_string=True,   
                                     output_precision=4,     
                                     print_detailed=True)   
    print(RESET)
