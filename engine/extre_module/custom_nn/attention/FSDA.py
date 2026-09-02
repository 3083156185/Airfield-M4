'''
本文件由BiliBili：魔傀面具整理  
engine/extre_module/module_images/FSDA.png  
engine/extre_module/module_images/FSDA.md
'''
     
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')   

import warnings
warnings.filterwarnings('ignore') 
from calflops import calculate_flops

import torch
import torch.nn as nn
 
from engine.extre_module.ultralytics_nn.conv import DSConv  


class FSDA(nn.Module):
    """  
    Frequency-Segregated Dual Attention (FSDA)   

    Decomposes input features into low-frequency (global structure) and   
    high-frequency (edge/detail) components via FFT with a learnable Gaussian 
    soft mask, then applies Channel Attention to low-frequency and Spatial
    Attention to high-frequency, fusing results through an adaptive gate.
    """     

    def __init__(self, in_channels, reduction_ratio=16): 
        """     
        Args:     
            in_channels (int): Number of input channels
            reduction_ratio (int): Reduction ratio for CAM FC layers
        """ 
        super(FSDA, self).__init__() 
     
        # Feature Refinement    
        self.feature_refine = DSConv(in_channels, in_channels, act=nn.LeakyReLU)    
     
        # Learnable soft frequency mask parameter  
        # sigma controls the cutoff frequency of the Gaussian mask   
        self.sigma = nn.Parameter(torch.tensor(4.0))    

        # Channel Attention Module (CAM) - for low-frequency branch
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)    
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.cam_fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.cam_fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
     
        # Spatial Attention Module (SAM) - for high-frequency branch
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

        # Adaptive Fusion Gate  
        self.gate_fc_cam = nn.Linear(in_channels, in_channels)
        self.gate_fc_sam = nn.Linear(in_channels, in_channels)
    
        # Activations
        self.silu = nn.SiLU()    
        self.sigmoid = nn.Sigmoid()   

    def _build_gaussian_mask(self, H, W, device):
        """
        Build low-pass and high-pass Gaussian soft masks.   
  
        Args:
            H (int): Height of feature map
            W (int): Width of feature map
            device: torch device 
 
        Returns:   
            mask_low [H, W]: Gaussian low-pass mask
            mask_high [H, W]: Complementary high-pass mask 
        """
        # Create distance matrix from center 
        cy, cx = H / 2.0, W / 2.0
        y = torch.arange(H, dtype=torch.float32, device=device) - cy
        x = torch.arange(W, dtype=torch.float32, device=device) - cx   
        yy, xx = torch.meshgrid(y, x, indexing='ij')   
        dist_sq = yy ** 2 + xx ** 2  # [H, W]   
  
        # Gaussian low-pass mask with learnable sigma
        # Clamp sigma to avoid division by zero or negative values     
        sigma_clamped = torch.clamp(self.sigma, min=1.0)  
        mask_low = torch.exp(-dist_sq / (2.0 * sigma_clamped ** 2))  # [H, W]
        mask_high = 1.0 - mask_low  # [H, W]    
     
        return mask_low, mask_high 

    def frequency_decompose(self, x): 
        """ 
        Decompose input into low-frequency and high-frequency components via FFT. 

        Args:
            x (torch.Tensor): Input feature map [B, C, H, W] 

        Returns:
            x_low (torch.Tensor): Low-frequency component [B, C, H, W]     
            x_high (torch.Tensor): High-frequency component [B, C, H, W]  
        """
        _, _, H, W = x.shape

        # FFT and shift to center  
        f = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')  
        f_shifted = torch.fft.fftshift(f, dim=(-2, -1))   
 
        # Build learnable Gaussian masks  
        mask_low, mask_high = self._build_gaussian_mask(H, W, x.device)
     
        # Apply masks in frequency domain     
        f_low = f_shifted * mask_low.unsqueeze(0).unsqueeze(0)   # [B, C, H, W]   
        f_high = f_shifted * mask_high.unsqueeze(0).unsqueeze(0)  # [B, C, H, W]
     
        # Inverse FFT back to spatial domain
        x_low = torch.fft.ifft2(
            torch.fft.ifftshift(f_low, dim=(-2, -1)), 
            dim=(-2, -1), norm='ortho'  
        ).real  
        x_high = torch.fft.ifft2(
            torch.fft.ifftshift(f_high, dim=(-2, -1)),  
            dim=(-2, -1), norm='ortho'    
        ).real  

        return x_low, x_high    
 
    def channel_attention(self, x):  
        """
        Channel Attention Module (CAM) for low-frequency branch.

        Args:
            x (torch.Tensor): Low-frequency feature map [B, C, H, W]
    
        Returns:
            torch.Tensor: Channel attention weighted feature map [B, C, H, W]
        """
        b, c, _, _ = x.size()

        # Global Average Pooling + Global Max Pooling
        gap = self.global_avg_pool(x).view(b, c)  # [B, C]
        gmp = self.global_max_pool(x).view(b, c)  # [B, C] 

        # Combine and pass through FC layers    
        combined = gap + gmp  # [B, C]
        channel_att = self.cam_fc1(combined)       # [B, C//r]     
        channel_att = self.silu(channel_att)  
        channel_att = self.cam_fc2(channel_att)    # [B, C]
        channel_att = self.sigmoid(channel_att)     
    
        # Reshape and apply
        channel_att = channel_att.view(b, c, 1, 1)  # [B, C, 1, 1]  
        return x * channel_att
   
    def spatial_attention(self, x):   
        """
        Spatial Attention Module (SAM) for high-frequency branch.   
     
        Args:     
            x (torch.Tensor): High-frequency feature map [B, C, H, W]

        Returns:
            torch.Tensor: Spatial attention weighted feature map [B, C, H, W]
        """
        # Channel-wise pooling
        mean_pool = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]

        # Concatenate and apply 7x7 conv
        pooled = torch.cat([mean_pool, max_pool], dim=1)  # [B, 2, H, W]     
        spatial_att = self.spatial_conv(pooled)             # [B, 1, H, W]    
        spatial_att = self.silu(spatial_att) 
        spatial_att = self.sigmoid(spatial_att)

        return x * spatial_att     

    def adaptive_fusion(self, x_cam, x_sam):     
        """     
        Adaptive Fusion Gate: channel-wise learned gating between CAM and SAM outputs.  
    
        Args:
            x_cam (torch.Tensor): CAM output [B, C, H, W]     
            x_sam (torch.Tensor): SAM output [B, C, H, W]    

        Returns:
            torch.Tensor: Fused feature map [B, C, H, W]   
        """
        b, c, _, _ = x_cam.size()     
    
        # Global context from both branches
        gap_cam = self.global_avg_pool(x_cam).view(b, c)  # [B, C]    
        gap_sam = self.global_avg_pool(x_sam).view(b, c)   # [B, C]     

        # Compute fusion gate   
        gate = self.sigmoid(
            self.gate_fc_cam(gap_cam) + self.gate_fc_sam(gap_sam)
        )  # [B, C]  
        gate = gate.view(b, c, 1, 1)  # [B, C, 1, 1]
   
        # Weighted fusion
        fused = gate * x_cam + (1.0 - gate) * x_sam  # [B, C, H, W]     
        return fused  

    def forward(self, x):   
        """
        Forward pass of FSDA module.

        Args:   
            x (torch.Tensor): Input feature map [B, C, H, W]   

        Returns:    
            torch.Tensor: Output feature map [B, C, H, W]
        """    
        # Feature refinement
        x_refined = self.feature_refine(x)     
   
        # Frequency-domain decomposition  
        x_low, x_high = self.frequency_decompose(x_refined)
    
        # Branch-specific attention   
        x_cam = self.channel_attention(x_low)    # CAM on low-frequency  
        x_sam = self.spatial_attention(x_high)   # SAM on high-frequency  
  
        # Adaptive fusion + residual connection
        output = self.adaptive_fusion(x_cam, x_sam) + x     
  
        return output     


if __name__ == '__main__':    
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel, height, width = 1, 16, 32, 32
    inputs = torch.randn((batch_size, channel, height, width)).to(device)

    module = FSDA(channel).to(device)

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)   

    print(ORANGE)     
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width),     
                                     output_as_string=True,     
                                     output_precision=4,     
                                     print_detailed=True)
    print(RESET)
