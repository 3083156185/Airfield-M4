'''   
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/自研模块-DPFGA.png
engine/extre_module/module_images/自研模块-DPFGA.md
'''    
    
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')   
  
import warnings
warnings.filterwarnings('ignore')   
from calflops import calculate_flops     
 
import torch
import torch.nn as nn  
import torch.nn.functional as F 
    
from engine.extre_module.ultralytics_nn.conv import Conv     

class GRN(nn.Module):  
    """Global Response Normalization."""
  
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1)) 
        self.eps = eps 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)  
        nx = x / (gx + self.eps)
        return x + self.gamma * nx + self.beta   


class DirectionalBranch(nn.Module):
    """Direction-adaptive peripheral branch.    

    Maintains two decomposition paths:
    - path_hv: horizontal (1,K) then vertical (K,1) — favored for horizontal textures     
    - path_vh: vertical (K,1) then horizontal (1,K) — favored for vertical textures 
 
    A lightweight direction router computes horizontal vs. vertical energy from
    the input to generate a soft mixing coefficient alpha ∈ (0,1), producing:
        output = alpha * path_hv(x) + (1 - alpha) * path_vh(x) 
     
    This enables the branch to adaptively weight its decomposition order based     
    on the dominant texture orientation in the current feature map.
    """

    def __init__(self, dim: int, K: int, center_suppress: bool = True) -> None:
        super().__init__()
        self.dim = dim
  
        # path_hv: horizontal → vertical
        self.dw_h  = nn.Conv2d(dim, dim, (1, K), padding=(0, K // 2), groups=dim, bias=False) 
        self.dw_v  = nn.Conv2d(dim, dim, (K, 1), padding=(K // 2, 0), groups=dim, bias=False)

        # path_vh: vertical → horizontal (independent weights)
        self.dw_v2 = nn.Conv2d(dim, dim, (K, 1), padding=(K // 2, 0), groups=dim, bias=False)
        self.dw_h2 = nn.Conv2d(dim, dim, (1, K), padding=(0, K // 2), groups=dim, bias=False)
 
        # Directional router: maps (B, 2*dim) energy vector to scalar alpha ∈ (0,1) 
        self.dir_router = nn.Sequential(  
            nn.Linear(dim * 2, max(dim // 4, 4), bias=False),     
            nn.ReLU(inplace=True),  
            nn.Linear(max(dim // 4, 4), 1, bias=False),
            nn.Sigmoid(), 
        )    

        self.center_suppress = center_suppress  
        if center_suppress:   
            self.dw_c  = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
            self.beta  = nn.Parameter(torch.zeros(1, dim, 1, 1))    
        else:
            self.register_parameter("beta", None)  
            self.dw_c = None
   
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute horizontal and vertical intermediate responses   
        h_resp = self.dw_h(x)   # (B, dim, H, W)  horizontal response
        v_resp = self.dw_v2(x)  # (B, dim, H, W)  vertical response 
     
        # Direction energy: global average of absolute values → (B, dim) each
        h_energy = F.adaptive_avg_pool2d(h_resp.abs(), 1).view(x.size(0), self.dim)
        v_energy = F.adaptive_avg_pool2d(v_resp.abs(), 1).view(x.size(0), self.dim) 
        dir_feat = torch.cat([h_energy, v_energy], dim=1)   # (B, 2*dim)
    
        alpha = self.dir_router(dir_feat).view(-1, 1, 1, 1)  # (B,1,1,1) ∈ (0,1)
   
        # path_hv: use pre-computed h_resp, then apply vertical conv    
        out_hv = self.dw_v(h_resp)

        # path_vh: use pre-computed v_resp, then apply horizontal conv 
        out_vh = self.dw_h2(v_resp)
    
        # Soft direction-adaptive mixing
        y = alpha * out_hv + (1.0 - alpha) * out_vh    
   
        # Learnable center suppression     
        if self.center_suppress:  
            center = self.dw_c(x)     
            y = y - torch.tanh(self.beta) * center 

        return y

     
class DPFGA(nn.Module):
    """Directional Peripheral-Frequency Guided Aggregation (DPFGA). 

    Extends PFGA with two innovations:
    1. Directional frequency descriptor: replaces the isotropic gradient 
       magnitude with four directional gradient responses (horizontal, vertical, 
       diagonal, anti-diagonal) plus Laplacian and local variance, yielding a    
       6-channel descriptor that explicitly encodes texture orientation.     
    2. Direction-adaptive branch decomposition: each branch dynamically weights
       two separable 1-D decomposition orders (HV vs VH) based on the  
       dominant texture direction inferred from the input feature map.

    Args: 
        dim: Number of input/output channels. 
        K_list: Kernel sizes for each branch.  
        use_grn: Whether to apply Global Response Normalization on output.
        center_suppress: Whether each branch applies learnable center suppression.
    """
  
    def __init__(     
        self,     
        in_dim: int,     
        out_dim: int,   
        K_list: tuple = (9, 15, 31),
        use_grn: bool = False,
        center_suppress: bool = True,
    ) -> None:
        super().__init__()
        self.dim = in_dim
        self.K_list = K_list 
  
        self.branches = nn.ModuleList(    
            [DirectionalBranch(in_dim, K, center_suppress) for K in K_list]
        )   

        # Fixed directional filter kernels (1,1,3,3)  
        sobel_x   = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],    
                                   dtype=torch.float32).view(1, 1, 3, 3)     
        sobel_y   = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                   dtype=torch.float32).view(1, 1, 3, 3)
        diag      = torch.tensor([[-1, 0, 0], [0, 0, 0], [0, 0, 1]],  
                                   dtype=torch.float32).view(1, 1, 3, 3)
        anti_diag = torch.tensor([[0, 0, -1], [0, 0, 0], [1, 0, 0]],
                                   dtype=torch.float32).view(1, 1, 3, 3)
        laplace   = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                   dtype=torch.float32).view(1, 1, 3, 3) 

        self.register_buffer("sobel_x",   sobel_x,   persistent=False)
        self.register_buffer("sobel_y",   sobel_y,   persistent=False)     
        self.register_buffer("diag",      diag,      persistent=False)
        self.register_buffer("anti_diag", anti_diag, persistent=False)     
        self.register_buffer("laplace",   laplace,   persistent=False)     
     
        # Gate head accepts 6-channel directional frequency descriptor
        self.gate_head = nn.Conv2d(6, len(K_list), kernel_size=1, bias=True)

        self.use_grn = use_grn
        if use_grn:    
            self.grn = GRN(in_dim)  
     
        self.conv_1x1 = Conv(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
     
    def _depthwise_filter(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor: 
        C = x.shape[1]     
        weight = kernel.repeat(C, 1, 1, 1)  
        return F.conv2d(x, weight, padding=1, groups=C)  
  
    def _directional_freq_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Compute 6-channel directional frequency descriptor.   

        Returns:
            (B, 6, H, W) tensor with channels:
            [f_h, f_v, f_d1, f_d2, f_lap, f_var] 
        """ 
        gx   = self._depthwise_filter(x, self.sobel_x)   
        gy   = self._depthwise_filter(x, self.sobel_y)   
        gd1  = self._depthwise_filter(x, self.diag)  
        gd2  = self._depthwise_filter(x, self.anti_diag)
        lap  = self._depthwise_filter(x, self.laplace)

        mean  = F.avg_pool2d(x, 3, 1, 1)    
        mean2 = F.avg_pool2d(x * x, 3, 1, 1)
        var   = torch.clamp(mean2 - mean * mean, min=0.0)  
    
        f_h   = gx.abs().mean(dim=1, keepdim=True)     # horizontal gradient energy
        f_v   = gy.abs().mean(dim=1, keepdim=True)     # vertical gradient energy   
        f_d1  = gd1.abs().mean(dim=1, keepdim=True)    # diagonal gradient energy
        f_d2  = gd2.abs().mean(dim=1, keepdim=True)    # anti-diagonal gradient energy     
        f_lap = lap.abs().mean(dim=1, keepdim=True)    # Laplacian (curvature) 
        f_var = var.mean(dim=1, keepdim=True)          # local variance (texture energy)
  
        return torch.cat([f_h, f_v, f_d1, f_d2, f_lap, f_var], dim=1)   # (B, 6, H, W)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        peris = [branch(x) for branch in self.branches]     
    
        freq   = self._directional_freq_maps(x)
        logits = self.gate_head(freq)
        alpha  = torch.softmax(logits, dim=1)   # (B, K, H, W)
   
        y = sum(peris[i] * alpha[:, i:i+1, :, :] for i in range(len(peris)))    

        if self.use_grn:   
            y = self.grn(y)    
 
        y = self.conv_1x1(y)
        return y

 
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m" 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32    
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = DPFGA(in_channel, out_channel).to(device)   

    outputs = module(inputs)   
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
  
    print(ORANGE)     
    flops, macs, _ = calculate_flops(model=module,  
                                     input_shape=(batch_size, in_channel, height, width),    
                                     output_as_string=True,  
                                     output_precision=4,     
                                     print_detailed=True)     
    print(RESET)   
