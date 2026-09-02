'''
本文件由BiliBili：魔傀面具整理     
engine/extre_module/module_images/自研模块-HPFGA.png 
engine/extre_module/module_images/自研模块-HPFGA.md   
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')    

import warnings     
warnings.filterwarnings('ignore')
from calflops import calculate_flops
  
import torch
import torch.nn as nn     
import torch.nn.functional as F    
import torch.fft as fft
    
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

 
class HPFGA(nn.Module):   
    """Hierarchical Progressive Frequency-Guided Aggregation (HPFGA).   

    Extends PFGA with two innovations:     
    1. Progressive cross-scale injection: small-kernel branches inject local
       detail into larger-kernel branches, building a fine-to-coarse hierarchy.  
    2. Dual-domain frequency descriptor: augments the original spatial-domain
       frequency cues (gradient magnitude, Laplacian, local variance) with
       FFT-based global spectral statistics (low/mid/high band energies),  
       yielding a 6-channel gating signal with richer frequency awareness.   

    Args:
        dim: Number of input/output channels.   
        K_list: Kernel sizes for each branch (ordered small to large).
        use_grn: Whether to apply Global Response Normalization on output.  
        center_suppress: Whether each branch applies learnable center suppression.
    """
    
    class _Branch(nn.Module):    
        """Single peripheral branch with optional center suppression."""

        def __init__(self, dim: int, K: int, center_suppress: bool = True) -> None:    
            super().__init__()
            self.center_suppress = center_suppress
            self.dw_h = nn.Conv2d(dim, dim, kernel_size=(1, K), padding=(0, K // 2),
                                   groups=dim, bias=False)     
            self.dw_v = nn.Conv2d(dim, dim, kernel_size=(K, 1), padding=(K // 2, 0),
                                   groups=dim, bias=False)
            if self.center_suppress:   
                self.dw_c = nn.Conv2d(dim, dim, kernel_size=3, padding=1,    
                                       groups=dim, bias=False)
                self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
            else:
                self.register_parameter("beta", None)
                self.dw_c = None
  
        def forward(self, x: torch.Tensor) -> torch.Tensor: 
            y = self.dw_v(self.dw_h(x)) 
            if self.center_suppress:     
                center = self.dw_c(x)
                y = y - torch.tanh(self.beta) * center     
            return y   
  
    def __init__(
        self, 
        in_dim: int, 
        out_dim:int,
        K_list: tuple = (9, 15, 31),     
        use_grn: bool = False,
        center_suppress: bool = True,
    ) -> None:    
        super().__init__()
        self.dim = in_dim
        self.K_list = K_list    
 
        self.branches = nn.ModuleList(
            [HPFGA._Branch(in_dim, K, center_suppress) for K in K_list]   
        )
    
        # Progressive injection projections: branch_i output -> inject into branch_{i+1} input
        # dim -> dim, one projection per adjacent pair (len(K_list) - 1 total)    
        self.inject_projs = nn.ModuleList([ 
            nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
            for _ in range(len(K_list) - 1)  
        ]) 
 
        # Fixed spatial-domain filter kernels 
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],  
                                dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],     
                                dtype=torch.float32).view(1, 1, 3, 3)     
        laplace = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],     
                                dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)     
        self.register_buffer("laplace", laplace, persistent=False)
     
        # Gate head now accepts 6-channel dual-domain descriptor    
        self.gate_head = nn.Conv2d(6, len(K_list), kernel_size=1, bias=True)

        self.use_grn = use_grn
        if use_grn:
            self.grn = GRN(in_dim)    
    
        self.conv_1x1 = Conv(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
    
    def _depthwise_filter(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        C = x.shape[1]
        weight = kernel.repeat(C, 1, 1, 1) 
        return F.conv2d(x, weight, padding=1, groups=C)
    
    @staticmethod
    def _radial_masks(H: int, W: int, device: torch.device):
        """Radial low/mid/high frequency boolean masks (centered)."""  
        cy, cx = H // 2, W // 2    
        ys = torch.arange(H, device=device).float() - cy
        xs = torch.arange(W, device=device).float() - cx
        r = torch.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2)
        r_max = min(H, W) / 2.0
        low  = r < r_max * 0.25
        high = r > r_max * 0.65
        mid  = ~low & ~high
        return low, mid, high   
   
    def _dual_freq_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Compute 6-channel dual-domain frequency descriptor.    

        Channels 1-3: spatial-domain cues (gradient magnitude, Laplacian, local variance).
        Channels 4-6: FFT-based global spectral statistics broadcast to spatial dimensions.   
        """    
        B, C, H, W = x.shape 
 
        # --- Spatial-domain cues (channels 1-3) ---
        gx = self._depthwise_filter(x, self.sobel_x)    
        gy = self._depthwise_filter(x, self.sobel_y)
        lap = self._depthwise_filter(x, self.laplace)  

        grad_mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)
        mean  = F.avg_pool2d(x, 3, 1, 1)   
        mean2 = F.avg_pool2d(x * x, 3, 1, 1)     
        var   = torch.clamp(mean2 - mean * mean, min=0.0)

        f1 = grad_mag.mean(dim=1, keepdim=True)           # (B,1,H,W) edge strength
        f2 = lap.abs().mean(dim=1, keepdim=True)           # (B,1,H,W) curvature 
        f3 = var.mean(dim=1, keepdim=True)                 # (B,1,H,W) local variance     
  
        # --- Frequency-domain cues (channels 4-6): global stats broadcast spatially ---     
        amp = torch.abs(fft.fftshift(fft.fft2(x.float())))   # (B, C, H, W)
        low_m, mid_m, high_m = self._radial_masks(H, W, x.device)
   
        # Per-band mean amplitude across channels → (B, 1) scalars
        f4_val = amp[:, :, low_m ].mean(dim=-1).mean(dim=1, keepdim=True)   # (B,1)
        f5_val = amp[:, :, mid_m ].mean(dim=-1).mean(dim=1, keepdim=True)   # (B,1)
        f6_val = amp[:, :, high_m].mean(dim=-1).mean(dim=1, keepdim=True)   # (B,1)     
  
        # Broadcast to spatial maps: same scalar at every (h, w) position 
        f4 = f4_val.view(B, 1, 1, 1).expand(B, 1, H, W)   # (B,1,H,W)
        f5 = f5_val.view(B, 1, 1, 1).expand(B, 1, H, W)   
        f6 = f6_val.view(B, 1, 1, 1).expand(B, 1, H, W)  
 
        return torch.cat([f1, f2, f3, f4, f5, f6], dim=1)   # (B, 6, H, W)  
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:     
        # --- Progressive cross-scale branch execution ---  
        peris = []     
        acc = None   # accumulated cross-scale injection in input (dim-channel) space  
        for i, branch in enumerate(self.branches):  
            inp = x if acc is None else x + acc
            out = branch(inp)     
            peris.append(out)   
            if i < len(self.inject_projs):   
                new_inject = self.inject_projs[i](out)
                acc = new_inject if acc is None else acc + new_inject   
    
        # --- Dual-domain frequency gating --- 
        freq   = self._dual_freq_maps(x)
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
    
    module = HPFGA(in_channel, out_channel).to(device)

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
   
    print(ORANGE)  
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),    
                                     output_as_string=True,
                                     output_precision=4,   
                                     print_detailed=True)
    print(RESET)   
