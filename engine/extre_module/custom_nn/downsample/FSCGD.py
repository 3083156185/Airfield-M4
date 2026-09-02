'''   
本文件由BiliBili：魔傀面具整理     
engine/extre_module/module_images/自研模块-FSCGD.png
engine/extre_module/module_images/自研模块-FSCGD.md
'''

import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops

import torch   
import torch.nn as nn    
import torch.nn.functional as F 
import torch.fft as fft     

    
class _FreqRouter(nn.Module):
    """Frequency-spectral router for multi-dilation context paths.    
   
    Computes a 2D FFT amplitude spectrum on the downsampled feature map,
    splits it into low / mid / high frequency bands via radial masks,
    extracts per-band global statistics (channel-wise mean amplitude), 
    and maps them to dilation-path softmax weights.     
    """ 

    def __init__(self, in_ch: int, n_paths: int, hidden: int = 16) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3 * in_ch, hidden, bias=False),
            nn.ReLU(inplace=True),  
            nn.Linear(hidden, n_paths, bias=False),
        )
        self.n_paths = n_paths    

    @staticmethod     
    def _radial_masks(H: int, W: int, device: torch.device): 
        """Return low / mid / high boolean masks based on radial frequency distance."""
        cy, cx = H // 2, W // 2
        ys = torch.arange(H, device=device).float() - cy
        xs = torch.arange(W, device=device).float() - cx    
        r = torch.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2)
        r_max = min(H, W) / 2.0
        low = r < r_max * 0.25     
        high = r > r_max * 0.65
        mid = ~low & ~high  
        return low, mid, high
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape    
        # 2D FFT amplitude spectrum, shifted so DC is at center
        amp = torch.abs(fft.fftshift(fft.fft2(x.float())))  # (B, C, H, W)     
        low_m, mid_m, high_m = self._radial_masks(H, W, x.device)
        # Per-band mean amplitude across spatial dims -> (B, C)
        low_stat = amp[:, :, low_m].mean(dim=-1)
        mid_stat = amp[:, :, mid_m].mean(dim=-1)  
        high_stat = amp[:, :, high_m].mean(dim=-1) 
        stat = torch.cat([low_stat, mid_stat, high_stat], dim=1)  # (B, 3C)     
        weights = torch.softmax(self.fc(stat), dim=-1)  # (B, n_paths) 
        return weights


class _HFCompensator(nn.Module):
    """High-Frequency Compensation Bypass.
 
    Extracts high-frequency residual from the *pre-downsample* input via
    subtracting the low-pass component (AvgPool + Upsample), encodes it     
    through a lightweight depthwise pipeline, and downsamples to match 
    the target resolution.  A learnable scalar alpha controls the    
    compensation strength. 
    """    
 
    def __init__(self, in_ch: int, out_ch: int) -> None: 
        super().__init__()    
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.dw = nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)    
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2) 
        # Learnable compensation strength, initialised small for training stability    
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Low-pass: AvgPool ↓2 then bilinear ↑2
        x_low = F.interpolate(
            F.avg_pool2d(x, kernel_size=2, stride=2),
            size=x.shape[2:], mode='bilinear', align_corners=False,    
        )
        # High-frequency residual   
        x_hf = x - x_low     
        # Lightweight encoding + downsample
        out = self.pw(x_hf)    
        out = self.dw(out)
        out = self.bn(out)    
        out = self.pool(out)    
        return self.alpha * out    
 

class _DualDomainGlo(nn.Module):
    """Dual-Domain Global Refinement.     

    Combines *spatial-domain* channel attention (SE-style: GAP -> FC ->     
    Sigmoid) with *frequency-domain* channel attention (FFT amplitude   
    spectrum -> global mean -> FC -> Sigmoid).  The two sets of channel
    weights are averaged to produce the final modulation coefficients. 
    """

    def __init__(self, channel: int, reduction: int = 16) -> None:     
        super().__init__()
        mid = max(channel // reduction, 4)  
        # Spatial-domain branch (classic SE)
        self.spatial_gap = nn.AdaptiveAvgPool2d(1) 
        self.spatial_fc = nn.Sequential(
            nn.Linear(channel, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channel),
            nn.Sigmoid(), 
        ) 
        # Frequency-domain branch    
        self.freq_fc = nn.Sequential( 
            nn.Linear(channel, mid),   
            nn.ReLU(inplace=True),
            nn.Linear(mid, channel),     
            nn.Sigmoid(), 
        )   
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:    
        B, C, _, _ = x.shape  

        # Spatial attention 
        y_s = self.spatial_gap(x).view(B, C)
        w_s = self.spatial_fc(y_s).view(B, C, 1, 1)  
  
        # Frequency attention: per-channel mean of FFT amplitude spectrum    
        amp = torch.abs(fft.fftshift(fft.fft2(x.float())))  
        y_f = amp.mean(dim=[2, 3])  # (B, C)  
        w_f = self.freq_fc(y_f).view(B, C, 1, 1)

        # Dual-domain fusion
        w = (w_s + w_f) * 0.5
        return x * w

    
class FSCGD(nn.Module):
    """Frequency-Spectral Context-Guided Downsampling (FSCGD).

    A drop-in replacement for ContextGuidedBlock_Down that introduces three
    key innovations:

    1. **Frequency-Aware Context Routing** — an FFT-based spectral router 
       dynamically assigns softmax weights to multiple dilation paths
       (d=2, 4, 8) according to the input's low / mid / high frequency    
       energy distribution.
    2. **High-Frequency Compensation Bypass** — a lightweight side-branch
       that explicitly extracts and preserves high-frequency details lost
       during stride-2 downsampling.    
    3. **Dual-Domain Global Refinement** — extends the original FGlo    
       (spatial-only SE attention) to a dual-domain version that fuses
       spatial GAP statistics with frequency-domain amplitude statistics
       for richer channel recalibration.   
  
    Interface:   
        Input  — (B, nIn,  H,   W)    
        Output — (B, nOut, H/2, W/2) 
    """

    def __init__(
        self,
        nIn: int,    
        nOut: int,
        dilations: tuple = (2, 4, 8),  
        reduction: int = 16, 
    ) -> None:
        super().__init__()
        n_paths = len(dilations)     
    
        # [1] Stride-2 downsampling convolution
        self.conv_down = nn.Sequential(  
            nn.Conv2d(nIn, nOut, 3, stride=2, padding=1, bias=False),     
            nn.BatchNorm2d(nOut),
            nn.SiLU(inplace=True),
        ) 
     
        # [2] Frequency-spectral router   
        self.freq_router = _FreqRouter(nOut, n_paths)   
     
        # [3] Multi-dilation context paths (depthwise convolutions)     
        self.ctx_paths = nn.ModuleList()   
        for d in dilations:     
            self.ctx_paths.append(  
                nn.Conv2d(nOut, nOut, 3, padding=d, dilation=d,
                          groups=nOut, bias=False)
            )   
        self.ctx_bn = nn.BatchNorm2d(nOut)
        self.ctx_act = nn.SiLU(inplace=True)

        # [4] High-frequency compensation bypass
        self.hf_comp = _HFCompensator(nIn, nOut)  
  
        # [5] Dual-domain global refinement
        self.dual_glo = _DualDomainGlo(nOut, reduction)
  
    def forward(self, input: torch.Tensor) -> torch.Tensor:     
        # [1] Downsample: (B, nIn, H, W) -> (B, nOut, H/2, W/2)
        x_down = self.conv_down(input) 

        # [2] Frequency routing weights: (B, n_paths)
        weights = self.freq_router(x_down)  

        # [3] Multi-dilation context with frequency-guided weighting  
        ctx_feats = []
        for i, path in enumerate(self.ctx_paths):
            w = weights[:, i].view(-1, 1, 1, 1) 
            ctx_feats.append(path(x_down) * w)  
        f_ctx = sum(ctx_feats)   
        f_ctx = self.ctx_act(self.ctx_bn(f_ctx))
  
        # [4] High-frequency compensation (operates on pre-downsample input)
        f_hf = self.hf_comp(input)    
 
        # [5] Feature fusion     
        f_fused = f_ctx + f_hf 
     
        # [6] Dual-domain global refinement  
        output = self.dual_glo(f_fused) 

        return output     

    
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"  
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32     
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)  

    module = FSCGD(in_channel, out_channel).to(device)     

    outputs = module(inputs)     
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)    
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True, 
                                     output_precision=4,  
                                     print_detailed=True) 
    print(RESET)    
