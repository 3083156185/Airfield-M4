'''     
本文件由BiliBili：魔傀面具整理   
engine/extre_module/module_images/自研模块-FMSI.png
engine/extre_module/module_images/自研模块-FMSI.md
'''

import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops    

import torch   
import torch.nn as nn
import torch.fft as fft   
    

class _RepDWLite(nn.Module):
    """Lightweight re-parameterized depthwise composition (shared with MSInit).""" 

    def __init__(self, dim: int, K: int, stride: int = 1) -> None: 
        super().__init__()    
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(1, K), stride=(1, stride), 
                              padding=(0, K // 2), groups=dim, bias=False)   
        self.dw_v = nn.Conv2d(dim, dim, kernel_size=(K, 1), stride=(stride, 1),     
                              padding=(K // 2, 0), groups=dim, bias=False)
        self.dw_s = nn.Conv2d(dim, dim, kernel_size=3, stride=stride,    
                              padding=1, groups=dim, bias=False)  
        self.dw_i = nn.Conv2d(dim, dim, kernel_size=1, stride=stride,   
                              groups=dim, bias=False) 
        nn.init.dirac_(self.dw_i.weight)
     
    def forward(self, x: torch.Tensor) -> torch.Tensor:   
        return self.dw_v(self.dw_h(x)) + self.dw_s(x) + self.dw_i(x)    
   
     
class _FreqBandRouter(nn.Module):
    """Lightweight frequency-band router.

    Computes a 2D FFT amplitude spectrum, splits it into low / mid / high
    frequency bands via radial masks, extracts per-band global statistics  
    (mean amplitude), and maps them to M scale weights via a shared FC.
  
    The output is a (B, M) softmax weight vector — one scalar per branch.  
    No inverse FFT is performed; the frequency path is purely analytical.
    """ 
    
    def __init__(self, in_ch: int, n_scales: int, hidden: int = 16) -> None:
        super().__init__() 
        # 3 band statistics (low/mid/high) × in_ch channels -> n_scales weights
        self.fc = nn.Sequential( 
            nn.Linear(3 * in_ch, hidden, bias=False),
            nn.ReLU(inplace=True),    
            nn.Linear(hidden, n_scales, bias=False),    
        )
        self.n_scales = n_scales

    @staticmethod
    def _radial_masks(H: int, W: int, device: torch.device):
        """Return low/mid/high boolean masks based on radial frequency distance.""" 
        cy, cx = H // 2, W // 2  
        ys = torch.arange(H, device=device).float() - cy
        xs = torch.arange(W, device=device).float() - cx 
        r = torch.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2)   
        r_max = (min(H, W) / 2.0)
        low  = r < r_max * 0.25  
        high = r > r_max * 0.65  
        mid  = ~low & ~high  
        return low, mid, high    

    def forward(self, x: torch.Tensor) -> torch.Tensor:     
        B, C, H, W = x.shape
        # 2D FFT amplitude spectrum, shifted so DC is at center
        amp = torch.abs(fft.fftshift(fft.fft2(x.float())))  # (B, C, H, W)
        low_m, mid_m, high_m = self._radial_masks(H, W, x.device)
        # Per-band mean amplitude across spatial dims -> (B, C) 
        low_stat  = amp[:, :, low_m ].mean(dim=-1)
        mid_stat  = amp[:, :, mid_m ].mean(dim=-1)  
        high_stat = amp[:, :, high_m].mean(dim=-1)
        stat = torch.cat([low_stat, mid_stat, high_stat], dim=1)  # (B, 3C)  
        weights = torch.softmax(self.fc(stat), dim=-1)             # (B, M)  
        return weights    
 
   
class FMSI(nn.Module):
    """Frequency-guided Multi-Scale Initialization (FMSI).

    Extends MSInit with a lightweight FFT-based frequency router that    
    generates per-scale attention weights from the input's spectral 
    statistics.  Each branch output is multiplied by its corresponding   
    frequency-derived scalar weight before concatenation, enabling   
    cross-domain (frequency → spatial) adaptive multi-scale selection.

    Data flow:
        Z  ->  FreqBandRouter  ->  softmax weights [w_1, ..., w_M]   (B, M)
        Z  ->  branch_m: RepDWLite(k_m) -> Conv1x1  ->  T_m          (B, C_m, H, W)
        Y_m = T_m * w_m                                               (broadcast)
        Y   = Concat(Y_1, ..., Y_M)  ->  GroupNorm  ->  GELU 
    """

    def __init__(
        self,
        in_ch: int,  
        out_ch: int,
        k_list: tuple = (3, 5, 7), 
        stride: int = 1,
        use_gn: bool = True,     
        router_hidden: int = 16,  
    ) -> None:
        super().__init__() 
        n = len(k_list)
        branch_ch = out_ch // n     
     
        self.branches = nn.ModuleList()
        for k in k_list:
            self.branches.append(nn.Sequential(
                _RepDWLite(in_ch, K=k, stride=stride),    
                nn.Conv2d(in_ch, branch_ch, 1, bias=False),
            ))     
     
        gap = out_ch - branch_ch * n 
        if gap == 0:  
            self.tail = None  
        else:
            self.tail = nn.Sequential(  
                _RepDWLite(in_ch, K=k_list[0], stride=stride),
                nn.Conv2d(in_ch, gap, 1, bias=False),
            )   
    
        total_branches = n + (1 if self.tail is not None else 0)
        self.router = _FreqBandRouter(in_ch, total_branches, hidden=router_hidden)   
  
        self.norm = nn.GroupNorm(1, out_ch) if use_gn else nn.Identity()
        self.act = nn.GELU()    
   
    def forward(self, x: torch.Tensor) -> torch.Tensor:    
        weights = self.router(x)  # (B, M)

        parts = []
        for i, branch in enumerate(self.branches):   
            w = weights[:, i].view(-1, 1, 1, 1)  
            parts.append(branch(x) * w)  
     
        if self.tail is not None:   
            w = weights[:, len(self.branches)].view(-1, 1, 1, 1)   
            parts.append(self.tail(x) * w)
     
        y = torch.cat(parts, dim=1)   
        return self.act(self.norm(y))     
  
   
if __name__ == '__main__':    
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = FMSI(in_channel, out_channel).to(device)
     
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)     
    
    print(ORANGE)  
    flops, macs, _ = calculate_flops(model=module,    
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)   
