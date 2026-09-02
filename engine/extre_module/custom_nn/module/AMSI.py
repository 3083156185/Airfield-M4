'''
本文件由BiliBili：魔傀面具整理   
engine/extre_module/module_images/自研模块-AMSI.png  
engine/extre_module/module_images/自研模块-AMSI.md
'''

import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops 

import torch 
import torch.nn as nn 

     
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
 

class _BranchSEGate(nn.Module): 
    """Per-branch SE-style self-gating: GAP -> FC -> Sigmoid -> channel-wise scale."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.gate = nn.Sequential(   
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),  
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),   
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),     
        )   
   
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.gate(x).view(x.size(0), x.size(1), 1, 1)  
        return x * w
 
  
class AMSI(nn.Module):
    """Adaptive Multi-Scale Initialization (AMSI).    

    Extends MSInit with per-branch SE self-gating so that each scale's
    contribution is dynamically weighted by the input content rather than
    being fixed at concatenation time.   

    Data flow (per branch m):
        Z  ->  _RepDWLite(K=k_m)  ->  Conv1x1  ->  _BranchSEGate  ->  gated_m     
        gated_1, gated_2, ..., gated_M  ->  Concat  ->  GroupNorm  ->  GELU  ->  Y  
    """
     
    def __init__(   
        self,   
        in_ch: int,   
        out_ch: int,    
        k_list: tuple = (3, 5, 7),
        stride: int = 1,
        use_gn: bool = True,  
        se_reduction: int = 4,
    ) -> None:
        super().__init__()
        n = len(k_list)
        branch_ch = out_ch // n 
 
        self.branches = nn.ModuleList()    
        self.gates = nn.ModuleList()
        for k in k_list:     
            self.branches.append(nn.Sequential(    
                _RepDWLite(in_ch, K=k, stride=stride),
                nn.Conv2d(in_ch, branch_ch, 1, bias=False),
            ))
            self.gates.append(_BranchSEGate(branch_ch, reduction=se_reduction))
    
        gap = out_ch - branch_ch * n
        if gap == 0: 
            self.tail = None     
            self.tail_gate = None
        else:     
            self.tail = nn.Sequential(     
                _RepDWLite(in_ch, K=k_list[0], stride=stride),
                nn.Conv2d(in_ch, gap, 1, bias=False),     
            )  
            self.tail_gate = _BranchSEGate(gap, reduction=se_reduction)

        self.norm = nn.GroupNorm(1, out_ch) if use_gn else nn.Identity() 
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [gate(branch(x)) for branch, gate in zip(self.branches, self.gates)]    
        if self.tail is not None:
            parts.append(self.tail_gate(self.tail(x)))
        y = torch.cat(parts, dim=1)
        return self.act(self.norm(y))   


if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32     
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)
    
    module = AMSI(in_channel, out_channel).to(device)     
 
    outputs = module(inputs)     
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET) 

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,    
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,   
                                     print_detailed=True)     
    print(RESET)
