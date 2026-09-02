'''  
本文件由BiliBili：魔傀面具整理     
engine/extre_module/module_images/CVPR2026-MSInit.png  
engine/extre_module/module_images/CVPR2026-MSInit.md     
论文链接：https://arxiv.org/pdf/2602.20537
'''
     
import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops

import torch
import torch.nn as nn   
   
class _RepDWLite(nn.Module):   
    """Lightweight re-parameterized depthwise composition."""    
    
    def __init__(self, dim: int, K: int, stride: int = 1) -> None:    
        super().__init__()
  
        self.dw_h = nn.Conv2d(    
            dim, 
            dim,  
            kernel_size=(1, K),
            stride=(1, stride),     
            padding=(0, K // 2),
            groups=dim,  
            bias=False,
        )
        self.dw_v = nn.Conv2d(
            dim,
            dim,
            kernel_size=(K, 1),    
            stride=(stride, 1),    
            padding=(K // 2, 0),    
            groups=dim, 
            bias=False,
        )    
        self.dw_s = nn.Conv2d(
            dim,     
            dim,
            kernel_size=3,
            stride=stride, 
            padding=1, 
            dilation=1,
            groups=dim,
            bias=False,
        )
        self.dw_i = nn.Conv2d(     
            dim,
            dim, 
            kernel_size=1,   
            stride=stride,
            groups=dim,
            bias=False, 
        )     
        nn.init.dirac_(self.dw_i.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   
        return self.dw_v(self.dw_h(x)) + self.dw_s(x) + self.dw_i(x)
  
 
class MSInit(nn.Module): 
    """Multi-scale initialization block with depthwise branch fusion."""    
 
    def __init__(
        self, 
        in_ch: int,  
        out_ch: int,     
        k_list: tuple[int, ...] = (3, 5, 7),
        stride: int = 1,
        use_gn: bool = True,
    ) -> None:
        super().__init__()  
        self.branches = nn.ModuleList(    
            [     
                nn.Sequential(
                    _RepDWLite(in_ch, K=k, stride=stride),     
                    nn.Conv2d(in_ch, out_ch // len(k_list), 1, bias=False),    
                )
                for k in k_list   
            ] 
        )
     
        gap = out_ch - (out_ch // len(k_list)) * len(k_list)
        self.tail = (     
            nn.Identity()
            if gap == 0  
            else nn.Sequential( 
                _RepDWLite(in_ch, K=k_list[0], stride=stride),
                nn.Conv2d(in_ch, gap, 1, bias=False),     
            )
        )

        self.fuse = nn.Identity()
        self.norm = nn.GroupNorm(1, out_ch) if use_gn else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:     
        parts = [branch(x) for branch in self.branches]
        if not isinstance(self.tail, nn.Identity): 
            parts.append(self.tail(x))   
        y = torch.cat(parts, dim=1)
        return self.act(self.norm(y))
  

if __name__ == '__main__':     
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)    
 
    module = MSInit(in_channel, out_channel).to(device)  
  
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)    
   
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,     
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)
