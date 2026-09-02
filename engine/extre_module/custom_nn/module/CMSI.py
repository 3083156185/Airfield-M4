'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/自研模块-CMSI.png
engine/extre_module/module_images/自研模块-CMSI.md    
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


class CMSI(nn.Module):
    """Cross-scale Multi-Scale Initialization (CMSI).
    
    Extends MSInit with progressive cross-scale residual injection:
    each branch (ordered from small to large kernel) receives an additive
    projection of the accumulated output from all previous branches before  
    its own depthwise extraction.  This builds a fine-to-coarse dependency    
    chain so that large-kernel branches are aware of local details already
    captured by small-kernel branches.     
   
    Data flow:
        acc = 0
        for m = 1..M (k_1 < k_2 < ... < k_M):   
            if m > 1:  acc = acc + proj_m(prev_branch_output)   
            T_m = RepDWLite(k_m)(Z + acc)  ->  Conv1x1  ->  branch_out_m  
            acc = branch_out_m  (projected back to in_ch for next injection)
        Y = Concat(branch_out_1, ..., branch_out_M)  ->  GroupNorm  ->  GELU
    """    

    def __init__(   
        self,
        in_ch: int,     
        out_ch: int,
        k_list: tuple = (3, 5, 7),
        stride: int = 1,
        use_gn: bool = True,
    ) -> None:
        super().__init__() 
        n = len(k_list)
        branch_ch = out_ch // n

        self.dw_blocks = nn.ModuleList(   
            [_RepDWLite(in_ch, K=k, stride=stride) for k in k_list]   
        )
        self.proj_1x1 = nn.ModuleList(
            [nn.Conv2d(in_ch, branch_ch, 1, bias=False) for _ in k_list]
        )
        # Cross-scale injection projections: project branch_ch -> in_ch
        # (only needed for branches 1..M-1 to feed into the next branch) 
        self.inject_proj = nn.ModuleList(
            [nn.Conv2d(branch_ch, in_ch, 1, bias=False) for _ in range(n - 1)] 
        )

        gap = out_ch - branch_ch * n    
        if gap == 0: 
            self.tail_dw = None     
        else:
            self.tail_dw = _RepDWLite(in_ch, K=k_list[0], stride=stride)   
            self.tail_proj = nn.Conv2d(in_ch, gap, 1, bias=False)
    
        self.norm = nn.GroupNorm(1, out_ch) if use_gn else nn.Identity() 
        self.act = nn.GELU()  
   
    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        parts = []     
        acc = None  # accumulated cross-scale context (in_ch space)
   
        for i, (dw, proj) in enumerate(zip(self.dw_blocks, self.proj_1x1)): 
            inp = x if acc is None else x + acc
            feat = proj(dw(inp))          # (B, branch_ch, H, W)
            parts.append(feat)   
            if i < len(self.inject_proj):
                acc_new = self.inject_proj[i](feat)  
                acc = acc_new if acc is None else acc + acc_new

        if self.tail_dw is not None:    
            inp = x if acc is None else x + acc
            parts.append(self.tail_proj(self.tail_dw(inp)))   
     
        y = torch.cat(parts, dim=1)
        return self.act(self.norm(y))

    
if __name__ == '__main__':     
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device) 
 
    module = CMSI(in_channel, out_channel).to(device)

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width), 
                                     output_as_string=True, 
                                     output_precision=4,   
                                     print_detailed=True)     
    print(RESET)
