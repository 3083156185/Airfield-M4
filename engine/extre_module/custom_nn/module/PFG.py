'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/CVPR2026-PFG.png  
engine/extre_module/module_images/CVPR2026-PFG.md   
论文链接：https://arxiv.org/pdf/2602.20537   
'''
    
import warnings
warnings.filterwarnings('ignore') 
from calflops import calculate_flops

import torch    
import torch.nn as nn
import torch.nn.functional as F

def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:   
    """Stochastic depth per sample."""   
    if drop_prob == 0.0 or not training:    
        return x
    keep_prob = 1.0 - drop_prob   
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)   
    random_tensor.floor_()    
    return x.div(keep_prob) * random_tensor   

 
class DropPath(nn.Module): 
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0) -> None:    
        super().__init__()   
        self.drop_prob = drop_prob 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)   

    
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

    
class PFGA(nn.Module):
    """Peripheral-Frequency Guided Aggregation token mixer."""   

    class Branch(nn.Module):    
        def __init__(self, dim: int, K: int, center_suppress: bool = True) -> None:   
            super().__init__()
            self.center_suppress = center_suppress
            self.dw_h = nn.Conv2d(    
                dim, 
                dim,     
                kernel_size=(1, K),
                padding=(0, K // 2), 
                groups=dim,  
                bias=False, 
            )
            self.dw_v = nn.Conv2d(
                dim,
                dim,
                kernel_size=(K, 1),   
                padding=(K // 2, 0),
                groups=dim, 
                bias=False,  
            ) 

            if self.center_suppress:  
                self.dw_c = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
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
        dim: int, 
        K_list: tuple[int, ...] = (9, 15, 31),
        use_grn: bool = False,
        center_suppress: bool = True, 
    ) -> None:
        super().__init__()     
        self.dim = dim     
        self.K_list = K_list
        self.branches = nn.ModuleList(    
            [PFGA.Branch(dim, K, center_suppress=center_suppress) for K in K_list]    
        )
   
        sobel_x = torch.tensor(   
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)   
        sobel_y = torch.tensor(    
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        laplace = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)     
 
        self.register_buffer("sobel_x", sobel_x, persistent=False)   
        self.register_buffer("sobel_y", sobel_y, persistent=False)  
        self.register_buffer("laplace", laplace, persistent=False) 

        self.gate_head = nn.Conv2d(3, len(K_list), kernel_size=1, bias=True)
        self.use_grn = use_grn     
        if use_grn:
            self.grn = GRN(dim)     
    
    def _depthwise_filter(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        _, channels, _, _ = x.shape
        weight = kernel.repeat(channels, 1, 1, 1)     
        return F.conv2d(x, weight, padding=1, groups=channels)    
    
    def _freq_maps(self, x: torch.Tensor) -> torch.Tensor:
        gx = self._depthwise_filter(x, self.sobel_x)
        gy = self._depthwise_filter(x, self.sobel_y)
        lap = self._depthwise_filter(x, self.laplace)   
   
        grad_mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)
        mean = F.avg_pool2d(x, 3, 1, 1)   
        mean2 = F.avg_pool2d(x * x, 3, 1, 1)     
        var = torch.clamp(mean2 - mean * mean, min=0.0)
  
        f1 = grad_mag.mean(dim=1, keepdim=True)
        f2 = lap.abs().mean(dim=1, keepdim=True)
        f3 = var.mean(dim=1, keepdim=True)     
        return torch.cat([f1, f2, f3], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:    
        peris = [branch(x) for branch in self.branches] 
        freq = self._freq_maps(x) 
        logits = self.gate_head(freq)
        alpha = torch.softmax(logits, dim=1)   

        y = 0.0
        for i, branch_out in enumerate(peris):
            y = y + branch_out * alpha[:, i : i + 1, :, :]
 
        if self.use_grn:
            y = self.grn(y)
        return y
     
  
class PFG(nn.Module):
    """PFG block with PFGA token mixing and GLU-style channel mixing."""    

    def __init__(  
        self,     
        in_ch: int | None = None,
        out_ch: int | None = None,    
        groups_pw: int = 1,
        layerscale_init: float = 1e-6, 
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
        drop_path: float = 0.0,   
        pfga_K: tuple[int, ...] = (9, 15, 31),
        mlp_ratio: float = 4.0,
        dw_kernel: int = 3,
    ) -> None:
        super().__init__()
        out_ch = in_ch if out_ch is None else out_ch

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.dim = in_ch 
        self.norm_dw = nn.GroupNorm(num_groups=min(32, in_ch), num_channels=in_ch) 
        self.norm_pw = nn.GroupNorm(num_groups=min(32, in_ch), num_channels=in_ch)  

        self.tm = PFGA(in_ch, K_list=pfga_K, use_grn=False)   
        self.grn_dw = GRN(in_ch)     
        self.grn_pw = GRN(in_ch)

        self.mlp_ratio = mlp_ratio
        self.dw_kernel = dw_kernel
   
        expanded_dim = max(in_ch, int(in_ch * self.mlp_ratio))   
        self.pw_in = nn.Conv2d(in_ch, 2 * expanded_dim, kernel_size=1, bias=True, groups=groups_pw) 
        self.dw_v = nn.Conv2d(     
            expanded_dim,
            expanded_dim, 
            kernel_size=self.dw_kernel,
            padding=1,     
            groups=expanded_dim,
            bias=False,
        )
        self.pw_out = nn.Conv2d(expanded_dim, in_ch, kernel_size=1, bias=True, groups=groups_pw)     

        self.act = act_layer()
        self.gamma_dw = nn.Parameter(torch.ones(in_ch) * layerscale_init)
        self.gamma_pw = nn.Parameter(torch.ones(in_ch) * layerscale_init)  
   
        self.dropout_dw = nn.Dropout(drop) if drop > 0 else nn.Identity() 
        self.dropout_pw = nn.Dropout(drop) if drop > 0 else nn.Identity()  
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.out_proj = (
            nn.Identity()    
            if in_ch == out_ch    
            else nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) 
        )   
     
        self._init_params() 
    
    @torch.jit.ignore     
    def no_weight_decay(self) -> set[str]:
        return {"gamma_dw", "gamma_pw"}     
     
    def _init_params(self) -> None:
        for module in self.modules():    
            if isinstance(module, nn.Conv2d):    
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")    
                if module.bias is not None:  
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)  
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm_dw(x)   
        y = self.tm(y)
        y = self.act(y)  
        y = self.grn_dw(y)  
        y = self.dropout_dw(y)
        x = x + self.drop_path(y * self.gamma_dw.view(1, self.dim, 1, 1))

        z = self.norm_pw(x)    
        uv = self.pw_in(z)
        u, v = torch.chunk(uv, 2, dim=1)  
        v = self.dw_v(v)
        z = F.silu(u) * v     
        z = self.pw_out(z)

        z = self.grn_pw(z) 
        z = self.dropout_pw(z)    
        x = x + self.drop_path(z * self.gamma_pw.view(1, self.dim, 1, 1))    
        return self.out_proj(x)

     
if __name__ == '__main__':   
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"     
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)  
 
    module = PFG(in_channel, out_channel).to(device)     
 
    outputs = module(inputs)    
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
   
    print(ORANGE) 
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,    
                                     print_detailed=True)  
    print(RESET)  
