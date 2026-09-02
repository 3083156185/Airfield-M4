'''
本文件由BiliBili：魔傀面具整理     
engine/extre_module/module_images/AAAI2026-GradMamba.md
engine/extre_module/module_images/AAAI2026-GradMamba.png 
论文链接：https://arxiv.org/pdf/2508.03336    
'''
  
import warnings
warnings.filterwarnings('ignore') 
from calflops import calculate_flops
     
import math

import torch
import torch.nn as nn 
import torch.nn.functional as F     

try:    
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref     
except ImportError as e:    
    pass
  
def semantic_neighbor(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor: 
    dim = index.dim() 
    assert x.shape[:dim] == index.shape, (   
        f"x ({x.shape}) and index ({index.shape}) shape incompatible"
    )

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape) 
    return torch.gather(x, dim=dim - 1, index=index)


def index_reverse(index: torch.Tensor) -> torch.Tensor:    
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1], device=index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r     
     

class GradientExtractor(nn.Module):     
    def __init__(self) -> None:
        super().__init__()  
        self.register_buffer(
            "sobel_x",
            torch.tensor(   
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32     
            ).unsqueeze(0).unsqueeze(0),  
        )    
        self.register_buffer(
            "sobel_y",
            torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32  
            ).unsqueeze(0).unsqueeze(0),
        )
        self.register_buffer(  
            "laplacian",    
            torch.tensor(
                [[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32  
            ).unsqueeze(0).unsqueeze(0),
        )    
        self.channel_fusion = nn.Conv2d(3, 1, 1, bias=False)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        gray = x.mean(dim=1, keepdim=True) if x.size(1) > 1 else x 
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)    
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)     
        grad_lap = F.conv2d(gray, self.laplacian, padding=1)  
        grad_stack = torch.cat([grad_x, grad_y, grad_lap], dim=1)     
        gradient_magnitude = self.channel_fusion(grad_stack)   
        return torch.abs(gradient_magnitude)

    
class SimplifiedGradientToPriority(nn.Module):   
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))   
        self.offset = nn.Parameter(torch.tensor(0.0))    
   
    def forward(self, gradient_map: torch.Tensor) -> torch.Tensor:
        gradient = gradient_map.squeeze(1)
        b, h, w = gradient.shape
        gradient_flat = gradient.view(b, -1)   
        grad_min = gradient_flat.min(dim=1, keepdim=True)[0]
        grad_max = gradient_flat.max(dim=1, keepdim=True)[0]
        gradient_norm = (gradient_flat - grad_min) / (grad_max - grad_min + 1e-8) 
        gradient_norm = gradient_norm.view(b, h, w)  
        priority_score = self.scale * gradient_norm + self.offset
        return torch.sigmoid(priority_score)   


class GradientGuidedSelectiveScan(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: float = 2.0,     
        dt_rank="auto",  
        dt_min: float = 0.001,  
        dt_max: float = 0.1,
        dt_init: str = "random",  
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,  
        device=None,  
        dtype=None,   
    ) -> None:    
        super().__init__()   
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state   
        self.expand = expand    
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank 
  
        self.x_proj = (
            nn.Linear(
                self.d_inner,
                self.dt_rank + self.d_state * 2,  
                bias=False,
                **factory_kwargs,   
            ),     
        )   
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj    

        self.dt_projs = ( 
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,  
                dt_min,    
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
        )     
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs], dim=0)
        )     
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) 
        del self.dt_projs 

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)
     
        self.gradient_c_enhancer = nn.Linear(1, self.d_state)
        self.gradient_c_weight = nn.Parameter(torch.tensor(0.1))
        self.feedback_a_weight = nn.Parameter(torch.tensor(0.05))
        self.selective_scan = selective_scan_fn
        self.selective_scan_ref = selective_scan_ref  
   
    @staticmethod
    def dt_init(
        dt_rank: int,     
        d_inner: int,
        dt_scale: float = 1.0,
        dt_init: str = "random", 
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        **factory_kwargs, 
    ) -> nn.Linear:
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)    
        elif dt_init == "random":   
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)    
        else:
            raise NotImplementedError(f"Unsupported dt_init: {dt_init}")

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  
        with torch.no_grad():    
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj    

    @staticmethod     
    def A_log_init(
        d_state: int,     
        d_inner: int,
        copies: int = 1,
        device=None,
        merge: bool = True,
    ) -> nn.Parameter: 
        a = torch.arange(1, d_state + 1, dtype=torch.float32, device=device)
        a = a.repeat(d_inner, 1).contiguous()   
        a_log = torch.log(a)
        if copies > 1:
            a_log = a_log.repeat(copies, 1, 1)
            if merge:
                a_log = a_log.flatten(0, 1)
        a_log = nn.Parameter(a_log)
        a_log._no_weight_decay = True     
        return a_log

    @staticmethod 
    def D_init(   
        d_inner: int,   
        copies: int = 1,
        device=None,  
        merge: bool = True,
    ) -> nn.Parameter:
        d = torch.ones(d_inner, device=device)
        if copies > 1:     
            d = d.repeat(copies, 1)   
            if merge:
                d = d.flatten(0, 1)     
        d = nn.Parameter(d) 
        d._no_weight_decay = True
        return d  

    def forward_core(self, x: torch.Tensor, gradient_score: torch.Tensor) -> torch.Tensor:
        b, l, c = x.shape
        k = 1
        xs = x.permute(0, 2, 1).view(b, 1, c, l).contiguous()

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(b, k, -1, l), self.x_proj_weight)  
        dts, bs, cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)   
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(b, k, -1, l), self.dt_projs_weight)  
 
        gradient_flat = gradient_score.view(b, -1, 1)
        gradient_enhancement = self.gradient_c_enhancer(gradient_flat)   
        gradient_enhancement = gradient_enhancement.view(b, k, self.d_state, l)     
        cs_enhanced = cs + self.gradient_c_weight * gradient_enhancement   
  
        a_s = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        xs = xs.float().view(b, -1, l)
        dts = dts.contiguous().float().view(b, -1, l)     
        bs = bs.float().view(b, k, -1, l)     
        cs_enhanced = cs_enhanced.float().view(b, k, -1, l)     
        ds = self.Ds.float().view(-1)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
 
        selective_scan_impl = self.selective_scan if xs.is_cuda else self.selective_scan_ref
        out_y = selective_scan_impl(
            xs,
            dts,  
            a_s,
            bs, 
            cs_enhanced,     
            ds,
            z=None,
            delta_bias=dt_projs_bias, 
            delta_softplus=True, 
            return_last_state=False,  
        ).view(b, k, -1, l)   
        return out_y[:, 0]  

    def forward(self, x: torch.Tensor, gradient_score: torch.Tensor) -> torch.Tensor:
        y = self.forward_core(x, gradient_score)
        return y.permute(0, 2, 1).contiguous()


class GradientGuidedMamba(nn.Module):   
    def __init__(self, dim: int, d_state: int, mlp_ratio: float = 2.0) -> None:     
        super().__init__()     
        self.dim = dim  
        self.expand = mlp_ratio 
        hidden = int(self.dim * self.expand)
        self.d_state = d_state  
   
        self.selective_scan = GradientGuidedSelectiveScan(
            d_model=hidden, 
            d_state=self.d_state,
            expand=1,
        )   
        self.out_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, dim, bias=True)
        self.in_proj = nn.Sequential(nn.Conv2d(self.dim, hidden, 1, 1, 0))
        self.cpe = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden),   
        )   

    def forward(self, x: torch.Tensor, gradient_score: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape    
        _, h, w = gradient_score.shape
        assert n == h * w, f"Token count ({n}) does not match spatial size ({h}x{w})."  
  
        gradient_score_flat = gradient_score.view(b, -1)
        _, x_sort_indices = torch.sort(gradient_score_flat, dim=-1, stable=False)
        x_sort_indices_reverse = index_reverse(x_sort_indices)
    
        x_spatial = x.permute(0, 2, 1).reshape(b, c, h, w).contiguous()
        x_proj = self.in_proj(x_spatial)
        x_proj = x_proj * torch.sigmoid(self.cpe(x_proj))     

        cc = x_proj.shape[1] 
        x_proj = x_proj.view(b, cc, -1).permute(0, 2, 1).contiguous()
        semantic_x = semantic_neighbor(x_proj, x_sort_indices)
        y = self.selective_scan(semantic_x, gradient_score)
        y = self.out_proj(self.out_norm(y))
        return semantic_neighbor(y, x_sort_indices_reverse)
 

class GradMamba(nn.Module):
    def __init__(self, dim: int, d_state: int = 16, mlp_ratio: float = 2.0, scale: float = 1.0) -> None:
        super().__init__()
        self.dim = dim   
        self.d_state = d_state   
        self.scale = scale     
  
        self.norm1 = nn.LayerNorm(dim)    
        self.norm2 = nn.LayerNorm(dim)
        self.gradient_extractor = GradientExtractor()
        self.gradient_to_priority = SimplifiedGradientToPriority()
        self.gradient_mamba = GradientGuidedMamba(dim, d_state, mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), 
            nn.SiLU(),
            nn.Linear(int(dim * mlp_ratio), dim),     
        )

    @staticmethod     
    def _spatial_to_tokens(x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        return x.view(b, c, h * w).permute(0, 2, 1).contiguous()  
  
    @staticmethod
    def _tokens_to_spatial(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        assert n == h * w, f"Token count ({n}) does not match spatial size ({h}x{w})."   
        return x.permute(0, 2, 1).reshape(b, c, h, w).contiguous()   

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4: 
            raise ValueError(   
                f"GradStateSpaceBlock expects input in BCHW format, but got shape {tuple(x.shape)}."     
            )    
  
        _, _, h, w = x.shape
        gradient_priority = self.gradient_to_priority(self.gradient_extractor(x))
        x_tokens = self._spatial_to_tokens(x)   
     
        residual = x_tokens
        x_tokens = self.norm1(x_tokens)
        x_tokens = self.gradient_mamba(x_tokens, gradient_priority)
        x_tokens = residual + self.scale * x_tokens   

        residual = x_tokens
        x_tokens = self.norm2(x_tokens) 
        x_tokens = self.mlp(x_tokens)   
        x_tokens = residual + self.scale * x_tokens

        return self._tokens_to_spatial(x_tokens, h, w) 


if __name__ == "__main__":     
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel, height, width = 1, 16, 20, 20
    inputs = torch.randn((batch_size, channel, height, width)).to(device)

    # 此模块需要编译,详细编译命令在 compile_module/mamba   
    # 对于Linux用户可以直接运行上述的make.sh文件   
    # 对于Windows用户需要逐行执行make.sh文件里的内容   
    module = GradMamba(channel).to(device)
    
    outputs = module(inputs)   
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)  

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,    
                                     input_shape=(batch_size, channel, height, width),   
                                     output_as_string=True,     
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET) 
