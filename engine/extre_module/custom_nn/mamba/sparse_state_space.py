'''
本文件由BiliBili：魔傀面具整理     
engine/extre_module/module_images/CVPR2026-SparseStateSpace.md   
engine/extre_module/module_images/CVPR2026-SparseStateSpace.png
论文链接：https://arxiv.org/pdf/2602.21917    
'''  
    
import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops    

import math
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F   
from einops import rearrange, repeat
     
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref    
except Exception as e:
    pass

def to_3d(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
 

class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape):    
        super().__init__()     
        if isinstance(normalized_shape, numbers.Integral):   
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight + self.bias
  

class LayerNorm(nn.Module):
    def __init__(self, dim: int):    
        super().__init__()
        self.body = WithBiasLayerNorm(dim)     
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]  
        return to_4d(self.body(to_3d(x)), h, w)     
 
  
def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    x1 = F.normalize(x1, dim=-1) 
    x2 = F.normalize(x2, dim=-1)   
    return torch.matmul(x1, x2.transpose(-2, -1))    
     

def selective_scan(   
    xs: torch.Tensor,  
    dts: torch.Tensor,   
    As: torch.Tensor,
    Bs_: torch.Tensor,
    Cs_: torch.Tensor,
    Ds: torch.Tensor, 
    dt_projs_bias: torch.Tensor,
) -> torch.Tensor:
    scan_impl = selective_scan_fn if xs.is_cuda else selective_scan_ref   
    return scan_impl(
        xs,
        dts,
        As,
        Bs_,
        Cs_,
        Ds,
        z=None,     
        delta_bias=dt_projs_bias,  
        delta_softplus=True,     
        return_last_state=False,     
    )
   
    
class SparseStateSpace(nn.Module):
    def __init__( 
        self,
        channels,  
        proposal_hw=2,  # Sparse center proposal size used for region aggregation.
        fold_hw=2,  # Spatial fold factor for splitting large feature maps into local regions.  
        heads=1,  # Number of sparse state-space heads.
        d_state=8,  # Hidden state size inside the state-space scan.     
        d_conv=3,  # Depthwise convolution kernel size before sparse scanning.
        expand=2,  # Channel expansion ratio used to build the inner dimension.    
        dt_rank="auto",  # Rank of the delta-time projection.  
        dt_min=0.001,  # Minimum value used when initializing delta-time.    
        dt_max=0.1,  # Maximum value used when initializing delta-time.   
        dt_init="random",  # Initialization mode for the delta-time projection. 
        dt_scale=1.0,  # Scaling factor applied to delta-time initialization.
        dt_init_floor=1e-4,  # Lower clamp to keep initialized delta-time numerically stable.  
        dropout=0.0,  # Dropout applied after the output projection.     
        conv_bias=True,  # Whether the depthwise convolution uses bias.
        bias=False,  # Whether linear projections use bias.
        device=None,     
        dtype=None, 
    ):   
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
    
        self.d_model = channels    
        self.proposal_hw = proposal_hw    
        self.fold_hw = fold_hw
        self.heads = heads     
        self.d_state = d_state   
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model) // self.heads
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank  

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)     
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,   
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,  
            **factory_kwargs,
        )    
        self.act = nn.SiLU()     

        x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)     
        self.x_proj_weight = nn.Parameter(x_proj.weight.unsqueeze(0))
  
        self.x_conv = nn.Conv1d(   
            in_channels=self.dt_rank + self.d_state * 2, 
            out_channels=self.dt_rank + self.d_state * 2,     
            kernel_size=7, 
            padding=3,     
            groups=self.dt_rank + self.d_state * 2,
        )    
     
        dt_proj = self.dt_init(     
            self.dt_rank, 
            self.d_inner,   
            dt_scale,    
            dt_init,
            dt_min,
            dt_max,  
            dt_init_floor,
            **factory_kwargs,
        )
        self.dt_projs_weight = nn.Parameter(dt_proj.weight.unsqueeze(0))  
        self.dt_projs_bias = nn.Parameter(dt_proj.bias.unsqueeze(0))   

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        self.f = nn.Conv2d(self.d_inner, self.d_inner * self.heads, kernel_size=1) 
        self.proj = nn.Conv2d(self.d_inner * self.heads, self.d_inner, kernel_size=1)  
        self.v = nn.Conv2d(self.d_inner, self.d_inner * self.heads, kernel_size=1)
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))  
        self.centers_proposal = nn.AdaptiveAvgPool2d((self.proposal_hw, self.proposal_hw))
   
    @staticmethod
    def dt_init(    
        dt_rank,
        d_inner,
        dt_scale=1.0, 
        dt_init="random", 
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,   
        **factory_kwargs,
    ):   
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
     
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":    
            nn.init.constant_(dt_proj.weight, dt_init_std) 
        elif dt_init == "random":     
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)    
        else:   
            raise NotImplementedError(f"Unsupported dt_init: {dt_init}")   

        dt = torch.exp( 
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)   
        ).clamp(min=dt_init_floor)    
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge: 
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True   
        return A_log    

    @staticmethod    
    def D_init(d_inner, copies=1, device=None, merge=True):  
        D = torch.ones(d_inner, device=device)
        if copies > 1:   
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)   
        D = nn.Parameter(D)
        D._no_weight_decay = True   
        return D
     
    def forward_core(self, x: torch.Tensor) -> torch.Tensor:
        value = self.v(x)
        x = self.f(x)
        x = rearrange(x, "b (e c) w h -> (b e) c w h", e=self.heads)   
        value = rearrange(value, "b (e c) w h -> (b e) c w h", e=self.heads)   
        if self.fold_hw > 1:    
            b0, c0, w0, h0 = x.shape     
            assert w0 % self.fold_hw == 0 and h0 % self.fold_hw == 0, (   
                f"Ensure the feature map size ({w0}*{h0}) can be divided by fold {self.fold_hw}*{self.fold_hw}"
            )
            x = rearrange(
                x,    
                "b c (f1 w) (f2 h) -> (b f1 f2) c w h",    
                f1=self.fold_hw,
                f2=self.fold_hw,
            )
            value = rearrange(
                value,    
                "b c (f1 w) (f2 h) -> (b f1 f2) c w h",
                f1=self.fold_hw,
                f2=self.fold_hw,   
            )
        b, c, w, h = x.shape
        centers = self.centers_proposal(x)
        value_centers = rearrange(self.centers_proposal(value), "b c w h -> b (w h) c")

        b, c, ww, hh = centers.shape  
        sim = torch.sigmoid( 
            self.sim_beta 
            + self.sim_alpha  
            * pairwise_cos_sim(
                centers.reshape(b, c, -1).permute(0, 2, 1),
                x.reshape(b, c, -1).permute(0, 2, 1), 
            )
        )  
    
        _, sim_max_idx = sim.max(dim=1, keepdim=True)   
        mask = torch.zeros_like(sim)
        mask.scatter_(1, sim_max_idx, 1.0)     
        sim = sim * mask
        value_flat = rearrange(value, "b c w h -> b (w h) c")
        out = (
            (value_flat.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers  
        ) / (sim.sum(dim=-1, keepdim=True) + 1.0)

        batch_size, length, _ = out.shape
        xs = rearrange(out, "b l c -> b c l")
        xs = torch.stack([xs], dim=1).view(batch_size, 1, -1, length)   
        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l",   
            xs.view(batch_size, 1, -1, length),    
            self.x_proj_weight,
        )     
        x_dbl = self.x_conv(x_dbl.squeeze(1)).unsqueeze(1)     
        dts, Bs_, Cs_ = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(batch_size, 1, -1, length), self.dt_projs_weight)    
        xs = xs.float().view(batch_size, -1, length) 
        dts = dts.contiguous().float().view(batch_size, -1, length)    
        Bs_ = Bs_.float().view(batch_size, 1, -1, length)
        Cs_ = Cs_.float().view(batch_size, 1, -1, length)    
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        out_y = selective_scan(xs, dts, As, Bs_, Cs_, Ds, dt_projs_bias).view(batch_size, 1, -1, length)    
        out = rearrange(out_y[:, 0], "b c l -> b l c")    
 
        out = (out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)
        out = rearrange(out, "b (w h) c -> b c w h", w=w)  
     
        if self.fold_hw > 1:
            out = rearrange(
                out,
                "(b f1 f2) c w h -> b c (f1 w) (f2 h)",
                f1=self.fold_hw,
                f2=self.fold_hw,
            )
        out = rearrange(out, "(b e) c w h -> b (e c) w h", e=self.heads)
        return self.proj(out)     
     
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b c h w -> b h w c")     
        batch_size, height, width, channels = x.shape 
 
        xz = self.in_proj(x)     
        x, z = xz.chunk(2, dim=-1)    
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))    
        y = self.forward_core(x) 
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(batch_size, height, width, -1)  
        y = self.out_norm(y) 
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out) 
        return rearrange(out, "b h w c -> b c h w")    

if __name__ == '__main__': 
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    batch_size, channel, height, width = 1, 16, 20, 20 
    inputs = torch.randn((batch_size, channel, height, width)).to(device) 
    
    # 此模块需要编译,详细编译命令在 compile_module/mamba
    # 对于Linux用户可以直接运行上述的make.sh文件    
    # 对于Windows用户需要逐行执行make.sh文件里的内容  
    module = SparseStateSpace(channel).to(device)     
    
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
  
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,     
                                     input_shape=(batch_size, channel, height, width), 
                                     output_as_string=True, 
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)    
