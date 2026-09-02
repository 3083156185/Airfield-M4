'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/TGRS2025-DSEBlock.png
engine/extre_module/module_images/TGRS2025-DSEBlock.md 
论文链接：https://arxiv.org/pdf/2601.16428
'''  

import os, sys 
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')    
  
import warnings    
warnings.filterwarnings('ignore')  
from calflops import calculate_flops 
    
import math
from functools import partial
from typing import Callable, Optional     
 
import torch
import torch.nn as nn     
import torch.nn.functional as F
 
try:   
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    pass

from engine.extre_module.ultralytics_nn.conv import Conv   

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()     
        self.drop_prob = float(drop_prob)     
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        if self.drop_prob == 0.0 or not self.training: 
            return x
        keep_prob = 1.0 - self.drop_prob  
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  
        return x.div(keep_prob) * random_tensor    

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"  
  

def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    batch_size, height, width, num_channels = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batch_size, height, width, groups, channels_per_group)   
    x = torch.transpose(x, 3, 4).contiguous()
    return x.view(batch_size, height, width, -1)  
     

class ChannelAttentionModule(nn.Module):   
    def __init__(self, in_channels: int, reduction: int = 4) -> None:
        super().__init__() 
        hidden_channels = max(1, in_channels // reduction) 
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),    
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:   
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))   
        return self.sigmoid(avg_out + max_out)
  
   
class SS2D(nn.Module):
    def __init__(     
        self, 
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,    
        expand: int = 2,
        dt_rank: str | int = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random", 
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        dropout: float = 0.0,
        conv_bias: bool = True, 
        bias: bool = False, 
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **_: object, 
    ) -> None:     
        super().__init__()  
        factory_kwargs = {"device": device, "dtype": dtype}  
        self.d_model = d_model    
        self.d_state = d_state  
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else int(dt_rank)

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

        x_proj = [
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
            for _ in range(4)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([layer.weight for layer in x_proj], dim=0))

        dt_projs = [     
            self.dt_init(  
                self.dt_rank,  
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,  
                dt_max,     
                dt_init_floor,
                **factory_kwargs,
            )  
            for _ in range(4)
        ]     
        self.dt_projs_weight = nn.Parameter(torch.stack([layer.weight for layer in dt_projs], dim=0))   
        self.dt_projs_bias = nn.Parameter(torch.stack([layer.bias for layer in dt_projs], dim=0))     

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True, device=device)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True, device=device)

        self.out_norm = nn.LayerNorm(self.d_inner)   
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)    
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None     
        self.channel_attention = ChannelAttentionModule(in_channels=self.d_inner)
   
    @staticmethod   
    def dt_init(     
        dt_rank: int,
        d_inner: int,  
        dt_scale: float = 1.0,
        dt_init: str = "random",
        dt_min: float = 0.001,
        dt_max: float = 0.1,  
        dt_init_floor: float = 1e-4,     
        **factory_kwargs: object,  
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
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
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
        device: Optional[torch.device] = None,     
        merge: bool = True,
    ) -> nn.Parameter:  
        a = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).unsqueeze(0).repeat(d_inner, 1)  
        a_log = torch.log(a)
        if copies > 1: 
            a_log = a_log.unsqueeze(0).repeat(copies, 1, 1)
            if merge:
                a_log = a_log.flatten(0, 1)   
        parameter = nn.Parameter(a_log)
        parameter._no_weight_decay = True
        return parameter 
     
    @staticmethod
    def D_init(
        d_inner: int, 
        copies: int = 1,
        device: Optional[torch.device] = None,    
        merge: bool = True,   
    ) -> nn.Parameter:     
        d = torch.ones(d_inner, device=device)  
        if copies > 1: 
            d = d.unsqueeze(0).repeat(copies, 1)   
            if merge:
                d = d.flatten(0, 1)    
        parameter = nn.Parameter(d) 
        parameter._no_weight_decay = True
        return parameter     
     
    def forward_core(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if selective_scan_fn is None:
            raise ImportError( 
                "SS_Conv_SSM requires a selective scan backend. Install mamba-ssm "
                "or provide a compatible selective_scan package."     
            ) 

        batch_size, _, height, width = x.shape
        length = height * width
        num_directions = 4
     
        x_hwwh = torch.stack(
            [   
                x.view(batch_size, -1, length),     
                torch.transpose(x, dim0=2, dim1=3).contiguous().view(batch_size, -1, length),
            ],
            dim=1,   
        ).view(batch_size, 2, -1, length)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) 
     
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(batch_size, num_directions, -1, length), self.x_proj_weight)
        dts, bs, cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(batch_size, num_directions, -1, length), self.dt_projs_weight)  

        xs = xs.float().view(batch_size, -1, length)     
        dts = dts.contiguous().float().view(batch_size, -1, length)     
        bs = bs.float().view(batch_size, num_directions, -1, length) 
        cs = cs.float().view(batch_size, num_directions, -1, length)
        ds = self.Ds.float().view(-1)  
        a_logs = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  
        dt_projs_bias = self.dt_projs_bias.float().view(-1)    

        out_y = selective_scan_fn(
            xs,     
            dts,     
            a_logs,
            bs,
            cs,     
            ds,  
            z=None,
            delta_bias=dt_projs_bias,  
            delta_softplus=True,
            return_last_state=False,
        ).view(batch_size, num_directions, -1, length)
    
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(batch_size, 2, -1, length)    
        wh_y = torch.transpose(out_y[:, 1].view(batch_size, -1, width, height), dim0=2, dim1=3).contiguous().view(batch_size, -1, length)
        invwh_y = torch.transpose(inv_y[:, 1].view(batch_size, -1, width, height), dim0=2, dim1=3).contiguous().view(batch_size, -1, length) 
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y
     
    def forward(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        batch_size, height, width, _ = x.shape     
   
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1) 
        z = z.permute(0, 3, 1, 2)
        z = self.channel_attention(z) * z
        z = z.permute(0, 2, 3, 1).contiguous()

        x_proj = x_proj.permute(0, 3, 1, 2).contiguous()    
        x_proj = self.act(self.conv2d(x_proj))    
     
        y1, y2, y3, y4 = self.forward_core(x_proj)   
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(batch_size, height, width, -1)
        y = self.out_norm(y)     
        y = y * F.silu(z)    
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out) 
        return out + x  
 
  
class SS_Conv_SSM(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,    
        drop_path: float = 0.0,    
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),    
        attn_drop_rate: float = 0.0,
        d_state: int = 16,
        **kwargs: object, 
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or hidden_dim % 2 != 0:  
            raise ValueError(f"hidden_dim must be a positive even integer, got {hidden_dim}")

        half_dim = hidden_dim // 2  
        self.ln_1 = norm_layer(half_dim)
        self.self_attention = SS2D(d_model=half_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)    
        self.drop_path = DropPath(drop_path)     
        self.conv33conv33conv11 = nn.Sequential(   
            nn.BatchNorm2d(half_dim),  
            nn.Conv2d(in_channels=half_dim, out_channels=half_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(half_dim),  
            nn.ReLU(),
            nn.Conv2d(in_channels=half_dim, out_channels=half_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(half_dim),
            nn.ReLU(),
            nn.Conv2d(in_channels=half_dim, out_channels=half_dim, kernel_size=1, stride=1),    
            nn.ReLU(),   
        )    
        self.channel_attention = ChannelAttentionModule(in_channels=half_dim)
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:     
        input_left, input_right = x.chunk(2, dim=-1)
        input_left = self.ln_1(input_left)
        input_right = self.ln_1(input_right)

        right_branch = self.drop_path(self.self_attention(input_right))     
   
        left_branch = input_left.permute(0, 3, 1, 2).contiguous()
        left_features = self.conv33conv33conv11(left_branch)     
        left_attention = self.channel_attention(left_branch) 
        left_features = left_features.permute(0, 2, 3, 1).contiguous()  
        left_attention = left_attention.permute(0, 2, 3, 1).contiguous()

        output = torch.cat((left_features * left_attention, right_branch), dim=-1)
        output = channel_shuffle(output, groups=2)     
        return output + x

class DSEBlock(nn.Module):   
    def __init__(self, in_channels, out_channels, d_state=16):   
        super().__init__() 

        self.m = SS_Conv_SSM(in_channels, in_channels, d_state=d_state)
        self.conv = Conv(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()  

    def forward(self, x):   
        x = self.conv(self.m(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))     
        return x

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32  
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)    
 
    # 此模块需要编译,详细编译命令在 compile_module/mamba
    # 对于Linux用户可以直接运行上述的make.sh文件  
    # 对于Windows用户需要逐行执行make.sh文件里的内容    
    module = DSEBlock(in_channel, out_channel).to(device)   
 
    outputs = module(inputs) 
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
     
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True, 
                                     output_precision=4,
                                     print_detailed=True)    
    print(RESET)