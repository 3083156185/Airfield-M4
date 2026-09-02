'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/CVPR2025-MRFFI.png
MobileMamba中的Multi-Receptive Field Feature Interaction (MRFFI) module 
论文链接：https://arxiv.org/pdf/2411.15941  
'''
   
import warnings     
warnings.filterwarnings('ignore')
from calflops import calculate_flops
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import repeat   
from functools import partial  
import pywt
import pywt.data  

try:
    from .csm_triton import cross_scan_fn, cross_merge_fn 
except:    
    from csm_triton import cross_scan_fn, cross_merge_fn   
     
try:   
    from .csm_tritonk2 import cross_scan_fn_k2, cross_merge_fn_k2   
    from .csm_tritonk2 import cross_scan_fn_k2_torch, cross_merge_fn_k2_torch
except:
    from csm_tritonk2 import cross_scan_fn_k2, cross_merge_fn_k2
    from csm_tritonk2 import cross_scan_fn_k2_torch, cross_merge_fn_k2_torch
   
try:  
    from .csms6s import selective_scan_fn, selective_scan_flop_jit 
except:
    from csms6s import selective_scan_fn, selective_scan_flop_jit
   
selective_scan_chunk_fn = None    

class LayerNorm2d(nn.LayerNorm):     
    def forward(self, x: torch.Tensor):   
        x = x.permute(0, 2, 3, 1)   
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)     
        x = x.permute(0, 3, 1, 2) 
        return x    
     
class Permute(nn.Module):
    def __init__(self, *args):    
        super().__init__()  
        self.args = args   
    
    def forward(self, x: torch.Tensor):     
        return x.permute(*self.args)   

class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):     
        if self.dim == -1:
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1).contiguous()).view(B, C, H, W).contiguous()   
        elif self.dim == 1:     
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C).contiguous()).view(B, H, W, C).contiguous()
        else:
            raise NotImplementedError

class Linear2d(nn.Linear):     
    def forward(self, x: torch.Tensor):   
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)   
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                             error_msgs)

class Conv2d_BN(torch.nn.Sequential):    
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,   
                 groups=1, bn_weight_init=1, resolution=-10000):  
        super().__init__()   
        self.add_module('c', torch.nn.Conv2d(     
            a, b, ks, stride, pad, dilation, groups, bias=False))    
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)  

    @torch.no_grad()
    def fuse(self):     
        c, bn = self._modules.values()    
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]     
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(     
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,     
                            groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m 

# =====================================================
class mamba_init:     
    @staticmethod 
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)  

        # Initialize special dt projection to preserve variance at initialization    
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":    
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random": 
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)   
        else:     
            raise NotImplementedError
  
        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))   
            + math.log(dt_min)     
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt) 
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit    
        # dt_proj.bias._no_reinit = True
     
        return dt_proj
 
    @staticmethod    
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization 
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()    
        A_log = torch.log(A)  # Keep A_log in fp32   
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous() 
            if merge:     
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):     
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)  
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True  
        return D    

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        # dt proj ============================
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)    
            for _ in range(k_group)  
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))  # (K, inner, rank)
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))  # (K, inner)
        del dt_projs   
   
        # A, D =======================================
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)   
        Ds = cls.D_init(d_inner, copies=k_group, merge=True)  # (K * D)
        return A_logs, Ds, dt_projs_weight, dt_projs_bias


class SS2Dv2: 
    def __initv2__(
            self,
            # basic dims ===========   
            d_model=96,
            d_state=16,     
            ssm_ratio=2.0,
            dt_rank="auto",    
            act_layer=nn.SiLU,     
            # dwconv ===============     
            d_conv=3,  # < 2 means no conv    
            conv_bias=True,    
            # ======================
            dropout=0.0,    
            bias=False,     
            # dt init ==============  
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",  
            dt_scale=1.0,  
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v05",
            channel_first=False, 
            # ======================
            k_group=4,  
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)   
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank    
        self.channel_first = channel_first     
        self.with_dconv = d_conv > 1   
        Linear = Linear2d if channel_first else nn.Linear   
        self.forward = self.forwardv2

        # tags for forward_type ==============================    
        checkpostfix = self.checkpostfix  
        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)    
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        self.out_norm, forward_type = self.get_outnorm(forward_type, d_inner, channel_first)
    
        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba", 
                        scan_force_torch=True),
            v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba"),     
            v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="oflex"),     
            v04=partial(self.forward_corev2, force_fp32=False),  # selective_scan_backend="oflex", scan_mode="cross2d"  
            v05=partial(self.forward_corev2, force_fp32=False, no_einsum=True),
            # selective_scan_backend="oflex", scan_mode="cross2d" 
            # ===============================     
            v051d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="unidi"),
            v052d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="bidi"),
            v052dc=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="cascade2d"),  
            # ===============================  
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="core"),
            v3=partial(self.forward_corev2, force_fp32=False, selective_scan_backend="oflex"),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)     
        self.k_group = k_group

        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2) 
        self.in_proj = Conv2d_BN(d_model, d_proj) 
        # self.in_proj = Linear(d_model, d_proj, bias=bias)     
        self.act: nn.Module = act_layer()    

        # conv =======================================
        if self.with_dconv:     
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,   
                groups=d_inner,    
                bias=conv_bias, 
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,     
                **factory_kwargs,     
            )     
     
        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            # torch.nn.Conv2d(d_inner, (dt_rank + d_state * 2), 1, bias=False)   
            for _ in range(k_group)    
        ] 
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)  
        del self.x_proj

        # out proj ======================================= 
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Conv2d_BN(d_inner, d_model)   
        # self.out_proj = Linear(d_inner, d_model, bias=bias)    
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(   
                d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=k_group,   
            )
        elif initialize in ["v1"]:    
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner))) 
            self.A_logs = nn.Parameter(
                torch.randn((k_group * d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.randn((k_group, d_inner, dt_rank)))  # 0.1 is added in 0430
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((k_group, d_inner)))  # 0.1 is added in 0430    
        elif initialize in ["v2"]:    
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))    
            self.A_logs = nn.Parameter(   
                torch.zeros((k_group * d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1    
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((k_group, d_inner, dt_rank)))     
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, d_inner)))     
   
    def forward_corev2(     
            self,   
            x: torch.Tensor = None,
            # ==============================
            force_fp32=False,  # True: input fp32   
            # ==============================
            ssoflex=True,  # True: input 16 or 32 output 32 False: output dtype as input
            no_einsum=False,  # replace einsum with linear or conv1d to raise throughput
            # ==============================
            selective_scan_backend=None,
            # ==============================   
            scan_mode="cross2d",
            scan_force_torch=False,  
            # ============================== 
            **kwargs,  
    ):     
        x_dtype = x.dtype
        assert scan_mode in ["unidi", "bidi", "cross2d", "cascade2d"]
        assert selective_scan_backend in [None, "oflex", "core", "mamba", "torch"] 
        delta_softplus = True   
        out_norm = self.out_norm    
        channel_first = self.channel_first   
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args) 
   
        B, D, H, W = x.shape     
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape 
        L = H * W    
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=3)[scan_mode]    
  
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            # print(u.device)
            # print(selective_scan_backend)
            if u.device == torch.device("cpu"):  
                selective_scan_backend = "torch" 
            else:    
                selective_scan_backend = "oflex"  
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex,     
                                     backend=selective_scan_backend)     
    
        if _scan_mode == 3:   
            x_proj_bias = getattr(self, "x_proj_bias", None)     
  
            def scan_rowcol(
                    x: torch.Tensor,  
                    proj_weight: torch.Tensor,
                    proj_bias: torch.Tensor,
                    dt_weight: torch.Tensor,
                    dt_bias: torch.Tensor,  # (2*c)    
                    _As: torch.Tensor,  # As = -torch.exp(A_logs.to(torch.float))[:2,] # (2*c, d_state)
                    _Ds: torch.Tensor,     
                    width=True,   
            ):  
                # x: (B, D, H, W)
                # proj_weight: (2 * D, (R+N+N))  
                XB, XD, XH, XW = x.shape   
                if width:    
                    _B, _D, _L = XB * XH, XD, XW
                    xs = x.permute(0, 2, 1, 3).contiguous()
                else:
                    _B, _D, _L = XB * XW, XD, XH
                    xs = x.permute(0, 3, 1, 2).contiguous() 
                xs = torch.stack([xs, xs.flip(dims=[-1])], dim=2)  # (B, H, 2, D, W)   
                if no_einsum:
                    x_dbl = F.conv1d(xs.view(_B, -1, _L), proj_weight.view(-1, _D, 1),  
                                     bias=(proj_bias.view(-1) if proj_bias is not None else None), groups=2)   
                    dts, Bs, Cs = torch.split(x_dbl.view(_B, 2, -1, _L), [R, N, N], dim=2)
                    dts = F.conv1d(dts.contiguous().view(_B, -1, _L), dt_weight.view(2 * _D, -1, 1), groups=2)  
                else:
                    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, proj_weight)  
                    if x_proj_bias is not None:
                        x_dbl = x_dbl + x_proj_bias.view(1, 2, -1, 1)  
                    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)     
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_weight)   
   
                xs = xs.view(_B, -1, _L)    
                dts = dts.contiguous().view(_B, -1, _L)
                As = _As.view(-1, N).to(torch.float)
                Bs = Bs.contiguous().view(_B, 2, N, _L) 
                Cs = Cs.contiguous().view(_B, 2, N, _L)
                Ds = _Ds.view(-1)
                delta_bias = dt_bias.view(-1).to(torch.float)   

                if force_fp32:
                    xs = xs.to(torch.float)
                dts = dts.to(xs.dtype) 
                Bs = Bs.to(xs.dtype)   
                Cs = Cs.to(xs.dtype)
 
                ys: torch.Tensor = selective_scan(
                    xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus  
                ).view(_B, 2, -1, _L)   
                return ys
    
            As = -self.A_logs.to(torch.float).exp().view(self.k_group, -1, N).contiguous()
            x = F.layer_norm(x.permute(0, 2, 3, 1), normalized_shape=(int(x.shape[1]),)).permute(0, 3, 1,  
                                                                                                 2).contiguous()  # added0510 to avoid nan
            y_row = scan_rowcol(
                x,
                proj_weight=self.x_proj_weight.view(self.k_group, -1, D)[:2].contiguous(),     
                proj_bias=(x_proj_bias.view(self.k_group, -1)[:2].contiguous() if x_proj_bias is not None else None), 
                dt_weight=self.dt_projs_weight.view(self.k_group, D, -1)[:2].contiguous(),   
                dt_bias=(self.dt_projs_bias.view(self.k_group, -1)[
                         :2].contiguous() if self.dt_projs_bias is not None else None),    
                _As=As[:2].contiguous().view(-1, N),
                _Ds=self.Ds.view(self.k_group, -1)[:2].contiguous().view(-1),    
                width=True,
            ).view(B, H, 2, -1, W).sum(dim=2).permute(0, 2, 1, 3).contiguous() # (B,C,H,W) 
            y_row = F.layer_norm(y_row.permute(0, 2, 3, 1), normalized_shape=(int(y_row.shape[1]),)).permute(0, 3, 1,
                                                                                                             2).contiguous()  # added0510 to avoid nan   
            y_col = scan_rowcol(
                y_row,    
                proj_weight=self.x_proj_weight.view(self.k_group, -1, D)[2:].contiguous().to(y_row.dtype),
                proj_bias=(
                    x_proj_bias.view(self.k_group, -1)[2:].contiguous().to(
                        y_row.dtype) if x_proj_bias is not None else None),     
                dt_weight=self.dt_projs_weight.view(self.k_group, D, -1)[2:].contiguous().to(y_row.dtype),
                dt_bias=(self.dt_projs_bias.view(self.k_group, -1)[2:].contiguous().to(
                    y_row.dtype) if self.dt_projs_bias is not None else None),
                _As=As[2:].contiguous().view(-1, N),
                _Ds=self.Ds.view(self.k_group, -1)[2:].contiguous().view(-1), 
                width=False,
            ).view(B, W, 2, -1, H).sum(dim=2).permute(0, 2, 3, 1).contiguous()
            y = y_col
        else: 
            x_proj_bias = getattr(self, "x_proj_bias", None)
            if self.k_group == 4:  
                xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                                   force_torch=scan_force_torch)   
            else:
                xs = cross_scan_fn_k2(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                                      force_torch=scan_force_torch)     
            if no_einsum:
                x_dbl = F.conv1d(xs.view(B, -1, L).contiguous(), self.x_proj_weight.view(-1, D, 1).contiguous(),
                                 bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
                dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L).contiguous(), [R, N, N], dim=2)   
                dts = F.conv1d(dts.contiguous().view(B, -1, L).contiguous(), self.dt_projs_weight.view(K * D, -1, 1).contiguous(), groups=K) 
            else:
                x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
                if x_proj_bias is not None:    
                    x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1).contiguous()
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)     
                dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)   

            xs = xs.view(B, -1, L).contiguous()    
            dts = dts.contiguous().view(B, -1, L).contiguous() 
            As = -self.A_logs.to(torch.float).exp()  # (k * c, d_state)  
            Ds = self.Ds.to(torch.float)  # (K * c)  
            Bs = Bs.contiguous().view(B, K, N, L).contiguous()    
            Cs = Cs.contiguous().view(B, K, N, L).contiguous()   
            delta_bias = self.dt_projs_bias.view(-1).contiguous().to(torch.float)     
     
            if force_fp32:  
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)     

            ys: torch.Tensor = selective_scan(   
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            ).view(B, K, -1, H, W).contiguous()  

            if self.k_group == 4:     
                y: torch.Tensor = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode,    
                                                 force_torch=scan_force_torch)  
            else:
                y: torch.Tensor = cross_merge_fn_k2(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode,    
                                                    force_torch=scan_force_torch)   
     
            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(  
                    A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y, H=H, W=W,
                ))

        y = y.view(B, -1, H, W).contiguous()
        if not channel_first:
            y = y.view(B, -1, H * W).contiguous().transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1).contiguous()  # (B, L, C)
        y = out_norm(y.to(x_dtype))   
 
        return y.to(x.dtype)
  
    def forwardv2(self, x: torch.Tensor, **kwargs):   
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, h, w, d)   
        z = self.act(z.contiguous().clone())
        x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y) 
        y = y * z 
        out = self.dropout(self.out_proj(y))
        return out
  
    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):     
        def checkpostfix(tag, value):     
            ret = value[-len(tag):] == tag  
            if ret:   
                value = value[:-len(tag)]   
            return ret, value  
  
        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm   

        out_norm_none, forward_type = checkpostfix("_onnone", forward_type) 
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)   
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type) 
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)
   
        out_norm = nn.Identity()
        if out_norm_none:
            out_norm = nn.Identity()     
        elif out_norm_cnorm:
            out_norm = nn.Sequential(
                LayerNorm(d_inner),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False), 
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),  
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),    
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),     
            )
        elif out_norm_softmax:
            out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:    
            out_norm = nn.Sigmoid()   
        else:    
            out_norm = LayerNorm(d_inner)     
     
        return out_norm, forward_type
   
    @staticmethod
    def checkpostfix(tag, value): 
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value

   
# mamba2 support ================================   
class SS2Dm0:  
    def __initm0__( 
            self, 
            # basic dims ===========   
            d_model=96,    
            d_state=16,  # now with mamba2, dstate should be bigger...  
            ssm_ratio=2.0,    
            dt_rank="auto",
            act_layer=nn.GELU,
            # dwconv =============== 
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,     
            bias=False,
            # dt init ==============  
            dt_min=0.001,
            dt_max=0.1, 
            dt_init="random", 
            dt_scale=1.0,    
            dt_init_floor=1e-4,
            initialize="v2",
            # ======================  
            forward_type="m0",     
            # ======================
            with_initial_state=False,     
            channel_first=False,  
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)    
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank  
        assert d_inner % dt_rank == 0  
        self.channel_first = channel_first     
        Linear = Linear2d if channel_first else nn.Linear     
        self.with_dconv = d_conv > 1
        self.forward = self.forwardm0

        # tags for forward_type ==============================   
        checkpostfix = SS2Dv2.checkpostfix
        self.disable_force32, forward_type = checkpostfix("_no32", forward_type) 
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type) 
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        self.out_norm, forward_type = SS2Dv2.get_outnorm(forward_type, d_inner, False)    
 
        # forward_type debug =======================================     
        FORWARD_TYPES = dict(
            m0=partial(self.forward_corem0, force_fp32=False, dstate=d_state),
        )     
        self.forward_core = FORWARD_TYPES.get(forward_type, None) 
        k_group = 4    

        # in proj =======================================  
        d_proj = d_inner if self.disable_z else (d_inner * 2) 
        # self.in_proj = Linear(d_model, d_proj, bias=bias)     
        self.in_proj = Conv2d_BN(d_model, d_proj)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(  
                in_channels=d_inner,
                out_channels=d_inner,    
                groups=d_inner,   
                bias=conv_bias, 
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,   
                **factory_kwargs,
            )
   
        # x proj ============================ 
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)    
        del self.x_proj

        # out proj =======================================  
        self.out_act = nn.GELU() if self.oact else nn.Identity()   
        # self.out_proj = Linear(d_inner, d_model, bias=bias)    
        self.out_proj = Conv2d_BN(d_inner, d_model) 
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v1"]:  
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group, dt_rank, int(d_inner // dt_rank))))    
            self.A_logs = nn.Parameter(torch.randn((k_group, dt_rank)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1  
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((k_group, dt_rank)))  # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds   
            self.Ds = nn.Parameter(torch.ones((k_group, dt_rank, int(d_inner // dt_rank)))) 
            self.A_logs = nn.Parameter(torch.zeros((k_group, dt_rank)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1  
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, dt_rank)))

        # init state ============================
        self.initial_state = None   
        if with_initial_state:
            self.initial_state = nn.Parameter(torch.zeros((1, k_group * dt_rank, int(d_inner // dt_rank), d_state)),
                                              requires_grad=False)     
 
    def forward_corem0(    
            self,
            x: torch.Tensor = None,   
            # ==============================
            force_fp32=False,  # True: input fp32
            chunk_size=64, 
            dstate=64,
            # ==============================    
            selective_scan_backend='torch',
            scan_mode="cross2d",
            scan_force_torch=False,
            # ==============================   
            **kwargs,
    ):
        x_dtype = x.dtype   
        assert scan_mode in ["unidi", "bidi", "cross2d"]    
        assert selective_scan_backend in [None, "triton", "torch"]
        x_proj_bias = getattr(self, "x_proj_bias", None)     
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)    

        N = dstate     
        B, H, W, RD = x.shape 
        K, R = self.A_logs.shape    
        K, R, D = self.Ds.shape  
        assert RD == R * D    
        L = H * W  
        KR = K * R
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=3)[scan_mode] 
  
        initial_state = None
        if self.initial_state is not None:   
            assert self.initial_state.shape[-1] == dstate
            initial_state = self.initial_state.detach().repeat(B, 1, 1, 1)    
        xs = cross_scan_fn(x.view(B, H, W, RD).contiguous(), in_channel_first=False, out_channel_first=False,
                           scans=_scan_mode, force_torch=scan_force_torch)  # (B, H, W, 4, D)
        x_dbl = torch.einsum("b l k d, k c d -> b l k c", xs, self.x_proj_weight)
        if x_proj_bias is not None:  
            x_dbl = x_dbl + x_proj_bias.view(1, -1, K, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=3)
        xs = xs.contiguous().view(B, L, KR, D).contiguous()
        dts = dts.contiguous().view(B, L, KR).contiguous()     
        Bs = Bs.contiguous().view(B, L, K, N).contiguous()  
        Cs = Cs.contiguous().view(B, L, K, N).contiguous()
        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)   
    
        As = -self.A_logs.to(torch.float).exp().view(KR).contiguous()
        Ds = self.Ds.to(torch.float).view(KR, D).contiguous()
        dt_bias = self.dt_projs_bias.view(KR).contiguous()     

        if force_fp32:   
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        ys, final_state = selective_scan_chunk_fn(
            xs, dts, As, Bs, Cs, chunk_size=chunk_size, D=Ds, dt_bias=dt_bias,
            initial_states=initial_state, dt_softplus=True, return_final_states=True,
            backend=selective_scan_backend,    
        )
        y: torch.Tensor = cross_merge_fn(ys.contiguous().view(B, H, W, K, RD).contiguous(), in_channel_first=False, 
                                         out_channel_first=False, scans=_scan_mode, force_torch=scan_force_torch)
    
        if getattr(self, "__DEBUG__", False):    
            setattr(self, "__data__", dict(   
                A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=self.Ds,
                us=xs, dts=dts, delta_bias=self.dt_projs_bias,
                initial_state=self.initial_state, final_satte=final_state,
                ys=ys, y=y, H=H, W=W, 
            ))  
        if self.initial_state is not None:
            self.initial_state = nn.Parameter(final_state.detach().sum(0, keepdim=True), requires_grad=False)
    
        y = self.out_norm(y.view(B, H, W, -1).contiguous().to(x_dtype))
  
        return y.to(x.dtype)   
    
    def forwardm0(self, x: torch.Tensor, **kwargs):   
        x = self.in_proj(x) 
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, h, w, d)
            if not self.disable_z_act: 
                z = self.act(z.contiguous())     
        if self.with_dconv:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)

        y = self.forward_core(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)     
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z   
        out = self.dropout(self.out_proj(y))
        return out

   
class SS2D(nn.Module, SS2Dv2, SS2Dm0):     
    def __init__(   
            self,
            # basic dims ===========    
            d_model=96,  
            d_state=16,     
            ssm_ratio=2.0,  
            dt_rank="auto",
            act_layer=nn.SiLU,  
            # dwconv ===============    
            d_conv=3,  # < 2 means no conv 
            conv_bias=True,
            # ======================
            dropout=0.0,   
            bias=False,     
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",  
            # ====================== 
            forward_type="v052d",
            channel_first=False, 
            # ======================    
            k_group=4,
            **kwargs,     
    ):     
        super().__init__()
        kwargs.update(     
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,     
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,    
            initialize=initialize, forward_type=forward_type, channel_first=channel_first, k_group=k_group,     
        )
        if forward_type in ["v0", "v0seq"]:
            self.__initv0__(seq=("seq" in forward_type), **kwargs)   
        elif forward_type.startswith("xv"):
            self.__initxv__(**kwargs)
        elif forward_type.startswith("m"):
            self.__initm0__(**kwargs)
        else: 
            self.__initv2__(**kwargs)

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1), 
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),     
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0) 
     
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])   
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),     
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)   
  
    return dec_filters, rec_filters 

def wavelet_transform(x, filters):
    b, c, h, w = x.shape   
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)   
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x
   
 
def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)    
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)  
    return x    
     
class _ScaleModule(nn.Module):    
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims  
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale) 
        self.bias = None  
   
    def forward(self, x):
        return torch.mul(self.weight, x)  

class MBWTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1',ssm_ratio=1,forward_type="v05",): 
        super(MBWTConv2d, self).__init__()    
  
        assert in_channels == out_channels

        self.in_channels = in_channels     
        self.wt_levels = wt_levels
        self.stride = stride  
        self.dilation = 1
    
        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)     
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False) 
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)     
 
        self.global_atten =SS2D(d_model=in_channels, d_state=1,    
             ssm_ratio=ssm_ratio, initialize="v2", forward_type=forward_type, channel_first=True, k_group=2)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,  
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]  
        )   

        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]   
        )    

        if self.stride > 1: 
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,    
                                                   groups=in_channels)
        else:     
            self.do_stride = None    
 
    def forward(self, x):
    
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []
     
        curr_x_ll = x   

        for i in range(self.wt_levels):    
            curr_shape = curr_x_ll.shape   
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)  
                curr_x_ll = F.pad(curr_x_ll, curr_pads)   
   
            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]   

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])  
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)     
     
            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])   

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()    
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop() 
 
            curr_x_ll = curr_x_ll + next_x_ll    
    
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)   

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]] 
 
        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.global_atten(x))
        x = x + x_tag
     
        if self.do_stride is not None:
            x = self.do_stride(x)

        return x    
    
def nearest_multiple_of_16(n):   
    if n % 16 == 0:    
        return n
    else:   
        lower_multiple = (n // 16) * 16
        upper_multiple = lower_multiple + 16
    
        if (n - lower_multiple) < (upper_multiple - n):    
            return lower_multiple
        else:   
            return upper_multiple

class DWConv2d_BN_ReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, bn_weight_init=1):
        super().__init__() 
        self.add_module('dwconv3x3',     
                        nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=in_channels,    
                                  bias=False))   
        self.add_module('bn1', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True)) 
        self.add_module('dwconv1x1',
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=in_channels,
                                  bias=False))    
        self.add_module('bn2', nn.BatchNorm2d(out_channels))
 
        # Initialize batch norm weights 
        nn.init.constant_(self.bn1.weight, bn_weight_init)
        nn.init.constant_(self.bn1.bias, 0)    
        nn.init.constant_(self.bn2.weight, bn_weight_init) 
        nn.init.constant_(self.bn2.bias, 0)

    @torch.no_grad()
    def fuse(self):
        # Fuse dwconv3x3 and bn1     
        dwconv3x3, bn1, relu, dwconv1x1, bn2 = self._modules.values()

        w1 = bn1.weight / (bn1.running_var + bn1.eps) ** 0.5     
        w1 = dwconv3x3.weight * w1[:, None, None, None]
        b1 = bn1.bias - bn1.running_mean * bn1.weight / (bn1.running_var + bn1.eps) ** 0.5    

        fused_dwconv3x3 = nn.Conv2d(w1.size(1) * dwconv3x3.groups, w1.size(0), w1.shape[2:], stride=dwconv3x3.stride,    
                                    padding=dwconv3x3.padding, dilation=dwconv3x3.dilation, groups=dwconv3x3.groups,
                                    device=dwconv3x3.weight.device)   
        fused_dwconv3x3.weight.data.copy_(w1)   
        fused_dwconv3x3.bias.data.copy_(b1)

        # Fuse dwconv1x1 and bn2     
        w2 = bn2.weight / (bn2.running_var + bn2.eps) ** 0.5
        w2 = dwconv1x1.weight * w2[:, None, None, None] 
        b2 = bn2.bias - bn2.running_mean * bn2.weight / (bn2.running_var + bn2.eps) ** 0.5
    
        fused_dwconv1x1 = nn.Conv2d(w2.size(1) * dwconv1x1.groups, w2.size(0), w2.shape[2:], stride=dwconv1x1.stride,
                                    padding=dwconv1x1.padding, dilation=dwconv1x1.dilation, groups=dwconv1x1.groups, 
                                    device=dwconv1x1.weight.device)    
        fused_dwconv1x1.weight.data.copy_(w2)
        fused_dwconv1x1.bias.data.copy_(b2)
  
        # Create a new sequential model with fused layers     
        fused_model = nn.Sequential(fused_dwconv3x3, relu, fused_dwconv1x1)
        return fused_model
     
class MobileMambaModule(torch.nn.Module):     
    def __init__(self, dim, global_ratio=0.25, local_ratio=0.25,
                 kernels=3, ssm_ratio=1, forward_type="v052d",):
        super().__init__()    
        self.dim = dim  
        self.global_channels = nearest_multiple_of_16(int(global_ratio * dim))
        if self.global_channels + int(local_ratio * dim) > dim:
            self.local_channels = dim - self.global_channels
        else:
            self.local_channels = int(local_ratio * dim)  
        self.identity_channels = self.dim - self.global_channels - self.local_channels   
        if self.local_channels != 0:  
            self.local_op = DWConv2d_BN_ReLU(self.local_channels, self.local_channels, kernels)     
        else:    
            self.local_op = nn.Identity()
        if self.global_channels != 0:
            self.global_op = MBWTConv2d(self.global_channels, self.global_channels, kernels, wt_levels=1, ssm_ratio=ssm_ratio, forward_type=forward_type,)  
        else: 
            self.global_op = nn.Identity()    
  
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(   
            dim, dim, bn_weight_init=0,)) 
  
    def forward(self, x):  # x (B,C,H,W)   
        x1, x2, x3 = torch.split(x, [self.global_channels, self.local_channels, self.identity_channels], dim=1)
        x1 = self.global_op(x1) 
        x2 = self.local_op(x2)    
        x = self.proj(torch.cat([x1, x2, x3], dim=1))    
        return x    

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"   
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel, height, width = 1, 16, 20, 20  
    inputs = torch.randn((batch_size, channel, height, width)).to(device)    

    # 此模块需要编译,详细编译命令在 compile_module/selective_scan   
    # 对于Linux用户可以直接运行上述的make.sh文件
    # 对于Windows用户需要逐行执行make.sh文件里的内容
    module = MobileMambaModule(channel).to(device)    

    outputs = module(inputs)     
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
     
    print(ORANGE)    
    flops, macs, _ = calculate_flops(model=module, 
                                     input_shape=(batch_size, channel, height, width),    
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)  
    print(RESET)