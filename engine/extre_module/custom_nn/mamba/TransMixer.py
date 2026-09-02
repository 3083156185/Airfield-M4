''' 
本文件由BiliBili：魔傀面具整理 
engine/extre_module/module_images/CVPR2026-TransMixer.png
engine/extre_module/module_images/CVPR2026-TransMixer.md     
论文链接：https://arxiv.org/pdf/2603.01361  
'''
     
import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops   
    
import math
from functools import partial
from typing import Any
   
import torch 
import torch.nn as nn  
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath   
    
class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight[:, :, None, None], self.bias) 

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,     
        local_metadata,
        strict,     
        missing_keys,  
        unexpected_keys, 
        error_msgs,
    ):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)    
        return super()._load_from_state_dict(    
            state_dict,   
            prefix, 
            local_metadata,  
            strict,
            missing_keys,  
            unexpected_keys,
            error_msgs, 
        )
 
  
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
 

class Mlp(nn.Module):   
    def __init__(   
        self,     
        in_features,  
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        channels_first=False,     
    ):
        super().__init__()
        out_features = out_features or in_features   
        hidden_features = hidden_features or in_features
        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)   

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)     
        x = self.drop(x)    
        return x


class gMlp(nn.Module):
    def __init__(
        self,
        in_features,  
        hidden_features=None, 
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        channels_first=False,     
    ):     
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)    
        self.act = act_layer()     
        self.fc2 = Linear(hidden_features, out_features) 
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor): 
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))    
        x = self.drop(x)
        return x    
     

class SoftmaxSpatial(nn.Softmax):    
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            b, c, h, w = x.shape     
            return super().forward(x.view(b, c, -1)).view(b, c, h, w)  
        if self.dim == 1: 
            b, h, w, c = x.shape  
            return super().forward(x.view(b, -1, c)).view(b, h, w, c) 
        raise NotImplementedError


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,  
        qk_norm=False, 
        attn_drop=0.0,   
        proj_drop=0.0, 
        norm_layer=nn.GroupNorm,
    ):  
        super().__init__()   
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads   
        self.scale = self.head_dim**-0.5
        self.fused_attn = True     
 
        self.in_norm = nn.GroupNorm(dim // 8, dim)  
 
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, l, c = x.shape
        x = self.in_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        qkv = self.qkv(x).reshape(b, l, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:    
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)   
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)    
            attn = attn.softmax(dim=-1) 
            attn = self.attn_drop(attn)   
            x = attn @ v
  
        x = x.transpose(1, 2).reshape(b, l, c)
        x = self.proj(x)  
        x = self.proj_drop(x)   
        return x     
    
     
class LocalModule(nn.Module):
    def __init__(self, dim): 
        super().__init__()
        self.in_norm = nn.GroupNorm(dim // 8, dim)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(dim, 1, 1)
        self.sigmoid = nn.Sigmoid()  
        self.conv2 = nn.Conv2d(dim, dim, 1)
 
    def forward(self, x, h, w):    
        b, _, c = x.shape   
        x = x.view(b, h, w, c).permute(0, 3, 1, 2)
        x = self.in_norm(x)
        x1 = self.sigmoid(self.conv1(self.maxpool(x)))
        x = x1 * x  
        x = self.conv2(x).permute(0, 2, 3, 1).view(b, -1, c)     
        return x
     

def get_global_local_index(delta, delta_bias, ratio=0.5):  
    if delta_bias is not None:
        delta = delta + delta_bias[..., None]     
    delta = torch.nn.functional.softplus(delta) 

    _, d, _ = delta.shape
    global_channel_num = int(d * ratio)
    _, sorted_indices = torch.sort(delta, descending=True, dim=1)
    global_indices, local_indices = torch.split(sorted_indices, [global_channel_num, d - global_channel_num], dim=1)
  
    global_indices = global_indices.permute(0, 2, 1)
    local_indices = local_indices.permute(0, 2, 1)
    return global_indices, local_indices     
  

# ==============================================     
# VMamba csm_triton.py (torch path)

   
def cross_scan_fwd(x: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):    
    if in_channel_first:
        b, c, h, w = x.shape    
        if scans == 0:     
            y = x.new_empty((b, 4, c, h * w))    
            y[:, 0, :, :] = x.flatten(2, 3)
            y[:, 1, :, :] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
            y[:, 2:4, :, :] = torch.flip(y[:, 0:2, :, :], dims=[-1])     
        elif scans == 1:
            y = x.view(b, 1, c, h * w).repeat(1, 4, 1, 1) 
        elif scans == 2:    
            y = x.view(b, 1, c, h * w).repeat(1, 2, 1, 1)
            y = torch.cat([y, y.flip(dims=[-1])], dim=1)
        else:  
            raise ValueError(f"unsupported scans={scans}")
    else:
        b, h, w, c = x.shape
        if scans == 0:    
            y = x.new_empty((b, h * w, 4, c))
            y[:, :, 0, :] = x.flatten(1, 2)   
            y[:, :, 1, :] = x.transpose(dim0=1, dim1=2).flatten(1, 2)     
            y[:, :, 2:4, :] = torch.flip(y[:, :, 0:2, :], dims=[1])  
        elif scans == 1:  
            y = x.view(b, h * w, 1, c).repeat(1, 1, 4, 1)
        elif scans == 2:    
            y = x.view(b, h * w, 1, c).repeat(1, 1, 2, 1) 
            y = torch.cat([y, y.flip(dims=[1])], dim=2)
        else:    
            raise ValueError(f"unsupported scans={scans}")
    
    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()     
    elif (not in_channel_first) and out_channel_first:     
        y = y.permute(0, 2, 3, 1).contiguous()
    return y    
  

def cross_merge_fwd(y: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):    
    if out_channel_first:
        b, k, d, h, w = y.shape
        y = y.view(b, k, d, -1)   
        if scans == 0:
            y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(b, 2, d, -1)
            y = y[:, 0] + y[:, 1].view(b, -1, w, h).transpose(dim0=2, dim1=3).contiguous().view(b, d, -1)
        elif scans == 1:
            y = y.sum(1)
        elif scans == 2: 
            y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(b, 2, d, -1)
            y = y.sum(1)    
        else:    
            raise ValueError(f"unsupported scans={scans}")
    else:
        b, h, w, k, d = y.shape
        y = y.view(b, -1, k, d)
        if scans == 0:
            y = y[:, :, 0:2] + y[:, :, 2:4].flip(dims=[1]).view(b, -1, 2, d)    
            y = y[:, :, 0] + y[:, :, 1].view(b, w, h, -1).transpose(dim0=1, dim1=2).contiguous().view(b, -1, d)   
        elif scans == 1:
            y = y.sum(2)
        elif scans == 2:
            y = y[:, :, 0:2] + y[:, :, 2:4].flip(dims=[1]).view(b, -1, 2, d)   
            y = y.sum(2)
        else: 
            raise ValueError(f"unsupported scans={scans}")
  
    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 2, 1).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 1).contiguous()
    return y


def cross_scan1b1_fwd(x: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):     
    if in_channel_first:
        b, _, c, h, w = x.shape
        if scans == 0:
            y = torch.stack(
                [
                    x[:, 0].flatten(2, 3),   
                    x[:, 1].transpose(dim0=2, dim1=3).flatten(2, 3),    
                    torch.flip(x[:, 2].flatten(2, 3), dims=[-1]),
                    torch.flip(x[:, 3].transpose(dim0=2, dim1=3).flatten(2, 3), dims=[-1]),
                ],  
                dim=1, 
            )
        elif scans == 1:    
            y = x.flatten(2, 3)
        elif scans == 2:
            y = torch.stack(   
                [    
                    x[:, 0].flatten(2, 3),   
                    x[:, 1].flatten(2, 3),
                    torch.flip(x[:, 2].flatten(2, 3), dims=[-1]), 
                    torch.flip(x[:, 3].flatten(2, 3), dims=[-1]),   
                ],    
                dim=1,
            )
        else:     
            raise ValueError(f"unsupported scans={scans}")
    else:
        b, h, w, _, c = x.shape
        if scans == 0:
            y = torch.stack( 
                [
                    x[:, :, :, 0].flatten(1, 2),     
                    x[:, :, :, 1].transpose(dim0=1, dim1=2).flatten(1, 2),
                    torch.flip(x[:, :, :, 2].flatten(1, 2), dims=[1]),  
                    torch.flip(x[:, :, :, 3].transpose(dim0=1, dim1=2).flatten(1, 2), dims=[1]),
                ],
                dim=2,   
            )   
        elif scans == 1:
            y = x.flatten(1, 2)
        elif scans == 2:  
            y = torch.stack(     
                [
                    x[:, 0].flatten(1, 2),
                    x[:, 1].flatten(1, 2), 
                    torch.flip(x[:, 2].flatten(1, 2), dims=[-1]),
                    torch.flip(x[:, 3].flatten(1, 2), dims=[-1]),   
                ],   
                dim=2,
            )
        else:    
            raise ValueError(f"unsupported scans={scans}")    
     
    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not in_channel_first) and out_channel_first: 
        y = y.permute(0, 2, 3, 1).contiguous() 
    return y


def cross_merge1b1_fwd(y: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0): 
    if out_channel_first:
        b, k, d, h, w = y.shape  
        y = y.view(b, k, d, -1)    
        if scans == 0:
            y = torch.stack(     
                [
                    y[:, 0],
                    y[:, 1].view(b, -1, w, h).transpose(dim0=2, dim1=3).flatten(2, 3),
                    torch.flip(y[:, 2], dims=[-1]),
                    torch.flip(y[:, 3].view(b, -1, w, h).transpose(dim0=2, dim1=3).flatten(2, 3), dims=[-1]),  
                ],
                dim=1,
            )   
        elif scans == 1:
            y = y    
        elif scans == 2:
            y = torch.stack( 
                [ 
                    y[:, 0],
                    y[:, 1],  
                    torch.flip(y[:, 2], dims=[-1]),    
                    torch.flip(y[:, 3], dims=[-1]),
                ],
                dim=1,  
            )
        else:
            raise ValueError(f"unsupported scans={scans}")  
    else:     
        b, h, w, k, d = y.shape  
        y = y.view(b, -1, k, d) 
        if scans == 0:    
            y = torch.stack(     
                [  
                    y[:, :, 0],  
                    y[:, :, 1].view(b, w, h, -1).transpose(dim0=1, dim1=2).flatten(1, 2),     
                    torch.flip(y[:, :, 2], dims=[1]),
                    torch.flip(y[:, :, 3].view(b, w, h, -1).transpose(dim0=1, dim1=2).flatten(1, 2), dims=[1]), 
                ],
                dim=2,  
            )     
        elif scans == 1:    
            y = y
        elif scans == 2:
            y = torch.stack(     
                [
                    y[:, :, 0],
                    y[:, :, 1],
                    torch.flip(y[:, :, 2], dims=[1]),    
                    torch.flip(y[:, :, 3], dims=[1]),
                ],    
                dim=2,    
            )    
        else:
            raise ValueError(f"unsupported scans={scans}")

    if out_channel_first and (not in_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not out_channel_first) and in_channel_first:    
        y = y.permute(0, 2, 3, 1).contiguous()
    return y


class CrossScanF(torch.autograd.Function): 
    @staticmethod  
    def forward(ctx, x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):   
        if one_by_one:   
            b, k, c, h, w = x.shape
            if not in_channel_first:     
                b, h, w, k, c = x.shape 
        else:
            b, c, h, w = x.shape   
            if not in_channel_first:   
                b, h, w, c = x.shape

        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans
        ctx.shape = (b, c, h, w)    
        _fn = cross_scan1b1_fwd if one_by_one else cross_scan_fwd     
        return _fn(x, in_channel_first, out_channel_first, scans)
     
    @staticmethod  
    def backward(ctx, ys: torch.Tensor):
        in_channel_first = ctx.in_channel_first 
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans     
        b, c, h, w = ctx.shape
  
        ys = ys.view(b, -1, c, h, w) if out_channel_first else ys.view(b, h, w, -1, c)
        _fn = cross_merge1b1_fwd if one_by_one else cross_merge_fwd     
        y = _fn(ys, in_channel_first, out_channel_first, scans)   
        if one_by_one:
            y = y.view(b, 4, -1, h, w) if in_channel_first else y.view(b, h, w, 4, -1)     
        else:   
            y = y.view(b, -1, h, w) if in_channel_first else y.view(b, h, w, -1)
        return y, None, None, None, None 


class CrossMergeF(torch.autograd.Function):
    @staticmethod    
    def forward(ctx, ys: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):  
        b, k, c, h, w = ys.shape  
        if not out_channel_first:
            b, h, w, k, c = ys.shape   

        ctx.in_channel_first = in_channel_first   
        ctx.out_channel_first = out_channel_first  
        ctx.one_by_one = one_by_one
        ctx.scans = scans
        ctx.shape = (b, c, h, w)
        _fn = cross_merge1b1_fwd if one_by_one else cross_merge_fwd
        return _fn(ys, in_channel_first, out_channel_first, scans)
     
    @staticmethod   
    def backward(ctx, x: torch.Tensor):
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one     
        scans = ctx.scans
        b, c, h, w = ctx.shape
     
        if not one_by_one:
            if in_channel_first:
                x = x.view(b, c, h, w)  
            else:    
                x = x.view(b, h, w, c)  
        else:   
            if in_channel_first:   
                x = x.view(b, 4, c, h, w)
            else:  
                x = x.view(b, h, w, 4, c)     

        _fn = cross_scan1b1_fwd if one_by_one else cross_scan_fwd
        x = _fn(x, in_channel_first, out_channel_first, scans) 
        x = x.view(b, 4, c, h, w) if out_channel_first else x.view(b, h, w, 4, c)
        return x, None, None, None, None
  
     
def cross_scan_fn(x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, force_torch=False):
    del force_torch
    return CrossScanF.apply(x, in_channel_first, out_channel_first, one_by_one, scans)
  
   
def cross_merge_fn(y: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, force_torch=False):  
    del force_torch
    return CrossMergeF.apply(y, in_channel_first, out_channel_first, one_by_one, scans)


# ==============================================    
# VMamba csms6s.py
     
WITH_SELECTIVESCAN_OFLEX = True 
WITH_SELECTIVESCAN_CORE = False 
WITH_SELECTIVESCAN_MAMBA = True

try:
    import selective_scan_cuda_oflex     
except ImportError:
    WITH_SELECTIVESCAN_OFLEX = False
    # warnings.warn("Can not import selective_scan_cuda_oflex. This affects speed.")  
   
try:
    import selective_scan_cuda_core   
except ImportError:    
    WITH_SELECTIVESCAN_CORE = False     

try:  
    import selective_scan_cuda   
except ImportError:     
    WITH_SELECTIVESCAN_MAMBA = False  

 
def selective_scan_torch(
    u: torch.Tensor,
    delta: torch.Tensor,  
    A: torch.Tensor, 
    B: torch.Tensor,
    C: torch.Tensor,   
    D: torch.Tensor = None,
    delta_bias: torch.Tensor = None,
    delta_softplus=True,
    oflex=True,    
    *args,     
    **kwargs,
):
    dtype_in = u.dtype    
    batch, k, n, l = B.shape
    kcdim = u.shape[1]
    cdim = int(kcdim / k) 
    assert u.shape == (batch, kcdim, l)   
    assert delta.shape == (batch, kcdim, l)
    assert A.shape == (kcdim, n)
    assert C.shape == B.shape    

    if delta_bias is not None:
        delta = delta + delta_bias[..., None]     
    if delta_softplus: 
        delta = torch.nn.functional.softplus(delta)

    u, delta, A, B, C = u.float(), delta.float(), A.float(), B.float(), C.float()  
    B = B.view(batch, k, 1, n, l).repeat(1, 1, cdim, 1, 1).view(batch, kcdim, n, l)  
    C = C.view(batch, k, 1, n, l).repeat(1, 1, cdim, 1, 1).view(batch, kcdim, n, l)
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    deltaB_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, B, u) 
   
    x = A.new_zeros((batch, kcdim, n))  
    ys = []
    for i in range(l):
        x = deltaA[:, :, i, :] * x + deltaB_u[:, :, i, :]
        y = torch.einsum("bdn,bdn->bd", x, C[:, :, :, i])
        ys.append(y)    
    y = torch.stack(ys, dim=2)

    out = y if D is None else y + u * D.unsqueeze(-1)
    return out if oflex else out.to(dtype=dtype_in)  

  
class SelectiveScanCuda(torch.autograd.Function): 
    @staticmethod    
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, oflex=True, backend=None):
        ctx.delta_softplus = delta_softplus
        backend = "oflex" if WITH_SELECTIVESCAN_OFLEX and (backend is None) else backend    
        backend = "core" if WITH_SELECTIVESCAN_CORE and (backend is None) else backend
        backend = "mamba" if WITH_SELECTIVESCAN_MAMBA and (backend is None) else backend    
        ctx.backend = backend     
    
        if backend == "oflex":     
            out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)  
        elif backend == "core":     
            out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)  
        elif backend == "mamba":
            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus) 
        else:   
            raise RuntimeError(f"Unknown backend {backend}")

        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
     
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):   
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors    
        backend = ctx.backend 
        if dout.stride(-1) != 1:  
            dout = dout.contiguous()   
        if backend == "oflex":   
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(    
                u,     
                delta,
                A,   
                B,    
                C,     
                D,
                delta_bias,    
                dout,
                x,    
                ctx.delta_softplus,
                1,     
            )
        elif backend == "core":  
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd( 
                u,
                delta,
                A,  
                B,
                C, 
                D,
                delta_bias,   
                dout,  
                x,
                ctx.delta_softplus,
                1,
            )
        elif backend == "mamba":     
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u,  
                delta,
                A,   
                B, 
                C, 
                D,
                None,
                delta_bias,  
                dout,    
                x,   
                None,
                None,
                ctx.delta_softplus,   
                False,
            )     
        else:
            raise RuntimeError(f"Unknown backend {backend}")    
        return du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None

     
def selective_scan_fn( 
    u: torch.Tensor,  
    delta: torch.Tensor,  
    A: torch.Tensor,  
    B: torch.Tensor,  
    C: torch.Tensor,
    D: torch.Tensor = None,    
    delta_bias: torch.Tensor = None,
    delta_softplus=True,   
    oflex=True,  
    backend=None,
):
    with_cuda = WITH_SELECTIVESCAN_OFLEX or WITH_SELECTIVESCAN_CORE or WITH_SELECTIVESCAN_MAMBA
    if not u.is_cuda:     
        with_cuda = False
    fn = selective_scan_torch if backend == "torch" or (not with_cuda) else SelectiveScanCuda.apply
    return fn(u, delta, A, B, C, D, delta_bias, delta_softplus, oflex, backend)     
   
   
# ==============================================
# VMamba vmamba.py (required subset)
   
 
class mamba_init:
    @staticmethod     
    def dt_init(
        dt_rank,
        d_inner,
        dt_scale=1.0,
        dt_init="random",     
        dt_min=0.001,   
        dt_max=0.1,
        dt_init_floor=1e-4,
    ):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)   
    
        dt_init_std = dt_rank**-0.5 * dt_scale 
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)     
        elif dt_init == "random":    
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)   
        else:   
            raise NotImplementedError

        dt = torch.exp(torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))   
        with torch.no_grad():  
            dt_proj.bias.copy_(inv_dt)
     
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):     
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)    
        if copies > 0:     
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:   
                A_log = A_log.flatten(0, 1)     
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True   
        return A_log
  
    @staticmethod    
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:     
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1) 
        D = nn.Parameter(D)   
        D._no_weight_decay = True
        return D  

    @classmethod
    def init_dt_A_D(
        cls,    
        d_state,
        dt_rank,
        d_inner,
        dt_scale, 
        dt_init,   
        dt_min, 
        dt_max,
        dt_init_floor,   
        k_group=4,   
    ):
        dt_projs = [   
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))    
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))
        del dt_projs   

        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True)  
        Ds = cls.D_init(d_inner, copies=k_group, merge=True)     
        return A_logs, Ds, dt_projs_weight, dt_projs_bias 


class SS2Dv2:
    def __initv2__( 
        self,
        d_model=96,   
        d_state=16,   
        ssm_ratio=2.0, 
        dt_rank="auto",
        act_layer=nn.SiLU,     
        d_conv=3,   
        conv_bias=True,   
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,   
        dt_init="random",   
        dt_scale=1.0,     
        dt_init_floor=1e-4,
        initialize="v0",   
        forward_type="v2",
        channel_first=False,
        compute_attn_matrix=False,     
        g_ratio=0.5,  
        **kwargs,    
    ):
        del kwargs     
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()  
        self.k_group = 4 
        self.d_model = int(d_model)     
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)   
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)  
        self.channel_first = channel_first  
        self.with_dconv = d_conv > 1 
        Linear = Linear2d if channel_first else nn.Linear     
        self.forward = self.forwardv2     
  
        checkpostfix = self.checkpostfix    
        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)   
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)    
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner, channel_first)
     
        FORWARD_TYPES = {   
            "v01": partial(     
                self.forward_corev2,  
                force_fp32=(not self.disable_force32),    
                selective_scan_backend="mamba",
                scan_force_torch=True,
            ),   
            "v02": partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba"),
            "v03": partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="oflex"),     
            "v04": partial(self.forward_corev2, force_fp32=False), 
            "v05": partial(self.forward_corev2, force_fp32=False, no_einsum=True),
            "v051d": partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="unidi"),  
            "v052d": partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="bidi"),
            "v052dc": partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="cascade2d"),
            "v052d3": partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode=3),  
            "v2": partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="oflex"),
            "v3": partial(self.forward_corev2, force_fp32=False, selective_scan_backend="oflex"),    
        } 
        self.forward_core = FORWARD_TYPES.get(forward_type, None)    
        if self.forward_core is None:
            raise NotImplementedError(f"Unsupported forward_type={forward_type}")  
    
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = Linear(self.d_model, d_proj, bias=bias)
        self.act = act_layer()

        if self.with_dconv:    
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,     
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        self.x_proj = [nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False) for _ in range(self.k_group)]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) 
        del self.x_proj
  
        self.mixer = True  
        if self.mixer:     
            self.compute_attn_matrix = compute_attn_matrix  
            self.g_ratio = g_ratio
            mix_dim = self.k_group * self.d_inner 
            global_dim = int(mix_dim * self.g_ratio)   
            local_dim = mix_dim - global_dim  
            self.global_module = Attention(dim=global_dim, num_heads=8)
            self.local_module = LocalModule(dim=local_dim)

        self.out_act = nn.GELU() if self.oact else nn.Identity()     
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias)   
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state,
                self.dt_rank,
                self.d_inner, 
                dt_scale,
                dt_init,  
                dt_min,
                dt_max, 
                dt_init_floor,
                k_group=self.k_group,
            ) 
        elif initialize in ["v1"]:   
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.k_group * self.d_inner, self.d_state)))
            self.dt_projs_weight = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner)))
        elif initialize in ["v2"]:
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner))) 
            self.A_logs = nn.Parameter(torch.zeros((self.k_group * self.d_inner, self.d_state)))
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))     
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))
        else:    
            raise NotImplementedError(f"Unsupported initialize={initialize}") 

    def forward_corev2(
        self,
        x: torch.Tensor = None,
        force_fp32=False,    
        ssoflex=True,  
        no_einsum=False,
        selective_scan_backend=None, 
        scan_mode="cross2d", 
        scan_force_torch=False,     
        **kwargs,
    ): 
        del kwargs    
        assert selective_scan_backend in [None, "oflex", "mamba", "torch"] 
        _scan_mode = (    
            {"cross2d": 0, "unidi": 1, "bidi": 2, "cascade2d": -1}.get(scan_mode, None)
            if isinstance(scan_mode, str)
            else scan_mode
        )  
        assert isinstance(_scan_mode, int)
        delta_softplus = True   
        out_norm = self.out_norm   
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
     
        b, d, h, w = x.shape    
        n = self.d_state     
        k, d, r = self.k_group, self.d_inner, self.dt_rank  
        l = h * w     
  
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, backend=selective_scan_backend)    
     
        if _scan_mode == -1:
            x_proj_bias = getattr(self, "x_proj_bias", None)

            def scan_rowcol(
                x: torch.Tensor,    
                proj_weight: torch.Tensor, 
                proj_bias: torch.Tensor,
                dt_weight: torch.Tensor, 
                dt_bias: torch.Tensor, 
                _As: torch.Tensor,     
                _Ds: torch.Tensor, 
                width=True,   
            ):
                xb, xd, xh, xw = x.shape
                if width:
                    _b, _d, _l = xb * xh, xd, xw  
                    xs = x.permute(0, 2, 1, 3).contiguous()    
                else:   
                    _b, _d, _l = xb * xw, xd, xh
                    xs = x.permute(0, 3, 1, 2).contiguous()
   
                xs = torch.stack([xs, xs.flip(dims=[-1])], dim=2)
                if no_einsum:
                    x_dbl = F.conv1d( 
                        xs.view(_b, -1, _l),    
                        proj_weight.view(-1, _d, 1), 
                        bias=(proj_bias.view(-1) if proj_bias is not None else None),     
                        groups=2,   
                    ) 
                    dts, Bs, Cs = torch.split(x_dbl.view(_b, 2, -1, _l), [r, n, n], dim=2)   
                    dts = F.conv1d(   
                        dts.contiguous().view(_b, -1, _l),
                        dt_weight.view(2 * _d, -1, 1),
                        groups=2, 
                    )    
                else:  
                    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, proj_weight)    
                    if x_proj_bias is not None:  
                        x_dbl = x_dbl + x_proj_bias.view(1, 2, -1, 1)
                    dts, Bs, Cs = torch.split(x_dbl, [r, n, n], dim=2)
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_weight)
   
                xs = xs.view(_b, -1, _l)  
                dts = dts.contiguous().view(_b, -1, _l)
                As = _As.view(-1, n).to(torch.float) 
                Bs = Bs.contiguous().view(_b, 2, n, _l)    
                Cs = Cs.contiguous().view(_b, 2, n, _l)     
                Ds = _Ds.view(-1)
                delta_bias = dt_bias.view(-1).to(torch.float) 
   
                if force_fp32:
                    xs = xs.to(torch.float)
                dts = dts.to(xs.dtype)    
                Bs = Bs.to(xs.dtype)
                Cs = Cs.to(xs.dtype)

                ys = selective_scan(xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus).view(_b, 2, -1, _l)  
                return ys
 
            As = -self.A_logs.to(torch.float).exp().view(4, -1, n)   
            x = F.layer_norm(x.permute(0, 2, 3, 1), normalized_shape=(int(x.shape[1]),)).permute(0, 3, 1, 2).contiguous()
 
            y_row = scan_rowcol(     
                x,
                proj_weight=self.x_proj_weight.view(4, -1, d)[:2].contiguous(),
                proj_bias=(x_proj_bias.view(4, -1)[:2].contiguous() if x_proj_bias is not None else None), 
                dt_weight=self.dt_projs_weight.view(4, d, -1)[:2].contiguous(),
                dt_bias=(self.dt_projs_bias.view(4, -1)[:2].contiguous() if self.dt_projs_bias is not None else None),
                _As=As[:2].contiguous().view(-1, n),
                _Ds=self.Ds.view(4, -1)[:2].contiguous().view(-1),
                width=True, 
            ).view(b, h, 2, -1, w).sum(dim=2).permute(0, 2, 1, 3)
 
            y_row = F.layer_norm(y_row.permute(0, 2, 3, 1), normalized_shape=(int(y_row.shape[1]),)).permute(0, 3, 1, 2).contiguous()    

            y_col = scan_rowcol(    
                y_row,
                proj_weight=self.x_proj_weight.view(4, -1, d)[2:].contiguous().to(y_row.dtype),
                proj_bias=(x_proj_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if x_proj_bias is not None else None),
                dt_weight=self.dt_projs_weight.view(4, d, -1)[2:].contiguous().to(y_row.dtype), 
                dt_bias=(self.dt_projs_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if self.dt_projs_bias is not None else None),     
                _As=As[2:].contiguous().view(-1, n),
                _Ds=self.Ds.view(4, -1)[2:].contiguous().view(-1),
                width=False,
            ).view(b, w, 2, -1, h).sum(dim=2).permute(0, 2, 3, 1)     
     
            y = y_col  
        else:
            x_proj_bias = getattr(self, "x_proj_bias", None)
            xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)    

            if no_einsum:
                x_dbl = F.conv1d(
                    xs.view(b, -1, l),   
                    self.x_proj_weight.view(-1, d, 1),    
                    bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None),     
                    groups=k,
                )   
                dts, Bs, Cs = torch.split(x_dbl.view(b, k, -1, l), [r, n, n], dim=2)
                if hasattr(self, "dt_projs_weight"):  
                    dts = F.conv1d(
                        dts.contiguous().view(b, -1, l),  
                        self.dt_projs_weight.view(k * d, -1, 1),    
                        groups=k,   
                    )
            else:     
                x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)  
                if x_proj_bias is not None:
                    x_dbl = x_dbl + x_proj_bias.view(1, k, -1, 1)    
                dts, Bs, Cs = torch.split(x_dbl, [r, n, n], dim=2)
                if hasattr(self, "dt_projs_weight"):
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

            xs = xs.view(b, -1, l)
            dts = dts.contiguous().view(b, -1, l)
            As = -self.A_logs.to(torch.float).exp()
            Ds = self.Ds.to(torch.float)    
            Bs = Bs.contiguous().view(b, k, n, l)
            Cs = Cs.contiguous().view(b, k, n, l)
            delta_bias = self.dt_projs_bias.view(-1).to(torch.float)    
    
            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs) 
   
            ys = selective_scan(xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus)

            if self.mixer:
                g_idx, l_idx = get_global_local_index(dts, delta_bias, self.g_ratio)
                batch_idx = torch.arange(b, device=ys.device)[:, None, None]
                token_idx = torch.arange(l, device=ys.device)[None, :, None]
     
                g_ys = ys[batch_idx, g_idx, token_idx]    
                g_ys = self.global_module(g_ys)
                ys[batch_idx, g_idx, token_idx] = g_ys
  
                l_ys = ys[batch_idx, l_idx, token_idx]
                l_ys = self.local_module(l_ys, h, w)     
                ys[batch_idx, l_idx, token_idx] = l_ys

            ys = ys.view(b, k, -1, h, w) 
            y = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)     

        y = y.view(b, -1, h, w)     
        if not channel_first:     
            y = y.view(b, -1, h * w).transpose(dim0=1, dim1=2).contiguous().view(b, h, w, -1)
   
        y = out_norm(y)   
        return y.to(x.dtype) 

    def forwardv2(self, x: torch.Tensor, **kwargs):
        del kwargs
        x = self.in_proj(x)    
        if not self.disable_z:     
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  
            if not self.disable_z_act:    
                z = self.act(z)
        if not self.channel_first:    
            x = x.permute(0, 3, 1, 2).contiguous()     
        if self.with_dconv:     
            x = self.conv2d(x)   
        x = self.act(x) 
        y = self.forward_core(x)     
        y = self.out_act(y)     
        if not self.disable_z:  
            y = y * z 
        out = self.dropout(self.out_proj(y))     
        return out  

    @staticmethod   
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):    
        def checkpostfix(tag, value):  
            ret = value[-len(tag) :] == tag
            if ret: 
                value = value[: -len(tag)]    
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
        ret = value[-len(tag) :] == tag
        if ret:  
            value = value[: -len(tag)] 
        return ret, value    
    
     
class SS2D(nn.Module, SS2Dv2):
    def __init__(
        self,    
        d_model=96,  
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,   
        d_conv=3,    
        conv_bias=True,  
        dropout=0.0,    
        bias=False, 
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,     
        dt_init_floor=1e-4,
        initialize="v0", 
        forward_type="v2",    
        channel_first=False,
        **kwargs,
    ):
        nn.Module.__init__(self)    
        kwargs.update(
            d_model=d_model,   
            d_state=d_state, 
            ssm_ratio=ssm_ratio,  
            dt_rank=dt_rank,
            act_layer=act_layer,
            d_conv=d_conv,    
            conv_bias=conv_bias, 
            dropout=dropout,     
            bias=bias,  
            dt_min=dt_min,
            dt_max=dt_max,     
            dt_init=dt_init,
            dt_scale=dt_scale, 
            dt_init_floor=dt_init_floor,   
            initialize=initialize,  
            forward_type=forward_type,   
            channel_first=channel_first, 
        )
        self.__initv2__(**kwargs)     


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, channel_first=None, in_channel_first=False, out_channel_first=False, **kwargs): 
        nn.LayerNorm.__init__(self, *args, **kwargs)
        if channel_first is not None: 
            in_channel_first = channel_first 
            out_channel_first = channel_first 
        self.in_channel_first = in_channel_first   
        self.out_channel_first = out_channel_first

    def forward(self, x: torch.Tensor):  
        if self.in_channel_first:
            x = x.permute(0, 2, 3, 1)   
        x = nn.LayerNorm.forward(self, x)
        if self.out_channel_first:   
            x = x.permute(0, 3, 1, 2)
        return x  
    
     
class TransMixer(nn.Module):   
    def __init__( 
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = LayerNorm,
        channel_first=False,
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True, 
        ssm_drop_rate: float = 0,     
        ssm_init="v0",
        forward_type="v2",   
        mlp_ratio=4.0, 
        mlp_act_layer=nn.GELU,   
        mlp_drop_rate: float = 0.0, 
        gmlp=False,
        use_checkpoint: bool = False,
        post_norm: bool = False,
        _SS2D: type = SS2D,    
        compute_attn_matrix_fn=False,  
        **kwargs,
    ):
        super().__init__()     
        del kwargs
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0    
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm 
    
        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim, channel_first=channel_first)    
            self.op = _SS2D(
                d_model=hidden_dim,  
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,   
                d_conv=ssm_conv,     
                conv_bias=ssm_conv_bias, 
                dropout=ssm_drop_rate,   
                initialize=ssm_init,   
                forward_type=forward_type,    
                channel_first=channel_first,    
                compute_attn_matrix_fn=compute_attn_matrix_fn, 
            )  

        self.drop_path = DropPath(drop_path)
   
        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = LayerNorm(hidden_dim, channel_first=channel_first)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(    
                in_features=hidden_dim,    
                hidden_features=mlp_hidden_dim,
                act_layer=mlp_act_layer,  
                drop=mlp_drop_rate,   
                channels_first=channel_first, 
            )

    def _forward(self, input: torch.Tensor):
        x = input 
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))    
            else:   
                x = x + self.drop_path(self.op(self.norm(x)))   
        if self.mlp_branch:
            if self.post_norm:   
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x     
  
    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)    
        return self._forward(input)

class TransMixerModule(nn.Module):
    def __init__(self, dims, mlp_ratio=4.0, state_dim=64):
        super().__init__() 
        self.transMixer = TransMixer(hidden_dim=dims, ssm_d_state=state_dim, mlp_ratio=mlp_ratio, channel_first=True)   

    def forward(self, x):  
        return self.transMixer(x)
  
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"  
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel, height, width = 1, 16, 20, 20
    inputs = torch.randn((batch_size, channel, height, width)).to(device)

    # 此模块需要编译,详细编译命令在 compile_module/selective_scan 
    # 对于Linux用户可以直接运行上述的make.sh文件
    # 对于Windows用户需要逐行执行make.sh文件里的内容
    module = TransMixerModule(channel).to(device)     
 
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)  

    print(ORANGE)    
    flops, macs, _ = calculate_flops(model=module,    
                                     input_shape=(batch_size, channel, height, width),
                                     output_as_string=True, 
                                     output_precision=4,  
                                     print_detailed=True)     
    print(RESET)