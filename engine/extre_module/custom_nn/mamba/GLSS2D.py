'''
本文件由BiliBili：魔傀面具整理 
engine/extre_module/module_images/TGRS2025-GLSS2D.png    
engine/extre_module/module_images/TGRS2025-GLSS2D.md
论文链接：https://ieeexplore.ieee.org/document/11014226  
''' 

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')
  
import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops    
  
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_  
from einops import repeat     
 
try:   
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref   
except:
    pass
     
class SELayer(nn.Module): 
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()     
        self.avg_pool = nn.AdaptiveAvgPool2d(1)    
        self.fc = nn.Sequential(  
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),   
            nn.Sigmoid()
        )
 
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x) 
  
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()   
        self.relu = nn.ReLU6(inplace=inplace)  
  
    def forward(self, x):
        return self.relu(x + 3) / 6
  
    
class ECALayer(nn.Module): 
    def __init__(self, channel, gamma=2, b=1, sigmoid=True):    
        super(ECALayer, self).__init__()   
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        if sigmoid: 
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = h_sigmoid()
    
    def forward(self, x): 
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))  
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)  
        return x * y.expand_as(x) 
 
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x) 
class LocalityFeedForward(nn.Module):   
    def __init__(self, in_dim, out_dim, stride, expand_ratio=4., act='hs+se', reduction=4,
                 wo_dp_conv=False, dp_first=False):
        """
        :param in_dim: the input dimension   
        :param out_dim: the output dimension. The input and output dimension should be the same.
        :param stride: stride of the depth-wise convolution.
        :param expand_ratio: expansion ratio of the hidden dimension.    
        :param act: the activation function.
                    relu: ReLU
                    hs: h_swish   
                    hs+se: h_swish and SE module  
                    hs+eca: h_swish and ECA module    
                    hs+ecah: h_swish and ECA module. Compared with eca, h_sigmoid is used.   
        :param reduction: reduction rate in SE module.
        :param wo_dp_conv: without depth-wise convolution.
        :param dp_first: place depth-wise convolution as the first layer.     
        """   
        super(LocalityFeedForward, self).__init__()
        hidden_dim = int(in_dim * expand_ratio)  
        kernel_size = 3     
     
        layers = []     
        # the first linear layer is replaced by 1x1 convolution.
        layers.extend([
            nn.Conv2d(in_dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)])

        # the depth-wise convolution between the two linear layers 
        if not wo_dp_conv:
            dp = [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False),   
                nn.BatchNorm2d(hidden_dim),    
                h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)   
            ]
            if dp_first:
                layers = dp + layers
            else:
                layers.extend(dp)
 
        if act.find('+') >= 0:
            attn = act.split('+')[1]
            if attn == 'se':   
                layers.append(SELayer(hidden_dim, reduction=reduction))
            elif attn.find('eca') >= 0:     
                layers.append(ECALayer(hidden_dim, sigmoid=attn == 'eca'))
            else:    
                raise NotImplementedError('Activation type {} is not implemented'.format(act))
   
        # the second linear layer is replaced by 1x1 convolution.  
        layers.extend([
            nn.Conv2d(hidden_dim, out_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_dim)
        ])  
        self.conv = nn.Sequential(*layers)     

    def forward(self, x):
        x = x + self.conv(x)
        return x
  
class DWConv(nn.Module):
    def __init__(self, dim=768):     
        super(DWConv, self).__init__() 
        self.dwconv = nn.Sequential(    
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim) 
        )   

    def forward(self, x):  
        x = self.dwconv(x)    
        return x    
    
class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()  
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features     
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.dwconv = DWConv(hidden_features) 
        self.act = act_layer()    
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)    
        self.drop = nn.Dropout(drop) 
     
        self.shift_size = shift_size   
        self.pad = shift_size // 2   

        self.conv = LocalityFeedForward(in_features, out_features, 1, expand_ratio=4., act='hs+se', reduction=4, wo_dp_conv=False, dp_first=False)
        self.apply(self._init_weights)
 
    def _init_weights(self, m):  
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)     
            if isinstance(m, nn.Linear) and m.bias is not None: 
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)   
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups     
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out)) 
            if m.bias is not None:     
                m.bias.data.zero_()

    def forward(self, x, y):
        B, C, H, W = x.shape
        xn = F.pad(x, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1) 
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]   
        x_cat = torch.cat(x_shift, 1)    
        x_cat = torch.narrow(x_cat, 2, self.pad, H)   
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x = self.fc1(x_s)   
        x = self.dwconv(x)     
        x = self.act(x)
        x = self.drop(x)

        xn = F.pad(x, (self.pad, self.pad, self.pad, self.pad), "constant", 0)  
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)   
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)   
        x = self.fc2(x_s)  
        x = self.drop(x)  
   
        # y_fc = self.fc1(y)
        # y_dw = self.dwconv(y_fc)
        # y_out = self.act(y_dw) 
        # y_out = self.fc2(y_out) 
        # y = self.drop(y_out)    
        y = self.conv(y)   
        x = torch.cat([x.unsqueeze(2), y.unsqueeze(2)], dim=2).reshape(B, 2 * C, H, W)   
        return x   
   
class shiftedBlock(nn.Module):     
    def __init__(self, dim, mlp_ratio=4., drop=0., shift_size=4,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm2d):
        super().__init__()
    
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, shift_size=shift_size)
        self.apply(self._init_weights)
  
    def _init_weights(self, m): 
        if isinstance(m, nn.Linear):     
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:    
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)  
            nn.init.constant_(m.weight, 1.0)     
        elif isinstance(m, nn.Conv2d):  
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels   
            fan_out //= m.groups     
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))    
            if m.bias is not None:   
                m.bias.data.zero_()

    def forward(self, x):    
    
        # x = x + self.drop_path(self.mlp(self.norm2(x)))  #origin   
        x = x + self.drop_path((self.norm2(x)))
        return x
     
    
class ConvBN(nn.Sequential):  
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, 
                 bias=False):  
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,    
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)    
        )     
     
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs    
    with shape (batch_size, channels, height, width).
    """     
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"): 
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))    
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format 
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )     
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)   
            x = (x - u) / torch.sqrt(s + self.eps) 
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x     
  
class SS2D(nn.Module): 
    def __init__(
        self,    
        d_model,     
        d_state=16,
        # d_state="auto", # 20240109     
        d_conv=3,   
        expand=0.5,
        dt_rank="auto",  
        dt_min=0.001,     
        dt_max=0.1,
        dt_init="random", 
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,    
        conv_bias=True,    
        bias=False,     
        device=None,
        dtype=None,
        **kwargs, 
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109     
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)   
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
    
        self.x_proj = (  
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),     
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),  
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj
   
        self.dt_projs = (    
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),   
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),     
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),    
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)    
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)  
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)   
  
        self.forward_core = self.forward_core_windows 
        # self.forward_core = self.forward_corev0_seq     
        # self.forward_core = self.forward_corev1

        self.out_norm = nn.LayerNorm(self.d_inner)     
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs) 
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
 

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs) 

        # Initialize special dt projection to preserve variance at initialization     
        dt_init_std = dt_rank**-0.5 * dt_scale    
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":     
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std) 
        else:  
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)    
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759  
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():     
            dt_proj.bias.copy_(inv_dt)  
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit  
        dt_proj.bias._no_reinit = True
  
        return dt_proj     

    @staticmethod     
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization 
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),    
            "n -> d n",   
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1) 
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log    

    @staticmethod    
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter     
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies) 
            if merge:
                D = D.flatten(0, 1)     
        D = nn.Parameter(D)  # Keep in fp32    
        D._no_weight_decay = True 
        return D 
   
 
    def forward_core_windows(self, x: torch.Tensor, layer=1):  
        return self.forward_corev0(x)  
        if layer == 1:
            return self.forward_corev0(x)   
        downsampled_4 = F.avg_pool2d(x, kernel_size=2, stride=2)    
        processed_4 = self.forward_corev0(downsampled_4) 
        processed_4 = processed_4.permute(0, 3, 1, 2) 
        restored_4 = F.interpolate(processed_4, scale_factor=2, mode='nearest') 
        restored_4 = restored_4.permute(0, 2, 3, 1)   
        if layer == 2:
            output = (self.forward_corev0(x) + restored_4) / 2.0

        downsampled_8 = F.avg_pool2d(x, kernel_size=4, stride=4)
        processed_8 = self.forward_corev0(downsampled_8)
        processed_8 = processed_8.permute(0, 3, 1, 2) 
        restored_8 = F.interpolate(processed_8, scale_factor=4, mode='nearest')   
        restored_8 = restored_8.permute(0, 2, 3, 1)   
 
        output = (self.forward_corev0(x) + restored_4 + restored_8) / 3.0   
        return output
        # B C H W   
 
        num_splits = 2 ** layer
        split_size = x.shape[2] // num_splits  # Assuming H == W and is divisible by 2**layer

        # Use unfold to create windows
        x_unfolded = x.unfold(2, split_size, split_size).unfold(3, split_size, split_size)   
        x_unfolded = x_unfolded.contiguous().view(-1, x.size(1), split_size, split_size)  

        # Process all splits at once
        processed_splits = self.forward_corev0(x_unfolded)
        processed_splits = processed_splits.permute(0, 3, 1, 2)
        # Reshape to get the splits back into their original positions and then permute to align dimensions
        processed_splits = processed_splits.view(x.size(0), num_splits, num_splits, x.size(1), split_size, split_size)    
        processed_splits = processed_splits.permute(0, 3, 1, 4, 2, 5).contiguous()
        processed_splits = processed_splits.view(x.size(0), x.size(1), x.size(2), x.size(3))    
        processed_splits = processed_splits.permute(0, 2, 3, 1)   
     
        return processed_splits

     
        # num_splits = 2 ** layer  
        # split_size = x.shape[2] // num_splits  # Assuming H == W and is divisible by 2**layer   
        # outputs = []
        # for i in range(num_splits):
        #     row_outputs = []
        #     for j in range(num_splits): 
        #         sub_x = x[:, :, i*split_size:(i+1)*split_size, j*split_size:(j+1)*split_size].contiguous()   
        #         processed = self.forward_corev0(sub_x)
        #         row_outputs.append(processed)
        #     # Concatenate all column splits for current row     
        #     outputs.append(torch.cat(row_outputs, dim=2)) 
        # # Concatenate all rows 
        # final_output = torch.cat(outputs, dim=1)   
    
        return final_output
   
 
    def forward_corev0(self, x: torch.Tensor):   
        self.selective_scan = selective_scan_fn
        B, C, H, W = x.shape    
        L = H * W

        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)  
  
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)  
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)     
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)    
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)    
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)   
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)     
    
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,   
        ).view(B, K, -1, L)  
        assert out_y.dtype == torch.float    

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)  
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y 
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype) 

        return y
    
    def forward_corev0_seq(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn    
  
        B, C, H, W = x.shape   
        L = H * W    
        K = 4   
    
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)  
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)   
 
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)   
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)    
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)     
     
        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)   
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
  
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)  
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
  
        out_y = []
        for i in range(4):
            yi = self.selective_scan(    
                xs[:, i], dts[:, i], 
                As[i], Bs[:, i], Cs[:, i], Ds[i],
                delta_bias=dt_projs_bias[i], 
                delta_softplus=True,   
            ).view(B, -1, L)
            out_y.append(yi)
        out_y = torch.stack(out_y, dim=1)    
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)  
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y   
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)
     
        return y

    def forward(self, x: torch.Tensor, layer=1, **kwargs):
        B, H, W, C = x.shape

   
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)  
  
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w) 
    
        y = self.forward_core(x, layer)
        y = y * F.silu(z) 
     
     
        out = self.out_proj(y)  
        if self.dropout is not None:
            out = self.dropout(out)
        return out 

class GLSS2D(nn.Module):  
    def __init__(self, dim, d_state=16, dropout=0.):   
        super().__init__() 
        self.local1 = ConvBN(dim // 2, dim // 2, kernel_size=3)
        self.local2 = ConvBN(dim // 2, dim // 2, kernel_size=1)
        self.pre_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.post_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first') 
        self.mff = shiftedBlock(     
            dim=dim, mlp_ratio=1, drop=0., drop_path=0, norm_layer=nn.BatchNorm2d, shift_size=4)    
        self.mlp = shiftmlp(in_features=dim // 2, hidden_features=2 * dim , act_layer=nn.GELU, drop=0., shift_size=4)
        self.SS2D = SS2D(d_model=dim // 2, dropout=dropout, d_state=d_state) 
        self.apply(self._init_weights)    
  
    def _init_weights(self, m): 
        if isinstance(m, nn.Linear):     
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None: 
                nn.init.constant_(m.bias, 0)  
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)     
            nn.init.constant_(m.weight, 1.0)    
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))    
            if m.bias is not None:    
                m.bias.data.zero_()
  
    def forward(self, x):  
        x = self.pre_norm(x) 
        x1, x2 = torch.chunk(x, 2, dim=1) 
        x1_local = self.local1(x1) + self.local2(x1)  
        B, C, a, b = x2.shape    
   
        x2 = x2.permute(0, 2, 3, 1).contiguous() 
 
        x2 = self.SS2D(x2)
     
        x2 = x2.permute(0, 3, 1, 2).contiguous()
        x = self.mlp(x1_local,x2)
        x = self.mff(x)  
        x = self.post_norm(x)    
        return x 

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"   
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel, height, width = 1, 16, 20, 20
    inputs = torch.randn((batch_size, channel, height, width)).to(device) 
   
    # 此模块需要编译,详细编译命令在 compile_module/mamba 
    # 对于Linux用户可以直接运行上述的make.sh文件
    # 对于Windows用户需要逐行执行make.sh文件里的内容     
    module = GLSS2D(channel).to(device) 
 
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)  

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,     
                                     input_shape=(batch_size, channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)     
    print(RESET)     
