'''
本文件由BiliBili：魔傀面具整理  
engine/extre_module/module_images/A3FPN.png
engine/extre_module/module_images/A3FPN.md
论文链接：https://arxiv.org/pdf/2604.10210
'''    
import math 
import os    
import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union  
import warnings
   
import torch
import torch.nn as nn
import torch.nn.functional as F    
  
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../..")  
warnings.filterwarnings("ignore")
  
try:   
    from calflops import calculate_flops
except Exception: 
    calculate_flops = None     

try:
    from compile_module.DCNv4_op.DCNv4.functions import DCNv4Function
except Exception:     
    DCNv4Function = None 

from engine.deim.utils import get_activation
   
__all__ = ["A3Conv", "A3Fusion2", "A3Fusion3", "A3FPNTri"]


def autopad(k, p=None, d=1):   
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:     
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]    
    return p    

 
def _safe_group_count(channels: int, preferred: Optional[int]) -> int:    
    if preferred is None:
        preferred = channels
    preferred = max(1, min(preferred, channels))     
    while channels % preferred != 0 and preferred > 1:  
        preferred -= 1
    return preferred   


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()     
        self.weight = nn.Parameter(torch.ones(channels))     
        self.bias = nn.Parameter(torch.zeros(channels))    
        self.eps = eps  
     
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2).contiguous()
 
   
def build_norm(norm_cfg: Optional[Union[bool, str, nn.Module, Dict]], channels: int) -> nn.Module: 
    if norm_cfg is False or norm_cfg is None:
        return nn.Identity()
    if isinstance(norm_cfg, nn.Module):
        return norm_cfg    
    if isinstance(norm_cfg, str):
        norm_type = norm_cfg.lower()   
    elif isinstance(norm_cfg, dict):
        norm_type = str(norm_cfg.get("type", "BN")).lower()   
    else:
        raise TypeError(f"Unsupported norm config: {type(norm_cfg)}")
  
    if norm_type in {"bn", "batchnorm", "batchnorm2d", "syncbn"}: 
        return nn.BatchNorm2d(channels)
    if norm_type in {"ln", "layernorm", "layernorm2d"}:  
        return LayerNorm2d(channels)
    if norm_type in {"gn", "groupnorm"}:
        return nn.GroupNorm(_safe_group_count(channels, 32), channels)  
    raise ValueError(f"Unsupported norm type: {norm_type}")     
   

class A3Conv(nn.Module):
    default_act = nn.GELU()

    def __init__(     
        self,    
        c1: int,   
        c2: int,
        k: int = 1,   
        s: int = 1,
        p: Optional[int] = None,  
        g: int = 1,
        d: int = 1,
        act: Union[bool, str, nn.Module] = True,   
        bias: bool = False,     
        norm_cfg: Optional[Union[bool, str, nn.Module, Dict]] = "BN",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=bias)    
        self.norm = build_norm(norm_cfg, c2)
        self.act = self.default_act if act is True else get_activation(act) if isinstance(act, str) else act
        if self.act is False or self.act is None:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))    


class RepVggBlock(nn.Module): 
    def __init__(
        self,     
        ch_in: int,
        ch_out: int,    
        act: Union[str, nn.Module, bool] = "gelu", 
        norm_cfg: Optional[Union[bool, str, nn.Module, Dict]] = "BN",
    ) -> None:   
        super().__init__()
        self.conv1 = A3Conv(ch_in, ch_out, 3, 1, 1, act=False, norm_cfg=norm_cfg)
        self.conv2 = A3Conv(ch_in, ch_out, 1, 1, 0, act=False, norm_cfg=norm_cfg)
        self.act = nn.Identity() if act is False or act is None else get_activation(act) if isinstance(act, str) else act

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.act(self.conv1(x) + self.conv2(x))   
   
     
class CSPRepLayer(nn.Module):  
    def __init__(
        self,    
        in_channels: int,   
        out_channels: int,
        num_blocks: int = 1,
        expansion: float = 1.0,    
        act: Union[str, nn.Module, bool] = "gelu",    
        norm_cfg: Optional[Union[bool, str, nn.Module, Dict]] = "BN",     
    ) -> None: 
        super().__init__()  
        hidden_channels = int(out_channels * expansion)
        self.conv1 = A3Conv(in_channels, hidden_channels, 1, 1, act=act, norm_cfg=norm_cfg) 
        self.conv2 = A3Conv(in_channels, hidden_channels, 1, 1, act=act, norm_cfg=norm_cfg)     
        self.bottlenecks = nn.Sequential(     
            *[RepVggBlock(hidden_channels, hidden_channels, act=act, norm_cfg=norm_cfg) for _ in range(num_blocks)]
        )
        self.conv3 = (    
            A3Conv(hidden_channels, out_channels, 1, 1, bias=True, act=False, norm_cfg=False)   
            if hidden_channels != out_channels
            else nn.Identity() 
        )    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.bottlenecks(self.conv1(x))    
        x2 = self.conv2(x) 
        return self.conv3(x1 + x2)


class ContextGuidedDCNv4(nn.Module):
    def __init__(  
        self,
        channels: int,   
        kernel_size: int = 3,
        stride: int = 1,    
        pad: int = 1,    
        dilation: int = 1,  
        group: int = 4,
        offset_scale: float = 1.0,
        dw_kernel_size: Optional[int] = None,
        center_feature_scale: bool = False,
        remove_center: bool = False,
        output_bias: bool = True,
        without_pointwise: bool = False,
    ) -> None: 
        super().__init__()    
        if channels % group != 0:     
            raise ValueError(f"channels must be divisible by group, but got {channels} and {group}")  
   
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride     
        self.dilation = dilation  
        self.pad = pad 
        self.group = group     
        self.group_channels = channels // group 
        self.offset_scale = offset_scale
        self.dw_kernel_size = dw_kernel_size
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)
        self.without_pointwise = without_pointwise
        self.K = group * (kernel_size * kernel_size - self.remove_center)   

        if dw_kernel_size is not None:
            self.offset_mask_dw = nn.Conv2d(   
                channels,
                channels,    
                dw_kernel_size, 
                stride=1,     
                padding=(dw_kernel_size - 1) // 2,   
                groups=channels,   
            )   

        self.offset_mask = nn.Linear(channels, int(math.ceil((self.K * 3) / 8) * 8))     
        if not without_pointwise:
            self.value_proj = nn.Linear(channels, channels)
            self.output_proj = nn.Linear(channels, channels, bias=output_bias) 
  
        if center_feature_scale:  
            self.center_feature_scale_proj_weight = nn.Parameter(torch.zeros((group, channels), dtype=torch.float)) 
            self.center_feature_scale_proj_bias = nn.Parameter(torch.zeros(group, dtype=torch.float))    

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.constant_(self.offset_mask.weight.data, 0.0)    
        nn.init.constant_(self.offset_mask.bias.data, 0.0)
        if not self.without_pointwise:
            nn.init.xavier_uniform_(self.value_proj.weight.data)
            nn.init.constant_(self.value_proj.bias.data, 0.0)
            nn.init.xavier_uniform_(self.output_proj.weight.data)
            if self.output_proj.bias is not None:
                nn.init.constant_(self.output_proj.bias.data, 0.0)   

    def forward(self, sampled_feat: torch.Tensor, context_info: torch.Tensor) -> torch.Tensor:   
        if DCNv4Function is None:
            raise RuntimeError(     
                "DCNv4Function is unavailable" \
                "# 此模块需要编译,详细编译命令在 compile_module/DCNv4_op" \
                "# 对于Linux用户可以直接运行上述的make.sh文件" \
                "# 对于Windows用户需要逐行执行make.sh文件里的内容")    

        n, c, h, w = sampled_feat.shape   
        x = sampled_feat.permute(0, 2, 3, 1).contiguous()  
        if not self.without_pointwise:
            x = self.value_proj(x.view(n, -1, c)).reshape(n, h, w, -1)

        if self.dw_kernel_size is not None:
            offset_mask_input = self.offset_mask_dw(context_info)     
            offset_mask_input = offset_mask_input.permute(0, 2, 3, 1).contiguous().view(n, -1, c)   
        else:
            offset_mask_input = context_info.permute(0, 2, 3, 1).contiguous().view(n, -1, c) 
        offset_mask = self.offset_mask(offset_mask_input).reshape(n, h, w, -1)    
    
        x = DCNv4Function.apply(    
            x,     
            offset_mask,
            self.kernel_size,     
            self.kernel_size, 
            self.stride,
            self.stride, 
            self.pad,  
            self.pad,    
            self.dilation,   
            self.dilation,  
            self.group,    
            self.group_channels,
            self.offset_scale,   
            256,
            self.remove_center,   
        )     

        if not self.without_pointwise:  
            x = self.output_proj(x.view(n, -1, c)).reshape(n, h, w, c)
        return x.permute(0, 3, 1, 2).contiguous()   


class A3Resampler(nn.Module):     
    def __init__(
        self,
        channels: int,    
        act: Union[str, nn.Module, bool] = "gelu",   
        dcn_norm: Optional[Union[bool, str, nn.Module, Dict]] = "LN",
        dcn_group: int = 4,     
        offset_scale: float = 0.5,
        dw_kernel_size: int = 3,     
        dcn_output_bias: bool = False,
        center_feature_scale: bool = False,   
        remove_center: bool = False,
        without_pointwise: bool = False,
    ) -> None:     
        super().__init__() 
        self.resampling = ContextGuidedDCNv4(     
            channels,
            3,    
            stride=1,
            pad=1,    
            dilation=1,
            group=dcn_group,
            offset_scale=offset_scale,
            dw_kernel_size=dw_kernel_size, 
            output_bias=dcn_output_bias,
            center_feature_scale=center_feature_scale,
            remove_center=remove_center,     
            without_pointwise=without_pointwise,
        )     
        self.norm = build_norm(dcn_norm, channels) if dcn_norm else nn.Identity()
        self.act = nn.Identity() if act is False or act is None else get_activation(act) if isinstance(act, str) else act 
        self.fallback = A3Conv(channels, channels, 3, 1, 1, act=act, norm_cfg="BN") 

    def forward(self, context_info: torch.Tensor, sampled_feat: torch.Tensor) -> torch.Tensor:  
        if DCNv4Function is not None and sampled_feat.is_cuda and context_info.is_cuda:
            x = self.resampling(sampled_feat, context_info)
            return self.act(self.norm(x))
        return self.fallback(sampled_feat)   

 
class Fusion(nn.Module):
    def __init__(
        self,
        in_channels: int,    
        out_channels: int,
        num_fusion: int = 2,  
        compress_c: int = 16,
        act: Union[str, nn.Module, bool] = "gelu",
        norm_cfg: Optional[Union[bool, str, nn.Module, Dict]] = "BN",
        num_blocks: int = 1,   
        expansion: float = 1.0,   
        using_resampling: bool = False,  
        dcn_group: int = 4,
        dcn_config: Optional[Dict] = None,
    ) -> None:   
        super().__init__()    
        dcn_config = dcn_config or {}    
        self.using_resampling = using_resampling
        self.weight_layers = nn.ModuleList(     
            [A3Conv(in_channels, compress_c, 1, 1, bias=False, act=act, norm_cfg=norm_cfg) for _ in range(num_fusion)]
        )
        self.weight_levels = CSPRepLayer(   
            in_channels=compress_c * num_fusion,
            out_channels=num_fusion,
            num_blocks=num_blocks, 
            expansion=expansion,    
            act=act,   
            norm_cfg=norm_cfg,     
        ) 
        if using_resampling:
            self.context_conv = A3Conv(
                in_channels * num_fusion,    
                in_channels * (num_fusion - 1),
                1,     
                1,
                0, 
                bias=False,
                act=act,
                norm_cfg=norm_cfg,
            )
            self.resamplers = nn.ModuleList(
                [A3Resampler(in_channels, act=act, dcn_group=dcn_group, **dcn_config) for _ in range(num_fusion - 1)] 
            )  
        self.conv = ( 
            A3Conv(in_channels, out_channels, 1, 1, act=False, norm_cfg=False, bias=True)     
            if in_channels != out_channels     
            else nn.Identity()    
        )  

    def forward(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        sampled_feats = list(x[:-1])
        unsampled_feat = x[-1]
   
        if self.using_resampling and sampled_feats:   
            context_info = self.context_conv(torch.cat(list(x), dim=1))  
            split_context = torch.split(context_info, sampled_feats[0].size(1), dim=1)     
            sampled_feats = [    
                resampler(ctx, feat) for ctx, feat, resampler in zip(split_context, sampled_feats, self.resamplers)
            ]
  
        fusion_feats = [*sampled_feats, unsampled_feat]
        levels_weight = self.weight_levels(torch.cat([layer(feat) for layer, feat in zip(self.weight_layers, fusion_feats)], dim=1))    
        levels_weight = torch.sigmoid(levels_weight)    
  
        fused = 0    
        for index, feat in enumerate(fusion_feats):    
            fused = fused + feat * levels_weight[:, index : index + 1] 
        return self.conv(fused)
     
 
class Reassemble(nn.Module):
    def __init__(     
        self, 
        out_channels: int,
        group_num: int = 16,    
        gate_threshold: float = 0.5,
        act: Union[str, nn.Module, bool] = "gelu",
        norm: Optional[Union[bool, str, nn.Module, Dict]] = "LN",
    ) -> None:
        super().__init__()
        self.gn = nn.GroupNorm(_safe_group_count(out_channels, group_num), out_channels) 
        self.gate_threshold = gate_threshold    
        self.sigmoid = nn.Sigmoid()   
        self.act = nn.Identity() if act is False or act is None else get_activation(act) if isinstance(act, str) else act  
        self.norm = build_norm(norm, out_channels) if norm else nn.Identity() 

    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        gn_x = self.gn(x)
        w_gamma = (self.gn.weight / self.gn.weight.sum()).view(1, -1, 1, 1).contiguous()
        reweights = self.sigmoid(gn_x * w_gamma)
        w1 = torch.where(reweights > self.gate_threshold, torch.ones_like(reweights), reweights)     
        w2 = torch.where(reweights > self.gate_threshold, torch.zeros_like(reweights), reweights)     
        x1 = w1 * x  
        x2 = w2 * x
        y = x1 + x2.flip(1).contiguous()  
        return self.act(self.norm(y))
   
  
class SampleBlock(nn.Module):    
    def __init__(   
        self,
        in_channels: int,  
        out_channels: int,    
        sample: str = "upsample",
        scale_factor: int = 2,
        upsample_method: str = "bilinear",
        align_corners: bool = False,     
        group: int = 1,
        act: Union[str, nn.Module, bool] = "gelu",
        norm_cfg: Optional[Union[bool, str, nn.Module, Dict]] = "BN",
    ) -> None:
        super().__init__()   
        if sample == "upsample":
            layers = []
            if in_channels != out_channels:
                layers.append(A3Conv(in_channels, out_channels, 1, act=act, norm_cfg=norm_cfg))
            else:     
                layers.append(nn.Identity())
            layers.append(nn.Upsample(scale_factor=scale_factor, mode=upsample_method, align_corners=align_corners))    
            self.sample_layer = nn.Sequential(*layers)
        elif sample == "downsample":
            self.sample_layer = A3Conv(   
                in_channels,     
                out_channels,
                scale_factor,
                scale_factor,    
                0,  
                g=in_channels if in_channels == out_channels else group,
                act=act, 
                norm_cfg=norm_cfg,  
            )  
        else:
            raise ValueError("sample must be 'upsample' or 'downsample'") 
 
    def forward(self, feat: torch.Tensor) -> torch.Tensor:    
        return self.sample_layer(feat)
   
   
class A3Fusion2(nn.Module):
    def __init__(  
        self,   
        level: int,
        channels: Sequence[int],     
        act: Union[str, nn.Module, bool] = "gelu",     
        norm_cfg: Optional[Union[bool, str, nn.Module, Dict]] = "BN",    
        compress_channel: int = 16, 
        group_num: int = 16,  
        num_blocks: int = 1,  
        expansion: float = 1.0,
        using_resampling: bool = False,
        dcn_group: int = 4,
        dcn_config: Optional[Dict] = None,   
    ) -> None:   
        super().__init__()     
        self.level = level
        if level == 0:     
            self.sample = SampleBlock(channels[1], channels[0], "upsample", 2, act=act, norm_cfg=norm_cfg)   
        else:     
            self.sample = SampleBlock(channels[0], channels[1], "downsample", 2, act=act, norm_cfg=norm_cfg)
        self.mca = Fusion(
            in_channels=channels[level],
            out_channels=channels[level],
            num_fusion=2,
            compress_c=compress_channel, 
            act=act,
            norm_cfg=norm_cfg,
            num_blocks=num_blocks,
            expansion=expansion,
            using_resampling=using_resampling,
            dcn_group=dcn_group,
            dcn_config=dcn_config,     
        )
        self.ica = Reassemble(channels[level], group_num=group_num, act=act, norm="LN")
  
    def forward(self, x: Sequence[torch.Tensor]) -> torch.Tensor: 
        input1, input2 = x   
        if self.level == 0:    
            input2 = self.sample(input2) 
            out = self.mca((input2, input1))   
        else:     
            input1 = self.sample(input1)
            out = self.mca((input1, input2))
        return self.ica(out)   

 
class A3Fusion3(nn.Module):    
    def __init__(     
        self,
        level: int,
        channels: Sequence[int], 
        act: Union[str, nn.Module, bool] = "gelu",    
        norm_cfg: Optional[Union[bool, str, nn.Module, Dict]] = "BN",   
        compress_channel: int = 16,
        group_num: int = 16,   
        num_blocks: int = 1,     
        expansion: float = 1.0,    
        using_resampling: bool = False,    
        dcn_group: int = 4,  
        dcn_config: Optional[Dict] = None, 
    ) -> None:    
        super().__init__()
        self.level = level
        if level == 0:   
            self.upsample4x = SampleBlock(channels[2], channels[0], "upsample", 4, act=act, norm_cfg=norm_cfg)
            self.upsample2x = SampleBlock(channels[1], channels[0], "upsample", 2, act=act, norm_cfg=norm_cfg)
        elif level == 1:    
            self.upsample2x1 = SampleBlock(channels[2], channels[1], "upsample", 2, act=act, norm_cfg=norm_cfg)  
            self.downsample2x1 = SampleBlock(channels[0], channels[1], "downsample", 2, act=act, norm_cfg=norm_cfg) 
        else:     
            self.downsample2x = SampleBlock(channels[1], channels[2], "downsample", 2, act=act, norm_cfg=norm_cfg) 
            self.downsample4x = SampleBlock(channels[0], channels[2], "downsample", 4, act=act, norm_cfg=norm_cfg)
   
        self.mca = Fusion(  
            in_channels=channels[level],
            out_channels=channels[level],  
            num_fusion=3,
            compress_c=compress_channel,
            act=act,   
            norm_cfg=norm_cfg,
            num_blocks=num_blocks,
            expansion=expansion, 
            using_resampling=using_resampling,
            dcn_group=dcn_group, 
            dcn_config=dcn_config,     
        )  
        self.ica = Reassemble(channels[level], group_num=group_num, act=act, norm="LN") 
     
    def forward(self, x: Sequence[torch.Tensor]) -> torch.Tensor:  
        input1, input2, input3 = x
        if self.level == 0: 
            input2 = self.upsample2x(input2)   
            input3 = self.upsample4x(input3)    
            out = self.mca((input2, input3, input1))
        elif self.level == 1:
            input3 = self.upsample2x1(input3)  
            input1 = self.downsample2x1(input1)
            out = self.mca((input3, input1, input2))
        else:
            input1 = self.downsample4x(input1)
            input2 = self.downsample2x(input2)
            out = self.mca((input2, input1, input3))    
        return self.ica(out)   
    
   
class A3FPNTri(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int],  
        out_channels: int,   
        act: Union[str, nn.Module, bool] = "gelu", 
        norm_cfg: Optional[Union[bool, str, nn.Module, Dict]] = "BN",
        compress_channel: Union[int, Sequence[int]] = (16, 16, 16),
        group_num: Union[int, Sequence[int]] = (16, 16, 16),    
        num_repblocks: int = 1,   
        expansion: float = 1.0,
        using_resampling: bool = False,
        dcn_group: int = 4,   
        dcn_config: Optional[Dict] = None,
    ) -> None:   
        super().__init__()
        if len(in_channels) != 3: 
            raise ValueError(f"A3FPNTri expects three input scales, got {len(in_channels)}")   

        if isinstance(compress_channel, int):
            compress_channel = [compress_channel] * 3    
        if isinstance(group_num, int):
            group_num = [group_num] * 3
  
        self.reduce = nn.ModuleList(     
            [  
                A3Conv(channel, out_channels, 1, act=act, norm_cfg=norm_cfg, bias=False)  
                if channel != out_channels
                else nn.Identity()
                for channel in in_channels  
            ]
        )    

        tri_channels = [out_channels, out_channels, out_channels] 
        self.a3fpn_2_level0 = A3Fusion2(
            0,
            tri_channels[-2:],
            act=act,
            norm_cfg=norm_cfg,
            compress_channel=compress_channel[-2],
            group_num=group_num[-2],
            num_blocks=num_repblocks,
            expansion=expansion,
            using_resampling=using_resampling,     
            dcn_group=dcn_group,
            dcn_config=dcn_config,
        )
        self.a3fpn_2_level1 = A3Fusion2(    
            1,
            tri_channels[-2:], 
            act=act,
            norm_cfg=norm_cfg,     
            compress_channel=compress_channel[-1],    
            group_num=group_num[-1],
            num_blocks=num_repblocks,
            expansion=expansion,   
            using_resampling=using_resampling,
            dcn_group=dcn_group,     
            dcn_config=dcn_config,
        )
        self.a3fpn_3_level0 = A3Fusion3( 
            0,    
            tri_channels,     
            act=act,     
            norm_cfg=norm_cfg,
            compress_channel=compress_channel[0],
            group_num=group_num[0],
            num_blocks=num_repblocks,
            expansion=expansion,   
            using_resampling=using_resampling,
            dcn_group=dcn_group,     
            dcn_config=dcn_config,
        )
        self.a3fpn_3_level1 = A3Fusion3(
            1,
            tri_channels,
            act=act,
            norm_cfg=norm_cfg,  
            compress_channel=compress_channel[1],
            group_num=group_num[1],
            num_blocks=num_repblocks,
            expansion=expansion,
            using_resampling=using_resampling,
            dcn_group=dcn_group,     
            dcn_config=dcn_config,
        )   
        self.a3fpn_3_level2 = A3Fusion3(     
            2,
            tri_channels,     
            act=act,  
            norm_cfg=norm_cfg,  
            compress_channel=compress_channel[2],
            group_num=group_num[2],
            num_blocks=num_repblocks,
            expansion=expansion,
            using_resampling=using_resampling,   
            dcn_group=dcn_group,   
            dcn_config=dcn_config,
        )  
        self.output_convs = nn.ModuleList( 
            [A3Conv(out_channels, out_channels, 3, 1, 1, act=act, norm_cfg=norm_cfg) for _ in range(3)]
        )
   
    def forward(self, x: Sequence[torch.Tensor]) -> List[torch.Tensor]:    
        p3, p4, p5 = [layer(feat) for layer, feat in zip(self.reduce, x)] 

        p4_seed = self.a3fpn_2_level0((p4, p5))    
        p5_seed = self.a3fpn_2_level1((p4, p5))  

        out_p3 = self.a3fpn_3_level0((p3, p4_seed, p5_seed))   
        out_p4 = self.a3fpn_3_level1((p3, p4_seed, p5_seed))   
        out_p5 = self.a3fpn_3_level2((p3, p4_seed, p5_seed))     
    
        return [conv(feat) for conv, feat in zip(self.output_convs, (out_p3, out_p4, out_p5))]   

  
if __name__ == "__main__": 
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"   

    def _print_header(name: str) -> None:    
        print(RED + "-" * 20 + f" {name} " + "-" * 20 + RESET)    
  
    def _profile_module(module: nn.Module, args) -> None:
        if calculate_flops is None:
            print(YELLOW + "calflops is not installed, skip FLOPs/MACs profiling." + RESET)  
            return     
        try:
            print(ORANGE)  
            calculate_flops(    
                model=module,
                args=args,
                output_as_string=True,   
                output_precision=4,
                print_detailed=True,
            )
        except Exception as exc:  
            print(YELLOW + f"calflops profiling skipped: {exc}" + RESET)
        finally:  
            print(RESET)    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    torch.manual_seed(0)
    
    batch_size, channel = 1, 64    
    p3 = torch.randn((batch_size, channel, 40, 40), device=device)
    p4 = torch.randn((batch_size, channel, 20, 20), device=device)
    p5 = torch.randn((batch_size, channel, 10, 10), device=device)

    _print_header("A3Conv")
    a3conv = A3Conv(channel, channel, 3, 1, 1).to(device).eval()
    a3conv_out = a3conv(p4)
    print(GREEN + f"input.size:{p4.size()} output.size:{a3conv_out.size()}" + RESET)   
    _profile_module(a3conv, [p4])   
    
    _print_header("A3Fusion2 level=0")
    a3fusion2_level0 = A3Fusion2(0, [channel, channel]).to(device).eval()
    a3fusion2_level0_out = a3fusion2_level0((p4, p5))     
    print(GREEN + f"p4.size:{p4.size()} p5.size:{p5.size()} output.size:{a3fusion2_level0_out.size()}" + RESET)
    _profile_module(a3fusion2_level0, [[p4, p5]])    
 
    _print_header("A3Fusion2 level=1")
    a3fusion2_level1 = A3Fusion2(1, [channel, channel]).to(device).eval()
    a3fusion2_level1_out = a3fusion2_level1((p4, p5))
    print(GREEN + f"p4.size:{p4.size()} p5.size:{p5.size()} output.size:{a3fusion2_level1_out.size()}" + RESET)   
    _profile_module(a3fusion2_level1, [[p4, p5]])  

    _print_header("A3Fusion3 level=0")
    a3fusion3_level0 = A3Fusion3(0, [channel, channel, channel]).to(device).eval()
    a3fusion3_level0_out = a3fusion3_level0((p3, p4, p5))
    print(GREEN + f"p3.size:{p3.size()} p4.size:{p4.size()} p5.size:{p5.size()} output.size:{a3fusion3_level0_out.size()}" + RESET)    
    _profile_module(a3fusion3_level0, [[p3, p4, p5]])

    _print_header("A3Fusion3 level=1")
    a3fusion3_level1 = A3Fusion3(1, [channel, channel, channel]).to(device).eval()
    a3fusion3_level1_out = a3fusion3_level1((p3, p4, p5))    
    print(GREEN + f"p3.size:{p3.size()} p4.size:{p4.size()} p5.size:{p5.size()} output.size:{a3fusion3_level1_out.size()}" + RESET)
    _profile_module(a3fusion3_level1, [[p3, p4, p5]])  

    _print_header("A3Fusion3 level=2")
    a3fusion3_level2 = A3Fusion3(2, [channel, channel, channel]).to(device).eval()     
    a3fusion3_level2_out = a3fusion3_level2((p3, p4, p5))  
    print(GREEN + f"p3.size:{p3.size()} p4.size:{p4.size()} p5.size:{p5.size()} output.size:{a3fusion3_level2_out.size()}" + RESET)
    _profile_module(a3fusion3_level2, [[p3, p4, p5]])

    _print_header("A3FPNTri")
    a3fpn_tri = A3FPNTri([channel, channel, channel], channel).to(device).eval()  
    a3fpn_tri_out = a3fpn_tri((p3, p4, p5))    
    print(     
        GREEN  
        + f"p3.size:{p3.size()} p4.size:{p4.size()} p5.size:{p5.size()} "     
        + f"out_p3.size:{a3fpn_tri_out[0].size()} out_p4.size:{a3fpn_tri_out[1].size()} out_p5.size:{a3fpn_tri_out[2].size()}"
        + RESET    
    )
    _profile_module(a3fpn_tri, [[p3, p4, p5]])
