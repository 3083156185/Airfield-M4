'''  
本文件由BiliBili：魔傀面具整理   
engine/extre_module/module_images/CVPR2025-wca.png    
engine/extre_module/module_images/CVPR2025-wca.md    
论文链接：https://openaccess.thecvf.com/content/CVPR2025/papers/Yan_Wavelet_and_Prototype_Augmented_Query-based_Transformer_for_Pixel-level_Surface_Defect_CVPR_2025_paper.pdf  
''' 

import warnings  
warnings.filterwarnings('ignore')
from calflops import calculate_flops
  
import math  
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F 
   

def _is_perfect_square(value: int) -> bool: 
    root = int(math.isqrt(value))
    return root * root == value     
   
  
class _WavePool(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.LL = self._build_wavelet_conv(in_channels, low_x=True, low_y=True)
        self.LH = self._build_wavelet_conv(in_channels, low_x=True, low_y=False)
        self.HL = self._build_wavelet_conv(in_channels, low_x=False, low_y=True)
        self.HH = self._build_wavelet_conv(in_channels, low_x=False, low_y=False)
 
    @staticmethod  
    def _build_wavelet_conv(in_channels: int, low_x: bool, low_y: bool) -> nn.Conv2d:  
        conv = nn.Conv2d( 
            in_channels,
            in_channels,   
            kernel_size=2,    
            stride=2,
            padding=0,     
            bias=False,    
            groups=in_channels,   
        )
        low = torch.tensor([1.0, 1.0], dtype=torch.float32) / math.sqrt(2.0)     
        high = torch.tensor([-1.0, 1.0], dtype=torch.float32) / math.sqrt(2.0)
        base_x = low if low_x else high 
        base_y = low if low_y else high     
        kernel = torch.outer(base_y, base_x).view(1, 1, 2, 2)
        with torch.no_grad():   
            conv.weight.copy_(kernel.expand(in_channels, -1, -1, -1))    
        conv.weight.requires_grad_(False)     
        return conv

    def forward(
        self, x: torch.Tensor 
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:   
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)     
     
   
class _ConvBnRelu(nn.Module):
    def __init__(
        self,
        in_channel: int,  
        out_channel: int,
        k: int = 3,
        s: int = 1,
        p: int = 1, 
        g: int = 1,     
        d: int = 1,  
        bias: bool = False, 
        bn: bool = True,
        relu: bool = True,
    ) -> None:
        super().__init__()  
        layers = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)] 
        if bn:   
            layers.append(nn.BatchNorm2d(out_channel)) 
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)     
     
    def forward(self, x: torch.Tensor) -> torch.Tensor:     
        return self.conv(x)
     

class _DSConv3x3(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, stride: int = 1, dilation: int = 1) -> None:   
        super().__init__()  
        self.conv = nn.Sequential(
            _ConvBnRelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
            _ConvBnRelu(in_channel, out_channel, k=1, s=1, p=0, relu=False),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:    
        return self.conv(x)    
  
  
class _MSCW(nn.Module):
    def __init__(self, d_model: int = 64) -> None:  
        super().__init__()
        self.local_attn = nn.Sequential(     
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),    
        )
        self.global_attn = nn.Sequential(    
            nn.Linear(d_model, d_model),    
            nn.ReLU(), 
            nn.Linear(d_model, d_model),
        )
        self.sigmoid = nn.Sigmoid()   
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:     
        pool = torch.mean(x, dim=1, keepdim=True)
        attn = self.local_attn(x) + self.global_attn(pool)
        return self.sigmoid(attn)


class WCA(nn.Module): 
    """Standalone WPFormer MultiheadAttention.

    Plug-and-play path (recommended): 
    - input ``x``: [B, C, H, W]
    - output: [B, C, H, W]
 
    Compatibility path:
    - ``query/key/value`` with ``seq_first`` ([N, B, C]) or  
      ``batch_first`` ([B, N, C]) layout.  
    - ``key/value`` also support [B, C, H, W].
    """     

    def __init__(
        self,     
        d_model: int,
        h: int = 8,
        dropout: float = 0.0,     
        proto_size: int = 16, 
        input_layout: str = "auto",
    ) -> None:
        super().__init__() 
        if d_model % h != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by heads ({h}).")
        if input_layout not in {"auto", "seq_first", "batch_first"}:  
            raise ValueError("input_layout must be one of: auto, seq_first, batch_first")
    
        self.d_model = d_model     
        self.h = h    
        self.proto_size = proto_size     
        self.input_layout = input_layout  
 
        self.norm1 = nn.LayerNorm(d_model)  
        self.norm2 = nn.LayerNorm(d_model)
        self.pool = _WavePool(d_model) 
        self.self_attn1 = nn.MultiheadAttention(d_model, h, dropout=dropout, batch_first=True)
        self.mscw1 = _MSCW(d_model=d_model) 
        self.mscw2 = _MSCW(d_model=d_model)
        self.conv3x3 = _DSConv3x3(d_model, d_model)
        self.Mheads = nn.Linear(d_model, proto_size, bias=False)

    def _resolve_layout(self, x: torch.Tensor, name: str) -> str: 
        if x.ndim == 4: 
            return "feature_map"
        if x.ndim != 3:    
            raise ValueError(f"{name} must be 3D ([N,B,C] or [B,N,C]) or 4D ([B,C,H,W]), got {tuple(x.shape)}")    
        if x.shape[-1] != self.d_model:
            raise ValueError(f"{name}.shape[-1] must equal d_model={self.d_model}, got {x.shape[-1]}")
   
        if self.input_layout != "auto":    
            return self.input_layout     
        if x.shape[0] == x.shape[1]:
            return "seq_first"
        return "batch_first" if x.shape[0] < x.shape[1] else "seq_first"    

    def _to_batch_first(
        self, x: torch.Tensor, *, name: str
    ) -> Tuple[torch.Tensor, str, Optional[Tuple[int, int]]]:    
        layout = self._resolve_layout(x, name)     
        if layout == "feature_map":
            b, c, h, w = x.shape
            if c != self.d_model:
                raise ValueError(f"{name} channel dimension must equal d_model={self.d_model}, got {c}")     
            return x.flatten(2).transpose(1, 2).contiguous(), layout, (h, w)   
        if layout == "batch_first":
            return x.contiguous(), layout, None
        return x.transpose(0, 1).contiguous(), layout, None

    def _restore_layout(  
        self, x: torch.Tensor, query_layout: str, query_hw: Optional[Tuple[int, int]]
    ) -> torch.Tensor:
        if query_layout == "seq_first":     
            return x.transpose(0, 1).contiguous()    
        if query_layout == "feature_map": 
            if query_hw is None:
                raise RuntimeError("Missing query_hw for feature_map restoration.")
            b, n, c = x.shape    
            h, w = query_hw
            if n != h * w: 
                raise ValueError(f"Cannot restore query feature map: token count {n} != {h}*{w}")     
            return x.transpose(1, 2).reshape(b, c, h, w).contiguous()
        return x     
 
    @staticmethod   
    def _validate_hw_from_tokens(tokens: torch.Tensor, spatial_shape: Optional[Tuple[int, int]]) -> Tuple[int, int]:
        if spatial_shape is not None:   
            return spatial_shape
        token_len = tokens.size(1)  
        if _is_perfect_square(token_len):
            side = int(math.isqrt(token_len))    
            return (side, side)
        raise ValueError(  
            "Cannot infer 2D shape from token length. "
            "Please provide spatial_shape=(H, W) or pass feature map input [B, C, H, W]."    
        )
  
    def _forward_batch_first_tokens(    
        self,
        query_tokens: torch.Tensor,
        key_tokens: torch.Tensor,    
        value_tokens: torch.Tensor,
        hw: Tuple[int, int],
        attn_mask: Optional[torch.Tensor] = None,  
    ) -> torch.Tensor:
        if query_tokens.size(0) != key_tokens.size(0) or key_tokens.size(0) != value_tokens.size(0):
            raise ValueError("Batch size of query, key, value must be equal.")
        if key_tokens.size(1) != value_tokens.size(1):
            raise ValueError("Sequence length of key and value must be equal.")
        if hw[0] * hw[1] != key_tokens.size(1): 
            raise ValueError(f"Invalid spatial_shape={hw}: H*W != key length ({key_tokens.size(1)}).")  
  
        source_tokens = 0.5 * (key_tokens + value_tokens)  
   
        b, _, c = source_tokens.size()
        feat = source_tokens.transpose(1, 2).reshape(b, c, hw[0], hw[1]) 

        ll, lh, hl, hh = self.pool(feat)  
        high_fre = (hl + lh + hh).flatten(2).transpose(1, 2)
        low_fre = ll.flatten(2).transpose(1, 2)
        wei = self.mscw1(high_fre + low_fre)
        fre = wei * high_fre + low_fre

        x1 = self.self_attn1(query=query_tokens, key=fre, value=fre, attn_mask=attn_mask)[0]     
        x1 = self.norm1(x1 + query_tokens) 

        feat_tokens = self.conv3x3(feat).flatten(2).transpose(1, 2)
        multi_heads_weights = F.softmax(self.Mheads(feat_tokens), dim=1)
        protos = multi_heads_weights.transpose(-1, -2) @ source_tokens
        if protos.size(1) != query_tokens.size(1):
            protos = F.interpolate(
                protos.transpose(1, 2), size=query_tokens.size(1), mode="linear", align_corners=False    
            ).transpose(1, 2)   

        attn = self.mscw2(protos + query_tokens)
        x2 = self.norm2(query_tokens * attn + query_tokens)  
    
        return x1 + x2 
   
    def forward(    
        self, 
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,  
        value: Optional[torch.Tensor] = None,     
        attn_mask: Optional[torch.Tensor] = None,     
        spatial_shape: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        # Plug-and-play path: only one feature-map input [B, C, H, W]  
        if key is None and value is None:
            if query.ndim != 4: 
                raise ValueError(
                    "Single-input mode expects query as [B, C, H, W]. "
                    "For token mode, provide query/key/value together."   
                ) 
            b, c, h, w = query.shape
            if c != self.d_model: 
                raise ValueError(f"query channel dimension must equal d_model={self.d_model}, got {c}")
            tokens = query.flatten(2).transpose(1, 2).contiguous()  
            out_tokens = self._forward_batch_first_tokens(tokens, tokens, tokens, (h, w), attn_mask=attn_mask)
            return out_tokens.transpose(1, 2).reshape(b, c, h, w).contiguous()   

        if (key is None) != (value is None):  
            raise ValueError("key and value must both be provided or both be None.")
        if key is None or value is None:   
            raise ValueError("Unexpected key/value state.")    
   
        query_tokens, query_layout, query_hw = self._to_batch_first(query, name="query") 
        key_tokens, _, key_hw = self._to_batch_first(key, name="key")
        value_tokens, _, value_hw = self._to_batch_first(value, name="value")  
  
        if key_hw is not None:
            hw = key_hw
        elif value_hw is not None:    
            hw = value_hw    
        else:
            hw = self._validate_hw_from_tokens(key_tokens, spatial_shape)  

        out = self._forward_batch_first_tokens(query_tokens, key_tokens, value_tokens, hw, attn_mask=attn_mask)     
        return self._restore_layout(out, query_layout, query_hw)    


if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m" 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel, height, width = 1, 64, 20, 20  
    inputs = torch.randn((batch_size, channel, height, width)).to(device)    
    
    module = WCA(channel, h=8).to(device) 

    outputs = module(inputs)   
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)    

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width),  
                                     output_as_string=True,   
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)   
