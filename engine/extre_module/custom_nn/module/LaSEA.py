''' 
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/TGRS2025-LaSEA.png   
engine/extre_module/module_images/TGRS2025-LaSEA.md     
论文链接：https://arxiv.org/pdf/2601.16428
'''   
    
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')     

import warnings  
warnings.filterwarnings('ignore')
from calflops import calculate_flops     

import math, random    
from typing import Callable, Optional    
import torch
import torch.nn as nn
from torch import Tensor  

from engine.extre_module.ultralytics_nn.conv import Conv
   
def make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:   
    if min_value is None:     
        min_value = divisor    
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor  
    return new_v

 
def shuffle_tensor(feature: Tensor, mode: int = 1) -> Tensor:
    batch, channels, height, width = feature.shape
    if mode == 1:
        feature = feature.flatten(2)
        indices = torch.randperm(feature.shape[-1], device=feature.device)  
        feature = feature[:, :, indices] 
        return feature.reshape(batch, channels, height, width)
 
    h_indices = torch.randperm(height, device=feature.device) 
    w_indices = torch.randperm(width, device=feature.device) 
    feature = feature[:, :, h_indices]
    feature = feature[:, :, :, w_indices]   
    return feature
     
   
NORM_LAYER_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,  
    nn.BatchNorm3d,   
    nn.SyncBatchNorm,  
    nn.LayerNorm,  
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.GroupNorm, 
)
 
   
def init_weight(module: nn.Module) -> None:    
    if module is None:  
        return
    if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)):    
        nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        if module.bias is not None:  
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight) 
            if fan_in != 0:  
                bound = 1 / math.sqrt(fan_in)    
                nn.init.uniform_(module.bias, -bound, bound)  
        return
    if isinstance(module, NORM_LAYER_TYPES):
        if module.weight is not None:    
            nn.init.ones_(module.weight)     
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        return
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))    
        if module.bias is not None:  
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight) 
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0  
            nn.init.uniform_(module.bias, -bound, bound)

  
class BaseConv2d(nn.Module):
    def __init__(
        self, 
        in_channels: int,  
        out_channels: int,
        kernel_size: int,     
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,     
        bias: Optional[bool] = None,
        use_bn: bool = False,
        act_layer: Optional[Callable[..., nn.Module]] = None,     
        dilation: int = 1, 
        momentum: float = 0.1, 
    ) -> None:  
        super().__init__()
        if padding is None:     
            padding = int((kernel_size - 1) // 2 * dilation)
        if bias is None:    
            bias = not use_bn 

        self.conv = nn.Conv2d(    
            in_channels,
            out_channels,   
            kernel_size,    
            stride,
            padding,
            dilation,   
            groups,    
            bias,
        )   
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=momentum) if use_bn else nn.Identity()
  
        if act_layer is None:
            self.act = None 
        elif isinstance(act_layer(), nn.Sigmoid):    
            self.act = act_layer()
        else:   
            self.act = act_layer(inplace=True)    
  
        self.apply(init_weight)    
 
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x) 
        if self.act is not None:
            x = self.act(x)    
        return x


class Attention(nn.Module):
    def __init__(     
        self,    
        in_channels: int,
        hidden_channels: Optional[int] = None,
        squeeze_factor: int = 4,     
        pool_res: Optional[list[int]] = None,
        act: Callable[..., nn.Module] = nn.ReLU,
        scale_act: Callable[..., nn.Module] = nn.Sigmoid,   
        moc_order: bool = True,
    ) -> None:   
        super().__init__()    
        if pool_res is None:   
            pool_res = [1, 2, 3]
        if hidden_channels is None:
            hidden_channels = max(make_divisible(in_channels // squeeze_factor, 8), 32)  
    
        all_pool_res = list(pool_res)
        if 1 not in all_pool_res:    
            all_pool_res.append(1)     
        self.pools = nn.ModuleDict({str(k): nn.AdaptiveAvgPool2d(k) for k in all_pool_res})    
        self.pool_res = pool_res
        self.moc_order = moc_order  
        self.se_layer = nn.Sequential(
            BaseConv2d(in_channels, hidden_channels, 1, act_layer=act),
            BaseConv2d(hidden_channels, in_channels, 1, act_layer=scale_act), 
        )

    def random_sample(self, x: Tensor) -> Tensor:    
        if not self.training:
            return self.pools["1"](x)
    
        pool_keep = random.choice(self.pool_res)
        pooled_input = shuffle_tensor(x) if self.moc_order else x 
        attn_map = self.pools[str(pool_keep)](pooled_input)     
        if attn_map.shape[-1] > 1:
            attn_map = attn_map.flatten(2)     
            index = torch.randperm(attn_map.shape[-1], device=attn_map.device)[0]  
            attn_map = attn_map[:, :, index][:, :, None, None]   
        return attn_map 

    def forward(self, x: Tensor) -> Tensor:    
        attn_map = self.random_sample(x)     
        return x * self.se_layer(attn_map)


def channel_shuffle(x: Tensor, groups: int) -> Tensor: 
    batch_size, num_channels, height, width = x.shape     
    channels_per_group = num_channels // groups   
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(batch_size, -1, height, width)


class LaSEA(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:  
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels     
        self.conv_1 = Conv(in_channels, in_channels, 3, d=1, act=nn.ReLU)     
        self.conv_2 = Conv(in_channels, in_channels, 3, d=2, act=nn.ReLU)
        self.conv_3 = Conv(in_channels, in_channels, 3, d=3, act=nn.ReLU)
        self.conv_4 = Conv(in_channels, in_channels, 3, d=4, act=nn.ReLU)
        self.fuse = Conv(in_channels * 4, in_channels, 3, act=nn.ReLU) 
        self.mca = Attention(in_channels=in_channels, hidden_channels=16)
        self.conv1x1 = Conv(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()    
     
    def forward(self, x: Tensor) -> Tensor:     
        identity = x 
        c1 = self.conv_1(x)     
        c2 = self.conv_2(x)  
        c3 = self.conv_3(x)
        c4 = self.conv_4(x)
        fused = torch.cat([c1, c2, c3, c4], dim=1)    
        fused = channel_shuffle(fused, groups=4)
        fused = self.fuse(fused) 
        fused = self.mca(fused)
        return self.conv1x1(fused + identity)


if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32   
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)    

    module = LaSEA(in_channel, out_channel).to(device)

    outputs = module(inputs)    
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)  

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),     
                                     output_as_string=True,   
                                     output_precision=4,   
                                     print_detailed=True)
    print(RESET)
