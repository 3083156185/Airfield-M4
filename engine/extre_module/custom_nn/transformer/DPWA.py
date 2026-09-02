'''
本文件由BiliBili：魔傀面具整理   
engine/extre_module/module_images/TGRS2025-DPWA.png  
engine/extre_module/module_images/TGRS2025-DPWA.md   
论文链接：https://ieeexplore.ieee.org/document/11146454     
'''

import warnings     
warnings.filterwarnings('ignore') 
from calflops import calculate_flops
    
import torch 
import torch.nn as nn   
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_

def window_partition(x, window_size): 
    """
    Args: 
        x: (B, H, W, C)    
        window_size (int): window size

    Returns:     
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows  
    

def window_reverse(windows, window_size, H, W):     
    """   
    Args:
        windows: (num_windows*B, window_size, window_size, C)   
        window_size (int): Window size
        H (int): Height of image     
        W (int): Width of image

    Returns: 
        x: (B, H, W, C)    
    """    
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x   
   
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.     
    It supports both of shifted and non-shifted window.

    Args: 
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set    
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0     
    """ 

    def __init__(self, dim, window_size, num_heads, qk_scale=None, attn_drop=0.):    
 
        super().__init__()
        self.dim = dim 
        self.window_size = window_size  # Wh, Ww   
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
    
        # define a parameter table of relative position bias  
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
  
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])     
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2  
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1  
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww 
        self.register_buffer("relative_position_index", relative_position_index)    
        trunc_normal_(self.relative_position_bias_table, std=.02) 

        self.attn_drop = nn.Dropout(attn_drop)  
   
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):     
        """    
        Args:     
            q: queries with shape of (num_windows*B, N, C)   
            k: keys with shape of (num_windows*B, N, C)
            v: values with shape of (num_windows*B, N, C)    
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None    
        """    
        B_, N, C = q.shape
        # print(B_, N, C)   
        q = q.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)    
        k = k.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
     
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(     
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH  
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # print(relative_position_bias)    
        attn = attn + relative_position_bias.unsqueeze(0)     

        if mask is not None:
            # print(mask.size()) 
            nW = mask.shape[0] 
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)   
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)   
        else:
            attn = self.softmax(attn)    

        attn = self.attn_drop(attn)    
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    
        return x

class DPWA(nn.Module):
    """ DynamicParallelWindowAttention.     
     
    Args:
        dim (int): Number of input channels.     
        input_resolution (tuple[int]): Input resulotion.  
        num_heads (int): Number of attention heads.
        window_size (int): Window size.    
        shift_size (int): Shift size for SW-MSA.  
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.    
        drop (float, optional): Dropout rate. Default: 0.0 
        attn_drop (float, optional): Attention dropout rate. Default: 0.0 
        drop_path (float, optional): Stochastic depth rate. Default: 0.0 
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU   
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    
    def __init__(self, dim, num_heads, window_size=4, shift_size=2, alternate=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,    
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads    
        self.window_size = window_size     
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)   
        self.proj = nn.Linear(dim, dim)
        self.alternate = alternate
        self.attn = nn.ModuleList([
            WindowAttention(
                dim // 2, window_size=to_2tuple(self.window_size), num_heads=num_heads // 2,    
                qk_scale=qk_scale, attn_drop=attn_drop),
            WindowAttention(
                dim // 2, window_size=to_2tuple(self.window_size), num_heads=num_heads // 2,
                qk_scale=qk_scale, attn_drop=attn_drop),
        ])  
   
    def forward(self, x): 
        B, C, H, W = x.size()
        L = H * W
        
        x = x.flatten(2).permute(0, 2, 1) 

        attn_mask1 = None
        attn_mask2 = None

        if self.shift_size > 0:     
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),   
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),   
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0    
            for h in h_slices: 
                for w in w_slices:     
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1,
                                             self.window_size * self.window_size)
            attn_mask2 = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask2 = attn_mask2.masked_fill(  
                attn_mask2 != 0, float(-100.0)).masked_fill(attn_mask2 == 0, float(0.0)).to( 
                x.device)    
   
        x = self.norm1(x)
    
        # double attn
        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3).reshape(3 * B, H, W, C)
        # print(self.alternate)    
        if self.alternate == 0: 
            qkv_1 = qkv[:, :, :, : C // 2].reshape(3, B, H, W, C // 2)  
  
            if self.shift_size > 0:
                qkv_2 = torch.roll(qkv[:, :, :, C // 2:], shifts=(-self.shift_size, -self.shift_size),     
                                   dims=(1, 2)).reshape(
                    3, B, H, W, C // 2)
            else:  
                qkv_2 = qkv[:, :, :, C // 2:].reshape(3, B, H, W, C // 2)   
        else:
            qkv_1 = qkv[:, :, :, C // 2:].reshape(3, B, H, W, C // 2) 

            if self.shift_size > 0:
                qkv_2 = torch.roll(qkv[:, :, :, :C // 2], shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2)).reshape(
                    3, B, H, W, C // 2) 
            else:   
                qkv_2 = qkv[:, :, :, :C // 2].reshape(3, B, H, W, C // 2)  

        q1_windows, k1_windows, v1_windows = self.get_window_qkv(qkv_1)   
        q2_windows, k2_windows, v2_windows = self.get_window_qkv(qkv_2)

        x1 = self.attn[0](q1_windows, k1_windows, v1_windows, attn_mask1)     
        x2 = self.attn[1](q2_windows, k2_windows, v2_windows, attn_mask2) 

        x1 = window_reverse(x1.view(-1, self.window_size * self.window_size, C // 2), self.window_size, H, W)    
        x2 = window_reverse(x2.view(-1, self.window_size * self.window_size, C // 2), self.window_size, H, W)

        if self.shift_size > 0:     
            x2 = torch.roll(x2, shifts=(self.shift_size, self.shift_size), dims=(1, 2))    
        else:     
            x2 = x2 
        if self.alternate == 0: 
            x = torch.cat([x1.reshape(B, H * W, C // 2), x2.reshape(B, H * W, C // 2)], dim=2)
        else:
            x = torch.cat([x2.reshape(B, H * W, C // 2), x1.reshape(B, H * W, C // 2)], dim=2)
        x = self.proj(x)     

        return x.permute(0, 2, 1).reshape((B, C, H, W))
    
    def get_window_qkv(self, qkv):  
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, H, W, C
        C = q.shape[-1]
        q_windows = window_partition(q, self.window_size).view(-1, self.window_size * self.window_size,    
                                                               C)  # nW*B, window_size*window_size, C
        k_windows = window_partition(k, self.window_size).view(-1, self.window_size * self.window_size,
                                                               C)  # nW*B, window_size*window_size, C
        v_windows = window_partition(v, self.window_size).view(-1, self.window_size * self.window_size,
                                                               C)  # nW*B, window_size*window_size, C
        return q_windows, k_windows, v_windows     

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    batch_size, channel, height, width = 1, 512, 20, 20 
    inputs = torch.randn((batch_size, channel, height, width)).to(device)

    # alternate 参数控制 是否反转两个流的注意力类型 设置0或者1  
    module = DPWA(channel, num_heads=8, alternate=0).to(device)
     
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)   
    print(RESET)