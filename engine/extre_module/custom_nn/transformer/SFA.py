'''    
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/IJCAI2024-Spatial Frequency Attention.png
论文链接：https://www.ijcai.org/proceedings/2024/0081.pdf  
'''

import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops  
  
import torch    
import torch.nn as nn
import torch.nn.functional as F   
from einops import reduce
  
def img2windows(img, H_sp, W_sp):
    """     
    Input: Image (B, C, H, W)  
    Output: Window Partition (B', N, C)
    """   
    B, C, H, W = img.shape 
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)     
    return img_perm
 
def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """    
    Input: Window Partition (B', N, C)   
    Output: Image (B, H, W, C)
    """   
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))     
    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)     
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)    
    return img     

class FrequencyProjection(nn.Module):    
    """ Frequency Projection.
    Args:
        dim (int): input channels.
    """
    def __init__(self, dim):
        super().__init__()  
        self.conv_1 = nn.Conv2d(dim, dim // 2, 1, 1, 0) 
        self.act = nn.GELU()   
        self.res_2 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),   
            nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
            nn.GELU()
        ) 
        self.conv_out = nn.Conv2d(dim // 2, dim, 1, 1, 0)

    def forward(self, x):     
        """
        Input: x: (B, C, H, W)  
        Output: x: (B, C, H, W)
        """
        res = x
        x = self.conv_1(x)    
        x1, x2 = x.chunk(2, dim=1)  
        out = torch.cat((self.act(x1), self.res_2(x2)), dim=1)
        out = self.conv_out(out)
        return out + res
 
class ChannelProjection(nn.Module):   
    """ Channel Projection.   
    Args:
        dim (int): input channels.
    """
    def __init__(self, dim):     
        super().__init__() 
        self.pro_in = nn.Conv2d(dim, dim // 6, 1, 1, 0)    
        self.CI1 = nn.Sequential(  
            nn.AdaptiveAvgPool2d(1),   
            nn.Conv2d(dim // 6, dim // 6, kernel_size=1) 
        )   
        self.CI2 = nn.Sequential(  
            nn.Conv2d(dim // 6, dim // 6, kernel_size=3, stride=1, padding=1, groups=dim // 6),     
            nn.Conv2d(dim // 6, dim // 6, 7, stride=1, padding=9, groups=dim // 6, dilation=3),
            nn.Conv2d(dim // 6, dim // 6, kernel_size=1)
        )
        self.pro_out = nn.Conv2d(dim // 6, dim, kernel_size=1)    
 
    def forward(self, x):  
        """ 
        Input: x: (B, C, H, W)    
        Output: x: (B, C, H, W)
        """ 
        x = self.pro_in(x)
        res = x
        ci1 = self.CI1(x) 
        ci2 = self.CI2(x)
        out = self.pro_out(res * ci1 * ci2)
        return out 

class SpatialProjection(nn.Module):     
    """ Spatial Projection.    
    Args:
        dim (int): input channels.
    """    
    def __init__(self, dim):
        super().__init__()
        self.pro_in = nn.Conv2d(dim, dim // 2, 1, 1, 0)  
        self.dwconv = nn.Conv2d(dim // 2,  dim // 2, kernel_size=3, stride=1, padding=1, groups= dim // 2)     
        self.pro_out = nn.Conv2d(dim // 4, dim, kernel_size=1)
 
    def forward(self, x):
        """   
        Input: x: (B, C, H, W) 
        Output: x: (B, C, H, W)
        """
        x = self.pro_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2     
        x = self.pro_out(x)
        return x
   
class DynamicPosBias(nn.Module):     
    # The implementation builds on Crossformer code https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py
    """ Dynamic Relative Position Bias.
    Args:
        dim (int): Number of input channels.   
        num_heads (int): Number of attention heads.
        residual (bool):  If True, use residual strage to connect conv.
    """     
    def __init__(self, dim, num_heads, residual):
        super().__init__()     
        self.residual = residual 
        self.num_heads = num_heads
        self.pos_dim = dim // 4 
        self.pos_proj = nn.Linear(2, self.pos_dim)    
        self.pos1 = nn.Sequential(   
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True), 
            nn.Linear(self.pos_dim, self.pos_dim), 
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim), 
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential( 
            nn.LayerNorm(self.pos_dim),     
            nn.ReLU(inplace=True),  
            nn.Linear(self.pos_dim, self.num_heads)
        )    
    def forward(self, biases):  
        if self.residual:   
            pos = self.pos_proj(biases) # 2Gh-1 * 2Gw-1, heads
            pos = pos + self.pos1(pos)  
            pos = pos + self.pos2(pos)     
            pos = self.pos3(pos)  
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))    
        return pos  

class Spatial_Attention(nn.Module):   
    """ Spatial Self-Attention. 
    It supports rectangle window (containing square window). 
    Args:    
        dim (int): Number of input channels.
        idx (int): The indentix of window. (0/1)
        split_size (tuple(int)): Height and Width of spatial window.    
        dim_out (int | None): The dimension of the attention output. Default: None
        num_heads (int): Number of attention heads. Default: 6
        attn_drop (float): Dropout ratio of attention weight. Default: 0.0 
        proj_drop (float): Dropout ratio of output. Default: 0.0
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set
        position_bias (bool): The dynamic relative position bias. Default: True     
    """
    def __init__(self, dim, idx, split_size=[8,8], dim_out=None, num_heads=6, attn_drop=0., proj_drop=0., qk_scale=None, position_bias=True):   
        super().__init__()     
        self.dim = dim  
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx 
        self.position_bias = position_bias
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5   

        if idx == 0: 
            H_sp, W_sp = self.split_size[0], self.split_size[1]   
        elif idx == 1:
            W_sp, H_sp = self.split_size[0], self.split_size[1]
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp

        if self.position_bias:    
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)     
            # generate mother-set
            position_bias_h = torch.arange(1 - self.H_sp, self.H_sp)
            position_bias_w = torch.arange(1 - self.W_sp, self.W_sp)  
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()
            self.register_buffer('rpe_biases', biases)    
    
            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.H_sp)
            coords_w = torch.arange(self.W_sp)     
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)  
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()    
            relative_coords[:, :, 0] += self.H_sp - 1
            relative_coords[:, :, 1] += self.W_sp - 1   
            relative_coords[:, :, 0] *= 2 * self.W_sp - 1 
            relative_position_index = relative_coords.sum(-1)   
            self.register_buffer('relative_position_index', relative_position_index) 
     
        self.attn_drop = nn.Dropout(attn_drop)     

    def im2win(self, x, H, W):
        B, N, C = x.shape  
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)     
        # (b win_num_h win_num_w) (win_h win_w) c     
        # -> (b win_num_h win_num_w) (win_h win_w) num_heads d
        # -> (b win_num_h win_num_w) num_heads (win_h win_w) d   
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous() 
        return x 
     
    def forward(self, qkv, H, W, mask=None):
        """   
        Input: qkv: (B, 3*L, C), H, W, mask: (B, N, N), N is the window size 
        Output: x (B, H, W, C)     
        """
        q,k,v = qkv[0], qkv[1], qkv[2]   

        B, L, C = q.shape    
        assert L == H * W, "flatten img_tokens has wrong size"   

        # partition the q,k,v, image to window
        q = self.im2win(q, H, W) 
        k = self.im2win(k, H, W)     
        v = self.im2win(v, H, W)

        q = q * self.scale     
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N

        # calculate drpe 
        if self.position_bias:
            pos = self.pos(self.rpe_biases) 
            # select position bias   
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1)    
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0) 

        N = attn.shape[3]

        # use mask for shift window
        if mask is not None:  
            nW = mask.shape[0]
            attn = attn.view(B, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn) 
 
        x = (attn @ v) 
        x = x.transpose(1, 2).reshape(-1, self.H_sp* self.W_sp, C)  # B head N N @ B head N C  

        # merge the window, window to image     
        x = windows2img(x, self.H_sp, self.W_sp, H, W)  # B H' W' C   
 
        return x   

class Spatial_Frequency_Attention(nn.Module):
    # The implementation builds on CAT code https://github.com/Zhengchen1999/CAT
    """ Spatial Frequency Self-Attention
    Args:    
        dim (int): Number of input channels.   
        num_heads (int): Number of attention heads. Default: 6     
        split_size (tuple(int)): Height and Width of spatial window.    
        shift_size (tuple(int)): Shift size for spatial window.     
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True   
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set.     
        drop (float): Dropout rate. Default: 0.0
        attn_drop (float): Attention dropout rate. Default: 0.0
        b_idx (int): The index of Block    
    """    
    def __init__(self, dim, reso, num_heads=8,
                 split_size=[8,8], shift_size=[1,2], qkv_bias=False, qk_scale=None,  
                 drop=0., attn_drop=0., b_idx=0): 
        super().__init__()
        self.dim = dim    
        self.num_heads = num_heads   
        self.split_size = split_size   
        self.shift_size = shift_size
        self.b_idx  = b_idx   
        self.patches_resolution = reso
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)     
        self.hf = nn.Linear(dim, dim, bias=qkv_bias)
 
        assert 0 <= self.shift_size[0] < self.split_size[0], "shift_size must in 0-split_size0"
        assert 0 <= self.shift_size[1] < self.split_size[1], "shift_size must in 0-split_size1"   
 
        self.branch_num = 2    
     
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.dw_block = nn.Sequential(   
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        )
    
        self.attns = nn.ModuleList([
                Spatial_Attention(
                    dim//2, idx = i,  
                    split_size=split_size, num_heads=num_heads//2, dim_out=dim//2,   
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, position_bias=True)
                for i in range(self.branch_num)])
        if self.b_idx > 0 and (self.b_idx - 2) % 4 == 0:    
            attn_mask = self.calculate_mask(*self.patches_resolution)
            self.register_buffer("attn_mask_0", attn_mask[0])   
            self.register_buffer("attn_mask_1", attn_mask[1])    
        else:
            self.register_buffer("attn_mask_0", None)    
            self.register_buffer("attn_mask_1", None)
    
        self.channel_projection = ChannelProjection(dim)     
        self.spatial_projection = SpatialProjection(dim)
        self.frequency_projection = FrequencyProjection(dim)    

    def calculate_mask(self, H, W): 
        # The implementation builds on Swin Transformer code https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py    
        # calculate attention mask for shift window

        img_mask_0 = torch.zeros((1, H, W, 1))  # 1 H W 1 idx=0
        img_mask_1 = torch.zeros((1, H, W, 1))  # 1 H W 1 idx=1    
        h_slices_0 = (slice(0, -self.split_size[0]), 
                    slice(-self.split_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))   
        w_slices_0 = (slice(0, -self.split_size[1]), 
                    slice(-self.split_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))

        h_slices_1 = (slice(0, -self.split_size[1]),    
                    slice(-self.split_size[1], -self.shift_size[1]),     
                    slice(-self.shift_size[1], None))  
        w_slices_1 = (slice(0, -self.split_size[0]),
                    slice(-self.split_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        cnt = 0
        for h in h_slices_0:  
            for w in w_slices_0:
                img_mask_0[:, h, w, :] = cnt
                cnt += 1 
        cnt = 0    
        for h in h_slices_1:   
            for w in w_slices_1: 
                img_mask_1[:, h, w, :] = cnt   
                cnt += 1
 
        # calculate mask for window-0 
        img_mask_0 = img_mask_0.view(1, H // self.split_size[0], self.split_size[0], W // self.split_size[1], self.split_size[1], 1) 
        img_mask_0 = img_mask_0.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size[0], self.split_size[1], 1) # nW, sw[0], sw[1], 1  
        mask_windows_0 = img_mask_0.view(-1, self.split_size[0] * self.split_size[1])
        attn_mask_0 = mask_windows_0.unsqueeze(1) - mask_windows_0.unsqueeze(2)
        attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 != 0, float(-100.0)).masked_fill(attn_mask_0 == 0, float(0.0))   

        # calculate mask for window-1    
        img_mask_1 = img_mask_1.view(1, H // self.split_size[1], self.split_size[1], W // self.split_size[0], self.split_size[0], 1) 
        img_mask_1 = img_mask_1.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size[1], self.split_size[0], 1) # nW, sw[1], sw[0], 1    
        mask_windows_1 = img_mask_1.view(-1, self.split_size[1] * self.split_size[0])
        attn_mask_1 = mask_windows_1.unsqueeze(1) - mask_windows_1.unsqueeze(2)
        attn_mask_1 = attn_mask_1.masked_fill(attn_mask_1 != 0, float(-100.0)).masked_fill(attn_mask_1 == 0, float(0.0))  

        return attn_mask_0, attn_mask_1
    
    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W
        x = x.flatten(2).permute(0, 2, 1).contiguous()    
    
        hf = self.hf(x).transpose(-2,-1).contiguous().view(B, C, H, W)   

        hf = self.frequency_projection(hf)   

        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3) # 3, B, HW, C
        v = qkv[2].transpose(-2,-1).contiguous().view(B, C, H, W)  
 
        # image padding
        max_split_size = max(self.split_size[0], self.split_size[1])  
        pad_l = pad_t = 0   
        pad_r = (max_split_size - W % max_split_size) % max_split_size
        pad_b = (max_split_size - H % max_split_size) % max_split_size

        qkv = qkv.reshape(3*B, H, W, C).permute(0, 3, 1, 2) # 3B C H W   
        # hw填充
        qkv = F.pad(qkv, (pad_l, pad_r, pad_t, pad_b)).reshape(3, B, C, -1).transpose(-2, -1) # l r t b
        _H = pad_b + H
        _W = pad_r + W
        _L = _H * _W 
    
        # window-0 and window-1 on split channels [C/2, C/2]; for square windows (e.g., 8x8), window-0 and window-1 can be merged
        # shift in block: (0, 4, 8, ...), (2, 6, 10, ...), (0, 4, 8, ...), (2, 6, 10, ...), ...     
        if self.b_idx  > 0 and (self.b_idx  - 2) % 4 == 0:
            qkv = qkv.view(3, B, _H, _W, C)
            qkv_0 = torch.roll(qkv[:,:,:,:,:C//2], shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(2, 3)) 
            qkv_0 = qkv_0.view(3, B, _L, C//2)
            qkv_1 = torch.roll(qkv[:,:,:,:,C//2:], shifts=(-self.shift_size[1], -self.shift_size[0]), dims=(2, 3))
            qkv_1 = qkv_1.view(3, B, _L, C//2)   
   
            if self.patches_resolution[0] != _H or self.patches_resolution[1] != _W:   
                mask_tmp = self.calculate_mask(_H, _W)
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=mask_tmp[0].to(x.device)) 
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=mask_tmp[1].to(x.device))
            else:
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=self.attn_mask_0)
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=self.attn_mask_1)
     
            x1 = torch.roll(x1_shift, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))    
            x2 = torch.roll(x2_shift, shifts=(self.shift_size[1], self.shift_size[0]), dims=(1, 2)) 
            x1 = x1[:, :H, :W, :].reshape(B, L, C//2)
            x2 = x2[:, :H, :W, :].reshape(B, L, C//2)
            # attention output
            attened_x = torch.cat([x1,x2], dim=2) 
        else:
            x1 = self.attns[0](qkv[:,:,:,:C//2], _H, _W)[:, :H, :W, :].reshape(B, L, C//2)
            x2 = self.attns[1](qkv[:,:,:,C//2:], _H, _W)[:, :H, :W, :].reshape(B, L, C//2)
            # attention output
            attened_x = torch.cat([x1,x2], dim=2)

        conv_x = self.dw_block(v) 
     
        # C-Map (before sigmoid)
        channel_map = self.channel_projection(conv_x)    
        conv_x = conv_x  + channel_map    
        # high_fre info mix channel
        hf = hf + channel_map
        channel_map = reduce(channel_map, 'b c h w -> b c 1 1', 'mean').permute(0, 2, 3, 1).contiguous().view(B, 1, C)


        # S-Map (before sigmoid)
        attention_reshape = attened_x.transpose(-2,-1).contiguous().view(B, C, H, W)  
        spatial_map = self.spatial_projection(attention_reshape)
        # high_fre info mix spatial
        hf = hf + attention_reshape

        # C-I
        attened_x = attened_x * torch.sigmoid(channel_map) * torch.sigmoid(reduce(hf, 'b c h w -> b c 1 1', 'mean').permute(0, 2, 3, 1).contiguous().view(B, 1, C))  
        # S-I
        conv_x = torch.sigmoid(spatial_map) * conv_x * torch.sigmoid(hf)   
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, L, C)

        x = attened_x + conv_x + hf.permute(0, 2, 3, 1).contiguous().view(B, L, C)  
    
        x = self.proj(x)    
     
        x = self.proj_drop(x)     
  
        return x.permute(0, 2, 1).view(B, C, H, W).contiguous() 
    
if __name__ == '__main__':    
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel, height, width = 1, 16, 20, 20
    inputs = torch.randn((batch_size, channel, height, width)).to(device)    

    # 此模块不支持多尺度训练，reso的参数是一个元组，其为当前特征图的height,width。
    module = Spatial_Frequency_Attention(channel, reso=(height, width), num_heads=8).to(device)   

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
    
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width), 
                                     output_as_string=True, 
                                     output_precision=4,  
                                     print_detailed=True)    
    print(RESET)