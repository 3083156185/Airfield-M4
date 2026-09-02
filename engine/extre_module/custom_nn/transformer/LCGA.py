'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/TGRS2026-LCGA.png
engine/extre_module/module_images/TGRS2026-LCGA.md  
论文链接：https://ieeexplore.ieee.org/document/11373098 
'''

import warnings 
warnings.filterwarnings('ignore')
from calflops import calculate_flops 
   
import torch
import torch.nn as nn    
from torch.nn import functional as F  
 

def img2windows(img, h_sp, w_sp):
    b, c, h, w = img.shape
    img_reshape = img.view(b, c, h // h_sp, h_sp, w // w_sp, w_sp)
    img_perm = (
        img_reshape.permute(0, 2, 4, 3, 5, 1)     
        .contiguous()  
        .reshape(-1, h_sp * w_sp, c)
    ) 
    return img_perm
  
     
def windows2img(img_splits_hw, h_sp, w_sp, h, w): 
    b = int(img_splits_hw.shape[0] / (h * w / h_sp / w_sp))
    img = img_splits_hw.view(b, h // h_sp, w // w_sp, h_sp, w_sp, -1) 
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return img
   

def _meshgrid_ij(*tensors):     
    try:  
        return torch.meshgrid(*tensors, indexing="ij")
    except TypeError: 
        return torch.meshgrid(*tensors)


class DynamicPosBias(nn.Module):    
    """Dynamic relative position bias used inside window attention.""" 
   
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
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos3 = nn.Sequential(  
            nn.LayerNorm(self.pos_dim),   
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads),
        )   
  
    def forward(self, biases):
        if self.residual:    
            pos = self.pos_proj(biases)    
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)    
        else:     
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))  
        return pos     

   
class WindowAttention(nn.Module):
    def __init__(     
        self,
        dim,
        idx,    
        split_size=(8, 8),   
        dim_out=None,
        num_heads=6,
        attn_drop=0.0,  
        proj_drop=0.0,    
        qk_scale=None,
        position_bias=True,   
    ): 
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
            h_sp, w_sp = self.split_size[0], self.split_size[1] 
        elif idx == 1:  
            w_sp, h_sp = self.split_size[0], self.split_size[1]
        else:
            raise ValueError(f"Unsupported window attention branch index: {idx}")
        self.H_sp = h_sp
        self.W_sp = w_sp 

        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
            position_bias_h = torch.arange(1 - self.H_sp, self.H_sp)    
            position_bias_w = torch.arange(1 - self.W_sp, self.W_sp)
            biases = torch.stack(_meshgrid_ij(position_bias_h, position_bias_w))
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()   
            self.register_buffer("rpe_biases", biases)
    
            coords_h = torch.arange(self.H_sp)   
            coords_w = torch.arange(self.W_sp)  
            coords = torch.stack(_meshgrid_ij(coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()   
            relative_coords[:, :, 0] += self.H_sp - 1 
            relative_coords[:, :, 1] += self.W_sp - 1
            relative_coords[:, :, 0] *= 2 * self.W_sp - 1 
            relative_position_index = relative_coords.sum(-1)   
            self.register_buffer("relative_position_index", relative_position_index)    

        self.curvature_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        self.alpha = nn.Parameter(torch.zeros(1))    
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.attn_drop = nn.Dropout(attn_drop)  

    def im2win(self, x, h, w):
        b, _, c = x.shape
        x = x.transpose(-2, -1).contiguous().view(b, c, h, w)    
        x = img2windows(x, self.H_sp, self.W_sp) 
        x = ( 
            x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, c // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()   
        ) 
        return x
    
    def forward(self, qkv, h, w, mask=None):
        q, k, v = qkv[0], qkv[1], qkv[2]

        b, l, c = q.shape    
        assert l == h * w, "flatten img_tokens has wrong size"
    
        q = self.im2win(q, h, w)
        k = self.im2win(k, h, w)
        v = self.im2win(v, h, w)
   
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale  
  
        if self.position_bias:   
            pos = self.pos(self.rpe_biases)
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1 
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn_logits = attn_logits + relative_position_bias.unsqueeze(0)
   
        n = attn_logits.shape[3]

        if mask is not None:
            n_w = mask.shape[0]   
            attn_logits = (
                attn_logits.view(b, n_w, self.num_heads, n, n)
                + mask.unsqueeze(1).unsqueeze(0)   
            )
            attn_logits = attn_logits.view(-1, self.num_heads, n, n)
  
        x2d = qkv[0].transpose(1, 2).view(b, c, h, w)
        curvature = self.curvature_conv(x2d).mean(dim=1, keepdim=True).view(b, -1)
        curvature = F.layer_norm(curvature, curvature.shape[1:])    
        curvature_map = curvature.view(b, 1, h, w) 
        curvature_patches = img2windows(curvature_map, self.H_sp, self.W_sp)
        curvature_win = curvature_patches.view(-1, self.H_sp * self.W_sp)    
        curvature_win = curvature_win.unsqueeze(1).unsqueeze(-1)    
 
        modulated_logits = attn_logits * (1 + self.alpha * curvature_win)
  
        attn_std = F.softmax(attn_logits, dim=-1, dtype=attn_logits.dtype)
        attn_cga = F.softmax(modulated_logits, dim=-1, dtype=attn_logits.dtype)
        attn = self.beta * attn_std + (1 - self.beta) * attn_cga
        attn = self.attn_drop(attn) 

        x = attn @ v 
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, c)
        x = windows2img(x, self.H_sp, self.W_sp, h, w)
        return x


class LCGA(nn.Module):
    """
    Local Curvature-Guided Attention (LCGA).
    """
    
    def __init__(     
        self,     
        dim,    
        num_heads=8,
        split_size=(2, 4),   
        shift_size=(1, 2),   
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,    
        attn_drop=0.0,    
        idx=0,
        reso=64,     
        rs_id=0,  
    ):  
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size    
        self.shift_size = shift_size
        self.idx = idx
        self.rs_id = rs_id
        self.patches_resolution = reso    
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  
   
        assert 0 <= self.shift_size[0] < self.split_size[0], "shift_size must in 0-split_size0"
        assert 0 <= self.shift_size[1] < self.split_size[1], "shift_size must in 0-split_size1"     

        self.branch_num = 2  
   
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)   
 
        self.attns = nn.ModuleList(  
            [   
                WindowAttention(
                    dim // 2,
                    idx=i,
                    split_size=split_size, 
                    num_heads=num_heads // 2,
                    dim_out=dim // 2,   
                    qk_scale=qk_scale,   
                    attn_drop=attn_drop,
                    proj_drop=drop,  
                    position_bias=True,
                )
                for i in range(self.branch_num)
            ]    
        )

        if (self.rs_id % 2 == 0 and self.idx > 0 and (self.idx - 2) % 4 == 0) or (
            self.rs_id % 2 != 0 and self.idx % 4 == 0
        ):    
            attn_mask = self.calculate_mask(self.patches_resolution, self.patches_resolution)   
            self.register_buffer("attn_mask_0", attn_mask[0])  
            self.register_buffer("attn_mask_1", attn_mask[1])  
        else:    
            self.register_buffer("attn_mask_0", None)
            self.register_buffer("attn_mask_1", None)

        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def calculate_mask(self, h, w):   
        img_mask_0 = torch.zeros((1, h, w, 1))   
        img_mask_1 = torch.zeros((1, h, w, 1))   
        h_slices_0 = (     
            slice(0, -self.split_size[0]),
            slice(-self.split_size[0], -self.shift_size[0]),  
            slice(-self.shift_size[0], None),
        ) 
        w_slices_0 = (
            slice(0, -self.split_size[1]),
            slice(-self.split_size[1], -self.shift_size[1]),   
            slice(-self.shift_size[1], None), 
        )

        h_slices_1 = (   
            slice(0, -self.split_size[1]),   
            slice(-self.split_size[1], -self.shift_size[1]),  
            slice(-self.shift_size[1], None),     
        )
        w_slices_1 = (
            slice(0, -self.split_size[0]),
            slice(-self.split_size[0], -self.shift_size[0]),  
            slice(-self.shift_size[0], None),
        )     
        cnt = 0
        for h_item in h_slices_0:
            for w_item in w_slices_0:  
                img_mask_0[:, h_item, w_item, :] = cnt
                cnt += 1    
        cnt = 0     
        for h_item in h_slices_1:
            for w_item in w_slices_1: 
                img_mask_1[:, h_item, w_item, :] = cnt
                cnt += 1

        img_mask_0 = img_mask_0.view(
            1,     
            h // self.split_size[0],
            self.split_size[0],
            w // self.split_size[1],
            self.split_size[1],
            1,
        )  
        img_mask_0 = (
            img_mask_0.permute(0, 1, 3, 2, 4, 5) 
            .contiguous()
            .view(-1, self.split_size[0], self.split_size[1], 1)
        )
        mask_windows_0 = img_mask_0.view(-1, self.split_size[0] * self.split_size[1])
        attn_mask_0 = mask_windows_0.unsqueeze(1) - mask_windows_0.unsqueeze(2)
        attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 != 0, float(-100.0)).masked_fill(   
            attn_mask_0 == 0, float(0.0)     
        )   

        img_mask_1 = img_mask_1.view(
            1, 
            h // self.split_size[1],
            self.split_size[1], 
            w // self.split_size[0],    
            self.split_size[0],
            1,
        )     
        img_mask_1 = (    
            img_mask_1.permute(0, 1, 3, 2, 4, 5)    
            .contiguous()
            .view(-1, self.split_size[1], self.split_size[0], 1)    
        )  
        mask_windows_1 = img_mask_1.view(-1, self.split_size[1] * self.split_size[0])
        attn_mask_1 = mask_windows_1.unsqueeze(1) - mask_windows_1.unsqueeze(2) 
        attn_mask_1 = attn_mask_1.masked_fill(attn_mask_1 != 0, float(-100.0)).masked_fill(
            attn_mask_1 == 0, float(0.0)
        )   

        return attn_mask_0, attn_mask_1

    def forward(self, x):
        b, c, h, w = x.shape  
        l = h * w
        x = x.flatten(2).transpose(1, 2).contiguous() 
 
        qkv = self.qkv(x).reshape(b, -1, 3, c).permute(2, 0, 1, 3)   
        v = qkv[2].transpose(-2, -1).contiguous().view(b, c, h, w)  

        max_split_size = max(self.split_size[0], self.split_size[1])   
        pad_l = pad_t = 0     
        pad_r = (max_split_size - w % max_split_size) % max_split_size   
        pad_b = (max_split_size - h % max_split_size) % max_split_size    
    
        qkv = qkv.reshape(3 * b, h, w, c).permute(0, 3, 1, 2)
        qkv = F.pad(qkv, (pad_l, pad_r, pad_t, pad_b)).reshape(3, b, c, -1).transpose(-2, -1)  
        padded_h = pad_b + h 
        padded_w = pad_r + w    
        padded_l = padded_h * padded_w   
    
        if (self.rs_id % 2 == 0 and self.idx > 0 and (self.idx - 2) % 4 == 0) or (   
            self.rs_id % 2 != 0 and self.idx % 4 == 0
        ):
            qkv = qkv.view(3, b, padded_h, padded_w, c)    
            qkv_0 = torch.roll(   
                qkv[:, :, :, :, : c // 2],    
                shifts=(-self.shift_size[0], -self.shift_size[1]),
                dims=(2, 3),    
            )     
            qkv_0 = qkv_0.view(3, b, padded_l, c // 2)
            qkv_1 = torch.roll( 
                qkv[:, :, :, :, c // 2 :],
                shifts=(-self.shift_size[1], -self.shift_size[0]),
                dims=(2, 3),    
            )  
            qkv_1 = qkv_1.view(3, b, padded_l, c // 2)
     
            if self.patches_resolution != padded_h or self.patches_resolution != padded_w:
                mask_tmp = self.calculate_mask(padded_h, padded_w) 
                x1_shift = self.attns[0](qkv_0, padded_h, padded_w, mask=mask_tmp[0].to(x.device))
                x2_shift = self.attns[1](qkv_1, padded_h, padded_w, mask=mask_tmp[1].to(x.device))
            else:
                x1_shift = self.attns[0](qkv_0, padded_h, padded_w, mask=self.attn_mask_0)     
                x2_shift = self.attns[1](qkv_1, padded_h, padded_w, mask=self.attn_mask_1)
  
            x1 = torch.roll(
                x1_shift,    
                shifts=(self.shift_size[0], self.shift_size[1]),     
                dims=(1, 2),    
            )
            x2 = torch.roll(
                x2_shift,     
                shifts=(self.shift_size[1], self.shift_size[0]),    
                dims=(1, 2),
            ) 
            x1 = x1[:, :h, :w, :].reshape(b, l, c // 2)
            x2 = x2[:, :h, :w, :].reshape(b, l, c // 2)
            attended_x = torch.cat([x1, x2], dim=2)  
        else:     
            x1 = self.attns[0](qkv[:, :, :, : c // 2], padded_h, padded_w)[:, :h, :w, :].reshape(
                b, l, c // 2
            )  
            x2 = self.attns[1](qkv[:, :, :, c // 2 :], padded_h, padded_w)[:, :h, :w, :].reshape(
                b, l, c // 2   
            )    
            attended_x = torch.cat([x1, x2], dim=2)  
 
        lcm = self.get_v(v)    
        lcm = lcm.permute(0, 2, 3, 1).contiguous().view(b, l, c)    
   
        x = attended_x + lcm  
        x = self.proj(x)
        x = self.proj_drop(x)  
        return x.transpose(1, 2).reshape(b, c, h, w).contiguous()   


if __name__ == '__main__':     
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"     
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   
    batch_size, channel, height, width = 1, 64, 20, 20   
    inputs = torch.randn((batch_size, channel, height, width)).to(device)  
    
    module = LCGA(channel, num_heads=8).to(device).eval()
  
    outputs = module(inputs)    
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET) 
   
    print(ORANGE)   
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,   
                                     print_detailed=True)
    print(RESET)
