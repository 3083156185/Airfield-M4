'''    
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/CVPR2026-WDAM.png    
engine/extre_module/module_images/CVPR2026-WDAM.md
论文链接：https://arxiv.org/pdf/2603.22794
'''

import warnings
warnings.filterwarnings('ignore')    
from calflops import calculate_flops

import torch    
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse

class WDAM(nn.Module):  
    def __init__(self, dim, num_heads=8, window_size=5, shift_size=2, bias=False):     
        super(WDAM, self).__init__()
        self.dim = dim    
        self.num_heads = num_heads
        self.shift_size = shift_size 
        self.window_size = window_size  

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))   
        # DWT 
        self.dwt = DWTForward(J=1, wave='haar')
        self.idwt = DWTInverse(wave='haar')
     
        # 高频信息卷积     
        self.high_conv = nn.Sequential(
            nn.Conv2d(dim*2, dim*2, kernel_size=3, padding=1, groups=2, bias=bias),
            nn.ReLU(inplace=True),  
            nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias),     
            nn.ReLU(inplace=True)    
        )   
        self.high_out = nn.Sequential(
            nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, groups=3, bias=bias),
            nn.ReLU(inplace=True)
        ) 

        # QKV for low-frequency attention
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)    
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # 相对位置偏置（Swin Transformer style）
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*window_size-1)*(2*window_size-1), num_heads)    
        )   
        coords = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size)))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2*window_size - 1
        relative_position_index = relative_coords.sum(-1)    
        self.register_buffer("relative_position_index", relative_position_index)
     
    # ========================= 工具函数 =========================
    def window_partition(self, x):
        B, C, H, W = x.shape  
        ws = self.window_size
        x = x.view(B, C, H//ws, ws, W//ws, ws)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()     
        x = x.view(-1, C, ws, ws) 
        return x
    
    def window_reverse(self, windows, H, W):
        B = int(windows.shape[0] / (H * W / self.window_size / self.window_size))
        C = windows.shape[1]   
        ws = self.window_size
        x = windows.view(B, H//ws, W//ws, C, ws, ws)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()     
        x = x.view(B, C, H, W)
        return x 
 
    def shift(self, x, shift_size):   
        if shift_size > 0:
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(2, 3))     
        return x
    
    def reverse_shift(self, x, shift_size):  
        if shift_size > 0:
            x = torch.roll(x, shifts=(shift_size, shift_size), dims=(2, 3))
        return x
    
    def window_attention(self, q, k, v):
        """  
        q,k,v: (B_win, num_heads, head_dim, N) 
        返回: (B_win, num_heads, head_dim, N)  
        """  
        # q^T * k -> (B_win, num_heads, N, N)
        q = F.normalize(q, dim=-2)  # 沿 head_dim 归一化  
        k = F.normalize(k, dim=-2)
        attn = torch.matmul(q.transpose(-2, -1), k)  # (B_win, head, N, N)

        # 相对位置偏置     
        N = self.window_size * self.window_size 
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]    
        relative_position_bias = relative_position_bias.view(N, N, -1).permute(2,0,1).unsqueeze(0)  # (1, head, N, N)
        attn = attn + relative_position_bias     
     
        attn = attn * self.temperature     
        attn = attn.softmax(dim=-1)   
     
        # 注意力矩阵乘 v -> (B_win, head, head_dim, N)    
        out = torch.matmul(v, attn.transpose(-2, -1))  # (B_win, head, head_dim, N)
        return out

   
    def forward(self, x):
        B, C, H, W = x.shape    
  
        # ----------- DWT -----------     
        LL, Yh = self.dwt(x) 
        Yh = Yh[0]
        LH, HL, HH = Yh[:, :, 0, :, :], Yh[:, :, 1, :, :], Yh[:, :, 2, :, :]
    
        # ----------- 高频信息卷积 -----------
        filter_hv = self.high_conv(torch.cat([LH, HL], dim=1))

        # ----------- QKV & 加权 V -----------  
        qkv = self.qkv_dwconv(self.qkv(LL))
        q, k, v_inp = qkv.chunk(3, dim=1)  
        v = v_inp * filter_hv + v_inp     

        # ----------- Shifted Window Attention ----------- 
        x_shifted = self.shift(LL, self.shift_size)
        q = self.window_partition(x_shifted)  
        k = self.window_partition(x_shifted) 
        v = self.window_partition(v)
     
        # reshape for multi-head
        B_win, Cq, ws, _ = q.shape
        q = q.view(B_win, self.num_heads, Cq//self.num_heads, ws*ws)
        k = k.view(B_win, self.num_heads, Cq//self.num_heads, ws*ws)   
        v = v.view(B_win, self.num_heads, Cq//self.num_heads, ws*ws)  

        out = self.window_attention(q, k, v)     
        out = out.view(B_win, Cq, ws, ws)    
        out = self.window_reverse(out, H//2, W//2)  
        out = self.reverse_shift(out, self.shift_size)   
        out = self.project_out(out)
 
        # ----------- 高频信息重建 -----------    
        Yh = self.high_out(torch.cat([LH, HL, HH], dim=1))
        LH, HL, HH = Yh.chunk(3, dim=1) 
        Yh = torch.stack([LH, HL, HH], dim=2)   
        x_hat = self.idwt((out, [Yh]))

        return x_hat  
    
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    batch_size, channel, height, width = 1, 64, 20, 20
    inputs = torch.randn((batch_size, channel, height, width)).to(device)    

    module = WDAM(channel, num_heads=8).to(device)
    
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)   
    flops, macs, _ = calculate_flops(model=module,    
                                     input_shape=(batch_size, channel, height, width),
                                     output_as_string=True,  
                                     output_precision=4,  
                                     print_detailed=True)
    print(RESET)
