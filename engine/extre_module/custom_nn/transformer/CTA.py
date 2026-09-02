'''     
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/IJCAI2024-Channel Transposed Attention.png
论文链接：https://www.ijcai.org/proceedings/2024/0081.pdf
'''  

import warnings
warnings.filterwarnings('ignore')  
from calflops import calculate_flops
    
import torch
import torch.nn as nn    
import torch.nn.functional as F
from einops import reduce

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
    
class Channel_Transposed_Attention(nn.Module): 
    # The implementation builds on XCiT code https://github.com/facebookresearch/xcit  
    """ Channel Transposed Self-Attention
    Args:     
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 6
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set.
        attn_drop (float): Attention dropout rate. Default: 0.0 
        drop_path (float): Stochastic depth rate. Default: 0.0  
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
  
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim) 
        self.proj_drop = nn.Dropout(proj_drop)    
     
        self.channel_projection = ChannelProjection(dim) 
        self.spatial_projection = SpatialProjection(dim) 
        self.dwconv = nn.Sequential(   
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
        )

        # self.frequency_projection = FrequencyProjection(dim)
     
    def forward(self, x):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)   
        """
        B, C, H, W = x.shape
        N = H * W  
        x = x.flatten(2).permute(0, 2, 1).contiguous()
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)    
        qkv = qkv.permute(2, 0, 3, 1, 4) #  3 B num_heads N D   
        q, k, v = qkv[0], qkv[1], qkv[2]    
     
        #  B num_heads D N    
        q = q.transpose(-2, -1)   
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)  

        v_ = v.reshape(B, C, N).contiguous().view(B, C, H, W) 

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
    
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
   
        # attention output   
        attened_x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
 
        # convolution output    
        conv_x = self.dwconv(v_)
 
        # C-Map (before sigmoid)
        attention_reshape = attened_x.transpose(-2,-1).contiguous().view(B, C, H, W)  
        channel_map = self.channel_projection(attention_reshape)    
        attened_x = attened_x + channel_map.permute(0, 2, 3, 1).contiguous().view(B, N, C)    
        channel_map = reduce(channel_map, 'b c h w -> b c 1 1', 'mean')

        # S-Map (before sigmoid)    
        spatial_map = self.spatial_projection(conv_x).permute(0, 2, 3, 1).contiguous().view(B, N, C)
     
        # S-I    
        attened_x = attened_x * torch.sigmoid(spatial_map)
        # C-I
        conv_x = conv_x * torch.sigmoid(channel_map)
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, N, C)
     
        x = attened_x + conv_x

        x = self.proj(x) 

        x = self.proj_drop(x)  
     
        return x.permute(0, 2, 1).view(B, C, H, W).contiguous()
    
if __name__ == '__main__':     
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    batch_size, channel, height, width = 1, 16, 20, 20  
    inputs = torch.randn((batch_size, channel, height, width)).to(device)
  
    # 此模块不支持多尺度训练，reso的参数是一个元组，其为当前特征图的height,width。
    module = Channel_Transposed_Attention(channel, num_heads=8).to(device)   
  
    outputs = module(inputs)   
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
    
    print(ORANGE) 
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width),    
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)  
    print(RESET)