'''
本文件由BiliBili：魔傀面具整理
  
论文链接：
'''
 
import warnings     
warnings.filterwarnings('ignore')    
from calflops import calculate_flops   

import torch
import torch.nn as nn
import torch.nn.functional as F    

class MaskUnitAttention(nn.Module):
    """
    Computes either Mask Unit or Global Attention. Also is able to perform q pooling.     

    Note: this assumes the tokens have already been flattened and unrolled into mask units.    
    See `Unroll` for more details.
    """  
  
    def __init__(   
        self,   
        dim: int,
        heads: int,     
        q_stride: int = 1, 
        window_size: int = 0,
        use_mask_unit_attn: bool = False, 
    ): 
        """
        Args:    
        - dim, dim_out: The input and output feature dimensions.
        - heads: The number of attention heads. 
        - q_stride: If greater than 1, pool q with this stride. The stride should be flattened (e.g., 2x2 = 4).
        - window_size: The current (flattened) size of a mask unit *after* pooling (if any).   
        - use_mask_unit_attn: Use Mask Unit or Global Attention.
        """
        super().__init__()
     
        self.dim = dim 
        self.heads = heads
        self.q_stride = q_stride   
     
        self.head_dim = dim // heads 
        self.scale = (self.head_dim) ** -0.5     
   
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)  

        self.window_size = window_size    
        self.use_mask_unit_attn = use_mask_unit_attn
    
    @staticmethod    
    def _to_nlc(x):  
        if x.dim() == 4: 
            b, c, h, w = x.shape
            return x.flatten(2).transpose(1, 2).contiguous(), (b, c, h, w)
        if x.dim() == 3:
            return x, None
        raise ValueError(f"MaskUnitAttention expects BCHW or BNC input, got {tuple(x.shape)}")    

    @staticmethod    
    def _restore_layout(x, bchw_shape): 
        if bchw_shape is None:     
            return x     
        b, c, h, w = bchw_shape
        if x.shape[1] != h * w:     
            raise ValueError(
                f"Cannot restore {x.shape[1]} tokens to BCHW shape {(b, c, h, w)}"  
            )
        return x.transpose(1, 2).reshape(b, c, h, w).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """ Input should be of shape [batch, tokens, channels]. """ 
        x, bchw_shape = self._to_nlc(x)
        B, N, _ = x.shape    
        num_windows = (    
            (N // (self.q_stride * self.window_size)) if self.use_mask_unit_attn else 1 
        )

        qkv = ( 
            self.qkv(x)
            .reshape(B, -1, num_windows, 3, self.heads, self.head_dim)
            .permute(3, 0, 4, 2, 1, 5)
        )    
        q, k, v = qkv[0], qkv[1], qkv[2] 

        if self.q_stride > 1:
            # Refer to Unroll to see how this performs a maxpool-Nd  
            q = (
                q.view(B, self.heads, num_windows, self.q_stride, -1, self.head_dim)  
                .max(dim=3)   
                .values   
            ) 
     
        if hasattr(F, "scaled_dot_product_attention"):
            # Note: the original paper did *not* use SDPA, it's a free boost!     
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            attn = (q * self.scale) @ k.transpose(-1, -2)
            attn = attn.softmax(dim=-1) 
            x = (attn @ v)    

        x = x.transpose(1, 3).reshape(B, -1, self.dim)
        x = self.proj(x)
        return self._restore_layout(x, bchw_shape)
    
if __name__ == '__main__':     
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   
    batch_size, channel, height, width = 1, 128, 20, 20
    inputs = torch.randn((batch_size, channel, height, width)).to(device)
     
    module = MaskUnitAttention(channel, heads=8).to(device)    
  
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)   
    
    print(ORANGE)    
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width),    
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True) 
    print(RESET)    
