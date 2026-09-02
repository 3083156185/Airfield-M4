'''   
本文件由BiliBili：魔傀面具整理 
engine/extre_module/module_images/CVPR2026-BinaryAttention.png
engine/extre_module/module_images/CVPR2026-BinaryAttention.md    
论文链接：https://arxiv.org/abs/2602.00701 
'''

import warnings    
warnings.filterwarnings('ignore')
from calflops import calculate_flops
     
import torch  
import torch.nn as nn
from typing import Any, NewType
from torch.autograd import Function     

BinaryTensor = NewType('BinaryTensor', torch.Tensor)  # A type where each element is in {-1, 1}
    
def round_ste(z):
    """Round with straight through gradients."""
    zhat = z.round()  
    return z + (zhat - z).detach()   
  
def binary_sign(x: torch.Tensor) -> BinaryTensor: 
    """Return -1 if x < 0, 1 if x >= 0.""" 
    return x.sign() + (x == 0).type(torch.float)  

 
class STESign(Function):   
    """  
    Binarize tensor using sign function. 
    Straight-Through Estimator (STE) is used to approximate the gradient of sign function.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> BinaryTensor: 
        """     
        Return a Sign tensor.     

        Args:   
            ctx: context     
            x: input tensor 

        Returns:
            Sign(x) = (x>=0) - (x<0)  
            Output type is float tensor where each element is either -1 or 1.   
        """
        ctx.save_for_backward(x)
        sign_x = binary_sign(x)
        return sign_x

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:    
        """   
        Compute gradient using STE. 

        Args:
            ctx: context
            grad_output: gradient w.r.t. output of Sign  

        Returns:
            Gradient w.r.t. input of the Sign function
        """  
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()     
        grad_input[x.gt(1)] = 0    
        grad_input[x.lt(-1)] = 0
        return grad_input
 
binarize = STESign.apply    
 
class SymQuantizer(Function):     
    """     
    uniform quantization   
    """
    @staticmethod  
    def forward(ctx, input, clip_val, num_bits, layerwise=False):
        """     
        :param ctx:  
        :param input: tensor to be quantized  
        :param clip_val: clip val
        :param num_bits: number of bits   
        :return: quantized tensor     
        """   
        ctx.save_for_backward(input, clip_val)    
 
        if layerwise:
            max_input = torch.max(torch.abs(input)).expand_as(input) 
        else: 
            assert input.ndimension() == 4 
            max_input = (     
                    torch.max(torch.abs(input), dim=-2, keepdim=True)[0]
                    .expand_as(input)
                    .detach()
                )

        s = (2 ** (num_bits - 1) - 1) / (max_input + 1e-6)    
 
        output = torch.round(input * s).div(s + 1e-6)     
   
        return output
     
    @staticmethod   
    def backward(ctx, grad_output):  
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val     
        :param grad_output: gradient ert the quantized tensor    
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors     
        grad_input = grad_output.clone()    
        grad_input[input.ge(clip_val[1])] = 0 
        grad_input[input.le(clip_val[0])] = 0     
        return grad_input, None, None, None

symquantize = SymQuantizer.apply

class BinaryAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., attn_quant=False, attn_bias=False, pv_quant=False, input_size=None):
        super().__init__()   
        self.num_heads = num_heads
        head_dim = dim // num_heads     
        self.scale = head_dim ** -0.5
        self.dim = dim 

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
   
        self.attn_drop = nn.Dropout(attn_drop)    
     
        self.proj = nn.Linear(dim, dim)
 
        self.proj_drop = nn.Dropout(proj_drop)  

        self.attn_quant = attn_quant     
        self.attn_bias = attn_bias 
        self.pv_quant = pv_quant
 
        if self.attn_bias: # dense bias
            self.input_size = input_size  
            self.num_relative_distance = (2 * input_size[0] - 1) * (2 * input_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls     
 
            # get pair-wise relative position index for each token inside the window     
            coords_h = torch.arange(input_size[0])  
            coords_w = torch.arange(input_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww    
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2    
            relative_coords[:, :, 0] += input_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += input_size[1] - 1
            relative_coords[:, :, 0] *= 2 * input_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(input_size[0] * input_size[1] + 1, ) * 2, dtype=relative_coords.dtype)   
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww  
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2     
            relative_position_index[0, 0] = self.num_relative_distance - 1
     
            self.register_buffer("relative_position_index", relative_position_index)   
    
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)     
 
    @staticmethod
    def _quantize(x): 
        s = x.abs().mean(dim=-2, keepdim=True).mean(dim=-1, keepdim=True)
        sign = binarize(x)  
        return s * sign     
    
    @staticmethod     
    def _quantize_p(x):
        qmax = 255
        s = 1.0 / qmax 
        q = round_ste(x / s).clamp(0, qmax)
        return s * q 
  
    @staticmethod     
    def _quantize_v(x, bits=8):    
        act_clip_val = torch.tensor([-2.0, 2.0])
        return symquantize(x, act_clip_val, bits, False)  
   
    def forward(self, x):
        B, C, H, W = x.size()
        N = H * W  
        x = x.flatten(2).permute(0, 2, 1)     
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) 
 
        if self.attn_quant:

            q = self._quantize(q)
            k = self._quantize(k)     
     
            attn = (q @ k.transpose(-2, -1)) * self.scale 
 
            if self.attn_bias: 
                relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view( 
                            self.input_size[0] * self.input_size[1] + 1,     
                            self.input_size[0] * self.input_size[1] + 1, -1)
                relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
                attn = attn + relative_position_bias.unsqueeze(0)  

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            if self.pv_quant:    
                attn = self._quantize_p(attn)
                v = self._quantize_v(v, 8)
 
        else: 
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)    
            attn = self.attn_drop(attn)     
     
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)   
        x = self.proj(x)
        x = self.proj_drop(x)  

        x = x.permute(0, 2, 1).reshape((B, C, H, W))   
        return x    
   
if __name__ == '__main__': 
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m" 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')     
    batch_size, channel, height, width = 1, 128, 20, 20     
    inputs = torch.randn((batch_size, channel, height, width)).to(device)

    module = BinaryAttention(channel, num_heads=8).to(device)
     
    outputs = module(inputs)     
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
     
    print(ORANGE)   
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width),
                                     output_as_string=True,     
                                     output_precision=4,
                                     print_detailed=True)  
    print(RESET)