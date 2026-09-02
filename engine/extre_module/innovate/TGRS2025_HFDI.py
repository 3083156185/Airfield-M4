'''    
本文件由BiliBili：魔傀面具整理
论文链接：https://arxiv.org/pdf/2406.02037     
'''

import os, sys 
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../..')
  
import warnings   
warnings.filterwarnings('ignore')   
from calflops import calculate_flops

import torch
import torch.nn as nn  
from torch.autograd import Function
import pywt

from engine.extre_module.ultralytics_nn.conv import Conv    
from engine.extre_module.custom_nn.downsample.ADown import ADown    
from engine.extre_module.custom_nn.downsample.SPDConv import SPDConv  

class DWT_2D(nn.Module):   
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)    
        dec_hi = torch.Tensor(w.dec_hi[::-1])  
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)   
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)  
   
        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))   
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))
    
        self.w_ll = self.w_ll.to(dtype=torch.float32)   
        self.w_lh = self.w_lh.to(dtype=torch.float32)
        self.w_hl = self.w_hl.to(dtype=torch.float32)
        self.w_hh = self.w_hh.to(dtype=torch.float32)  
    
    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)   

 
class DWT_Function(Function):  
    @staticmethod     
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh): 
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape
        dim = x.shape[1]
  
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, padding=0, groups=dim)    
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, padding=0, groups=dim)     
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, padding=0, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, padding=0,     
                                          groups=dim)
        x = torch.cat([x_lh, x_hl, x_hh], dim=1)  
        # x =  x_hh
        return x 
     
    @staticmethod 
    def backward(ctx, dx):  
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors   
            B, C, H, W = ctx.shape  

            dx = dx.view(B, 3, -1, H // 2, W // 2)
            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            filters = torch.cat([w_lh, w_hl, w_hh], dim=0)
            # filters = filters.repeat(C, 1, 1, 1).to(dtype=torch.float16)   
            filters = filters.repeat(C, 1, 1, 1).to(dx.device, dx.dtype) 
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        return dx, None, None, None, None

class Hfrequencyfeature(nn.Module):  #
    def __init__(self):    
        super().__init__()     
        self.DWT_2D = DWT_2D('haar')   
   
    def forward(self, x):
        out = self.DWT_2D(x)  
        return out    

class Hfrequency(nn.Module):   
    def __init__(self):
        super(Hfrequency, self).__init__()  
        self.Hfrequencyfeature = Hfrequencyfeature()
     
    def forward(self, inputs):
        x, out_2 = inputs
        out_1 = self.Hfrequencyfeature(x)     
        out = torch.cat([out_1, out_2], dim=1)   
        return out
 
class HighFrequencyDirectionInjectionModule(nn.Module):   
    def __init__(self, inc, ouc) -> None:
        super().__init__() 
   
        hf_channel = inc * 3  
        self.branch = Conv(inc, ouc - hf_channel, 3, 2)
        # self.branch = nn.Sequential(   
        #     Conv(inc, ouc // 2, 1),   
        #     ADown(ouc // 2, ouc),    
        #     Conv(ouc, ouc - hf_channel, 1)  
        # ) 
        # self.branch = SPDConv(inc, ouc - hf_channel)    
        self.hf = Hfrequency()
  
    def forward(self, x):
        b_out = self.branch(x)
        out = self.hf((x, b_out))
        return out
    
# class HighFrequencyDirectionInjectionModule(nn.Module):
#     def __init__(self, inc, ouc) -> None:    
#         super().__init__()    
  
#         self.branch = Conv(inc, ouc, 3, 2) 
#         # self.branch = nn.Sequential(
#         #     Conv(inc, ouc // 2, 1),
#         #     ADown(ouc // 2, ouc),    
#         #     Conv(ouc, ouc - hf_channel, 1)
#         # )
#         # self.branch = SPDConv(inc, ouc - hf_channel)
#         self.hf = Hfrequency()
#         self.conv_1x1 = Conv(ouc + inc * 3, ouc, 1)
    
#     def forward(self, x):   
#         b_out = self.branch(x)
#         out = self.hf((x, b_out))     
#         return self.conv_1x1(out)
   
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')     
    batch_size, in_channel, out_channel, height, width = 1, 3, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device) 
    
    module = HighFrequencyDirectionInjectionModule(in_channel, out_channel).to(device)
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)  

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),     
                                     output_as_string=True, 
                                     output_precision=4,
                                     print_detailed=True)     
    print(RESET)