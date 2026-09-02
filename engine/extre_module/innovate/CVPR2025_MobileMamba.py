'''
本文件由BiliBili：魔傀面具整理
论文链接：https://arxiv.org/pdf/2411.15941 
'''     

import os, sys     
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../..') 

import warnings     
warnings.filterwarnings('ignore')
from calflops import calculate_flops
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt   
import pywt.data  
from functools import partial 
    
from engine.extre_module.ultralytics_nn.conv import Conv 
from engine.extre_module.custom_nn.module.MSBlock import MSBlock
from engine.extre_module.custom_nn.mamba.SAVSS import SAVSS

def nearest_multiple_of_16(n):
    if n % 16 == 0:
        return n
    else:   
        lower_multiple = (n // 16) * 16
        upper_multiple = lower_multiple + 16   

        if (n - lower_multiple) < (upper_multiple - n):
            return lower_multiple  
        else:
            return upper_multiple 

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):  
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),  
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)     
    
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0]) 
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])     
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),  
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)
     
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
 
    return dec_filters, rec_filters
  
def wavelet_transform(x, filters):    
    b, c, h, w = x.shape   
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)  
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x 
  
    
def inverse_wavelet_transform(x, filters):    
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)     
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x  
     
class _ScaleModule(nn.Module):   
    def __init__(self, dims, init_scale=1.0, init_bias=0):   
        super(_ScaleModule, self).__init__()   
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None 

    def forward(self, x):    
        return torch.mul(self.weight, x)   

class LongRange_WTEModule(nn.Module):  
    def __init__(self, in_channels, out_channels, kernel_size=5, wt_levels=1, wt_type='db1'):
        super(LongRange_WTEModule, self).__init__()    

        self.in_channels = in_channels
        self.wt_levels = wt_levels     
        self.dilation = 1   

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False) 
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
    
        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)
 
        self.global_atten = SAVSS(in_channels)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]    
        )    

        self.wavelet_scale = nn.ModuleList(   
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]  
        )     
 
        self.conv1x1 = Conv(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()  

    def forward_wt(self, x):  
        x_ll_in_levels = []  
        x_h_in_levels = []  
        shapes_in_levels = []     

        curr_x_ll = x 
 
        for i in range(self.wt_levels):  
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)    
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)    

            curr_x = self.wt_function(curr_x_ll) 
            curr_x_ll = curr_x[:, :, 0, :, :] 
     
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))     
            curr_x_tag = curr_x_tag.reshape(shape_x)
   
            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])   
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :]) 

        next_x_ll = 0
     
        for i in range(self.wt_levels - 1, -1, -1):    
            curr_x_ll = x_ll_in_levels.pop() 
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()  

            curr_x_ll = curr_x_ll + next_x_ll
   
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2) 
            next_x_ll = self.iwt_function(curr_x)  

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll     
        assert len(x_ll_in_levels) == 0    
        return x_tag    
  
    def forward(self, x):
  
        x_tag = self.forward_wt(x)
        x = self.base_scale(self.global_atten(x))
        x = x + x_tag 
  
        return x     

class Progressive_WTEModule(nn.Module):   
    def __init__(self, in_channels, out_channels, kernel_size=5, wt_levels=1, wt_type='db1'):
        super(Progressive_WTEModule, self).__init__()

        assert in_channels == out_channels  

        self.in_channels = in_channels   
        self.wt_levels = wt_levels
        self.dilation = 1 
   
        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float) 
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)     
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)    

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        self.global_atten = SAVSS(in_channels)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(     
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,   
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
 
        self.wavelet_scale = nn.ModuleList(    
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )
        
        self.conv1x1 = Conv(in_channels * 2, out_channels, 1)   

    def forward_wt(self, x):  
        x_ll_in_levels = []
        x_h_in_levels = [] 
        shapes_in_levels = []    

        curr_x_ll = x    
  
        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape    
            shapes_in_levels.append(curr_shape)     
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):  
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])   
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x) 

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0   
   
        for i in range(self.wt_levels - 1, -1, -1):   
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()  
  
            curr_x_ll = curr_x_ll + next_x_ll
   
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)    
            next_x_ll = self.iwt_function(curr_x)  

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]
     
        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0 
        return x_tag

    def forward(self, x):
        x_global = self.base_scale(self.global_atten(x))  
        x_wt = self.forward_wt(x_global)
        x = torch.cat([x_global, x_wt], dim=1)
        x = self.conv1x1(x)    
        return x    
 
class MobileMamba(nn.Module):  
    def __init__(self, inc, ouc, global_ratio=0.25, local_ratio=0.25) -> None:
        super().__init__()     
   
        self.global_channels = int(nearest_multiple_of_16(int(inc * global_ratio)))
        self.local_channels = int(inc * local_ratio)
        self.identity_channels = inc - self.global_channels - self.local_channels 
     
        # self.global_branch = SAVSS(self.global_channels)  
        self.global_branch = Progressive_WTEModule(self.global_channels, self.global_channels)    
        self.local_branch = MSBlock(self.local_channels, self.local_channels, kernel_sizes=[1, 3, 5])

        self.proj = Conv(inc, ouc, 1) 
    
    def forward(self, x):
        x1, x2, x3 = torch.split(x, (self.global_channels, self.local_channels, self.identity_channels), dim=1)
    
        x1 = self.global_branch(x1)
        x2 = self.local_branch(x2)   

        y = torch.cat([x1, x2, x3], dim=1)
        y = self.proj(y) 
        return y

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    batch_size, in_channel, out_channel, height, width = 1, 128, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)    
  
    print(RED + '-'*20 + " MobileMamba " + '-'*20 + RESET)

    module = MobileMamba(in_channel, out_channel).to(device)  

    outputs = module(inputs)  
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)  
   
    print(ORANGE)   
    flops, macs, _ = calculate_flops(model=module,     
                                     input_shape=(batch_size, in_channel, height, width),    
                                     output_as_string=True,    
                                     output_precision=4,    
                                     print_detailed=True)
    print(RESET)
     
    print(RED + '-'*20 + " LongRange_WTEModule " + '-'*20 + RESET)    

    module = LongRange_WTEModule(in_channel, out_channel).to(device)  

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)     

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),  
                                     output_as_string=True,  
                                     output_precision=4,  
                                     print_detailed=True)
    print(RESET)
 
    print(RED + '-'*20 + " Progressive_WTEModule " + '-'*20 + RESET)

    module = Progressive_WTEModule(in_channel, out_channel).to(device)

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)    

    print(ORANGE) 
    flops, macs, _ = calculate_flops(model=module, 
                                     input_shape=(batch_size, in_channel, height, width),  
                                     output_as_string=True, 
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)