'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/MIA2025-FourierConv.png
engine/extre_module/module_images/MIA2025-FourierConv.md
论文链接：https://www.sciencedirect.com/science/article/abs/pii/S1361841524002743
'''

import os, sys 
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings 
warnings.filterwarnings('ignore')   
from calflops import calculate_flops    
 
import torch  
import torch.nn as nn   
import torch.nn.functional as F
import torch.fft    
import numpy as np
   
from engine.extre_module.ultralytics_nn.conv import Conv, autopad

   
def complexinit(weights_real, weights_imag, criterion):
    output_chs, input_chs, num_rows, num_cols = weights_real.shape
    fan_in = input_chs   
    fan_out = output_chs 
    if criterion == 'glorot':
        s = 1. / np.sqrt(fan_in + fan_out) / 4.
    elif criterion == 'he':  
        s = 1. / np.sqrt(fan_in) / 4.  
    else:
        raise ValueError('Invalid criterion: ' + criterion)
 
    rng = np.random.RandomState()
    kernel_shape = weights_real.shape
    modulus = rng.rayleigh(scale=s, size=kernel_shape)
    phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)
    weight_real = modulus * np.cos(phase)   
    weight_imag = modulus * np.sin(phase) 
    weights_real.data = torch.Tensor(weight_real) 
    weights_imag.data = torch.Tensor(weight_imag)     
  
class DeepSparse(nn.Module):
    def __init__(self, input_chs, size, init='he'):   
        super(DeepSparse, self).__init__() 
        h, w = size  
        self.weights_real = nn.Parameter(torch.Tensor(1, input_chs, h, int(w//2 + 1)))  
        self.weights_imag = nn.Parameter(torch.Tensor(1, input_chs, h, int(w//2 + 1)))  
        complexinit(self.weights_real, self.weights_imag, init)
        self.size = size
  
    def forward(self, x):    
        original_dtype = x.dtype
        x = x.float()
        x = torch.fft.rfftn(x, dim=(-2, -1), norm=None)
        x_real, x_imag = x.real, x.imag  
        y_real = torch.mul(x_real, self.weights_real) - torch.mul(x_imag, self.weights_imag)
        y_imag = torch.mul(x_real, self.weights_imag) + torch.mul(x_imag, self.weights_real)
        x = torch.fft.irfftn(torch.complex(y_real, y_imag), s=self.size, dim=(-2, -1), norm=None)
        x = x.to(original_dtype)   
        return x     
   
class FourierConv(nn.Module):
    def __init__(self, inc, ouc, size, s=1, act=True) -> None:     
        super().__init__()
 
        self.deepsparse = DeepSparse(inc, size if s == 1 else [i * 2 for i in size])
        self.conv = Conv(inc, ouc, 1 if s == 1 else 3, s=s, act=act)  
 
    def forward(self, x):   
        x = self.deepsparse(x)    
        x = self.conv(x)
        return x 
  
if __name__ == '__main__':    
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32  
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)
    
    # 此模块不支持多尺度训练，size的参数是一个元组，其为当前特征图的height,width。    
    module = FourierConv(in_channel, out_channel, size=(height, width),s=1).to(device)   
    
    outputs = module(inputs)   
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)   

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),  
                                     output_as_string=True,  
                                     output_precision=4,
                                     print_detailed=True)  
    print(RESET)