'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/TGRS2025-MAC.png 
engine/extre_module/module_images/TGRS2025-MAC.md  
论文链接：https://ieeexplore.ieee.org/document/11017756  
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')     

import warnings
warnings.filterwarnings('ignore')  
from calflops import calculate_flops  
     
import numpy as np
import torch  
import torch.nn as nn
import torch.nn.functional as F  

from engine.extre_module.ultralytics_nn.conv import Conv  
    
kernels_all = [[] for i in range(5)]
num_cycle1 = [1, 2, 3, 4, 5] 

kernels_all2 = [[] for i in range(7)]
num_cycle2 = [1, 2, 3, 4, 5, 6, 7] 

kernels_all3 = [[] for i in range(1)]   
   
kernels_all4 = [[] for i in range(1)]
 
def GenerateKernels():   
    """
    生成固定权值卷积核
    :return: None
    """
    for i in num_cycle1:     
        kernels = [] 
        for j in range(i):  
            k_size = (2 * i) + 1  
            kernel = np.zeros(shape=(k_size, k_size)).astype(np.float32)  
            lt_y = lt_x = k_size // 2 - ((j + 1) * 2 - 1) // 2 
            red_size = (j + 1) * 2 - 1
            red_val = 1 / kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size].size 
            kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size] = red_val 
            blue_val = -1 / (k_size ** 2 - kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size].size)     
            kernel[0:lt_x, :] = kernel[lt_x + red_size:, :] = kernel[:, :lt_y] = kernel[:, lt_y + red_size:] = blue_val  
    
            kernels.append(kernel)     
        kernels_all[i - 1] = kernels  
        pass     
    return kernels_all 

def GenerateKernels2(): 
    """
    生成固定权值卷积核    
    :return: None    
    """    
    for i in num_cycle2:  
        kernels = []    
        for j in range(1): 
            k_size = (2 * i) + 1     
            kernel = np.zeros(shape=(k_size, k_size)).astype(np.float32)   
            lt_y = lt_x = k_size // 2 - ((j + 1) * 2 - 1) // 2    
            red_size = (j + 1) * 2 - 1  
            red_val = 1 / kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size].size  
            kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size] = red_val    
            blue_val = -1 / (k_size ** 2 - kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size].size)     
            kernel[0:lt_x, :] = kernel[lt_x + red_size:, :] = kernel[:, :lt_y] = kernel[:, lt_y + red_size:] = blue_val  
            kernels.append(kernel)
        kernels_all2[i - 1] = kernels
        pass
    return kernels_all2 
    
def GenerateKernels3():
    kernel = np.ones(shape=(3, 3)).astype(np.float32)
    kernel = kernel / 9.0
    kernels_all3[0].append(kernel)
    return kernels_all3

def GenerateKernels4():
    kernel = np.ones(shape=(3, 3)).astype(np.float32) 
    kernel = kernel / 8.0 * -1   
    kernel[1, 1] = 0
    kernels_all4[0].append(kernel)   
    return kernels_all4
 
def _to_kernel_tensor(kernel):
    return torch.as_tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def _cycle_kernel(kernel_size):     
    kernels = [k for ks in GenerateKernels() for k in ks]     
    return _to_kernel_tensor(kernels[kernel_size // 2 - 1]).clone()
    
def _avg_kernel(): 
    kernel = np.ones(shape=(3, 3)).astype(np.float32)
    kernel = kernel / 9.0 
    return _to_kernel_tensor(kernel)  
    
def _edge_kernel(): 
    kernel = np.ones(shape=(3, 3)).astype(np.float32)
    kernel = kernel / 8.0 * -1  
    kernel[1, 1] = 0 
    return _to_kernel_tensor(kernel)
 
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, max(1, in_planes // 16), 1, bias=False)   
        self.relu1 = nn.ReLU()   
        self.fc2   = nn.Conv2d(max(1, in_planes // 16), in_planes, 1, bias=False) 
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):     
    def __init__(self, kernel_size=7):  
        super(SpatialAttention, self).__init__()    
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'     
        padding = 3 if kernel_size == 7 else 1    
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)     
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):     
        avg_out = torch.mean(x, dim=1, keepdim=True)     
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)     
        x = self.conv1(x)
        return self.sigmoid(x) 

class MAC(nn.Module):
    def __init__(self, inplanes, outplanes, one=3, two=3, three=3, scales = 4):   
        super(MAC, self).__init__()
        if outplanes % scales != 0: 
            raise ValueError('Planes must be divisible by scales')  
        self.scales = scales
        self.relu = nn.ReLU(inplace = True)  
        self.spx = outplanes // scales
        self.inconv = Conv(inplanes, outplanes, act=False)  

        self.register_buffer('conv1_kernel', _cycle_kernel(one), persistent=False)  
        self.register_buffer('conv2_kernel', _cycle_kernel(two), persistent=False)     
        self.register_buffer('avg_kernel', _avg_kernel(), persistent=False)
        self.register_buffer('edge_kernel', _edge_kernel(), persistent=False)  
 
        self.conv1 = Conv(self.spx, self.spx, k=one, g=self.spx, act=False)   
 
        self.conv2 = Conv(self.spx, self.spx, k=one, g=self.spx, act=False)

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.spx, self.spx, three, 1, 1, groups = self.spx),   
        )     
    
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.spx, self.spx, three, 1, 2, groups = self.spx, dilation=2),    
        )
        with torch.no_grad():
            self.conv1.conv.weight.copy_(self.conv1_kernel.repeat(self.spx, 1, 1, 1))
            self.conv2.conv.weight.copy_(self.conv2_kernel.repeat(self.spx, 1, 1, 1))  
            self.conv3[0].weight.copy_(self.avg_kernel.repeat(self.spx, 1, 1, 1)) 
            self.conv4[0].weight.copy_(self.edge_kernel.repeat(self.spx, 1, 1, 1)) 
        
        self.conv5 = nn.Sequential(   
            nn.BatchNorm2d(self.spx) 
        )
        self.outconv = Conv(outplanes, outplanes, k=3, act=nn.ReLU)
        self.ca = ChannelAttention(outplanes) 
        self.sa = SpatialAttention()   
 
    def forward(self, x):  
        x = self.inconv(x)
        input = x
        xs = torch.chunk(x, self.scales, 1)    
        ys = [] 
        ys.append(xs[0])    
        ys.append(self.relu(self.conv1(xs[1])))     
        ys.append(self.relu(self.conv2(xs[2] + ys[1])))
        temp = xs[3] + ys[2]
        temp1 = self.conv5(self.conv3(temp) + self.conv4(temp))  
        ys.append(self.relu(temp1))
        y = torch.cat(ys, 1)
     
        y = self.outconv(y)
    
        output = self.relu(y + input)    
        return output

if __name__ == '__main__':   
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32     
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = MAC(in_channel, out_channel).to(device)

    outputs = module(inputs) 
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)   

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),   
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)
