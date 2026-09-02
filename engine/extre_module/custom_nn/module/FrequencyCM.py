'''  
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/CVPR2026-FrequencyCM.png     
engine/extre_module/module_images/CVPR2026-FrequencyCM.md     
论文链接：https://arxiv.org/pdf/2604.00381 
'''    

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings  
warnings.filterwarnings('ignore')
from calflops import calculate_flops

import math
import torch 
import torch.nn as nn   

from engine.extre_module.ultralytics_nn.conv import Conv

class Branch(nn.Module):  
    '''
    Branch that lasts lonly the dilated convolutions   
    ''' 
    def __init__(self, c, DW_Expand, dilation = 1):    
        super().__init__()
        self.dw_channel = DW_Expand * c 
        
        self.branch = nn.Sequential(
                       nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel, kernel_size=3, padding=dilation, stride=1, groups=self.dw_channel,
                                            bias=True, dilation = dilation) # the dconv
        )
    def forward(self, input):
        return self.branch(input)

class SimpleGate(nn.Module):
    def forward(self, x):   
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2    

class LayerNormFunction(torch.autograd.Function):    
 
    @staticmethod
    def forward(ctx, x, weight, bias, eps):  
        ctx.eps = eps
        N, C, H, W = x.size()     
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)  
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)    
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1) 
        return y 
  
    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables  
        g = grad_output * weight.view(1, C, 1, 1)     
        mean_g = g.mean(dim=1, keepdim=True)    

        mean_gy = (g * y).mean(dim=1, keepdim=True)   
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum( 
            dim=0), None

class LayerNorm2d(nn.Module):    

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()    
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels))) 
        self.eps = eps     

    def forward(self, x):    
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps) 
     
class FreMLP(nn.Module):  
    
    def __init__(self, nc, expand = 2):  
        super(FreMLP, self).__init__()
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 3, 1, 1),    
            )
        # self.fft = nn.Parameter(torch.zeros((1, nc, 1, 1)), requires_grad=True)    

    def forward(self, x):
        _, _, H, W = x.shape     
        x_freq = torch.fft.rfft2(x, norm='backward')
        _, _, H_fft, W_fft = x_freq.shape  
        x_freq = x_freq #+ x_freq*self.fft
  
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)   

        mag = self.process1(mag)  
        real = mag * torch.cos(pha)  
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')    
        return x_out

class FrequencyCM(nn.Module):
    def __init__(self, inc, ouc, DW_Expand=2, dilations = [1], extra_depth_wise = True):
        super().__init__()
        #we define the 2 branches
        self.dw_channel = DW_Expand * inc 
        self.extra_conv = nn.Conv2d(inc, inc, kernel_size=3, padding=1, stride=1, groups=inc, bias=True, dilation=1) if extra_depth_wise else nn.Identity() #optional extra dw
        # self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)    
        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=self.dw_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True, dilation = 1)
    
        self.branches = nn.ModuleList() 
        for dilation in dilations:
            self.branches.append(Branch(inc, DW_Expand, dilation = dilation))  
            
        assert len(dilations) == len(self.branches)     
        self.dw_channel = DW_Expand * inc  
        self.sca = nn.Sequential(    
                       nn.AdaptiveAvgPool2d(1),
                       nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0, stride=1,   
                       groups=1, bias=True, dilation = 1),  
        ) 
        self.sg1 = SimpleGate()
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=inc, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
        # second step 
   
        self.norm1 = LayerNorm2d(inc)
        self.norm2 = LayerNorm2d(inc)  
        self.freq = FreMLP(nc = inc, expand=1)    
        self.gamma = nn.Parameter(torch.zeros((1, inc, 1, 1)), requires_grad=True) 
        self.beta = nn.Parameter(torch.zeros((1, inc, 1, 1)), requires_grad=True) 
        self.act = nn.GELU()    
  
        self.conv_final = Conv(inc, ouc, k=1) if inc != ouc else nn.Identity()
   
    def forward(self, inp):  
        # step1   
        x_step1 = self.norm1(inp) # size [B, 2*C, H, W]
        x_freq = self.freq(x_step1) # size [B, C, H, W]     
        x = inp + x_freq * self.gamma
        x_low = x    
        x_hf = x

        x_hf = self.norm2(x_hf)
        x_hf = self.conv1(self.extra_conv(x_hf))   
        z = 0     
        for branch in self.branches:
            z += branch(x_hf)
 
        z = self.sg1(z)
        x_hf = self.sca(z) * z   
        x_high = self.conv3(x_hf) 
        y = x_low + x_high * self.beta
  
        return self.conv_final(y)    
    
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)  

    module = FrequencyCM(in_channel, out_channel).to(device)  

    outputs = module(inputs)     
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)    

    print(ORANGE)    
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,    
                                     output_precision=4,     
                                     print_detailed=True)     
    print(RESET)     
