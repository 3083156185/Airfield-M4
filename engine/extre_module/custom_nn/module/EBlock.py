'''     
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/CVPR2025-EBlock.png  
论文链接：https://arxiv.org/abs/2412.13443   
'''
  
import os, sys     
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')    
   
import warnings
warnings.filterwarnings('ignore') 
from calflops import calculate_flops

import torch 
import torch.nn as nn     
from engine.extre_module.ultralytics_nn.conv import Conv
    
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
     
class SimpleGate(nn.Module):     
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2    
 
class Adapter(nn.Module):
    
    def __init__(self, c, ffn_channel = None):
        super().__init__()
        if ffn_channel:
            ffn_channel = 2    
        else:  
            ffn_channel = c
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=ffn_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True) 
        self.depthwise = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=3, padding=1, stride=1, groups=c, bias=True, dilation=1)    

    def forward(self, input):
        
        x = self.conv1(input) + self.depthwise(input)
        x = self.conv2(x)
  
        return x   

class FreMLP(nn.Module):  
    
    def __init__(self, nc, expand = 2):
        super(FreMLP, self).__init__()  
        self.process1 = nn.Sequential( 
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),   
            nn.Conv2d(expand * nc, nc, 1, 1, 0))    
   
    def forward(self, x):
        _, _, H, W = x.shape
        x_dtype = x.dtype  
        x_freq = torch.fft.rfft2(x.float(), norm='backward') 
        mag = torch.abs(x_freq)  
        pha = torch.angle(x_freq)     
        mag = self.process1(mag.to(x_dtype))    
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)    
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out.float(), s=(H, W), norm='backward')    
        return x_out.to(x_dtype)

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
    
class EBlock(nn.Module):   
    '''  
    Change this block using Branch
    '''
     
    def __init__(self, inc, c, DW_Expand=2, dilations = [1], extra_depth_wise = False):
        super().__init__()
        #we define the 2 branches 
        self.dw_channel = DW_Expand * c 
        self.extra_conv = nn.Conv2d(c, c, kernel_size=3, padding=1, stride=1, groups=c, bias=True, dilation=1) if extra_depth_wise else nn.Identity() #optional extra dw    
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
                
        self.branches = nn.ModuleList()     
        for dilation in dilations:   
            self.branches.append(Branch(c, DW_Expand, dilation = dilation))
            
        assert len(dilations) == len(self.branches)   
        self.dw_channel = DW_Expand * c 
        self.sca = nn.Sequential(     
                       nn.AdaptiveAvgPool2d(1),   
                       nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0, stride=1,
                       groups=1, bias=True, dilation = 1),  
        ) 
        self.sg1 = SimpleGate()     
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)    
        # second step 
   
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)  
        self.freq = FreMLP(nc = c, expand=2)  
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)     
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.conv1x1 = Conv(inc, c, 1) if inc != c else nn.Identity()   

    def forward(self, inp):
        inp = self.conv1x1(inp)     
        y = inp  
        x = self.norm1(inp)  
        x = self.conv1(self.extra_conv(x))
        z = 0     
        for branch in self.branches:
            z += branch(x)  
     
        z = self.sg1(z)  
        x = self.sca(z) * z  
        x = self.conv3(x)   
        y = inp + self.beta * x
        #second step
        x_step2 = self.norm2(y) # size [B, 2*C, H, W]
        x_freq = self.freq(x_step2) # size [B, C, H, W] 
        x = y * x_freq    
        x = y + x * self.gamma

        return x   
 
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m" 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32 
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device) 

    module = EBlock(in_channel, out_channel).to(device)
  
    outputs = module(inputs) 
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE) 
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),  
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)