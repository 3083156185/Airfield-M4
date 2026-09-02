'''
本文件由BiliBili：魔傀面具整理     
engine/extre_module/module_images/ECCV2022-Local Feature Extraction.png  
论文链接：https://arxiv.org/pdf/2203.06697
'''  

import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops  
    
import torch
import torch.nn as nn    
import torch.nn.functional as F
  
class ShiftConv2d0(nn.Module):   
    def __init__(self, inp_channels, out_channels):     
        super(ShiftConv2d0, self).__init__()     
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.n_div = 5
        g = inp_channels // self.n_div
    
        conv3x3 = nn.Conv2d(inp_channels, out_channels, 3, 1, 1)   
        mask = nn.Parameter(torch.zeros((self.out_channels, self.inp_channels, 3, 3)), requires_grad=False) 
        mask[:, 0*g:1*g, 1, 2] = 1.0 
        mask[:, 1*g:2*g, 1, 0] = 1.0     
        mask[:, 2*g:3*g, 2, 1] = 1.0
        mask[:, 3*g:4*g, 0, 1] = 1.0
        mask[:, 4*g:, 1, 1] = 1.0
        self.w = conv3x3.weight
        self.b = conv3x3.bias
        self.m = mask

    def forward(self, x):   
        y = F.conv2d(input=x, weight=self.w * self.m, bias=self.b, stride=1, padding=1) 
        return y

 
class ShiftConv2d1(nn.Module):    
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d1, self).__init__() 
        self.inp_channels = inp_channels     
        self.out_channels = out_channels   

        self.weight = nn.Parameter(torch.zeros(inp_channels, 1, 3, 3), requires_grad=False)    
        self.n_div = 5    
        g = inp_channels // self.n_div
        self.weight[0*g:1*g, 0, 1, 2] = 1.0 ## left
        self.weight[1*g:2*g, 0, 1, 0] = 1.0 ## right   
        self.weight[2*g:3*g, 0, 2, 1] = 1.0 ## up
        self.weight[3*g:4*g, 0, 0, 1] = 1.0 ## down
        self.weight[4*g:, 0, 1, 1] = 1.0 ## identity     
 
        self.conv1x1 = nn.Conv2d(inp_channels, out_channels, 1)
 
    def forward(self, x):   
        y = F.conv2d(input=x, weight=self.weight, bias=None, stride=1, padding=1, groups=self.inp_channels)
        y = self.conv1x1(y) 
        return y

    
class ShiftConv2d(nn.Module): 
    def __init__(self, inp_channels, out_channels, conv_type='fast-training-speed'):
        super(ShiftConv2d, self).__init__()    
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.conv_type = conv_type
        if conv_type == 'low-training-memory':    
            self.shift_conv = ShiftConv2d0(inp_channels, out_channels)
        elif conv_type == 'fast-training-speed':
            self.shift_conv = ShiftConv2d1(inp_channels, out_channels)
        else:     
            raise ValueError('invalid type of shift-conv2d')  
  
    def forward(self, x):
        y = self.shift_conv(x)     
        return y  
    
class LFE(nn.Module):     
    def __init__(self, inp_channels, out_channels, exp_ratio=4, act_type='relu'):    
        super(LFE, self).__init__()   
        self.exp_ratio = exp_ratio     
        self.act_type  = act_type

        self.conv0 = ShiftConv2d(inp_channels, out_channels*exp_ratio) 
        self.conv1 = ShiftConv2d(out_channels*exp_ratio, out_channels)

        if self.act_type == 'linear':     
            self.act = None
        elif self.act_type == 'relu':     
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'silu':  
            self.act = nn.SiLU(inplace=True)     
        elif self.act_type == 'gelu':
            self.act = nn.GELU()  
        else:
            raise ValueError('unsupport type of activation')   

    def forward(self, x):     
        y = self.conv0(x)  
        y = self.act(y)
        y = self.conv1(y)   
        return y

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32   
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)
    
    module = LFE(in_channel, out_channel).to(device)
     
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
   
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,   
                                     output_precision=4,     
                                     print_detailed=True)     
    print(RESET)     
