'''     
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/ICCV2025-MBRConv.png   
engine/extre_module/module_images/ICCV2025-MBRConv.md 
论文链接：https://www.arxiv.org/pdf/2507.01838     
'''    

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')     
    
import warnings     
warnings.filterwarnings('ignore') 
from calflops import calculate_flops

import torch
import torch.nn as nn     
import torch.nn.functional as F

from engine.extre_module.torch_utils import model_fuse_test 
     
class MBRConv5(nn.Module):
    def __init__(self, in_channels, out_channels, rep_scale=4):
        super(MBRConv5, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, 5, 1, 2)
        self.conv_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
        self.conv1_bn = nn.Sequential(   
            nn.BatchNorm2d(out_channels * rep_scale)     
        )
        self.conv2 = nn.Conv2d(in_channels, out_channels * rep_scale, 3, 1, 1)   
        self.conv2_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv_crossh = nn.Conv2d(in_channels, out_channels * rep_scale, (3, 1), 1, (1, 0))
        self.conv_crossh_bn = nn.Sequential(   
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv_crossv = nn.Conv2d(in_channels, out_channels * rep_scale, (1, 3), 1, (0, 1))     
        self.conv_crossv_bn = nn.Sequential(    
            nn.BatchNorm2d(out_channels * rep_scale)
        )     
        self.conv_out = nn.Conv2d(out_channels * rep_scale * 10, out_channels, 1)   
    
    def forward(self, inp):
        if hasattr(self, 'conv_rep'):   
            return self.conv_rep(inp)   
        else:  
            return self.forward_(inp)     

    def forward_(self, inp):   
        x1 = self.conv(inp)  
        x2 = self.conv1(inp)     
        x3 = self.conv2(inp)
        x4 = self.conv_crossh(inp)  
        x5 = self.conv_crossv(inp)    
        x = torch.cat(    
            [x1, x2, x3, x4, x5,
             self.conv_bn(x1),
             self.conv1_bn(x2),
             self.conv2_bn(x3), 
             self.conv_crossh_bn(x4),  
             self.conv_crossv_bn(x5)],
            1    
        )   
        out = self.conv_out(x)  
        return out    

    def slim(self):
        conv_weight = self.conv.weight  
        conv_bias = self.conv.bias 

        conv1_weight = self.conv1.weight  
        conv1_bias = self.conv1.bias
        conv1_weight = nn.functional.pad(conv1_weight, (2, 2, 2, 2)) 

        conv2_weight = self.conv2.weight     
        conv2_weight = nn.functional.pad(conv2_weight, (1, 1, 1, 1)) 
        conv2_bias = self.conv2.bias
    
        conv_crossv_weight = self.conv_crossv.weight    
        conv_crossv_weight = nn.functional.pad(conv_crossv_weight, (1, 1, 2, 2))
        conv_crossv_bias = self.conv_crossv.bias  

        conv_crossh_weight = self.conv_crossh.weight
        conv_crossh_weight = nn.functional.pad(conv_crossh_weight, (2, 2, 1, 1))
        conv_crossh_bias = self.conv_crossh.bias
  
        conv1_bn_weight = self.conv1.weight
        conv1_bn_weight = nn.functional.pad(conv1_bn_weight, (2, 2, 2, 2))   
  
        conv2_bn_weight = self.conv2.weight   
        conv2_bn_weight = nn.functional.pad(conv2_bn_weight, (1, 1, 1, 1))  

        conv_crossv_bn_weight = self.conv_crossv.weight
        conv_crossv_bn_weight = nn.functional.pad(conv_crossv_bn_weight, (1, 1, 2, 2)) 

        conv_crossh_bn_weight = self.conv_crossh.weight
        conv_crossh_bn_weight = nn.functional.pad(conv_crossh_bn_weight, (2, 2, 1, 1))
    
        bn = self.conv_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** .5
        b = - bn.running_mean / (bn.running_var + bn.eps) ** .5  

        conv_bn_weight = self.conv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_weight = conv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)    
        conv_bn_bias = self.conv.bias * k + b   
        conv_bn_bias = conv_bn_bias * bn.weight + bn.bias     

        bn = self.conv1_bn[0]     
        k = 1 / (bn.running_var + bn.eps) ** .5    
        b = - bn.running_mean / (bn.running_var + bn.eps) ** .5 
        conv1_bn_weight = conv1_bn_weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv1_bn_weight = conv1_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
        conv1_bn_bias = self.conv1.bias * k + b
        conv1_bn_bias = conv1_bn_bias * bn.weight + bn.bias    
     
        bn = self.conv2_bn[0] 
        k = 1 / (bn.running_var + bn.eps) ** .5     
        b = - bn.running_mean / (bn.running_var + bn.eps) ** .5
        conv2_bn_weight = conv2_bn_weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv2_bn_weight = conv2_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv2_bn_bias = self.conv2.bias * k + b  
        conv2_bn_bias = conv2_bn_bias * bn.weight + bn.bias 
 
        bn = self.conv_crossv_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** .5    
        b = - bn.running_mean / (bn.running_var + bn.eps) ** .5 
        conv_crossv_bn_weight = conv_crossv_bn_weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossv_bn_weight = conv_crossv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  
        conv_crossv_bn_bias = self.conv_crossv.bias * k + b     
        conv_crossv_bn_bias = conv_crossv_bn_bias * bn.weight + bn.bias
 
        bn = self.conv_crossh_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** .5  
        b = - bn.running_mean / (bn.running_var + bn.eps) ** .5
        conv_crossh_bn_weight = conv_crossh_bn_weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossh_bn_weight = conv_crossh_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossh_bn_bias = self.conv_crossh.bias * k + b 
        conv_crossh_bn_bias = conv_crossh_bn_bias * bn.weight + bn.bias

        weight = torch.cat(
            [conv_weight, conv1_weight, conv2_weight,  
             conv_crossh_weight, conv_crossv_weight,
             conv_bn_weight, conv1_bn_weight, conv2_bn_weight,   
             conv_crossh_bn_weight, conv_crossv_bn_weight],
            0  
        )    
        weight_compress = self.conv_out.weight.squeeze()
        weight = torch.matmul(weight_compress, weight.permute([2, 3, 0, 1])).permute([2, 3, 0, 1])
        bias_ = torch.cat( 
            [conv_bias, conv1_bias, conv2_bias,
             conv_crossh_bias, conv_crossv_bias,
             conv_bn_bias, conv1_bn_bias, conv2_bn_bias,
             conv_crossh_bn_bias, conv_crossv_bn_bias],
            0     
        )
        bias = torch.matmul(weight_compress, bias_)  
        if isinstance(self.conv_out.bias, torch.Tensor):
            bias = bias + self.conv_out.bias   
        return weight, bias    
    
    def convert_to_deploy(self):
        weight, bias = self.slim()   

        self.conv_rep = nn.Conv2d(self.in_channels, self.out_channels, 5, 1, 2).to(weight.device).to(weight.dtype)   
        self.conv_rep.weight = torch.nn.Parameter(weight)    
        self.conv.bias = torch.nn.Parameter(bias)

        del self.conv
        del self.conv_bn    
        del self.conv1 
        del self.conv1_bn
        del self.conv2    
        del self.conv2_bn
        del self.conv_crossh
        del self.conv_crossh_bn    
        del self.conv_crossv     
        del self.conv_crossv_bn
        del self.conv_out 

class MBRConv3(nn.Module):
    def __init__(self, in_channels, out_channels, rep_scale=4):
        super(MBRConv3, self).__init__()
   
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rep_scale = rep_scale     
        
        self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, 3, 1, 1)
        self.conv_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
        self.conv1_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv_crossh = nn.Conv2d(in_channels, out_channels * rep_scale, (3, 1), 1, (1, 0))
        self.conv_crossh_bn = nn.Sequential(    
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv_crossv = nn.Conv2d(in_channels, out_channels * rep_scale, (1, 3), 1, (0, 1))
        self.conv_crossv_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels * rep_scale)
        )     
        self.conv_out = nn.Conv2d(out_channels * rep_scale * 8, out_channels, 1)   

    def forward(self, inp):
        if hasattr(self, 'conv_rep'):
            return self.conv_rep(inp)    
        else: 
            return self.forward_(inp)

    def forward_(self, inp):    
        x0 = self.conv(inp) 
        x1 = self.conv1(inp)   
        x2 = self.conv_crossh(inp)     
        x3 = self.conv_crossv(inp)
        x = torch.cat(  
        [    x0,x1,x2,x3,   
             self.conv_bn(x0),    
             self.conv1_bn(x1),   
             self.conv_crossh_bn(x2), 
             self.conv_crossv_bn(x3)],
            1
        ) 
        out = self.conv_out(x)    
        return out
     
    def slim(self):     
        conv_weight = self.conv.weight
        conv_bias = self.conv.bias
   
        conv1_weight = self.conv1.weight     
        conv1_bias = self.conv1.bias
        conv1_weight = F.pad(conv1_weight, (1, 1, 1, 1))

        conv_crossh_weight = self.conv_crossh.weight    
        conv_crossh_bias = self.conv_crossh.bias    
        conv_crossh_weight = F.pad(conv_crossh_weight, (1, 1, 0, 0))

        conv_crossv_weight = self.conv_crossv.weight     
        conv_crossv_bias = self.conv_crossv.bias   
        conv_crossv_weight = F.pad(conv_crossv_weight, (0, 0, 1, 1))
   
        # conv_bn
        bn = self.conv_bn[0]     
        k = 1 / torch.sqrt(bn.running_var + bn.eps) 
        conv_bn_weight = self.conv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)     
        conv_bn_weight = conv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)     
        conv_bn_bias = self.conv.bias * k + (-bn.running_mean * k)
        conv_bn_bias = conv_bn_bias * bn.weight + bn.bias 

        # conv1_bn
        bn = self.conv1_bn[0]
        k = 1 / torch.sqrt(bn.running_var + bn.eps)
        conv1_bn_weight = self.conv1.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv1_bn_weight = conv1_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  
        conv1_bn_weight = F.pad(conv1_bn_weight, (1, 1, 1, 1))
        conv1_bn_bias = self.conv1.bias * k + (-bn.running_mean * k)
        conv1_bn_bias = conv1_bn_bias * bn.weight + bn.bias  
 
        # conv_crossh_bn 
        bn = self.conv_crossh_bn[0]    
        k = 1 / torch.sqrt(bn.running_var + bn.eps) 
        conv_crossh_bn_weight = self.conv_crossh.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)     
        conv_crossh_bn_weight = conv_crossh_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossh_bn_weight = F.pad(conv_crossh_bn_weight, (1, 1, 0, 0))  
        conv_crossh_bn_bias = self.conv_crossh.bias * k + (-bn.running_mean * k)     
        conv_crossh_bn_bias = conv_crossh_bn_bias * bn.weight + bn.bias   
    
        # conv_crossv_bn
        bn = self.conv_crossv_bn[0] 
        k = 1 / torch.sqrt(bn.running_var + bn.eps)     
        conv_crossv_bn_weight = self.conv_crossv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossv_bn_weight = conv_crossv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)   
        conv_crossv_bn_weight = F.pad(conv_crossv_bn_weight, (0, 0, 1, 1))     
        conv_crossv_bn_bias = self.conv_crossv.bias * k + (-bn.running_mean * k)
        conv_crossv_bn_bias = conv_crossv_bn_bias * bn.weight + bn.bias   

        weight = torch.cat([
            conv_weight,     
            conv1_weight,
            conv_crossh_weight,
            conv_crossv_weight,
            conv_bn_weight,
            conv1_bn_weight,    
            conv_crossh_bn_weight,   
            conv_crossv_bn_weight     
        ], dim=0)    

        bias = torch.cat([
            conv_bias,   
            conv1_bias,
            conv_crossh_bias, 
            conv_crossv_bias,
            conv_bn_bias,     
            conv1_bn_bias,   
            conv_crossh_bn_bias,    
            conv_crossv_bn_bias
        ], dim=0)
    
        weight_compress = self.conv_out.weight.squeeze()
        weight = torch.matmul(weight_compress, weight.view(weight.size(0), -1))
        weight = weight.view(self.conv_out.out_channels, self.in_channels, 3, 3)

        bias = torch.matmul(weight_compress, bias.unsqueeze(-1)).squeeze(-1)   
        if self.conv_out.bias is not None:    
            bias += self.conv_out.bias 

        return weight, bias   
  
    def convert_to_deploy(self):
        weight, bias = self.slim()

        self.conv_rep = nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1).to(weight.device).to(weight.dtype)  
        self.conv_rep.weight = torch.nn.Parameter(weight)
        self.conv.bias = torch.nn.Parameter(bias)

        del self.conv
        del self.conv_bn
        del self.conv1
        del self.conv1_bn
        del self.conv_crossh  
        del self.conv_crossh_bn
        del self.conv_crossv
        del self.conv_crossv_bn   
        del self.conv_out 

class MBRConv1(nn.Module):   
    def __init__(self, in_channels, out_channels, rep_scale=4):
        super(MBRConv1, self).__init__() 
    
        self.in_channels = in_channels   
        self.out_channels = out_channels
        self.rep_scale = rep_scale
        
        self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
        self.conv_bn = nn.Sequential(    
            nn.BatchNorm2d(out_channels * rep_scale)
        )
        self.conv_out = nn.Conv2d(out_channels * rep_scale * 2, out_channels, 1)    

    def forward(self, inp):
        if hasattr(self, 'conv_rep'):  
            return self.conv_rep(inp)    
        else:
            return self.forward_(inp) 
     
    def forward_(self, inp): 
        x0 = self.conv(inp)    
        x = torch.cat([x0, self.conv_bn(x0)], 1)
        out = self.conv_out(x)    
        return out 
   
    def slim(self):
        conv_weight = self.conv.weight     
        conv_bias = self.conv.bias    
     
        bn = self.conv_bn[0]  
        k = 1 / (bn.running_var + bn.eps) ** .5    
        b = - bn.running_mean / (bn.running_var + bn.eps) ** .5     
        conv_bn_weight = self.conv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_weight = conv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_bias = self.conv.bias * k + b
        conv_bn_bias = conv_bn_bias * bn.weight + bn.bias   

        weight = torch.cat([conv_weight, conv_bn_weight], 0)
        weight_compress = self.conv_out.weight.squeeze()
        weight = torch.matmul(weight_compress, weight.permute([2, 3, 0, 1])).permute([2, 3, 0, 1])

        bias = torch.cat([conv_bias, conv_bn_bias], 0)  
        bias = torch.matmul(weight_compress, bias)
   
        if isinstance(self.conv_out.bias, torch.Tensor):
            bias = bias + self.conv_out.bias
        return weight, bias
    
    def convert_to_deploy(self):
        weight, bias = self.slim()

        self.conv_rep = nn.Conv2d(self.in_channels, self.out_channels, 1).to(weight.device).to(weight.dtype)     
        self.conv_rep.weight = torch.nn.Parameter(weight)     
        self.conv.bias = torch.nn.Parameter(bias)   

        del self.conv  
        del self.conv_bn    
        del self.conv_out 
  
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)
    
    print(RED + '-'*20 + " MBRConv5(重参数化等于一个5x5的卷积) " + '-'*20 + RESET)
   
    module = MBRConv5(in_channel, out_channel).to(device) 

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)   

    print(GREEN + 'test reparameterization.' + RESET) 
    module = model_fuse_test(module)
    outputs = module(inputs)    
    print(GREEN + 'test reparameterization done.' + RESET)  

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,    
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)

    print(RED + '-'*20 + " MBRConv3(重参数化等于一个3x3的卷积) " + '-'*20 + RESET)
 
    module = MBRConv3(in_channel, out_channel).to(device)

    outputs = module(inputs)    
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
    
    print(GREEN + 'test reparameterization.' + RESET)
    module = model_fuse_test(module)     
    outputs = module(inputs)
    print(GREEN + 'test reparameterization done.' + RESET)    
 
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,  
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)     
   
    print(RED + '-'*20 + " MBRConv1(重参数化等于一个1x1的卷积) " + '-'*20 + RESET)     

    module = MBRConv1(in_channel, out_channel).to(device)
    
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
     
    print(GREEN + 'test reparameterization.' + RESET)     
    module = model_fuse_test(module)  
    outputs = module(inputs)
    print(GREEN + 'test reparameterization done.' + RESET)
     
    print(ORANGE)  
    flops, macs, _ = calculate_flops(model=module,  
                                     input_shape=(batch_size, in_channel, height, width),    
                                     output_as_string=True,
                                     output_precision=4, 
                                     print_detailed=True)   
    print(RESET)    
