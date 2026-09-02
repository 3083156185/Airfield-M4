'''
本文件由BiliBili：魔傀面具整理 
engine/extre_module/module_images/ACCV2024-RepMBConv.png
engine/extre_module/module_images/ACCV2024-RepMBConv.md  
论文链接：https://arxiv.org/pdf/2409.13435
'''
     
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')  
   
import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops
     
import torch
import torch.nn as nn   
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce

from engine.extre_module.torch_utils import model_fuse_test

__all__ = ['RepMBConv', 'LocalAttention']    

def pad_tensor(t, pattern):
    pattern = pattern.view(1, -1, 1, 1)   
    t = F.pad(t, (1, 1, 1, 1), 'constant', 0) 
    t[:, :, 0:1, :] = pattern   
    t[:, :, -1:, :] = pattern
    t[:, :, :, 0:1] = pattern     
    t[:, :, :, -1:] = pattern   
    
    return t  
 
class RepMBConv(nn.Module):
    def __init__(self, inc, ouc, ratio=2):
        super().__init__()     
        assert inc == ouc, 'This module only supports cases where the input and output channel numbers are the same.'
        i_feat = inc * ratio
        self.expand_conv = nn.Conv2d(inc,i_feat,1,1,0)  
        self.fea_conv = nn.Conv2d(i_feat,i_feat,3,1,0)
        self.reduce_conv = nn.Conv2d(i_feat,inc,1,1,0)
        self.se = ASR(i_feat)
        self.act = nn.SiLU()

    def forward(self, x):
        if hasattr(self, 'conv'):
            return self.act(self.conv(x))   
        else:
            out = self.expand_conv(x)   
            out_identity = out  
            
            # explicitly padding with bias for reparameterizing in the test phase
            b0 = self.expand_conv.bias 
            out = pad_tensor(out, b0) 
            out = self.fea_conv(out) 
            out = self.se(out) + out_identity  
            out = self.reduce_conv(out)   
            out = out + x

            return self.act(out)
 
    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            n_feat, _, _, _ = self.reduce_conv.weight.data.shape
            self.conv = nn.Conv2d(n_feat,n_feat,3,1,1)

            k0 = self.expand_conv.weight.data   
            b0 = self.expand_conv.bias.data

            k1 = self.fea_conv.weight.data   
            b1 = self.fea_conv.bias.data 
     
            k2 = self.reduce_conv.weight.data
            b2 = self.reduce_conv.bias.data 
   
            # first step: remove the ASR
            a = self.se.se(self.se.tensor)

            k1 = k1*(a.permute(1,0,2,3))
            b1 = b1*(a.view(-1))
 
            # second step: remove the middle identity     
            for i in range(2*n_feat):     
                k1[i,i,1,1] += 1.0 

            # third step: merge the first 1x1 convolution and the next 3x3 convolution 
            merge_k0k1 = F.conv2d(input=k1, weight=k0.permute(1, 0, 2, 3))
            merge_b0b1 = b0.view(1, -1, 1, 1) * torch.ones(1, 2*n_feat, 3, 3).to(b0.device) #.cuda()     
            merge_b0b1 = F.conv2d(input=merge_b0b1, weight=k1, bias=b1)  
 
            # third step: merge the remain 1x1 convolution 
            merge_k0k1k2 = F.conv2d(input=merge_k0k1.permute(1, 0, 2, 3), weight=k2).permute(1, 0, 2, 3)
            merge_b0b1b2 = F.conv2d(input=merge_b0b1, weight=k2, bias=b2).view(-1)

            # last step: remove the global identity     
            for i in range(n_feat):
                merge_k0k1k2[i, i, 1, 1] += 1.0    
 
            self.conv.weight.data = merge_k0k1k2.float()    
            self.conv.bias.data = merge_b0b1b2.float()   

            for para in self.parameters():
                para.detach_()     

            self.__delattr__('expand_conv')     
            self.__delattr__('fea_conv')     
            self.__delattr__('reduce_conv')     
            self.__delattr__('se')
     
     
class ASR(nn.Module):
    def __init__(self, n_feat, ratio=2):    
        super().__init__()     
        self.n_feat = n_feat   
        self.tensor = nn.Parameter(    
            0.1*torch.ones((1, n_feat, 1, 1)),  
            requires_grad=True   
        )     
        self.se = nn.Sequential(
            Reduce('b c 1 1 -> b c', 'mean'), 
            nn.Linear(n_feat, n_feat//4, bias = False),
            nn.SiLU(),
            nn.Linear(n_feat//4, n_feat, bias = False),    
            nn.Sigmoid(),    
            Rearrange('b c -> b c 1 1')
        )
        self.init_weights()
 
    def init_weights(self): 
        # to make sure the inital [0.5,0.5,...,0.5]  
        self.se[1].weight.data.fill_(1)     
        self.se[3].weight.data.fill_(1)
 
    def forward(self, x):    
        attn = self.se(self.tensor)
        x = attn*x  
        return x

if __name__ == '__main__':     
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"  
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    batch_size, in_channel, out_channel, height, width = 1, 16, 16, 32, 32  
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)    

    module = RepMBConv(in_channel, out_channel).to(device)

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
