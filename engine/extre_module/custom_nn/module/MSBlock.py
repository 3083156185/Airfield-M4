'''    
本文件由BiliBili：魔傀面具整理   
engine/extre_module/module_images/TPAMI2025-MSBlock.png     
论文链接：https://arxiv.org/abs/2308.05480
'''

import os, sys 
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings  
warnings.filterwarnings('ignore')  
from calflops import calculate_flops    

import torch   
import torch.nn as nn
 
from engine.extre_module.ultralytics_nn.conv import Conv
from engine.extre_module.custom_nn.attention.GQL import GQL, GQL_Weight  

class MSBlockLayer(nn.Module):
    def __init__(self, inc, ouc, k) -> None: 
        super().__init__() 
   
        self.in_conv = Conv(inc, ouc, 1) 
        self.mid_conv = Conv(ouc, ouc, k, g=ouc)   
        self.out_conv = Conv(ouc, inc, 1)     
     
    def forward(self, x): 
        return self.out_conv(self.mid_conv(self.in_conv(x)))  
 
class MSBlock(nn.Module):
    def __init__(self, inc, ouc, kernel_sizes=[1, 3, 3], in_expand_ratio=3., mid_expand_ratio=2., layers_num=3, in_down_ratio=2.) -> None:   
        super().__init__()
     
        in_channel = int(inc * in_expand_ratio // in_down_ratio)
        self.mid_channel = in_channel // len(kernel_sizes)
        groups = int(self.mid_channel * mid_expand_ratio)
        self.in_conv = Conv(inc, in_channel)  
     
        self.mid_convs = []     
        for kernel_size in kernel_sizes:    
            if kernel_size == 1: 
                self.mid_convs.append(nn.Identity())
                continue
            mid_convs = [MSBlockLayer(self.mid_channel, groups, k=kernel_size) for _ in range(int(layers_num))] 
            self.mid_convs.append(nn.Sequential(*mid_convs))
        self.mid_convs = nn.ModuleList(self.mid_convs)
        self.out_conv = Conv(in_channel, ouc, 1)    
        
        self.attention = None
    
    def forward(self, x):
        out = self.in_conv(x)
        channels = []
        for i,mid_conv in enumerate(self.mid_convs):  
            channel = out[:,i * self.mid_channel:(i+1) * self.mid_channel,...]     
            if i >= 1:     
                channel = channel + channels[i-1]
            channel = mid_conv(channel)     
            channels.append(channel)
        out = torch.cat(channels, dim=1)   
        out = self.out_conv(out)
        if self.attention is not None:
            out = self.attention(out)    
        return out
  
class MSBlock_GQL(nn.Module):   
    def __init__(self, inc, ouc, kernel_sizes, down_ratio=1.0, mid_expand_ratio=2.0, layers_num=3, gql_param=(3, 4), start_branch_id=1, channel_split_ratios=[1 / 3, 1 / 3, 1 / 3]) -> None:
        super().__init__()  
        
        self.channel_split_ratios = [    
            ratio / sum(channel_split_ratios) for ratio in channel_split_ratios    
        ]   
        self.layers_num = layers_num
        self.start_branch_id = start_branch_id
 
        # Input feature processing  
        self.in_channel = int((inc * len(kernel_sizes)) * down_ratio)
        self.in_conv = Conv(inc, self.in_channel, 1)

        # Middle feature processing    
        self.mid_channels = [     
            int(ratio * self.in_channel) for ratio in self.channel_split_ratios
        ]
        self.mid_expand_ratio = mid_expand_ratio    
        groups = [     
            int(mid_channel * self.mid_expand_ratio) for mid_channel in self.mid_channels
        ]   

        self.mid_convs = []  
        for kernel_size, group, mid_channel in zip(kernel_sizes, groups, self.mid_channels): 
            if kernel_size == 1:
                self.mid_convs.append(nn.Identity())    
                continue 
            mid_convs = [   
                MSBlockLayer(
                    mid_channel,  
                    group,
                    kernel_size  
                )
                for _ in range(int(self.layers_num))
            ]  
            self.mid_convs.append(nn.Sequential(*mid_convs))
        self.mid_convs = nn.ModuleList(self.mid_convs)    
  
        # Indices for channel slicing   
        mid_channels = [0] + self.mid_channels
        self.indices = [   
            (sum(mid_channels[: i + 1]), sum(mid_channels[: i + 2]))     
            for i in range(len(self.mid_convs))     
        ]

        self.out_conv = Conv(self.in_channel, ouc, 1)

        self.attention = GQL(self.in_channel, gql_param[0], gql_param[1])  
        self.out_attention = None
    
    def forward(self, inputs):    
        x, query = inputs    
        out = self.in_conv(x)
        channels = []
        n = len(self.mid_convs)
        for i in range(n):
            start, end = self.indices[i]     
            channel = out[:, start:end, ...]
            if i >= self.start_branch_id:
                channel = channel + channels[i - 1]
            channel = self.mid_convs[i](channel) 
            channels.append(channel)
        out = torch.cat(channels, dim=1)
        out = self.attention(out, query)   
        out = self.out_conv(out)
        if self.out_attention is not None: 
            out = self.out_attention(out)  
        return out
     
if __name__ == '__main__': 
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)  
    
    print(RED + '-'*20 + " MSBlock(2023年的版本) " + '-'*20 + RESET) 
    module = MSBlock(in_channel, out_channel, kernel_sizes=[1, 3, 3], layers_num=3).to(device)   

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),    
                                     output_as_string=True,    
                                     output_precision=4,
                                     print_detailed=True)   
    print(RESET)  
     

    print(RED + '-'*20 + " MSBlock(2025年的TPAMI的版本) " + '-'*20 + RESET)     
    module = MSBlock_GQL(in_channel, out_channel, kernel_sizes=[1, 3, 3], layers_num=3, gql_param=(3, 4)).to(device)  
    gql_weight = GQL_Weight(3, 4).to(device)

    outputs = module([inputs, gql_weight]) 
    print(GREEN + f'inputs.size:{inputs.size()} gql_weight.size:{gql_weight.query.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)  
    flops, macs, _ = calculate_flops(model=module,
                                     args=[[inputs, gql_weight]],  
                                     output_as_string=True,    
                                     output_precision=4, 
                                     print_detailed=True)
    print(RESET)    
