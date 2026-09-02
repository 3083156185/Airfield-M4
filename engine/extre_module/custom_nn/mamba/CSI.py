'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/INFFUS2025-CSI.md
engine/extre_module/module_images/INFFUS2025-CSI.png     
论文链接：https://arxiv.org/pdf/2505.23214
'''    

import os, sys    
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops    
 
import math    
import torch    
import torch.nn as nn    
import torch.nn.functional as F  
try:
    from mamba_ssm import Mamba
except Exception as e:
    pass   

from engine.extre_module.ultralytics_nn.conv import Conv    

class Mlp(nn.Module):    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.): 
        super().__init__()
        out_features = out_features or in_features    
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)  
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)    
        x = self.act(x)    
        x = self.drop(x)     
        x = self.fc2(x)
        x = self.drop(x)
        return x
 
class CSI(nn.Module):
    def __init__(self, input_dim, d_state = 16, d_conv = 4, expand = 2):     
        super().__init__()
        self.input_dim = input_dim   
        self.output_dim = input_dim
        self.norm1 = nn.LayerNorm(input_dim//4)
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(   
                d_model=input_dim//4, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )     
        self.proj = nn.Linear(input_dim//4, input_dim//4)
        self.skip_scale= nn.Parameter(torch.ones(1))
        self.cpe2 = nn.Conv2d(input_dim//4, input_dim//4, 3, padding=1, groups=input_dim//4)
        self.out = Conv(   
            input_dim,
            input_dim,     
            1
        )  
        self.mlp = Mlp(in_features=input_dim//4, hidden_features=int(input_dim//4 * 4))    
    def forward(self, x):
        x_dtype = x.dtype     
        if x.dtype == torch.float16:
            x = x.type(torch.float32) 
        B, C = x.shape[:2]
        assert C == self.input_dim  
        n_tokens = x.shape[2:].numel()     
        img_dims = x.shape[2:] 
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
 
        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2) 
        x_mamba1 = self.mlp(self.norm1(self.mamba(x1))) + self.skip_scale * x1 
        x_mamba2 = self.mlp(self.norm1(self.mamba(x2))) + self.skip_scale * x2  
        x_mamba3 = self.mlp(self.norm1(self.mamba(x3))) + self.skip_scale * x3   
        x_mamba4 = self.mlp(self.norm1(self.mamba(x4))) + self.skip_scale * x4

        x_mamba1 = x_mamba1.transpose(-1, -2).reshape(B, self.output_dim//4, *img_dims) 
        x_mamba2 = x_mamba2.transpose(-1, -2).reshape(B, self.output_dim//4, *img_dims)    
        x_mamba3 = x_mamba3.transpose(-1, -2).reshape(B, self.output_dim//4, *img_dims)     
        x_mamba4 = x_mamba4.transpose(-1, -2).reshape(B, self.output_dim//4, *img_dims)

        # 按通道逐一拆分    
        # 创建一个空列表，用于存储拆分后的张量  
        split_tensors = []
        for channel in range(x_mamba1.size(1)):
            channel_tensors = [tensor[:, channel:channel + 1, :, :] for tensor in [x_mamba1, x_mamba2, x_mamba3, x_mamba4]]
            concatenated_channel = torch.cat(channel_tensors, dim=1) # 拼接在 batch_size 维度上     
            split_tensors.append(concatenated_channel)
        x = torch.cat(split_tensors, dim=1)     
        out = self.out(x)
     
        return out.to(x_dtype)

if __name__ == '__main__':     
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"   
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel, height, width = 1, 16, 20, 20    
    inputs = torch.randn((batch_size, channel, height, width)).to(device)   
  
    # 此模块需要编译,详细编译命令在 compile_module/mamba
    # 对于Linux用户可以直接运行上述的make.sh文件
    # 对于Windows用户需要逐行执行make.sh文件里的内容
    module = CSI(channel).to(device)    

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
  
    print(ORANGE)   
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width),
                                     output_as_string=True, 
                                     output_precision=4,
                                     print_detailed=True) 
    print(RESET)