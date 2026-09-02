'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/TGRS2025-ERM.png     
engine/extre_module/module_images/TGRS2025-ERM.md
论文链接：https://ieeexplore.ieee.org/document/11232501
''' 

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')    

import warnings
warnings.filterwarnings('ignore')     
from calflops import calculate_flops
  
import math
import numpy as np
import torch
import torch.nn as nn     
import torch.nn.functional as F

from engine.extre_module.ultralytics_nn.conv import Conv, DSConv

def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)   
    g_y = conv_y(input)    
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return torch.sigmoid(g) * input   
     
def get_sobel(in_chan, out_chan):
    '''
    filter_x = np.array([    
        [3, 0, -3],
        [10, 0, -10],  
        [3, 0, -3],     
    ]).astype(np.float32) 
    filter_y = np.array([     
        [3, 10, 3],
        [0, 0, 0],
        [-3, -10, -3],
    ]).astype(np.float32)
    '''     
    filter_x = np.array([ 
        [1, 0, -1],
        [2, 0, -2],     
        [1, 0, -1],     
    ]).astype(np.float32)
    filter_y = np.array([ 
        [1, 2, 1],     
        [0, 0, 0], 
        [-1, -2, -1],
    ]).astype(np.float32)    
    filter_x = filter_x.reshape((1, 1, 3, 3)) 
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)   
    
    filter_y = filter_y.reshape((1, 1, 3, 3))  
    filter_y = np.repeat(filter_y, in_chan, axis=1) 
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)    
    filter_x = nn.Parameter(filter_x, requires_grad=False)   
    filter_y = nn.Parameter(filter_y, requires_grad=False)     
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)    
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y   
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))
    return sobel_x, sobel_y  

class ExternalAttention(nn.Module):    
    def __init__(self, in_planes, S=8): 
        super().__init__()   
 
        self.mk = nn.Linear(in_planes, S, bias=False) 
        self.mv = nn.Linear(S, in_planes, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()   

    def init_weights(self):    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):    
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))   
            elif isinstance(m, nn.Conv1d):  
                n = m.kernel_size[0] * m.out_channels     
                m.weight.data.normal_(0, math.sqrt(2. / n))   
            elif isinstance(m, nn.BatchNorm2d):  
                m.weight.data.fill_(1)     
                if m.bias is not None:  
                    m.bias.data.zero_()
   
    def forward(self, x):
        b, c, h, w = x.size()    
        n = h * w  
        queries = x.view(b, c, n)  # 即bs,n,d_model
        queries = queries.permute(0, 2, 1)
        attn = self.mk(queries)  # bs,n,S    
        attn = self.softmax(attn)  # bs,n,S   
        attn = attn / (1e-9 + torch.sum(attn, dim=2, keepdim=True))  # bs,n,S  
        attn = self.mv(attn)  # bs,n,d_model
        attn = attn.permute(0, 2, 1)
        x_attn = attn.view(b, c, h, w) 
        x = x + x_attn  
        x = F.relu(x)
        return x

class ERM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ERM, self).__init__() 

        self.conv_1 = DSConv(out_channel, out_channel, k=3, act=nn.ReLU)   
        self.ea = ExternalAttention(out_channel)  
        self.conv = nn.Conv2d(out_channel, 1, 1) 
        self.sigmoid = nn.Sigmoid()     
        self.relu = nn.ReLU(inplace=True)
   
        ###     
        self.sobel_x, self.sobel_y = get_sobel(out_channel, 1)   
        self.channel = out_channel   
        self.conv_query = nn.Sequential(DSConv(out_channel, out_channel, k=3, act=nn.ReLU))    
        self.conv_key = nn.Sequential(DSConv(out_channel, out_channel, k=3, act=nn.ReLU))     
        self.conv_value = nn.Sequential(DSConv(out_channel, out_channel, k=3, act=nn.ReLU))  ###
  
        self.conv1x1 = nn.ModuleList([])  
        for i in in_channel:
            if i != out_channel:
                self.conv1x1.append(Conv(i, out_channel, 1))
            else:
                self.conv1x1.append(nn.Identity())  

    def forward(self, inputs):
        g, x = inputs
     
        g = self.conv1x1[0](g)   
        x = self.conv1x1[1](x)
     
        fusion = self.conv_1(g + x)     
        fusion = self.ea(fusion)     
     
        ### 
        x_ee = run_sobel(self.sobel_x, self.sobel_y, fusion)
        fg = self.sigmoid(self.conv(x_ee))
        p = fg - .5
        fg = torch.clip(p, 0, 1)  # foreground
        cg = .5 - torch.abs(p)  # confusion area 
        prob = torch.cat([fg, cg], dim=1)     
        # reshape feature & prob    
        b, c, h, w = x.shape     
        f = x.view(b, h * w, -1)
        prob = prob.view(b, 2, h * w)    

        # compute context vector 
        context = torch.bmm(prob, f).permute(0, 2, 1).unsqueeze(3)  # b, 3, c 

        # k q v compute
        query = self.conv_query(x).view(b, self.channel, -1).permute(0, 2, 1)
        key = self.conv_key(context).view(b, self.channel, -1)
        value = self.conv_value(context).view(b, self.channel, -1).permute(0, 2, 1)   
   
        # compute similarity map
        sim = torch.bmm(query, key)  # b, hw, c x b, c, 2   
        sim = (self.channel ** -.5) * sim
        sim = F.softmax(sim, dim=-1)

        # compute refined feature 
        context = torch.bmm(sim, value).permute(0, 2, 1).contiguous().view(b, -1, h, w)
        x_refine_scale = self.sigmoid(self.conv_1(context))    
     
        out = self.relu(x * x_refine_scale)  ###

        return out     
  
if __name__ == '__main__':    
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel_1, channel_2, height, width = 1, 32, 16, 32, 32
    ouc_channel = 32
    inputs_1 = torch.randn((batch_size, channel_1, height, width)).to(device)     
    inputs_2 = torch.randn((batch_size, channel_2, height, width)).to(device)  

    module = ERM([channel_1, channel_2], ouc_channel).to(device)
    
    outputs = module([inputs_1, inputs_2])    
    print(GREEN + f'inputs1.size:{inputs_1.size()} inputs2.size:{inputs_2.size()} outputs.size:{outputs.size()}' + RESET) 
   
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     args=[[inputs_1, inputs_2]],
                                     output_as_string=True,     
                                     output_precision=4,    
                                     print_detailed=True)
    print(RESET)    
