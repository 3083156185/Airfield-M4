'''     
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/ACMMM2024-WFU.png  
论文链接：https://arxiv.org/pdf/2407.19768   
'''
 
import os, sys     
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')     
  
import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops 
   
import torch
import torch.nn as nn    
import torch.nn.functional as F

from engine.extre_module.ultralytics_nn.conv import Conv
   
class HaarWavelet(nn.Module):
    def __init__(self, in_channels, grad=False):   
        super(HaarWavelet, self).__init__()  
        self.in_channels = in_channels

        self.haar_weights = torch.ones(4, 1, 2, 2)
        #h
        self.haar_weights[1, 0, 0, 1] = -1 
        self.haar_weights[1, 0, 1, 1] = -1     
        #v
        self.haar_weights[2, 0, 1, 0] = -1 
        self.haar_weights[2, 0, 1, 1] = -1    
        #d
        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1 
   
        self.haar_weights = torch.cat([self.haar_weights] * self.in_channels, 0)   
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = grad    

    def forward(self, x, rev=False): 
        if not rev:  
            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.in_channels) / 4.0
            out = out.reshape([x.shape[0], self.in_channels, 4, x.shape[2] // 2, x.shape[3] // 2])     
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.in_channels * 4, x.shape[2] // 2, x.shape[3] // 2])  
            return out    
        else:
            out = x.reshape([x.shape[0], 4, self.in_channels, x.shape[2], x.shape[3]])     
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.in_channels * 4, x.shape[2], x.shape[3]])   
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.in_channels) 
 
class WFU(nn.Module):
    def __init__(self, in_chn, ou_chn):
        super(WFU, self).__init__()   
        dim_big, dim_small = in_chn
        self.dim = dim_big     
        self.HaarWavelet = HaarWavelet(dim_big, grad=False)    
        self.InverseHaarWavelet = HaarWavelet(dim_big, grad=False)
        self.RB = nn.Sequential(
            nn.Conv2d(dim_big, dim_big, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim_big, dim_big, kernel_size=3, padding=1),  
        )    

        self.channel_tranformation = nn.Sequential(
            nn.Conv2d(dim_big+dim_small, dim_big+dim_small // 1, kernel_size=1, padding=0),     
            nn.ReLU(),    
            nn.Conv2d(dim_big+dim_small // 1, dim_big*3, kernel_size=1, padding=0),     
        )
  
        self.conv1x1 = Conv(dim_big, ou_chn) if dim_big != ou_chn else nn.Identity()

    def forward(self, x): 
        x_big, x_small = x
        haar = self.HaarWavelet(x_big, rev=False)
        a = haar.narrow(1, 0, self.dim)
        h = haar.narrow(1, self.dim, self.dim)  
        v = haar.narrow(1, self.dim*2, self.dim) 
        d = haar.narrow(1, self.dim*3, self.dim)  
   
        hvd = self.RB(h + v + d)
        a_ = self.channel_tranformation(torch.cat([x_small, a], dim=1))   
        out = self.InverseHaarWavelet(torch.cat([hvd, a_], dim=1), rev=True)   
        return self.conv1x1(out)   

if __name__ == '__main__':  
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel_1, height_1, width_1 = 1, 32, 40, 40
    batch_size, channel_2, height_2, width_2 = 1, 64, 20, 20
    ouc_channel = 64
    inputs_1 = torch.randn((batch_size, channel_1, height_1, width_1)).to(device)
    inputs_2 = torch.randn((batch_size, channel_2, height_2, width_2)).to(device)   

    # 可以参考进阶视频中的 engine/extre_module/custom_nn/featurefusion/mpca.py 的教程    
    module = WFU([channel_1, channel_2], ouc_channel).to(device)

    outputs = module([inputs_1, inputs_2])    
    print(GREEN + f'inputs1.size:{inputs_1.size()} inputs2.size:{inputs_2.size()} outputs.size:{outputs.size()}' + RESET)  
  
    print(ORANGE)    
    flops, macs, _ = calculate_flops(model=module,    
                                     args=[[inputs_1, inputs_2]],   
                                     output_as_string=True, 
                                     output_precision=4,    
                                     print_detailed=True)
    print(RESET)