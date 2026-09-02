'''    
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/LEGBlock.png
论文链接：https://arxiv.org/abs/2503.14012  
'''    
  
import os, sys    
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..') 

import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops 
   
import torch, math
import torch.nn as nn    
from torch import Tensor 
from timm.layers import DropPath 
from typing import List 
     
from engine.extre_module.ultralytics_nn.conv import Conv   
   
class Conv_Extra(nn.Module):  
    def __init__(self, channel):
        super(Conv_Extra, self).__init__()    
        self.block = nn.Sequential(Conv(channel, 64, 1),     
                                   Conv(64, 64, 3),  
                                   Conv(64, channel, 1, act=False))
    def forward(self, x):
        out = self.block(x)     
        return out   
 
class Scharr(nn.Module):
    def __init__(self, channel):
        super(Scharr, self).__init__()
        # 定义Scharr滤波器  
        scharr_x = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        scharr_y = torch.tensor([[-3., -10., -3.], [0., 0., 0.], [3., 10., 3.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.conv_x = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)     
        self.conv_y = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        # 将Sobel滤波器分配给卷积层   
        self.conv_x.weight.data = scharr_x.repeat(channel, 1, 1, 1)    
        self.conv_y.weight.data = scharr_y.repeat(channel, 1, 1, 1)
        self.norm = nn.BatchNorm2d(channel)    
        self.conv_extra = Conv_Extra(channel)
  
    def forward(self, x): 
        # show_feature(x)
        # 应用卷积操作
        edges_x = self.conv_x(x)
        edges_y = self.conv_y(x)
        # 计算边缘和高斯分布强度（可以选择不同的方式进行融合，这里使用平方和开根号）    
        scharr_edge = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        scharr_edge = self.act(self.norm(scharr_edge))     
        out = self.conv_extra(x + scharr_edge)
        # show_feature(out)  

        return out   
     
class Gaussian(nn.Module):     
    def __init__(self, dim, size, sigma, feature_extra=True):
        super().__init__()  
        self.feature_extra = feature_extra   
        gaussian = self.gaussian_kernel(size, sigma)
        gaussian = nn.Parameter(data=gaussian, requires_grad=False).clone()     
        self.gaussian = nn.Conv2d(dim, dim, kernel_size=size, stride=1, padding=int(size // 2), groups=dim, bias=False)
        self.gaussian.weight.data = gaussian.repeat(dim, 1, 1, 1)    
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.SiLU()
        if feature_extra == True:
            self.conv_extra = Conv_Extra(dim)
     
    def forward(self, x):
        edges_o = self.gaussian(x)
        gaussian = self.act(self.norm(edges_o))  
        if self.feature_extra == True:
            out = self.conv_extra(x + gaussian)  
        else:
            out = gaussian 
        return out    
 
    def gaussian_kernel(self, size: int, sigma: float):    
        kernel = torch.FloatTensor([    
            [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
             for x in range(-size // 2 + 1, size // 2 + 1)]    
             for y in range(-size // 2 + 1, size // 2 + 1)
             ]).unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()

class LFEA(nn.Module):     
    def __init__(self, channel):
        super(LFEA, self).__init__()    
        self.channel = channel
        t = int(abs((math.log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d = self.block = Conv(channel, channel, 3)   
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)   
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.BatchNorm2d(channel)

    def forward(self, c, att):    
        att = c * att + c    
        att = self.conv2d(att)  
        wei = self.avg_pool(att)    
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        x = self.norm(c + att * wei)

        return x

class LEGBlock(nn.Module):
    def __init__(self,    
                 inc,    
                 dim,
                 stage=1,
                 mlp_ratio=2,
                 drop_path=0.1,  
                 ):
        super().__init__()
        self.stage = stage
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
     
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_layer: List[nn.Module] = [
            Conv(dim, mlp_hidden_dim, 1),     
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)] 

        self.mlp = nn.Sequential(*mlp_layer)
        self.LFEA = LFEA(dim)   
  
        if stage == 0:
            self.Scharr_edge = Scharr(dim)
        else:
            self.gaussian = Gaussian(dim, 5, 1.0)
        self.norm = nn.BatchNorm2d(dim)
     
        self.conv1x1 = Conv(inc, dim, 1) if inc != dim else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1x1(x)     
        if self.stage == 0:
            att = self.Scharr_edge(x)    
        else:   
            att = self.gaussian(x)  
        x_att = self.LFEA(x, att)
        x = x + self.norm(self.drop_path(self.mlp(x_att)))     
        return x   
  
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"     
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32     
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)
     
    module = LEGBlock(in_channel, out_channel).to(device)   
   
    outputs = module(inputs)   
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)     
    flops, macs, _ = calculate_flops(model=module,  
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,  
                                     print_detailed=True)
    print(RESET)