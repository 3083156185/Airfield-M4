'''  
本文件由BiliBili：魔傀面具整理  
engine/extre_module/module_images/自研模块-WGFS.png  
engine/extre_module/module_images/自研模块-WGFS.md 
'''
    
import warnings     
warnings.filterwarnings('ignore')  
import os, sys  
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')    
from calflops import calculate_flops    
    
import torch
import torch.nn as nn    
  

class HaarCut(nn.Module):
    """Haar wavelet 2D downsampling with learnable frequency band gating.     

    Decomposes 2x2 pixel neighborhoods into 4 orthogonal sub-bands
    (LL low-freq, LH horizontal-detail, HL vertical-detail, HH diagonal-detail) 
    with independent sigmoid gates to learn task-adaptive band importance.
    """ 
    def __init__(self, in_channels, out_channels):     
        super().__init__() 
        self.band_weight = nn.Parameter(torch.zeros(4))  # [LL, LH, HL, HH]
        self.conv_fusion = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, stride=1) 
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):     
        x00 = x[:, :, 0::2, 0::2]  # top-left 
        x01 = x[:, :, 0::2, 1::2]  # top-right   
        x10 = x[:, :, 1::2, 0::2]  # bottom-left  
        x11 = x[:, :, 1::2, 1::2]  # bottom-right

        # 2D Haar wavelet sub-band decomposition 
        LL = (x00 + x01 + x10 + x11) * 0.5  # low-frequency structure  
        LH = (x00 - x01 + x10 - x11) * 0.5  # horizontal detail    
        HL = (x00 + x01 - x10 - x11) * 0.5  # vertical detail
        HH = (x00 - x01 - x10 + x11) * 0.5  # diagonal detail

        bw = torch.sigmoid(self.band_weight)  # independent gating, each in (0, 1)    
        x = torch.cat([LL * bw[0], LH * bw[1], HL * bw[2], HH * bw[3]], dim=1)
        return self.batch_norm(self.conv_fusion(x))

 
class WGFS(nn.Module):    
    def __init__(self, in_channels=3, out_channels=96):
        super().__init__()
        out_c14 = int(out_channels / 4)
        out_c12 = int(out_channels / 2)

        self.conv_init = nn.Conv2d(in_channels, out_c14, kernel_size=7, stride=1, padding=3)     
   
        # Stage 1: original → 2x downsampling
        self.conv_1 = nn.Conv2d(out_c14, out_c12, kernel_size=3, stride=1, padding=1, groups=out_c14)    
        self.conv_x1 = nn.Conv2d(out_c12, out_c12, kernel_size=3, stride=2, padding=1, groups=out_c12)  
        self.batch_norm_x1 = nn.BatchNorm2d(out_c12)    
        self.haar_cut_c = HaarCut(out_c14, out_c12)    
        self.fusion1 = nn.Conv2d(out_channels, out_c12, kernel_size=1, stride=1)

        # Stage 2: 2x → 4x downsampling
        self.conv_2 = nn.Conv2d(out_c12, out_channels, kernel_size=3, stride=1, padding=1, groups=out_c12)  
        self.conv_x2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels)     
        self.batch_norm_x2 = nn.BatchNorm2d(out_channels)
        self.max_m = nn.MaxPool2d(kernel_size=2, stride=2)  
        self.batch_norm_m = nn.BatchNorm2d(out_channels)
        self.haar_cut_r = HaarCut(out_c12, out_channels) 
        self.fusion2 = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, stride=1)   

    def forward(self, x):
        x = self.conv_init(x)  # [B, C/4, H, W]

        # Stage 1
        c = self.haar_cut_c(x) 
        xc = self.batch_norm_x1(self.conv_x1(self.conv_1(x)))   
        x = self.fusion1(torch.cat([xc, c], dim=1))  # [B, C/2, H/2, W/2]
  
        # Stage 2     
        r = x
        x_exp = self.conv_2(r)          # [B, C, H/2, W/2]   
        conv_d = self.batch_norm_x2(self.conv_x2(x_exp))
        max_d  = self.batch_norm_m(self.max_m(x_exp))
        haar_d = self.haar_cut_r(r)
        x = self.fusion2(torch.cat([conv_d, haar_d, max_d], dim=1))  # [B, C, H/4, W/4]
        return x
    

if __name__ == '__main__':   
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device) 
  
    module = WGFS(in_channel, out_channel).to(device)
    
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
    
    print(ORANGE)  
    flops, macs, _ = calculate_flops(model=module,     
                                     input_shape=(batch_size, in_channel, height, width),    
                                     output_as_string=True,
                                     output_precision=4,    
                                     print_detailed=True)
    print(RESET)
