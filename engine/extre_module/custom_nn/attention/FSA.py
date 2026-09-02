'''   
本文件由BiliBili：魔傀面具整理 
engine/extre_module/module_images/BIBM2024-FSA.png  
论文链接：https://arxiv.org/pdf/2406.07952
'''
     
import warnings
warnings.filterwarnings('ignore')  
from calflops import calculate_flops   

import torch    
import torch.nn as nn 
    
class Adaptive_global_filter(nn.Module):  
    def __init__(self, ratio=10, dim=32, H=512, W=512):
        super().__init__()  
        self.ratio = ratio
        self.filter = nn.Parameter(torch.randn(dim, H, W, 2, dtype=torch.float32), requires_grad=True)     
        self.mask_low = nn.Parameter(data=torch.zeros(size=(H, W)), requires_grad=False)  
        self.mask_high = nn.Parameter(data=torch.ones(size=(H, W)), requires_grad=False) 

    def forward(self, x):
        b, c, h, w = x.shape   
        crow, ccol = int(h / 2), int(w / 2)     

        mask_lowpass = self.mask_low     
        mask_lowpass[crow - self.ratio:crow + self.ratio, ccol - self.ratio:ccol + self.ratio] = 1
  
        mask_highpass = self.mask_high   
        mask_highpass[crow - self.ratio:crow + self.ratio, ccol - self.ratio:ccol + self.ratio] = 0

        x_fre = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1), norm='ortho'))
        weight = torch.view_as_complex(self.filter)   

        x_fre_low = torch.mul(x_fre, mask_lowpass)
        x_fre_high = torch.mul(x_fre, mask_highpass) 

        x_fre_low = torch.mul(x_fre_low, weight)
        x_fre_new = x_fre_low + x_fre_high
        x_out = torch.fft.ifft2(torch.fft.ifftshift(x_fre_new, dim=(-2, -1))).real    
        return x_out    
 
class SpatialAttention(nn.Module):  # Spatial Attention Module
    def __init__(self):
        super(SpatialAttention, self).__init__() 
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)   
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): 
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True) 
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out) 
        out = self.sigmoid(out)
        result = x * out
        return result
    
class FSA(nn.Module):  
    def __init__(self, input_channel=64, size=(20, 20), ratio=10):
        super(FSA, self).__init__()    
        self.agf = Adaptive_global_filter(ratio=ratio, dim=input_channel, H=size[0], W=size[1])  
        self.sa = SpatialAttention()
     
    def forward(self, x):
        f_out = self.agf(x)
        sa_out = self.sa(x)  
        result = f_out + sa_out   
        return result
    
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')     
    batch_size, channel, height, width = 1, 16, 32, 32
    inputs = torch.randn((batch_size, channel, height, width)).to(device)

    # 此模型不支持多尺度训练，size参数为当前特征图的尺寸(h, w)
    module = FSA(channel, size=(height, width)).to(device)     

    outputs = module(inputs)     
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)   

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width),   
                                     output_as_string=True,    
                                     output_precision=4,
                                     print_detailed=True)   
    print(RESET)