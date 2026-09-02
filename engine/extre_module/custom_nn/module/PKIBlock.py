'''
本文件由BiliBili：魔傀面具整理 
engine/extre_module/module_images/CVPR2024-PKIBlock.png
engine/extre_module/module_images/CVPR2024-PKIBlock.md    
论文链接：https://openaccess.thecvf.com/content/CVPR2024/papers/Cai_Poly_Kernel_Inception_Network_for_Remote_Sensing_Detection_CVPR_2024_paper.pdf
'''  

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')    
     
import warnings     
warnings.filterwarnings('ignore')     
from calflops import calculate_flops 
 
import torch, math
import torch.nn as nn

from engine.extre_module.ultralytics_nn.conv import Conv, autopad  

def make_divisible(x, divisor):     
    """  
    Returns the nearest number that is divisible by the given divisor. 
     
    Args: 
        x (int): The number to make divisible.   
        divisor (int | torch.Tensor): The divisor.
     
    Returns:
        (int): The nearest number divisible by the divisor.     
    """ 
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor

class GSiLU(nn.Module):
    """Global Sigmoid-Gated Linear Unit, reproduced from paper <SIMPLE CNN FOR VISION>"""   
    def __init__(self):     
        super().__init__()
        self.adpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return x * torch.sigmoid(self.adpool(x))
     
class CAA(nn.Module):     
    def __init__(self, ch, h_kernel_size = 11, v_kernel_size = 11) -> None:     
        super().__init__()
        
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = Conv(ch, ch)
        self.h_conv = nn.Conv2d(ch, ch, (1, h_kernel_size), 1, (0, h_kernel_size // 2), 1, ch) 
        self.v_conv = nn.Conv2d(ch, ch, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), 1, ch)
        self.conv2 = Conv(ch, ch)
        self.act = nn.Sigmoid()
    
    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))  
        return attn_factor     
    
class PKIBlock(nn.Module):
    def __init__(self, inc, ouc, kernel_sizes=(3, 5, 7, 9, 11), expansion=1.0, with_caa=True, caa_kernel_size=11, add_identity=True) -> None:
        super().__init__()  
        hidc = make_divisible(int(ouc * expansion), 8)     
  
        self.pre_conv = Conv(inc, hidc)     
        self.dw_conv = nn.ModuleList(nn.Conv2d(hidc, hidc, kernel_size=k, padding=autopad(k), groups=hidc) for k in kernel_sizes)
        self.pw_conv = Conv(hidc, hidc)    
        self.post_conv = Conv(hidc, ouc)
 
        if with_caa:    
            self.caa_factor = CAA(hidc, caa_kernel_size, caa_kernel_size)
        else:    
            self.caa_factor = None     
        
        self.add_identity = add_identity and inc == ouc
   
    def forward(self, x): 
        x = self.pre_conv(x)
   
        y = x
        x = self.dw_conv[0](x)
        x = torch.sum(torch.stack([x] + [layer(x) for layer in self.dw_conv[1:]], dim=0), dim=0)  
        x = self.pw_conv(x) 
   
        if self.caa_factor is not None:
            y = self.caa_factor(y)
        if self.add_identity:
            y = x * y
            x = x + y 
        else: 
            x = x * y
  
        x = self.post_conv(x)   
        return x

if __name__ == '__main__':    
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32     
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)
  
    module = PKIBlock(in_channel, out_channel, kernel_sizes=(3, 5, 7, 9, 11)).to(device)  

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)  
 
    print(ORANGE)  
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True, 
                                     output_precision=4,
                                     print_detailed=True)     
    print(RESET) 
