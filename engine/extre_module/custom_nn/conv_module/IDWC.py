'''    
本文件由BiliBili：魔傀面具整理     
engine/extre_module/module_images/CVPR2024-InceptionDWConv2d.png 
论文链接：https://arxiv.org/pdf/2303.16900 
'''  
     
import os, sys   
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings     
warnings.filterwarnings('ignore')   
from calflops import calculate_flops    

import torch    
import torch.nn as nn

from engine.extre_module.ultralytics_nn.conv import Conv     

class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution
    """   
    def __init__(self, in_channels, out_chanels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):   
        super().__init__()    
  
        gc = int(in_channels * branch_ratio) # channel numbers of a convolution branch  
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size//2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)
 
        self.conv1x1 = Conv(in_channels, out_chanels) if in_channels != out_chanels else nn.Identity() 
        
    def forward(self, x):   
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)   
        x = torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), 
            dim=1,  
        )
        return self.conv1x1(x)    
 
if __name__ == '__main__':   
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)     

    module = InceptionDWConv2d(in_channel, out_channel).to(device)    
    
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)  

    print(ORANGE)    
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,   
                                     print_detailed=True) 
    print(RESET)