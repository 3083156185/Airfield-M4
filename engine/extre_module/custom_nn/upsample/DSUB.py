'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/CVPR2025-DSUB.png 
engine/extre_module/module_images/CVPR2025-DSUB.md
论文链接：https://openaccess.thecvf.com/content/CVPR2025/papers/Wazir_Rethinking_Decoder_Design_Improving_Biomarker_Segmentation_Using_Depth-to-Space_Restoration_and_CVPR_2025_paper.pdf
'''

import os, sys  
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')    
   
import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops
    
import torch     
import torch.nn as nn
import torch.nn.functional as F
   
from engine.extre_module.ultralytics_nn.conv import Conv, DSConv    
    
class DSUB(nn.Module):
    def __init__(self, inc):    
        super().__init__()
     
        self.conv3x3_1 = Conv(inc, inc, 3) 
        self.conv3x3_2 = Conv(inc // 4, inc // 4, 3)
        self.convblock = DSConv(inc // 4, inc // 4, 3)
    
    def forward(self, x):
        x = self.conv3x3_1(x)   
        x = F.pixel_shuffle(x, upscale_factor=2)  
        x = self.conv3x3_2(x)
        x = self.convblock(x)
        return x

if __name__ == '__main__':    
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   
    batch_size, channel, height, width = 1, 16, 32, 32     
    inputs = torch.randn((batch_size, channel, height, width)).to(device)    
     
    module = DSUB(channel).to(device)

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)    
  
    print(ORANGE)  
    flops, macs, _ = calculate_flops(model=module,    
                                     input_shape=(batch_size, channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)     
