'''     
本文件由BiliBili：魔傀面具整理    
engine/extre_module/module_images/自研模块-DAF.png
engine/extre_module/module_images/自研模块-DAF.md     
'''
  
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings    
warnings.filterwarnings('ignore')
from calflops import calculate_flops
 
import torch    
import torch.nn as nn     

from engine.extre_module.ultralytics_nn.conv import Conv, DSConv     

class DynamicAlignFusion(nn.Module): 
    def __init__(self, inc, ouc) -> None:     
        super().__init__()  
  
        self.conv_align1 = Conv(inc[0], ouc, 1)    
        self.conv_align2 = Conv(inc[1], ouc, 1)
  
        self.conv_concat = DSConv(ouc * 2, ouc * 2, 3)
        self.sigmoid = nn.Sigmoid()  
        
        self.x1_param = nn.Parameter(torch.ones((1, ouc, 1, 1)) * 0.5, requires_grad=True)
        self.x2_param = nn.Parameter(torch.ones((1, ouc, 1, 1)) * 0.5, requires_grad=True)
 
        self.conv_final = Conv(ouc, ouc, 1) 
     
    def forward(self, x):
        self._clamp_abs(self.x1_param.data, 1.0)
        self._clamp_abs(self.x2_param.data, 1.0)
  
        x1, x2 = x
        x1, x2 = self.conv_align1(x1), self.conv_align2(x2)
        x_concat = self.sigmoid(self.conv_concat(torch.cat([x1, x2], dim=1)))  
        x1_weight, x2_weight = torch.chunk(x_concat, 2, dim=1)
        x1, x2 = x1 * x1_weight, x2 * x2_weight 
        
        return self.conv_final(x1 * self.x1_param + x2 * self.x2_param)

    def _clamp_abs(self, data, value):
        with torch.no_grad(): 
            sign=data.sign()
            data.abs_().clamp_(value)     
            data*=sign     

if __name__ == '__main__':   
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"     
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel_1, channel_2, height, width = 1, 32, 16, 32, 32 
    ouc_channel = 32    
    inputs_1 = torch.randn((batch_size, channel_1, height, width)).to(device)   
    inputs_2 = torch.randn((batch_size, channel_2, height, width)).to(device)
 
    module = DynamicAlignFusion([channel_1, channel_2], ouc_channel).to(device)  

    outputs = module([inputs_1, inputs_2])
    print(GREEN + f'inputs1.size:{inputs_1.size()} inputs2.size:{inputs_2.size()} outputs.size:{outputs.size()}' + RESET)   

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,  
                                     args=[[inputs_1, inputs_2]],
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)   
    print(RESET) 
