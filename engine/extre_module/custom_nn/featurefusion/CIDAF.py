'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/自研模块-CIDAF.png  
engine/extre_module/module_images/自研模块-CIDAF.md   
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings    
warnings.filterwarnings('ignore')
from calflops import calculate_flops
 
import torch    
import torch.nn as nn    

from engine.extre_module.ultralytics_nn.conv import Conv    

class _InteractionEncoder(nn.Module):
    def __init__(self, channels):
        super().__init__()    
        merged_channels = channels * 4     
        self.local = Conv(merged_channels, merged_channels, 3, g=merged_channels)
        self.mix = Conv(merged_channels, channels, 1)
        self.out = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=True)

    def forward(self, x):
        return self.out(self.mix(self.local(x)))

  
class _ComplementaryResidual(nn.Module):     
    def __init__(self, channels):    
        super().__init__()
        self.discrepancy = Conv(channels, channels, 3, g=channels)     
        self.consistency = Conv(channels, channels, 1)
        self.project = Conv(channels, channels, 1)  
   
    def forward(self, discrepancy, consistency):   
        return self.project(self.discrepancy(discrepancy) + self.consistency(consistency))

 
class CIDAF(nn.Module):
    def __init__(self, inc, ouc):  
        super().__init__()
        if len(inc) != 2:     
            raise ValueError(f"CIDAF expects exactly two input channels, got {len(inc)}")

        self.align_top = Conv(inc[0], ouc, 1)
        self.align_bottom = Conv(inc[1], ouc, 1)
        self.interaction = _InteractionEncoder(ouc)
        self.complementary = _ComplementaryResidual(ouc)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.output = Conv(ouc, ouc, 1)    
  
    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("CIDAF expects a list or tuple with two feature maps")
   
        x_top, x_bottom = x
        if x_top.shape[-2:] != x_bottom.shape[-2:]:
            raise ValueError("CIDAF expects both inputs to have the same spatial shape")
    
        top = self.align_top(x_top)
        bottom = self.align_bottom(x_bottom)  

        discrepancy = torch.abs(top - bottom)
        consistency = top * bottom   
        interaction = torch.cat((top, bottom, discrepancy, consistency), dim=1)

        gate_logits = self.interaction(interaction)
        batch_size, doubled_channels, height, width = gate_logits.shape     
        gate_logits = gate_logits.view(batch_size, 2, doubled_channels // 2, height, width)
        gate_weights = torch.softmax(gate_logits, dim=1)  
        top_weight, bottom_weight = gate_weights.unbind(dim=1)

        residual = self.complementary(discrepancy, consistency)
        fused = top_weight * top + bottom_weight * bottom + self.residual_scale * residual
        return self.output(fused)

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    batch_size, channel_1, channel_2, height, width = 1, 32, 16, 32, 32
    ouc_channel = 32 
    inputs_1 = torch.randn((batch_size, channel_1, height, width)).to(device)  
    inputs_2 = torch.randn((batch_size, channel_2, height, width)).to(device)

    module = CIDAF([channel_1, channel_2], ouc_channel).to(device) 

    outputs = module([inputs_1, inputs_2]) 
    print(GREEN + f'inputs1.size:{inputs_1.size()} inputs2.size:{inputs_2.size()} outputs.size:{outputs.size()}' + RESET) 

    print(ORANGE)    
    flops, macs, _ = calculate_flops(model=module,  
                                     args=[[inputs_1, inputs_2]],   
                                     output_as_string=True,
                                     output_precision=4,   
                                     print_detailed=True)     
    print(RESET)