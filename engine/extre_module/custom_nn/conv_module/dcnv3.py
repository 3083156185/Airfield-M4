'''    
本文件由BiliBili：魔傀面具整理    
engine/extre_module/module_images/CVPR2023-DCNV3.md
论文链接：https://arxiv.org/abs/2211.05778     
'''

import os, sys 
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings     
warnings.filterwarnings('ignore') 
from calflops import calculate_flops

try:     
    from compile_module.ops_dcnv3.modules.dcnv3 import DCNv3     
except:
    pass    
 
import torch
from torch import nn
     
from engine.extre_module.ultralytics_nn.conv import Conv, autopad
  
class DeformConvV3(nn.Module):
    def __init__(self, inc, ouc, k=1, s=1, p=None, g=1, d=1, act=True): 
        super().__init__()  
        
        if inc != ouc:  
            self.stem_conv = Conv(inc, ouc, k=1)
        self.dcnv3 = DCNv3(ouc, kernel_size=k, stride=s, pad=autopad(k, p, d), group=g, dilation=d)
        self.bn = nn.BatchNorm2d(ouc)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()     
    
    def forward(self, x):    
        if hasattr(self, 'stem_conv'):
            x = self.stem_conv(x)   
        x = x.permute(0, 2, 3, 1)     
        x = self.dcnv3(x)  
        x = x.permute(0, 3, 1, 2)
        x = self.act(self.bn(x))
        return x

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"  
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')     
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)     
   
    # 此模块需要编译,详细编译命令在 compile_module/ops_dcnv3    
    # 对于Linux用户可以直接运行上述的make.sh文件     
    # 对于Windows用户需要逐行执行make.sh文件里的内容
    module = DeformConvV3(in_channel, out_channel).to(device)  

    outputs = module(inputs)    
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE) 
    flops, macs, _ = calculate_flops(model=module,  
                                     input_shape=(batch_size, in_channel, height, width),     
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)