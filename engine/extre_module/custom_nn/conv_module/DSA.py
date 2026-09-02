'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/DSA.png
论文链接：https://www.techrxiv.org/users/628671/articles/775010-deformable-spatial-attention-networks-enhancing-lightweight-convolutional-models-for-vision-tasks
'''    
   
import os, sys   
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings   
warnings.filterwarnings('ignore')
from calflops import calculate_flops 
    
import torch   
import torch.nn as nn
try:   
    from ops_dscn.modules import DSCNX, DSCNY   
except: 
    pass
     
from engine.extre_module.ultralytics_nn.conv import Conv
   
class DSCNPair(nn.Module):
    def __init__(self, d_model, kernel_size, dw_kernel_size, pad, stride, dilation, group): 
        super().__init__()
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.pad = pad     
        self.stride = stride
        self.dilation = dilation
        self.group = group
        self.conv0 = nn.Conv2d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)
     
        self.dscn_x = DSCNX(d_model, kernel_size, dw_kernel_size, stride=stride, pad=pad, dilation=dilation, group=group)#, offset_scale=0.4)     
        self.dscn_y = DSCNY(d_model, kernel_size, dw_kernel_size, stride=stride, pad=pad, dilation=dilation, group=group)#, offset_scale=0.4)    
        self.conv = nn.Conv2d(d_model, d_model, 1)

    def forward(self,x):   
        u = x.clone()     
        x = self.conv0(x)
        attn = x.permute(0,2,3,1)
        attn = self.dscn_x(attn,x) 
        attn = self.dscn_y(attn,x)
        attn = attn.permute(0,3,1,2)     
        attn = self.conv(attn)
        return u*attn
    
def autopad(kernel_size: int, padding: int = None, dilation: int = 1):
    assert kernel_size % 2 == 1, 'if use autopad, kernel size must be odd'    
    if dilation > 1:
        kernel_size = dilation * (kernel_size - 1) + 1    
    if padding is None:
        padding = kernel_size // 2
    return padding

class DSA(nn.Module):   
    def __init__(self, inc, d_model, kernel_size=7, dw_kernel_size=5, stride=1, dilation=1, group=1):  
        super().__init__()
        pad = autopad(kernel_size, None, dilation)    
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()    
        self.spatial_gating_unit = DSCNPair(d_model, kernel_size, dw_kernel_size, pad, stride, dilation, group) 
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)   

        self.conv1x1 = Conv(inc, d_model) if inc != d_model else nn.Identity()
 
    def forward(self, x):    
        x = self.conv1x1(x)
        shorcut = x.clone()  
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut   
        return x

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32 
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # 此模块需要编译,详细编译命令在 compile_module/ops_dscn
    # 对于Linux用户可以直接运行上述的make.sh文件     
    # 对于Windows用户需要逐行执行make.sh文件里的内容
    module = DSA(in_channel, out_channel, kernel_size=7).to(device)    
   
    outputs = module(inputs)     
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)     

    print(ORANGE)   
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,
                                     output_precision=4, 
                                     print_detailed=True)     
    print(RESET)