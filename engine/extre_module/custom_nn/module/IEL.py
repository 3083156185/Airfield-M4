'''   
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/CVPR2025-IEL.png
engine/extre_module/module_images/CVPR2025-IEL.md   
论文链接：https://arxiv.org/pdf/2502.20272    
'''

import warnings     
warnings.filterwarnings('ignore')
from calflops import calculate_flops     

import torch
import torch.nn as nn     

# Intensity Enhancement Layer
class IEL(nn.Module):    
    def __init__(self, in_dim, out_dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()
     
        hidden_features = int(in_dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(in_dim, hidden_features*2, kernel_size=1, bias=bias)
     
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
       
        self.project_out = nn.Conv2d(hidden_features, out_dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()    

    def forward(self, x):  
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)   
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x 

if __name__ == '__main__':   
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = IEL(in_channel, out_channel).to(device)
 
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,  
                                     input_shape=(batch_size, in_channel, height, width), 
                                     output_as_string=True,     
                                     output_precision=4,    
                                     print_detailed=True)
    print(RESET)