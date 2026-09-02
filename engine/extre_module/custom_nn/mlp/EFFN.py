''' 
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/ECCV2024-EFFN.png    
论文链接：https://arxiv.org/pdf/2409.01686
'''   

import warnings
warnings.filterwarnings('ignore')   
from calflops import calculate_flops  

import torch  
import torch.nn as nn   
import torch.nn.functional as F
    
class EFFN(nn.Module): 
    def __init__(self, in_features, hidden_features=None, out_features=None, bias=False):
        
        super(EFFN, self).__init__()  
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dwconv1 = nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1, groups=in_features, bias=bias)
        self.dwconv2 = nn.Conv2d(in_features*2, in_features*2, kernel_size=3, stride=1, padding=1, groups=in_features, bias=bias)
        self.project_out = nn.Conv2d(in_features*4, out_features, kernel_size=1, bias=bias)
        self.weight = nn.Sequential(    
            nn.Conv2d(in_features, in_features // 16, 1, bias=True),
            nn.BatchNorm2d(in_features // 16), 
            nn.ReLU(True),
            nn.Conv2d(in_features // 16, in_features, 1, bias=True),  
            nn.Sigmoid())
        self.weight1 = nn.Sequential(
            nn.Conv2d(in_features*2, in_features // 16, 1, bias=True),  
            nn.BatchNorm2d(in_features // 16),   
            nn.ReLU(True),
            nn.Conv2d(in_features // 16, in_features*2, 1, bias=True),
            nn.Sigmoid())
     
    def forward(self, x):   

        x_f = torch.abs(self.weight(torch.fft.fft2(x.float()).real)*torch.fft.fft2(x.float()))     
        x_f_gelu = F.gelu(x_f) * x_f

        x_s   = self.dwconv1(x)    
        x_s_gelu = F.gelu(x_s) * x_s     

        x_f = torch.fft.fft2(torch.cat((x_f_gelu,x_s_gelu),1))   
        x_f = torch.abs(torch.fft.ifft2(self.weight1(x_f.real) * x_f))   
     
        x_s = self.dwconv2(torch.cat((x_f_gelu,x_s_gelu),1))
        out = self.project_out(torch.cat((x_f,x_s),1))
 
        return out 
    
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, hidden_channel, out_channel, height, width = 1, 16, 64, 32, 32, 32  
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)   
 
    module = EFFN(in_features=in_channel, hidden_features=hidden_channel, out_features=out_channel).to(device)  

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module, 
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,    
                                     output_precision=4,
                                     print_detailed=True)    
    print(RESET)     
