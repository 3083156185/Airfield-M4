'''
本文件由BiliBili：魔傀面具整理  
engine/extre_module/module_images/CVPR2026-AFFN.png     
engine/extre_module/module_images/CVPR2026-AFFN.md
论文链接：https://arxiv.org/pdf/2603.22794
'''     

import warnings    
warnings.filterwarnings('ignore')
from calflops import calculate_flops   

import torch
import torch.nn as nn
import torch.nn.functional as F 
from einops import rearrange    
     
class AFFN(nn.Module):    
    def __init__(self, in_features, hidden_features, out_features, bias=False):
     
        super(AFFN, self).__init__() 
        self.patch_size = 4    
        self.dim = in_features    
   
        self.project_in = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1, bias=bias)     
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,     
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias)  
   
        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1))) 
     
        # 自相关融合权重
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 控制频域融合强度  
        self.beta = nn.Parameter(torch.tensor(0.5))   # 控制空间域融合强度

    def forward(self, x):
        x = self.project_in(x)   
     
        x_patch = rearrange(
            x, 'b c (h ph) (w pw) -> b c h w ph pw',
            ph=self.patch_size, pw=self.patch_size  
        )
     
        # FFT     
        Xf = torch.fft.rfft2(x_patch.float())
        Xf = Xf * self.fft    
        # 自相关功率谱 
        power = Xf * torch.conj(Xf)          # |X|^2
        R = torch.fft.irfft2(power, s=(self.patch_size, self.patch_size)) 
    
        # 融合（频域 + 空间域）
        Xf_new = Xf + self.alpha * power     # 频域增强周期结构    
        x_patch_new = torch.fft.irfft2(Xf_new, s=(self.patch_size, self.patch_size)) 
        x_patch_new = x_patch_new + self.beta * R  # 空间域增强    

        # 重组
        x = rearrange(     
            x_patch_new, 'b c h w ph pw -> b c (h ph) (w pw)',
            ph=self.patch_size, pw=self.patch_size
        )

        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x    

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"  
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   
    batch_size, in_channel, hidden_channel, out_channel, height, width = 1, 16, 64, 16, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)  
    
    module = AFFN(in_features=in_channel, hidden_features=hidden_channel, out_features=out_channel).to(device)

    outputs = module(inputs)    
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)  

    print(ORANGE)    
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True, 
                                     output_precision=4,     
                                     print_detailed=True)
    print(RESET)    
