''' 
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/CVPR2025-EDFFN.png   
论文链接：https://arxiv.org/pdf/2405.14343
'''     
 
import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops  

import torch   
import torch.nn as nn 
import torch.nn.functional as F  
from einops import rearrange

class EDFFN(nn.Module):     
    def __init__(self, in_features, hidden_features=None, out_features=None, bias=False): 
        super(EDFFN, self).__init__()     
 
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.patch_size = 8    

        self.dim = in_features
        self.project_in = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1, bias=bias)   
 
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)   

        self.fft = nn.Parameter(torch.ones((out_features, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias)
     
    def forward(self, x):    
        x_dtype = x.dtype    
        x = self.project_in(x)  
        x1, x2 = self.dwconv(x).chunk(2, dim=1)   
        x = F.gelu(x1) * x2
        x = self.project_out(x)     

        b, c, h, w = x.shape 
        h_n = (8 - h % 8) % 8  
        w_n = (8 - w % 8) % 8
        
        x = torch.nn.functional.pad(x, (0, w_n, 0, h_n), mode='reflect')    
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,   
                            patch2=self.patch_size)    
        x_patch_fft = torch.fft.rfft2(x_patch.float())     
        x_patch_fft = x_patch_fft * self.fft     
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))  
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
  
        x=x[:,:,:h,:w] 
        
        return x.to(x_dtype)  
   
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, hidden_channel, out_channel, height, width = 1, 16, 64, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = EDFFN(in_features=in_channel, hidden_features=hidden_channel, out_features=out_channel).to(device)    

    outputs = module(inputs)  
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)   
    flops, macs, _ = calculate_flops(model=module,  
                                     input_shape=(batch_size, in_channel, height, width), 
                                     output_as_string=True,  
                                     output_precision=4,     
                                     print_detailed=True) 
    print(RESET)