'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/ECCV2024-FSSA.png    
论文链接：https://arxiv.org/pdf/2409.01686     
'''  
  
import warnings  
warnings.filterwarnings('ignore')  
from calflops import calculate_flops

import torch    
import torch.nn as nn
import torch.nn.functional as F    
from einops import rearrange    

def custom_complex_normalization(input_tensor, dim=-1):
    real_part = input_tensor.real    
    imag_part = input_tensor.imag
    norm_real = F.softmax(real_part, dim=dim)
    norm_imag = F.softmax(imag_part, dim=dim)
     
    normalized_tensor = torch.complex(norm_real, norm_imag)    
    
    return normalized_tensor

class Attention_F(nn.Module):
    def __init__(self, dim, num_heads, bias,):
        super(Attention_F, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.project_out = nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias)  
        self.weight = nn.Sequential(  
            nn.Conv2d(dim, dim // 16, 1, bias=True), 
            nn.BatchNorm2d(dim // 16),
            nn.ReLU(True),  
            nn.Conv2d(dim // 16, dim, 1, bias=True),   
            nn.Sigmoid()) 
    def forward(self, x):
        b, c, h, w = x.shape
     
        q_f = torch.fft.fft2(x.float())     
        k_f = torch.fft.fft2(x.float())
        v_f = torch.fft.fft2(x.float())

        q_f = rearrange(q_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_f = rearrange(k_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_f = rearrange(v_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_f = torch.nn.functional.normalize(q_f, dim=-1) 
        k_f = torch.nn.functional.normalize(k_f, dim=-1)
        attn_f = (q_f @ k_f.transpose(-2, -1)) * self.temperature     
        attn_f = custom_complex_normalization(attn_f, dim=-1) 
        out_f = torch.abs(torch.fft.ifft2(attn_f @ v_f))     
        out_f = rearrange(out_f, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_f_l = torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(x.float()).real)*torch.fft.fft2(x.float())))
        out = self.project_out(torch.cat((out_f,out_f_l),1)) 
        return out     

class Attention_S(nn.Module):  
    def __init__(self, dim, num_heads, bias):
        super(Attention_S, self).__init__()
        self.num_heads = num_heads   
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
 
        self.qkv1conv_1 = nn.Conv2d(dim,dim,kernel_size=1)
        self.qkv2conv_1 = nn.Conv2d(dim, dim, kernel_size=1)  
        self.qkv3conv_1 = nn.Conv2d(dim, dim, kernel_size=1)


        self.qkv1conv_3 = nn.Conv2d(dim, dim//2, kernel_size=3, stride=1, padding=1, groups=dim//2, bias=bias)
        self.qkv2conv_3 = nn.Conv2d(dim, dim//2, kernel_size=3, stride=1, padding=1, groups=dim//2, bias=bias)   
        self.qkv3conv_3 = nn.Conv2d(dim, dim//2, kernel_size=3, stride=1, padding=1, groups=dim//2, bias=bias)

        self.qkv1conv_5 = nn.Conv2d(dim, dim // 2, kernel_size=5, stride=1, padding=2, groups=dim//2, bias=bias)  
        self.qkv2conv_5 = nn.Conv2d(dim, dim // 2, kernel_size=5, stride=1, padding=2, groups=dim//2, bias=bias)
        self.qkv3conv_5 = nn.Conv2d(dim, dim // 2, kernel_size=5, stride=1, padding=2, groups=dim//2, bias=bias)


        self.conv_3      = nn.Conv2d(dim, dim//2, kernel_size=3, stride=1, padding=1, groups=dim//2, bias=bias)
        self.conv_5      = nn.Conv2d(dim, dim // 2, kernel_size=5, stride=1, padding=2, groups=dim//2, bias=bias)  
        self.project_out = nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q_s = torch.cat((self.qkv1conv_3(self.qkv1conv_1(x)),self.qkv1conv_5(self.qkv1conv_1(x))),1) 
        k_s = torch.cat((self.qkv2conv_3(self.qkv2conv_1(x)),self.qkv2conv_5(self.qkv2conv_1(x))),1)
        v_s = torch.cat((self.qkv3conv_3(self.qkv3conv_1(x)),self.qkv3conv_5(self.qkv3conv_1(x))),1) 

        q_s = rearrange(q_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)   
        k_s = rearrange(k_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_s = rearrange(v_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)     

        q_s = torch.nn.functional.normalize(q_s, dim=-1)
        k_s = torch.nn.functional.normalize(k_s, dim=-1)     
        attn_s = (q_s @ k_s.transpose(-2, -1)) * self.temperature
        attn_s = attn_s.softmax(dim=-1) 
        out_s = (attn_s @ v_s)     
        out_s = rearrange(out_s, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)   
        out_s_l = torch.cat((self.conv_3(x),self.conv_5(x)),1)
        out = self.project_out(torch.cat((out_s,out_s_l),1))   

        return out

class FSSA(nn.Module):
    def __init__(self, dim, num_heads, bias=False) -> None:
        super().__init__()
  
        self.fsa = Attention_F(dim, num_heads, bias)    
        self.ssa = Attention_S(dim, num_heads, bias)
  
    def forward(self, x):  
        fsa = self.fsa(x)    
        ssa = self.ssa(x)
        return fsa + ssa 

if __name__ == '__main__':  
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"  
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    batch_size, channel, height, width = 1, 16, 20, 20
    inputs = torch.randn((batch_size, channel, height, width)).to(device)

    module = FSSA(channel, num_heads=8).to(device)     

    outputs = module(inputs)   
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)  
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width),
                                     output_as_string=True,    
                                     output_precision=4,  
                                     print_detailed=True) 
    print(RESET)   
