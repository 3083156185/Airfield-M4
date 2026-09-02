'''     
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/ACMMM2025-CPIA_SA.png  
engine/extre_module/module_images/ACMMM2025-CPIA_SA.md 
论文链接：https://arxiv.org/abs/2504.16455
'''    

import warnings    
warnings.filterwarnings('ignore')
from calflops import calculate_flops
 
import torch   
import torch.nn as nn
import torch.fft as fft 
import torch.nn.functional as F
from einops import rearrange

class ComplexFFT(nn.Module):
    def __init__(self):
        super(ComplexFFT, self).__init__()   
    
    def forward(self, x):    
        # 对输入进行复数FFT变换     
        x_fft = fft.fft2(x, dim=(-2, -1))  # 2D FFT，沿最后两个维度进行   
        real = x_fft.real  # 提取实部
        imag = x_fft.imag  # 提取虚部     
        return real, imag


class ComplexIFFT(nn.Module): 
    def __init__(self):
        super(ComplexIFFT, self).__init__() 

    def forward(self, real, imag):
        # 复合实部和虚部，作为复杂信号   
        x_complex = torch.complex(real, imag)
        # 进行复数IFFT逆变换
        x_ifft = fft.ifft2(x_complex, dim=(-2, -1))  # 2D IFFT    
        return x_ifft.real  # 输出结果的实部

    
class Conv1x1(nn.Module):
    def __init__(self, in_channels):   
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, padding=0,groups=in_channels*2) 

    def forward(self, x):  
        return self.conv(x)

  
class Stage2_fft(nn.Module):
    def __init__(self, in_channels):    
        super(Stage2_fft, self).__init__()    
        self.c_fft = ComplexFFT()
        self.conv1x1 = Conv1x1(in_channels)  
        self.c_ifft = ComplexIFFT()

    def forward(self, x):
        real, imag = self.c_fft(x)

        combined = torch.cat([real, imag], dim=1)  
        conv_out = self.conv1x1(combined)
 
        out_channels = conv_out.shape[1] // 2
        real_out = conv_out[:, :out_channels, :, :]  
        imag_out = conv_out[:, out_channels:, :, :] 

        output = self.c_ifft(real_out, imag_out)  

        return output    

class spr_sa(nn.Module):
    def __init__(self, dim, growth_rate=2.0): 
        super().__init__()     
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(  
            nn.Conv2d(dim,hidden_dim,3,1,1,groups=dim),
            nn.Conv2d(hidden_dim,hidden_dim,1,1,0)  
        )
        self.act =nn.GELU()    
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)
 
    def forward(self, x): 
        x = self.conv_0(x)  
        x1= F.adaptive_avg_pool2d(x, (1, 1))
        x1 = F.softmax(x1, dim=1)
        x = x1 * x
        x = self.act(x)     
        x = self.conv_1(x)    
        return x     
     
class SPC_SA(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):     
        super(SPC_SA, self).__init__()
        self.num_heads = num_heads 
     
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))   
        self.spr_sa=spr_sa(dim//2,2) 
        self.linear_0 = nn.Conv2d(dim, dim , 1, 1, 0)     
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)   
        self.qkv = nn.Conv2d(dim//2, dim//2 * 3, kernel_size=1, bias=bias)   
        self.qkv_dwconv = nn.Conv2d(dim//2 * 3, dim//2 * 3, kernel_size=3, stride=1, padding=1, groups=dim//2 * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)  
        self.attn_drop = nn.Dropout(0.)    
  
        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),     
            nn.Conv2d(dim//2, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim//2, kernel_size=1),    
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim//2, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),   
            nn.Conv2d(dim // 16, 1, kernel_size=1)
        )     
        self.fft=Stage2_fft(in_channels=dim)   
        self.gate = nn.Sequential(
            nn.Conv2d(dim//2, dim // 4, kernel_size=1), 
            nn.ReLU(),  
            nn.Conv2d(dim // 4, 1, kernel_size=1),  # 输出动态 K
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape  
        y, x = self.linear_0(x).chunk(2, dim=1)     
   
        y_d = self.spr_sa(y)    
   
        qkv = self.qkv_dwconv(self.qkv(x))  
        q, k, v = qkv.chunk(3, dim=1) 
     
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)    
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1) 
        k = torch.nn.functional.normalize(k, dim=-1)  
     
        _, _, C, _ = q.shape    
        dynamic_k = int(C * self.gate(x).view(b, -1).mean())    
        attn = (q @ k.transpose(-2, -1)) * self.temperature  
        mask = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        index = torch.topk(attn, k=dynamic_k, dim=-1, largest=True)[1]
        mask.scatter_(-1, index, 1.) 
        attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))    
  
        attn = attn.softmax(dim=-1)
        out1 = (attn @ v)
        out2 = (attn @ v)
        out3 = (attn @ v)
        out4 = (attn @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4
     
        out_att = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)    
 
        # Frequency Adaptive Interaction Module (FAIM)
        # stage1
        # C-Map (before sigmoid)    
        channel_map = self.channel_interaction(out_att)
        # S-Map (before sigmoid)
        spatial_map = self.spatial_interaction(y_d)   

        # S-I 
        attened_x = out_att * torch.sigmoid(spatial_map)
        # C-I 
        conv_x = y_d * torch.sigmoid(channel_map)
 
        x = torch.cat([attened_x, conv_x], dim=1)
        out = self.project_out(x)
        # stage 2    
        out=self.fft(out)  
        return out

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel, height, width = 1, 64, 20, 20 
    inputs = torch.randn((batch_size, channel, height, width)).to(device)
     
    module = SPC_SA(channel, num_heads=8).to(device).eval()  
     
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width),    
                                     output_as_string=True, 
                                     output_precision=4,
                                     print_detailed=True)     
    print(RESET)  
