'''   
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/CVPR2025-Heat2D.png     
论文链接：https://arxiv.org/pdf/2405.16555  
'''

import os, sys  
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')     
  
import warnings
warnings.filterwarnings('ignore')  
from calflops import calculate_flops

import torch   
import torch.nn as nn
import torch.nn.functional as F  
import math

from engine.extre_module.ultralytics_nn.conv import Conv

class Heat2D(nn.Module): 
    """
    du/dt -k(d2u/dx2 + d2u/dy2) = 0;  
    du/dx_{x=0, x=a} = 0
    du/dy_{y=0, y=b} = 0
    =>   
    A_{n, m} = C(a, b, n==0, m==0) * sum_{0}^{a}{ sum_{0}^{b}{\phi(x, y)cos(n\pi/ax)cos(m\pi/by)dxdy }}   
    core = cos(n\pi/ax)cos(m\pi/by)exp(-[(n\pi/a)^2 + (m\pi/b)^2]kt)
    u_{x, y, t} = sum_{0}^{\infinite}{ sum_{0}^{\infinite}{ core } }
   
    assume a = N, b = M; x in [0, N], y in [0, M]; n in [0, N], m in [0, M]; with some slight change     
    => 
    (\phi(x, y) = linear(dwconv(input(x, y))))
    A(n, m) = DCT2D(\phi(x, y))
    u(x, y, t) = IDCT2D(A(n, m) * exp(-[(n\pi/a)^2 + (m\pi/b)^2])**kt)     
    """    
    def __init__(self, in_dim, dim, res=(14, 14), infer_mode=False, **kwargs):    
        super().__init__()    
        hidden_dim = dim
        self.res = res
        self.dwconv = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)   
        self.out_norm = nn.LayerNorm(hidden_dim)  
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.infer_mode = infer_mode
        self.to_k = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),   
        )  
        self.freq_embed = nn.Parameter(torch.zeros(*res, hidden_dim), requires_grad=True) 
        self.infer_init_heat2d(self.freq_embed)     

        self.conv1x1 = Conv(in_dim, dim) if in_dim != dim else nn.Identity()
    
    def infer_init_heat2d(self, freq):  
        weight_exp = self.get_decay_map(self.res, device=freq.device)
        self.k_exp = nn.Parameter(torch.pow(weight_exp[:, :, None], self.to_k(freq)), requires_grad=False) 
        # del self.to_k  
   
    @staticmethod
    def get_cos_map(N=224, device=torch.device("cpu"), dtype=torch.float):   
        # cos((x + 0.5) / N * n * \pi) which is also the form of DCT and IDCT
        # DCT: F(n) = sum( (sqrt(2/N) if n > 0 else sqrt(1/N)) * cos((x + 0.5) / N * n * \pi) * f(x) )     
        # IDCT: f(x) = sum( (sqrt(2/N) if n > 0 else sqrt(1/N)) * cos((x + 0.5) / N * n * \pi) * F(n) )
        # returns: (Res_n, Res_x) 
        weight_x = (torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(1, -1) + 0.5) / N    
        weight_n = torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(-1, 1) 
        weight = torch.cos(weight_n * weight_x * torch.pi) * math.sqrt(2 / N)   
        weight[0, :] = weight[0, :] / math.sqrt(2)
        return weight  
  
    @staticmethod 
    def get_decay_map(resolution=(224, 224), device=torch.device("cpu"), dtype=torch.float):  
        # exp(-[(n\pi/a)^2 + (m\pi/b)^2])
        # returns: (Res_h, Res_w) 
        resh, resw = resolution
        weight_n = torch.linspace(0, torch.pi, resh + 1, device=device, dtype=dtype)[:resh].view(-1, 1)     
        weight_m = torch.linspace(0, torch.pi, resw + 1, device=device, dtype=dtype)[:resw].view(1, -1)
        weight = torch.pow(weight_n, 2) + torch.pow(weight_m, 2)
        weight = torch.exp(-weight)
        return weight    

    def forward(self, x: torch.Tensor):
        x = self.conv1x1(x)  
        B, C, H, W = x.shape    
        x = self.dwconv(x)
    
        x = self.linear(x.permute(0, 2, 3, 1).contiguous()) # B, H, W, 2C
        x, z = x.chunk(chunks=2, dim=-1) # B, H, W, C
  
        if ((H, W) == getattr(self, "__RES__", (0, 0))) and (getattr(self, "__WEIGHT_COSN__", None).device == x.device):
            weight_cosn = getattr(self, "__WEIGHT_COSN__", None)  
            weight_cosm = getattr(self, "__WEIGHT_COSM__", None)   
            weight_exp = getattr(self, "__WEIGHT_EXP__", None) 
            assert weight_cosn is not None    
            assert weight_cosm is not None
            assert weight_exp is not None
        else: 
            weight_cosn = self.get_cos_map(H, device=x.device).detach_()
            weight_cosm = self.get_cos_map(W, device=x.device).detach_()
            weight_exp = self.get_decay_map((H, W), device=x.device).detach_()
            setattr(self, "__RES__", (H, W))
            setattr(self, "__WEIGHT_COSN__", weight_cosn)    
            setattr(self, "__WEIGHT_COSM__", weight_cosm)
            setattr(self, "__WEIGHT_EXP__", weight_exp)

        N, M = weight_cosn.shape[0], weight_cosm.shape[0]   
 
        x = F.conv1d(x.contiguous().view(B, H, -1), weight_cosn.contiguous().view(N, H, 1).type_as(x))
        x = F.conv1d(x.contiguous().view(-1, W, C), weight_cosm.contiguous().view(M, W, 1).type_as(x)).contiguous().view(B, N, M, -1) 
  
        if not self.training:
            x = torch.einsum("bnmc,nmc->bnmc", x, self.k_exp.type_as(x)) 
        else:  
            weight_exp = torch.pow(weight_exp[:, :, None], self.to_k(self.freq_embed))  
            x = torch.einsum("bnmc,nmc -> bnmc", x, weight_exp) # exp decay
   
        x = F.conv1d(x.contiguous().view(B, N, -1), weight_cosn.t().contiguous().view(H, N, 1).type_as(x))
        x = F.conv1d(x.contiguous().view(-1, M, C), weight_cosm.t().contiguous().view(W, M, 1).type_as(x)).contiguous().view(B, H, W, -1) 
  
        x = self.out_norm(x)  
        
        x = x * nn.functional.silu(z)  
        x = self.out_linear(x)
 
        x = x.permute(0, 3, 1, 2).contiguous()     

        return x

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32   
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device) 
   
    # 此模块不支持多尺度训练，res的参数是一个元组，其为当前特征图的height,width。  
    module = Heat2D(in_channel, out_channel, res=(height, width)).to(device)
     
    outputs = module(inputs)  
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)    

    print(ORANGE)   
    flops, macs, _ = calculate_flops(model=module,    
                                     input_shape=(batch_size, in_channel, height, width),     
                                     output_as_string=True,
                                     output_precision=4,   
                                     print_detailed=True)
    print(RESET)   
