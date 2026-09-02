'''    
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/Wave2D.png     
engine/extre_module/module_images/Wave2D.md
论文链接：https://arxiv.org/html/2601.08602v1
''' 

import warnings    
warnings.filterwarnings('ignore')  
from calflops import calculate_flops

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_     

class Wave2D(nn.Module):     
    """
    Wave equation operator:  
    d2u/dt2 - c2(d2u/dx2 + d2u/dy2) + αdu/dt = 0;    
    du/dx_{x=0, x=a} = 0    
    du/dy_{y=0, y=b} = 0
    =>    
    A_{n, m} = C(a, b, n==0, m==0) * sum_{0}^{a}{ sum_{0}^{b}{\phi(x, y)cos(n\pi/ax)cos(m\pi/by)dxdy }}     
    core = cos(n\pi/ax)cos(m\pi/by) * (1 - [(n\pi/a)^2 + (m\pi/b)^2]c2t2) * e^(-αt)
    u_{x, y, t} = sum_{0}^{\infinite}{ sum_{0}^{\infinite}{ core } }
    
    assume a = N, b = M; x in [0, N], y in [0, M]; n in [0, N], m in [0, M]; with some slight change
    =>  
    (\phi(x, y) = linear(dwconv(input(x, y))))
    A(n, m) = DCT2D(\phi(x, y))   
    u(x, y, t) = IDCT2D(A(n, m) * (1 - [(n\pi/a)^2 + (m\pi/b)^2]c2t2) * e^(-αt))
    """ 
    def __init__(self, in_dim=96, out_dim=96, res=[14, 14], **kwargs):
        super().__init__()
        self.res = res  
        self.dwconv = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim)
        self.linear = nn.Linear(in_dim, 2 * in_dim, bias=True)
        self.out_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(in_dim, out_dim, bias=True)
        self.to_k = nn.Sequential(
            nn.Linear(in_dim, in_dim, bias=True),     
            nn.GELU(),   
        )
        self.c = nn.Parameter(torch.ones(1) * 1.0)
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)   
        self.save_attention = False
  
        self.freq_embed = nn.Parameter(torch.zeros(res[0], res[1], in_dim), requires_grad=True) 
        trunc_normal_(self.freq_embed, std=.015) 

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
        # (1 - [(n\pi/a)^2 + (m\pi/b)^2]c2t2) * e^(-αt)
        # returns: (Res_h, Res_w)
        resh, resw = resolution     
        weight_n = torch.linspace(0, torch.pi, resh + 1, device=device, dtype=dtype)[:resh].view(-1, 1)     
        weight_m = torch.linspace(0, torch.pi, resw + 1, device=device, dtype=dtype)[:resw].view(1, -1)
        # Quadratic term for wave equation
        weight = torch.pow(weight_n, 2) + torch.pow(weight_m, 2)     
        weight = torch.exp(-weight)     
        return weight
     
    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape   
        x = self.dwconv(x)  
        x = self.linear(x.permute(0, 2, 3, 1).contiguous())
        x, z = x.chunk(chunks=2, dim=-1)
     
        cached_weight_cosn = getattr(self, "__WEIGHT_COSN__", None)
        if ((H, W) == getattr(self, "__RES__", (0, 0))) and cached_weight_cosn is not None and (cached_weight_cosn.device == x.device):
            weight_cosn = cached_weight_cosn
            weight_cosm = getattr(self, "__WEIGHT_COSM__", None)
            weight_exp = getattr(self, "__WEIGHT_EXP__", None)
        else:  
            weight_cosn = self.get_cos_map(H, device=x.device).detach_()  
            weight_cosm = self.get_cos_map(W, device=x.device).detach_()    
            weight_exp = self.get_decay_map((H, W), device=x.device).detach_()     
            setattr(self, "__RES__", (H, W)) 
            setattr(self, "__WEIGHT_COSN__", weight_cosn)    
            setattr(self, "__WEIGHT_COSM__", weight_cosm)
            setattr(self, "__WEIGHT_EXP__", weight_exp)
 
        N, M = weight_cosn.shape[0], weight_cosm.shape[0]
        weight_cosn_kernel = weight_cosn.view(H, 1, H)
        weight_cosm_kernel = weight_cosm.view(W, 1, W)
        x_perm = x.permute(0, 3, 2, 1).contiguous() # [B, C, W, H] 
        x_flat_H = x_perm.view(-1, 1, H)            # [B*C*W, 1, H]     
        x_u0 = F.conv1d(x_flat_H, weight_cosn_kernel).squeeze(-1) # [B*C*W, H]
        x_u0 = x_u0.view(B, C, W, H).permute(0, 3, 2, 1).contiguous() # [B, H, W, C]     
        x_perm = x_u0.permute(0, 3, 1, 2).contiguous() # [B, C, H, W] 
        x_flat_W = x_perm.view(-1, 1, W)               # [B*C*H, 1, W]     
        x_u0 = F.conv1d(x_flat_W, weight_cosm_kernel).squeeze(-1) # [B*C*H, W]     
        x_u0 = x_u0.view(B, C, H, W).permute(0, 2, 3, 1).contiguous()     
        x_perm = x.permute(0, 3, 2, 1).contiguous() # [B, C, W, H]   
        x_flat_H = x_perm.view(-1, 1, H)            # [B*C*W, 1, H]
        x_v0 = F.conv1d(x_flat_H, weight_cosn_kernel).squeeze(-1) # [B*C*W, H]     
        x_v0 = x_v0.view(B, C, W, H).permute(0, 3, 2, 1).contiguous() # [B, H, W, C]
        x_perm = x_v0.permute(0, 3, 1, 2).contiguous() # [B, C, H, W]
        x_flat_W = x_perm.view(-1, 1, W)               # [B*C*H, 1, W] 
        x_v0 = F.conv1d(x_flat_W, weight_cosm_kernel).squeeze(-1) # [B*C*H, W]
        x_v0 = x_v0.view(B, C, H, W).permute(0, 2, 3, 1).contiguous()  
        t = self.to_k(self.freq_embed)
        c_t = self.c * t     
        cos_term = torch.cos(c_t)    
        eps = 1e-8   
        sin_term = torch.sin(c_t) / (self.c + eps)
        wave_term = cos_term * x_u0
        velocity_term = sin_term * (x_v0 + (self.alpha / 2) * x_u0)   
        final_term = wave_term + velocity_term  
        cached_weight_cosn_idct = getattr(self, "__WEIGHT_COSN_IDCT__", None) 
        cached_weight_cosm_idct = getattr(self, "__WEIGHT_COSM_IDCT__", None)     
        
        if ((H, W) == getattr(self, "__RES_IDCT__", (0, 0))) and \
           cached_weight_cosn_idct is not None and \
           cached_weight_cosn_idct.device == x.device:  
            weight_cosn = cached_weight_cosn_idct
            weight_cosm = cached_weight_cosm_idct  
        else:
            weight_cosn = self.get_cos_map(H, device=x.device).detach_()  # (H, H)
            weight_cosm = self.get_cos_map(W, device=x.device).detach_()  # (W, W)
            setattr(self, "__RES_IDCT__", (H, W))
            setattr(self, "__WEIGHT_COSN_IDCT__", weight_cosn)     
            setattr(self, "__WEIGHT_COSM_IDCT__", weight_cosm)
        x_w = final_term.permute(0, 1, 3, 2).contiguous().view(B * H * C, 1, W)  # (B*H*C, 1, W) 
        weight_cosm_kernel_t = weight_cosm.t().contiguous().view(W, 1, W)  # (W,1,W) 
        x_w = F.conv1d(x_w, weight_cosm_kernel_t).squeeze(-1)  # (B*H*C, W)     
        x_w = x_w.view(B, H, C, W).permute(0, 1, 3, 2).contiguous()  # (B,H,W,C)
        x_h = x_w.permute(0, 2, 3, 1).contiguous().view(B * W * C, 1, H)  # (B*W*C,1,H) 
        weight_cosn_kernel_t = weight_cosn.t().contiguous().view(H, 1, H)  # (H,1,H)
        x_h = F.conv1d(x_h, weight_cosn_kernel_t).squeeze(-1)  # (B*W*C,H)  
        x_final = x_h.view(B, W, C, H).permute(0, 3, 1, 2).contiguous()
        x = self.out_norm(x_final)
        gate = nn.functional.silu(z)   
        x_gated = x * gate
        x = self.out_linear(x_gated)    
        x = x.permute(0, 3, 1, 2).contiguous()
        # if test_index is not None and hasattr(self, 'save_attention') and self.save_attention:
        #     center_h, center_w = H // 2, W // 2   
        #     attention_map = (x_final * x_final[:, center_h:center_h+1, center_w:center_w+1, :]).sum(-1)
            
        #     import matplotlib.pyplot as plt
  
        #     # Ensure the save directory exists   
        #     save_dir = "./save/attention_map"     
        #     os.makedirs(save_dir, exist_ok=True)
  
        #     # Move attention_map to cpu and convert to numpy   
        #     att_map_np = attention_map.detach().cpu().numpy() 

        #     # Normalize for visualization
        #     att_map_norm = (att_map_np - att_map_np.min()) / (att_map_np.max() - att_map_np.min() + 1e-8)
  
        #     # Save each sample in the batch
        #     for i in range(att_map_norm.shape[0]):
        #         filename = os.path.join(save_dir, f"attention_map_{test_index}_{i}.png")
        #         plt.imsave(filename, att_map_norm[i], cmap='viridis')    
        return x     

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')     
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32    
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)    
  
    # 此模块不支持多尺度训练，res的参数是一个元组/列表，其为当前特征图的height,width。 
    module = Wave2D(in_channel, out_channel, res=(height, width)).to(device)    

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)    
 
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),  
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)   
    print(RESET)   
