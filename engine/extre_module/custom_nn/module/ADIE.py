'''  
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/自研模块-ADIE.png
engine/extre_module/module_images/自研模块-ADIE.md
'''     
 
import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops
     
import torch
import torch.nn as nn
 

class ADIE(nn.Module):
    """Adaptive Decomposition-Interaction Enhancement (ADIE). 

    Generalises the IEL dual-path gated structure with three key improvements:
    1. Content-adaptive soft decomposition (replaces fixed chunk)
    2. Asymmetric multi-scale dual paths (3x3 + 5x5 DWConv)    
    3. Learnable multiplicative-additive interaction interpolation
 
    Data flow:
        x -> Conv1x1 (expand 2*C_h) -> DWConv3 (pre-mix)
          -> Adaptive Decomposition Gate -> M1, M2 = complementary soft masks
          -> Path1 (3x3 DW + Tanh + Res), Path2 (5x5 DW + Tanh + Res)
          -> Interaction Controller -> alpha*(P1*P2) + beta*(P1+P2)   
          -> Conv1x1 (project out)    
    """

    def __init__(self, in_dim, out_dim, ffn_expansion_factor=2.66, bias=False):
        super(ADIE, self).__init__()     
  
        hidden_features = int(in_dim * ffn_expansion_factor)     

        # Input projection: expand to 2*hidden for dual-path 
        self.project_in = nn.Conv2d(in_dim, hidden_features * 2, kernel_size=1, bias=bias)     
   
        # Shared pre-mixing depthwise conv
        self.dwconv_pre = nn.Conv2d(hidden_features * 2, hidden_features * 2,
                                    kernel_size=3, stride=1, padding=1,  
                                    groups=hidden_features * 2, bias=bias)
  
        # --- Adaptive Decomposition Gate ---
        # Generates complementary soft masks M1, M2 (M1 + M2 = 1)   
        self.decomp_gate = nn.Sequential(   
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_features * 2, hidden_features // 2, bias=False),
            nn.ReLU(inplace=True),    
            nn.Linear(hidden_features // 2, hidden_features, bias=False),
            nn.Sigmoid(),
        )
  
        # --- Asymmetric enhancement paths ---
        # Path 1: 3x3 DWConv (local detail) 
        self.dwconv_p1 = nn.Conv2d(hidden_features, hidden_features,
                                   kernel_size=3, stride=1, padding=1,
                                   groups=hidden_features, bias=bias)
        # Path 2: 5x5 DWConv (broader context)
        self.dwconv_p2 = nn.Conv2d(hidden_features, hidden_features,
                                   kernel_size=5, stride=1, padding=2, 
                                   groups=hidden_features, bias=bias)
   
        self.tanh = nn.Tanh()

        # --- Interaction Controller ---     
        # Predicts alpha, beta from concatenated path features  
        self.interaction_ctrl = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),    
            nn.Flatten(),
            nn.Linear(hidden_features * 2, hidden_features // 2, bias=False),     
            nn.ReLU(inplace=True),   
            nn.Linear(hidden_features // 2, 2, bias=False),
            nn.Sigmoid(),     
        )     

        # Output projection
        self.project_out = nn.Conv2d(hidden_features, out_dim, kernel_size=1, bias=bias)

    def forward(self, x): 
        x = self.project_in(x)    
        x = self.dwconv_pre(x)
    
        # --- Adaptive decomposition ---
        # Generate soft mask M1; M2 = 1 - M1 (complementary)   
        B, C, H, W = x.shape
        C_h = C // 2 
        x1_raw, x2_raw = x[:, :C_h], x[:, C_h:]  

        m1 = self.decomp_gate(x).view(B, C_h, 1, 1)  # (B, C_h, 1, 1)
        m2 = 1.0 - m1
 
        f1 = x1_raw * m1 + x2_raw * m2  # content-adaptive blend for path 1
        f2 = x1_raw * m2 + x2_raw * m1  # complementary blend for path 2    
    
        # --- Asymmetric enhancement --- 
        p1 = self.tanh(self.dwconv_p1(f1)) + f1  # 3x3 local detail path
        p2 = self.tanh(self.dwconv_p2(f2)) + f2  # 5x5 broader context path     
    
        # --- Multiplicative-Additive interaction ---
        cat_feat = torch.cat([p1, p2], dim=1)     
        coeffs = self.interaction_ctrl(cat_feat)  # (B, 2)    
        alpha = coeffs[:, 0].view(B, 1, 1, 1)
        beta = coeffs[:, 1].view(B, 1, 1, 1)
    
        out = alpha * (p1 * p2) + beta * (p1 + p2)  
 
        return self.project_out(out)
     
   
if __name__ == '__main__':   
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"   
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32     
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = ADIE(in_channel, out_channel).to(device)
   
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)    
     
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,  
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)   
