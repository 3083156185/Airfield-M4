'''   
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/自研模块-CSIE.png
engine/extre_module/module_images/自研模块-CSIE.md
'''
     
import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops  
   
import torch   
import torch.nn as nn


class CSIE(nn.Module):     
    """Context-Steered Interaction Enhancement (CSIE).

    Introduces global context steering into the IEL dual-path framework:    
    1. Global Context Encoder (GCE): extracts a context vector from pooled features
    2. Context-Steered Convolution: context vector generates per-channel dynamic  
       offsets that modulate the depthwise convolution outputs
    3. Adaptive Activation Range: context vector predicts per-sample scale (gamma)
       and temperature (tau) for the Tanh activation, enabling input-adaptive
       saturation control
   
    Data flow:
        x -> Conv1x1 (expand 2*C_h) -> DWConv3 (pre-mix)     
          -> chunk(2) -> f1, f2
          -> GCE(concat(f1,f2)) -> context vector c
          -> Context-Steered Conv: DWConv(fi) + delta_i(c) * fi
          -> Adaptive Activation: gamma_i * Tanh(path_i / tau_i) + path_i
          -> p1 * p2
          -> Conv1x1 (project out)   
    """
  
    def __init__(self, in_dim, out_dim, ffn_expansion_factor=2.66, bias=False):  
        super(CSIE, self).__init__() 

        hidden_features = int(in_dim * ffn_expansion_factor)
  
        # Input projection: expand to 2*hidden for dual-path    
        self.project_in = nn.Conv2d(in_dim, hidden_features * 2, kernel_size=1, bias=bias)
     
        # Shared pre-mixing depthwise conv
        self.dwconv_pre = nn.Conv2d(hidden_features * 2, hidden_features * 2,    
                                    kernel_size=3, stride=1, padding=1,     
                                    groups=hidden_features * 2, bias=bias)
     
        # --- Global Context Encoder (GCE) ---
        ctx_hidden = max(hidden_features // 4, 8)   
        self.context_encoder = nn.Sequential(     
            nn.AdaptiveAvgPool2d(1),     
            nn.Flatten(), 
            nn.Linear(hidden_features * 2, ctx_hidden, bias=False),
            nn.ReLU(inplace=True),     
            nn.Linear(ctx_hidden, hidden_features, bias=False),
        )   

        # --- Context-Steered Convolution --- 
        # Path 1 & Path 2: depthwise convolutions 
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features,
                                 kernel_size=3, stride=1, padding=1,    
                                 groups=hidden_features, bias=bias) 
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features,
                                 kernel_size=3, stride=1, padding=1,    
                                 groups=hidden_features, bias=bias) 

        # Dynamic offset generators: context -> per-channel offset
        self.offset_gen1 = nn.Linear(hidden_features, hidden_features, bias=False)     
        self.offset_gen2 = nn.Linear(hidden_features, hidden_features, bias=False)

        # --- Adaptive Activation Range ---
        # context -> (gamma1, tau1, gamma2, tau2), total 4*C_h parameters  
        self.act_param_gen = nn.Linear(hidden_features, hidden_features * 4, bias=False)
 
        self.tanh = nn.Tanh()  
        self.softplus = nn.Softplus()  
  
        # Output projection
        self.project_out = nn.Conv2d(hidden_features, out_dim, kernel_size=1, bias=bias)
     
    def forward(self, x):  
        x = self.project_in(x) 
        x = self.dwconv_pre(x)   
 
        # --- Dual-path split ---    
        f1, f2 = x.chunk(2, dim=1)
    
        # --- Global context encoding ---  
        c = self.context_encoder(x)  # (B, C_h)  
  
        B, C_h = c.shape

        # --- Context-Steered Convolution ---
        # Dynamic offsets modulate convolution outputs
        delta1 = self.offset_gen1(c).view(B, C_h, 1, 1)  # (B, C_h, 1, 1)  
        delta2 = self.offset_gen2(c).view(B, C_h, 1, 1)

        path1 = self.dwconv1(f1) + delta1 * f1  # steered path 1   
        path2 = self.dwconv2(f2) + delta2 * f2  # steered path 2
  
        # --- Adaptive Activation Range ---     
        act_params = self.act_param_gen(c)  # (B, 4*C_h)   
        gamma1, tau1_raw, gamma2, tau2_raw = act_params.chunk(4, dim=1) 

        gamma1 = gamma1.view(B, C_h, 1, 1)  
        gamma2 = gamma2.view(B, C_h, 1, 1)  
        tau1 = self.softplus(tau1_raw).view(B, C_h, 1, 1) + 0.5  # ensure tau >= 0.5
        tau2 = self.softplus(tau2_raw).view(B, C_h, 1, 1) + 0.5
  
        p1 = gamma1 * self.tanh(path1 / tau1) + path1  # adaptive activation + residual     
        p2 = gamma2 * self.tanh(path2 / tau2) + path2

        # --- Multiplicative interaction ---
        out = p1 * p2  
 
        return self.project_out(out)


if __name__ == '__main__': 
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"   
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32  
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)  
 
    module = CSIE(in_channel, out_channel).to(device) 
     
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,  
                                     input_shape=(batch_size, in_channel, height, width),   
                                     output_as_string=True,   
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)
