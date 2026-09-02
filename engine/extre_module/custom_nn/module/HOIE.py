'''  
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/自研模块-HOIE.png
engine/extre_module/module_images/自研模块-HOIE.md
'''
     
import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops  

import torch     
import torch.nn as nn   
    

class HOIE(nn.Module):
    """Higher-Order Interaction Enhancement (HOIE).
    
    Extends IEL's second-order multiplicative interaction (x1*x2) to a   
    three-path architecture with explicit first-, second-, and third-order
    interaction terms and order-adaptive fusion.
 
    Data flow:
        x -> Conv1x1 (expand 3*C_h) -> DWConv3 (pre-mix)
          -> chunk(3) -> x1, x2, x3 
          -> Path1 (3x3 DW), Path2 (3x3 DW), Path3 (5x5 DW)
          -> Order computation:    
               o1 = x1 + x2 + x3           (first-order: linear)    
               o2 = x1*x2 + x2*x3 + x1*x3 (second-order: pairwise)
               o3 = x1*x2*x3               (third-order: full)
          -> Order-Adaptive Fusion: concat -> GAP -> FC -> Softmax -> weighted sum
          -> Conv1x1 (project out)
    """   

    def __init__(self, in_dim, out_dim, ffn_expansion_factor=2.66, bias=False): 
        super(HOIE, self).__init__()
 
        hidden_features = int(in_dim * ffn_expansion_factor)
        # Ensure divisible by 3
        self.C_h = hidden_features - (hidden_features % 3)
        total_expand = self.C_h * 3  

        # Input projection: expand to 3*C_h for three paths
        self.project_in = nn.Conv2d(in_dim, total_expand, kernel_size=1, bias=bias)

        # Shared pre-mixing depthwise conv
        self.dwconv_pre = nn.Conv2d(total_expand, total_expand,
                                    kernel_size=3, stride=1, padding=1,
                                    groups=total_expand, bias=bias)
   
        # --- Three enhancement paths ---
        # Path 1 & 2: 3x3 DWConv (standard detail capture)   
        self.dwconv1 = nn.Conv2d(self.C_h, self.C_h, kernel_size=3, stride=1,   
                                 padding=1, groups=self.C_h, bias=bias)    
        self.dwconv2 = nn.Conv2d(self.C_h, self.C_h, kernel_size=3, stride=1,
                                 padding=1, groups=self.C_h, bias=bias)  
        # Path 3: 5x5 DWConv (extended receptive field)
        self.dwconv3 = nn.Conv2d(self.C_h, self.C_h, kernel_size=5, stride=1,  
                                 padding=2, groups=self.C_h, bias=bias)     

        self.tanh = nn.Tanh()    

        # --- Order-Adaptive Fusion ---    
        # Maps concatenated order features to 3-way softmax weights
        self.order_fusion = nn.Sequential(  
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),    
            nn.Linear(self.C_h * 3, self.C_h // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.C_h // 2, 3, bias=False),
        )

        # Output projection   
        self.project_out = nn.Conv2d(self.C_h, out_dim, kernel_size=1, bias=bias)     
    
    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv_pre(x)
   
        # --- Three-path split and enhancement ---  
        x1, x2, x3 = x.chunk(3, dim=1)
     
        x1 = self.tanh(self.dwconv1(x1)) + x1  # 3x3 path
        x2 = self.tanh(self.dwconv2(x2)) + x2  # 3x3 path
        x3 = self.tanh(self.dwconv3(x3)) + x3  # 5x5 path    

        # --- Multi-order interaction computation ---     
        o1 = x1 + x2 + x3                              # first-order (linear superposition)   
        o2 = x1 * x2 + x2 * x3 + x1 * x3              # second-order (pairwise multiplicative) 
        o3 = x1 * x2 * x3                              # third-order (full multiplicative)

        # --- Order-adaptive fusion ---
        cat_orders = torch.cat([o1, o2, o3], dim=1)
        weights = torch.softmax(self.order_fusion(cat_orders), dim=1)  # (B, 3)
        B = x.size(0)
        w1 = weights[:, 0].view(B, 1, 1, 1)   
        w2 = weights[:, 1].view(B, 1, 1, 1)     
        w3 = weights[:, 2].view(B, 1, 1, 1)
 
        out = w1 * o1 + w2 * o2 + w3 * o3
  
        return self.project_out(out)
  

if __name__ == '__main__':  
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"   
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)
   
    module = HOIE(in_channel, out_channel).to(device)
  
    outputs = module(inputs)   
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET) 
 
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,   
                                     input_shape=(batch_size, in_channel, height, width),    
                                     output_as_string=True,
                                     output_precision=4,    
                                     print_detailed=True)
    print(RESET)
