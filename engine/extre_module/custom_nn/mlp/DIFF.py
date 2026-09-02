'''
本文件由BiliBili：魔傀面具整理
ultralytics/nn/module_images/AAAI2026-DIFF.png     
ultralytics/nn/module_images/AAAI2026-DIFF.md     
论文链接：https://arxiv.org/pdf/2504.09377 
'''     

import warnings
warnings.filterwarnings('ignore')     
from calflops import calculate_flops

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
 
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()     
    channels_per_group = num_channels // groups 
    x = x.view(batchsize, groups, channels_per_group, height, width)   
    x = torch.transpose(x, 1, 2).contiguous()    
    x = x.view(batchsize, -1, height, width)
    return x
     
class ElementScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()  
        self.scale = nn.Parameter(   
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad    
        )    
    def forward(self, x):     
        return x * self.scale

class DIFF(nn.Module):
    def __init__(
        self,  
        in_features=None,  
        hidden_features=None, 
        out_features=None,
        bias=False,     
        ffn_expansion_factor=2.667,
    ):
        super(DIFF, self).__init__()
        if in_features is None:    
            raise ValueError('`in_features` must be provided.')

        out_features = out_features or in_features
        hidden_features = hidden_features or int(in_features * ffn_expansion_factor)

        if hidden_features % 4 != 0:
            raise ValueError(f'`hidden_features` must be divisible by 4 for DIFF, but got {hidden_features}.')

        branch_channels = hidden_features // 4
        self.sigma = ElementScale(
            branch_channels, init_value=1e-5, requires_grad=True) 
        self.decompose = nn.Conv2d(
            in_channels=branch_channels,  # C -> 1 
            out_channels=1, kernel_size=1,
        )   
        self.decompose_act = nn.GELU()
        self.project_in = nn.Conv2d(in_features, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv_5 = nn.Conv2d(branch_channels, branch_channels, kernel_size=5,   
                                stride=1, padding=2, groups=branch_channels, bias=bias)  
        self.dwconv_dilated2_1 = nn.Conv2d(branch_channels, branch_channels, kernel_size=3,
                                         stride=1, padding=2, groups=branch_channels,
                                         bias=bias, dilation=2)    
        self.p_unshuffle = nn.PixelUnshuffle(2)    
        self.p_shuffle = nn.PixelShuffle(2)
        self.project_out = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias)     
   
    def feat_decompose(self, x):
        # x_d: [B, C, H, W] -> [B, 1, H, W]     
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x
  
    def forward(self, x):     
        x = self.project_in(x)    
        x = self.p_shuffle(x)
        x = channel_shuffle(x, groups=1)    
        x1, x2 = x.chunk(2, dim=1)  
        x1 = self.dwconv_5(x1)
        x2 = self.dwconv_dilated2_1(x2) 
        x = F.mish(x2) * x1
        x = self.feat_decompose(x)   
        x = self.p_unshuffle(x)
        x = self.project_out(x)
        return x
    

if __name__ == '__main__': 
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, hidden_channel, out_channel, height, width = 1, 16, 64, 16, 32, 32     
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = DIFF(in_features=in_channel, hidden_features=hidden_channel, out_features=out_channel).to(device)  

    outputs = module(inputs)   
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)   
  
    print(ORANGE) 
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),     
                                     output_as_string=True,
                                     output_precision=4,   
                                     print_detailed=True)    
    print(RESET)     
