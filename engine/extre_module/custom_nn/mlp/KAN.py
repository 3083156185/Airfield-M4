'''
本文件由BiliBili：魔傀面具整理
论文链接：https://arxiv.org/abs/2409.10594
'''
 
import warnings   
warnings.filterwarnings('ignore')   
from calflops import calculate_flops  
  
import torch     
import torch.nn as nn     
from timm.layers import to_2tuple 
from functools import partial
 
try:
    from kat_rational import KAT_Group
except ImportError as e: 
    # print(f'from kat_rational import KAT_Group Failure. message:{e}')     
    pass    

class KAN(nn.Module):  
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks    
    """     
    def __init__(
            self,
            in_features,
            hidden_features=None,    
            out_features=None,    
            act_layer=None, 
            norm_layer=None,
            bias=True,     
            drop=0.,
            use_conv=False, 
            act_init="gelu",     
    ):    
        super().__init__()    
        out_features = out_features or in_features     
        hidden_features = hidden_features or in_features 
        bias = to_2tuple(bias)    
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear  
    
        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act1 = KAT_Group(mode="identity")
        self.drop1 = nn.Dropout(drop_probs[0])    
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.act2 = KAT_Group(mode=act_init)    
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1]) 
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward_kan(self, x):
        x = self.act1(x)
        x = self.drop1(x)    
        x = self.fc1(x) 
        x = self.act2(x)   
        x = self.drop2(x)
        x = self.fc2(x)
        return x
   
    def forward(self, x):     
        B, C, H, W = x.size()  
        x_nlc = x.flatten(2).permute(0, 2, 1)
        x_nlc = self.forward_kan(x_nlc)
        x_nchw = x_nlc.permute(0, 2, 1).view([B, -1, H, W]).contiguous()
        return x_nchw

if __name__ == '__main__': 
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"  
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, hidden_channel, out_channel, height, width = 1, 16, 64, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # 此模块需要编译,详细编译命令在 compile_module/rational_kat_cu/make.sh  
    # 对于Linux用户可以直接运行上述的make.sh文件    
    # 对于Windows用户需要逐行执行make.sh文件里的内容
    module = KAN(in_features=in_channel, hidden_features=hidden_channel, out_features=out_channel).to(device)

    outputs = module(inputs)   
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
 
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,   
                                     output_precision=4,
                                     print_detailed=True)  
    print(RESET)