'''
本文件由BiliBili：魔傀面具整理  
engine/extre_module/module_images/CVPR2024-FRFN.png   
论文链接：https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Adapt_or_Perish_Adaptive_Sparse_Transformer_with_Attentive_Feature_Refinement_CVPR_2024_paper.pdf
'''   
 
import warnings  
warnings.filterwarnings('ignore')
from calflops import calculate_flops

import torch
import torch.nn as nn
     
class FRFN(nn.Module): 
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):    
        super().__init__()
        out_features = out_features or in_features 
        hidden_features = hidden_features or in_features   
     
        self.linear1 = nn.Sequential(nn.Conv2d(in_features, hidden_features*2, 1),    
                                act_layer())    
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_features,hidden_features,groups=hidden_features,kernel_size=3,stride=1,padding=1),
                        act_layer())
        self.linear2 = nn.Sequential(nn.Conv2d(hidden_features, out_features, 1)) 
 
        self.hidden_dim = hidden_features

        self.dim = in_features
        self.dim_conv = self.dim // 4
        self.dim_untouched = self.dim - self.dim_conv     
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)   
     
    def forward(self, x):    
        x1, x2,= torch.split(x, [self.dim_conv,self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)    
        x = torch.cat((x1, x2), 1)    

        x = self.linear1(x)
        #gate mechanism    
        x_1, x_2 = x.chunk(2,dim=1)
        x_1 = self.dwconv(x_1) 
        x = x_1 * x_2
     
        x = self.linear2(x)   
        return x
 
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"     
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    batch_size, in_channel, hidden_channel, out_channel, height, width = 1, 16, 64, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = FRFN(in_features=in_channel, hidden_features=hidden_channel, out_features=out_channel).to(device)
  
    outputs = module(inputs)    
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
 
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),    
                                     output_as_string=True,     
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)