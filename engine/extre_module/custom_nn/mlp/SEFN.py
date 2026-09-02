'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/WACV2025-SEFN.png  
论文链接：https://arxiv.org/abs/2411.06318 
'''
   
import warnings
warnings.filterwarnings('ignore')   
from calflops import calculate_flops
  
import torch     
import torch.nn as nn   
import torch.nn.functional as F
import numbers, math
from einops import rearrange    
 
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')   

  
def to_4d(x, h, w):    
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)  

  
class BiasFree_LayerNorm(nn.Module):     
    def __init__(self, normalized_shape):    
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral): 
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
  
        assert len(normalized_shape) == 1   
     
        self.weight = nn.Parameter(torch.ones(normalized_shape))     
        self.normalized_shape = normalized_shape
 
    def forward(self, x):     
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight    
 
  
class WithBias_LayerNorm(nn.Module):     
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)   
        normalized_shape = torch.Size(normalized_shape)     
   
        assert len(normalized_shape) == 1 

        self.weight = nn.Parameter(torch.ones(normalized_shape))   
        self.bias = nn.Parameter(torch.zeros(normalized_shape)) 
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True) 
        sigma = x.var(-1, keepdim=True, unbiased=False)  
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias   
 

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)   
  
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
   
class SEFN(nn.Module):     
    def __init__(self, in_features, hidden_features=None, out_features=None, bias=False):
        super(SEFN, self).__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
    
        self.project_in = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1, bias=bias)
        
        self.fusion = nn.Conv2d(hidden_features + in_features, hidden_features, kernel_size=1, bias=bias)    
        self.dwconv_afterfusion = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,  
                                groups=hidden_features, bias=bias)    

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)     


        self.project_out = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias)    

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)  
        self.conv = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3,stride=1,padding=1,bias=True),
            LayerNorm(in_features, 'WithBias'),
            nn.ReLU(inplace=True), 
            nn.Conv2d(in_features, in_features, kernel_size=3,stride=1,padding=1,bias=True),
            LayerNorm(in_features, 'WithBias'),
            nn.ReLU(inplace=True)
        ) 
        self.upsample = nn.Upsample(scale_factor=2)   
        

    def forward(self, x, spatial):
        
        x = self.project_in(x)
   
        #### Spatial branch
        y = self.avg_pool(spatial)  
        y = self.conv(y)   
        y = self.upsample(y)    
        ####  
        
        x1, x2 = self.dwconv(x).chunk(2, dim=1)  
        x1 = self.fusion(torch.cat((x1, y),dim=1))  
        x1 = self.dwconv_afterfusion(x1)
        
        x = F.gelu(x1) * x2 
        x = self.project_out(x)     
        return x
     
if __name__ == '__main__': 
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, hidden_channel, out_channel, height, width = 1, 16, 64, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # 此模块有使用教程在VideoBaiduYun.txt内     

    module = SEFN(in_features=in_channel, hidden_features=hidden_channel, out_features=out_channel).to(device)

    outputs = module(inputs, inputs) 
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET) 
 
    print(ORANGE) 
    flops, macs, _ = calculate_flops(model=module, 
                                     args=[inputs, inputs],     
                                     output_as_string=True,
                                     output_precision=4,     
                                     print_detailed=True)
    print(RESET)  
