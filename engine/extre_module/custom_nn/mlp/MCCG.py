''' 
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/自研模块-MCCG.md
engine/extre_module/module_images/自研模块-MCCG.png 
'''  

import warnings     
warnings.filterwarnings('ignore')  
  
import torch     
import torch.nn as nn


class _ContextRouter(nn.Module):     
    def __init__(self, channels, num_paths=3, reduction=4):    
        super().__init__()
        hidden = max(channels // reduction, num_paths) 
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, num_paths, 1, bias=True)

    def forward(self, x):     
        weight = self.fc2(self.act(self.fc1(self.pool(x))))
        return torch.softmax(weight.flatten(1), dim=1)

 
class MCCG(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()   
        out_features = out_features or in_features     
        hidden_features = hidden_features or in_features   
        if hidden_features % 2 != 0:
            raise ValueError('hidden_features must be divisible by 2')
   
        self.hidden_dim = hidden_features // 2   
        self.dim = in_features     
        self.dim_conv = max(1, self.dim // 4)
        self.dim_untouched = self.dim - self.dim_conv    
     
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False) 
        self.linear1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            act_layer()
        )   
        self.local_dwconv = nn.Sequential( 
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 1, 1, groups=self.hidden_dim, bias=False),    
            act_layer()   
        )
        self.dilated_dwconv = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 1, 2, dilation=2, groups=self.hidden_dim, bias=False),    
            act_layer()
        )
        self.strip_conv = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, (1, 5), 1, (0, 2), groups=self.hidden_dim, bias=False),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, (5, 1), 1, (2, 0), groups=self.hidden_dim, bias=False),   
            act_layer()
        )  
        self.router = _ContextRouter(self.hidden_dim, num_paths=3)
        self.linear2 = nn.Conv2d(self.hidden_dim, out_features, 1)

    def forward(self, x):    
        if self.dim_untouched > 0:
            x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
            x1 = self.partial_conv3(x1)  
            x = torch.cat((x1, x2), dim=1)
        else:
            x = self.partial_conv3(x) 

        x = self.linear1(x) 
        feat, gate = x.chunk(2, dim=1)
   
        weights = self.router(feat)  
        local = self.local_dwconv(feat)
        dilated = self.dilated_dwconv(feat)  
        strip = self.strip_conv(feat)
        feat = (
            local * weights[:, 0].view(-1, 1, 1, 1)
            + dilated * weights[:, 1].view(-1, 1, 1, 1)
            + strip * weights[:, 2].view(-1, 1, 1, 1)
        )     
        x = feat * gate
        x = self.linear2(x)
        return x    
    

if __name__ == '__main__':     
    from calflops import calculate_flops  

    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   
    batch_size, in_channel, hidden_channel, out_channel, height, width = 1, 16, 64, 32, 32, 32     
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)    

    module = MCCG(in_features=in_channel, hidden_features=hidden_channel, out_features=out_channel).to(device)
   
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)   

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),  
                                     output_as_string=True,
                                     output_precision=4, 
                                     print_detailed=True)  
    print(RESET)
