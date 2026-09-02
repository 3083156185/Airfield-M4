'''   
本文件由BiliBili：魔傀面具整理     
engine/extre_module/module_images/AAAI2025-PSConv.png
论文链接：https://www.arxiv.org/pdf/2408.04804
'''
  
import warnings  
warnings.filterwarnings('ignore')    
from calflops import calculate_flops    
   
import torch
import torch.nn as nn   

class MessageAgg(nn.Module): 
    def __init__(self, agg_method="mean"): 
        super().__init__()     
        self.agg_method = agg_method  
  
    def forward(self, X, path):     
        """     
            X: [n_node, dim]
            path: col(source) -> row(target)    
        """
        X = torch.matmul(path, X) 
        if self.agg_method == "mean": 
            norm_out = 1 / torch.sum(path, dim=2, keepdim=True)    
            norm_out[torch.isinf(norm_out)] = 0  
            X = norm_out * X  
            return X
        elif self.agg_method == "sum":
            pass
        return X   
 
class HyPConv(nn.Module):   
    def __init__(self, c1, c2):
        super().__init__()
        self.fc = nn.Linear(c1, c2)
        self.v2e = MessageAgg(agg_method="mean")
        self.e2v = MessageAgg(agg_method="mean")

    def forward(self, x, H):
        x = self.fc(x)
        # v -> e    
        E = self.v2e(x, H.transpose(1, 2).contiguous())   
        # e -> v
        x = self.e2v(E, H)    
    
        return x     

class HyperComputeModule(nn.Module):
    def __init__(self, c, threshold):    
        super().__init__()    
        self.threshold = threshold 
        self.hgconv = HyPConv(c, c)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU()

    def forward(self, x):    
        b, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]  
        x = x.view(b, c, -1).transpose(1, 2).contiguous()
        feature = x.clone()
        distance = torch.cdist(feature, feature)
        hg = distance < self.threshold    
        hg = hg.float().to(x.device).to(x.dtype)    
        x = self.hgconv(x, hg).to(x.device).to(x.dtype) + x    
        x = x.transpose(1, 2).contiguous().view(b, c, h, w)  
        x = self.act(self.bn(x))
        return x

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"     
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  
    batch_size, channel, height, width = 1, 16, 32, 32
    inputs = torch.randn((batch_size, channel, height, width)).to(device)     

    module = HyperComputeModule(channel, threshold=8).to(device)  
 
    outputs = module(inputs)    
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET) 

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,     
                                     print_detailed=True) 
    print(RESET) 
