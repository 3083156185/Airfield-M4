import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init   
 

class RMSNorm(nn.Module):    
    def __init__(self, dim: int, eps: float = 1e-12):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))     
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale


class GatedFFNBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w12 = nn.Linear(dim, 4 * dim)  
        self.w3 = nn.Linear(2 * dim, dim)
        self.norm = RMSNorm(dim)   

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x12 = self.w12(self.norm(x))
        x1, x2 = x12.chunk(2, dim=-1)   
        return x + self.w3(F.silu(x1) * x2)
  
   
class TextAdapter(nn.Module):    
    def __init__(self, text_dim: int, img_dim: int, num_layers: int = 1):    
        super().__init__()
        self.layers = nn.ModuleList([GatedFFNBlock(text_dim) for _ in range(num_layers)])
        self.proj_out = nn.Linear(text_dim, img_dim)
        self._reset_parameters()     

    def _reset_parameters(self) -> None:   
        for layer in self.layers:     
            init.xavier_uniform_(layer.w12.weight)
            init.constant_(layer.w12.bias, 0)
            init.xavier_uniform_(layer.w3.weight)
            init.constant_(layer.w3.bias, 0)
        init.xavier_uniform_(self.proj_out.weight)    
        init.constant_(self.proj_out.bias, 0) 
   
    def forward(self, text_feats: torch.Tensor) -> torch.Tensor: 
        x = text_feats   
        for layer in self.layers:
            x = layer(x)
        return self.proj_out(x)
