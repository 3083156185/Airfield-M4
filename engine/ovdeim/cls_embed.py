import math

import torch    
import torch.nn as nn 
import torch.nn.functional as F     


class ClassEmbed(nn.Module):
    def __init__(self, init_bias: float = 100.0, init_scale: float = 15.0):
        super().__init__()  
        self.lang_bias = nn.Parameter(torch.full((), -math.log(init_bias)))
        self.lang_scale = nn.Parameter(torch.tensor(init_scale).log())    
 
    def forward(self, image_embeds: torch.Tensor, lang_embeds: torch.Tensor, mask=None) -> torch.Tensor:  
        image_norm = F.normalize(image_embeds, p=2, dim=-1)
        lang_norm = F.normalize(lang_embeds, p=2, dim=-1)   
        logits = image_norm @ lang_norm.transpose(2, 1)
        logits = logits * torch.exp(self.lang_scale) + self.lang_bias
        if mask is not None: 
            logits = logits.masked_fill(~mask.unsqueeze(1), float("-inf")) 
        return logits
