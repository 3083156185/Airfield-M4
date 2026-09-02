'''
本文件由BiliBili：魔傀面具整理 
engine/extre_module/module_images/TGRS2026-GLCDM.png
engine/extre_module/module_images/TGRS2026-GLCDM.md   
论文链接：https://ieeexplore.ieee.org/document/11397410
'''

import warnings
warnings.filterwarnings('ignore')     
from calflops import calculate_flops

import torch  
import torch.nn as nn 

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)  
        self.norm2 = nn.LayerNorm(embed_dim)   
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)   
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))  
        return x 

 
class HorizontalTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers):
        super(HorizontalTransformer, self).__init__()
        self.layers = nn.ModuleList(     
            [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        )
   
    def forward(self, x):  
        batch_size, channels, height, width = x.size()  
        x = x.permute(0, 2, 3, 1).reshape(batch_size * height, width, channels)
        for layer in self.layers:     
            x = layer(x)
        x = x.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2) 
        return x  
  
   
class VerticalTransformer(nn.Module):  
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers):  
        super(VerticalTransformer, self).__init__()
        self.layers = nn.ModuleList(  
            [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        )     
   
    def forward(self, x):   
        batch_size, channels, height, width = x.size()     
        x = x.permute(0, 3, 2, 1).reshape(batch_size * width, height, channels)
        for layer in self.layers:
            x = layer(x)
        x = x.reshape(batch_size, width, height, channels).permute(0, 3, 2, 1)
        return x


class GLCDM(nn.Module):
    def __init__(self, in_channels, num_heads=8):     
        super(GLCDM, self).__init__()     
        self.in_channels = in_channels
        self.HTransformer = HorizontalTransformer(    
            embed_dim=in_channels,    
            num_heads=num_heads,   
            ff_dim=in_channels * 2, 
            num_layers=1,   
        )
        self.VTransformer = VerticalTransformer(
            embed_dim=in_channels,
            num_heads=num_heads,
            ff_dim=in_channels,   
            num_layers=1,
        )     
        self.fusion_conv1 = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1)  
        self.self_att = TransformerBlock(in_channels, num_heads, in_channels * 2) 
        self.fusion_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
  
        self.layers = nn.ModuleList(    
            [
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),   
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),  
            ] 
        )    

    def forward(self, feature_map): 
        batch_size, channels, height, width = feature_map.size()
        if channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {channels}."
            )   
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError("Feature map height and width must be even.")

        h_context = self.HTransformer(feature_map) 
        v_context = self.VTransformer(feature_map)
        fused_out = torch.cat([h_context, v_context], dim=1)
        global_map = self.fusion_conv1(fused_out)
   
        half_height = height // 2
        half_width = width // 2     
        blocks = [
            feature_map[:, :, :half_height, :half_width],
            feature_map[:, :, :half_height, half_width:],   
            feature_map[:, :, half_height:, :half_width],
            feature_map[:, :, half_height:, half_width:], 
        ]    

        attended_outputs = []
        for index, block in enumerate(blocks):
            attended_outputs.append(self.layers[index](block))
     
        stacked_outputs = torch.stack(attended_outputs)
        reshaped_tensor = stacked_outputs.reshape(     
            2, 2, batch_size, self.in_channels, half_height, half_width  
        )
        permuted_tensor = reshaped_tensor.permute(2, 3, 0, 4, 1, 5) 
        output = permuted_tensor.contiguous().reshape(   
            batch_size, self.in_channels, height, width
        ) 
    
        output = self.fusion_conv2(global_map + output)   
        return output     

if __name__ == '__main__':     
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')     
    batch_size, channel, height, width = 1, 16, 20, 20
    inputs = torch.randn((batch_size, channel, height, width)).to(device) 
 
    module = GLCDM(channel, num_heads=8).to(device)  

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET) 
    
    print(ORANGE)  
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width),     
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)