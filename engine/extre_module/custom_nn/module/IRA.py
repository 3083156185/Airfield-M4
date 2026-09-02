'''
本文件由BiliBili：魔傀面具整理   
engine/extre_module/module_images/CVPR2026-IRA.png
engine/extre_module/module_images/CVPR2026-IRA.md    
论文链接：https://arxiv.org/pdf/2512.08441
'''
   
import warnings
warnings.filterwarnings('ignore')  
from calflops import calculate_flops
 
import torch     
import torch.nn as nn   
import torch.nn.functional as F  

class AttentionBlock(nn.Module):    
    def __init__(self, dim: int): 
        super(AttentionBlock, self).__init__()
        self._spatial_attention_conv = nn.Conv2d(2, dim, kernel_size=3, padding=1)    

        # Channel attention MLP
        self._channel_attention_conv0 = nn.Conv2d(1, dim, kernel_size=1, padding=0)  
        self._channel_attention_conv1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0)

        self._out_conv = nn.Conv2d(2 * dim, dim, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor):
        if len(x.shape) != 4:     
            raise ValueError(f"Expected [B, C, H, W] input, got {x.shape}.")

        # Spatial attention    
        mean = torch.mean(x, dim=1, keepdim=True)  # Mean/Max on C axis 
        max, _ = torch.max(x, dim=1, keepdim=True)   
        spatial_attention = torch.cat([mean, max], dim=1)  # [B, 2, H, W]
        spatial_attention = self._spatial_attention_conv(spatial_attention)
        spatial_attention = torch.sigmoid(spatial_attention) * x  

        # Channel attention. TODO: Correct that it only uses average pool contrary to CBAM?  
        # NOTE/TODO: This differs from CBAM as it uses Channel pooling, not spatial pooling!
        # In a way, this is 2x spatial attention  
        channel_attention = torch.relu(self._channel_attention_conv0(mean))
        channel_attention = self._channel_attention_conv1(channel_attention)
        channel_attention = torch.sigmoid(channel_attention) * x   

        attention = torch.cat([spatial_attention, channel_attention], dim=1)  # [B, 2*dim, H, W]  
        attention = self._out_conv(attention) 
        return x + attention

class BaseBlock(nn.Module):
    def __init__(self, channels: int):     
        super(BaseBlock, self).__init__()

        self._conv0 = nn.Conv2d(channels, channels, kernel_size=1)
        self._dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)  
        self._conv1 = nn.Conv2d(channels, channels, kernel_size=1)  

        self._conv2 = nn.Conv2d(channels, channels, kernel_size=1)     
        self._conv3 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        features = self._conv0(x)
        features = F.elu(self._dw_conv(features))  # TODO: ELU or ReLU?   
        features = self._conv1(features)   
        x = x + features

        features = F.elu(self._conv2(x))   
        features = self._conv3(features)
        return x + features

class IRA(nn.Module):
    def __init__(self, inc, ouc):    
        super().__init__()     

        self.conv = nn.Conv2d(inc, ouc, kernel_size=3, padding=1)    
        self.baseBlock = nn.Sequential(BaseBlock(ouc), BaseBlock(ouc))  
        self.attn = AttentionBlock(ouc)

    def forward(self, x):
        x = self.conv(x)    
        x = self.baseBlock(x)    
        x = self.attn(x)
        return x

if __name__ == '__main__': 
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)    

    module = IRA(in_channel, out_channel).to(device)     

    outputs = module(inputs) 
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)   
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,  
                                     output_precision=4, 
                                     print_detailed=True)
    print(RESET)
