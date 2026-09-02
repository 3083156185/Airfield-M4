'''  
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/GLSA.png
engine/extre_module/module_images/GLSA.md
论文链接：https://arxiv.org/pdf/2212.11677    
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')    
from calflops import calculate_flops  
   
import torch
import torch.nn as nn  
from typing import Union   
    
from engine.extre_module.ultralytics_nn.conv import Conv   
 
class ContextBlock(nn.Module):   

    def __init__(self,
                 inplanes,
                 ratio, 
                 pooling_type='att',
                 fusion_types=('channel_mul', )):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))  
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])    
        assert len(fusion_types) > 0, 'at least one fusion should be used'   
        self.inplanes = inplanes 
        self.ratio = ratio     
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type     
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)   
        if 'channel_add' in fusion_types: 
            self.channel_add_conv = nn.Sequential(   
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1), 
                nn.LayerNorm([self.planes, 1, 1]),     
                nn.ReLU(inplace=True),  # yapf: disable   
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))   
        else:  
            self.channel_add_conv = None 
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(   
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),    
                nn.LayerNorm([self.planes, 1, 1]),  
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:     
            self.channel_mul_conv = None
        self.reset_parameters()    
     
    @staticmethod
    def last_zero_init(m: Union[nn.Module, nn.Sequential]) -> None:
        try:
            from mmengine.model import kaiming_init, constant_init
            if isinstance(m, nn.Sequential):
                constant_init(m[-1], val=0)
            else:     
                constant_init(m, val=0)  
        except ImportError as e:
            pass
 
    def reset_parameters(self):   
        try:
            from mmengine.model import kaiming_init   
            if self.pooling_type == 'att': 
                kaiming_init(self.conv_mask, mode='fan_in')    
                self.conv_mask.inited = True    
     
            if self.channel_add_conv is not None: 
                self.last_zero_init(self.channel_add_conv)   
            if self.channel_mul_conv is not None: 
                self.last_zero_init(self.channel_mul_conv)     
        except ImportError as e:  
            pass

    def spatial_pool(self, x):
        batch, channel, height, width = x.size() 
        if self.pooling_type == 'att':   
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)  
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]    
            context_mask = context_mask.view(batch, 1, height * width)   
            # [N, 1, H * W]    
            context_mask = self.softmax(context_mask)   
            # [N, 1, H * W, 1]    
            context_mask = context_mask.unsqueeze(-1)     
            # [N, 1, C, 1]     
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1) 
        else:   
            # [N, C, 1, 1]  
            context = self.avg_pool(x)
     
        return context
    
    def forward(self, x):    
        # [N, C, 1, 1]
        context = self.spatial_pool(x)   
 
        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out + out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
   
        return out
  
class GLSAChannelAttention(nn.Module):     
    def __init__(self, in_planes, ratio=16):   
        super(GLSAChannelAttention, self).__init__()  
        self.avg_pool = nn.AdaptiveAvgPool2d(1)   
        self.max_pool = nn.AdaptiveMaxPool2d(1)
     
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)   
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)   
   
        self.sigmoid = nn.Sigmoid()    

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))    
        out = avg_out + max_out   
        return self.sigmoid(out)   
 
    
class GLSASpatialAttention(nn.Module):  
    def __init__(self, kernel_size=7):  
        super(GLSASpatialAttention, self).__init__()   
   
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'    
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x) 
     
class GLSAConvBranch(nn.Module):    
    def __init__(self, in_features, hidden_features = None, out_features = None):     
        super().__init__()  
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.conv1 = Conv(in_features, hidden_features, 1, act=nn.ReLU(inplace=True))   
        self.conv2 = Conv(hidden_features, hidden_features, 3, g=hidden_features, act=nn.ReLU(inplace=True))     
        self.conv3 = Conv(hidden_features, hidden_features, 1, act=nn.ReLU(inplace=True))     
        self.conv4 = Conv(hidden_features, hidden_features, 3, g=hidden_features, act=nn.ReLU(inplace=True))
        self.conv5 = Conv(hidden_features, hidden_features, 1, act=nn.SiLU(inplace=True))
        self.conv6 = Conv(hidden_features, hidden_features, 3, g=hidden_features, act=nn.ReLU(inplace=True))  
        self.conv7 = nn.Sequential(    
            nn.Conv2d(hidden_features, out_features, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.ca = GLSAChannelAttention(64) 
        self.sa = GLSASpatialAttention()
        self.sigmoid_spatial = nn.Sigmoid()    
    
    def forward(self, x):     
        res1 = x
        res2 = x
        x = self.conv1(x) 
        x = x + self.conv2(x)
        x = self.conv3(x) 
        x = x + self.conv4(x) 
        x = self.conv5(x)    
        x = x + self.conv6(x)
        x = self.conv7(x)
        x_mask = self.sigmoid_spatial(x)
        res1 = res1 * x_mask  
        return res2 + res1    
 
class GLSA(nn.Module): 
   
    def __init__(self, input_dim=512, output_dim=32):
        super().__init__()
                      
        self.conv1_1 = Conv(output_dim*2, output_dim, 1) 
        self.conv1_1_1 = Conv(input_dim//2, output_dim,1)
        self.local_11conv = nn.Conv2d(input_dim//2,output_dim,1)    
        self.global_11conv = nn.Conv2d(input_dim//2,output_dim,1) 
        self.GlobelBlock = ContextBlock(inplanes= output_dim, ratio=2)
        self.local = GLSAConvBranch(in_features = output_dim, hidden_features = output_dim, out_features = output_dim) 

    def forward(self, x):
        b, c, h, w = x.size()
        x_0, x_1 = x.chunk(2,dim = 1)  
        
    # local block 
        local = self.local(self.local_11conv(x_0))
        
    # Globel block 
        Globel = self.GlobelBlock(self.global_11conv(x_1))
     
    # concat Globel + local
        x = torch.cat([local,Globel], dim=1)
        x = self.conv1_1(x)

        return x
   
if __name__ == '__main__':    
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"  
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = GLSA(in_channel, out_channel).to(device)

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)   
     
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,   
                                     input_shape=(batch_size, in_channel, height, width),  
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)     
    print(RESET)