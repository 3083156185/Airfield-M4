'''  
本文件由BiliBili：魔傀面具整理     
ultralytics/nn/module_images/NN2025-KSFA.md
ultralytics/nn/module_images/NN2025-KSFA.png
论文链接：https://arxiv.org/pdf/2410.03171v3
'''

import warnings  
warnings.filterwarnings('ignore')   
from calflops import calculate_flops

import torch
import torch.nn as nn     

class KSFA(nn.Module):
    def __init__(self, dim, r=16, L=32):
        super().__init__()     
        d = max(dim // r, L)  
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2)   
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)    
        self.conv2 = nn.Conv2d(dim, dim // 2, 1) 
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)    
        self.conv = nn.Conv2d(dim // 2, dim, 1)     

        self.global_pool = nn.AdaptiveAvgPool2d(1)    
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),  
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Conv2d(d, dim, 1, 1, bias=False)   
        self.softmax = nn.Softmax(dim=1) 

    def forward(self, x):     
        batch_size = x.size(0)     
        dim = x.size(1)
        attn1 = self.conv0(x)  # conv_3*3   
        attn2 = self.conv_spatial(attn1)  # conv_3*3 -> conv_5*5

        attn1 = self.conv1(attn1) # b, dim/2, h, w    
        attn2 = self.conv2(attn2) # b, dim/2, h, w   
  
        attn = torch.cat([attn1, attn2], dim=1)  # b,c,h,w  
        avg_attn = torch.mean(attn, dim=1, keepdim=True) # b,1,h,w
        max_attn, _ = torch.max(attn, dim=1, keepdim=True) # b,1,h,w     
        agg = torch.cat([avg_attn, max_attn], dim=1) # spa b,2,h,w    
  
        ch_attn1 = self.global_pool(attn) # b,dim,1, 1
        z = self.fc1(ch_attn1)
        a_b = self.fc2(z) 
        a_b = a_b.reshape(batch_size, 2, dim // 2, -1)  
        a_b = self.softmax(a_b)
     
        a1,a2 =  a_b.chunk(2, dim=1)  
        a1 = a1.reshape(batch_size,dim // 2,1,1)     
        a2 = a2.reshape(batch_size, dim // 2, 1, 1) 
     
        w1 = a1 * agg[:, 0, :, :].unsqueeze(1)
        w2 = a2 * agg[:, 1, :, :].unsqueeze(1)   

        attn = attn1 * w1 + attn2 * w2
        attn = self.conv(attn).sigmoid()   
 
        return x * attn

if __name__ == '__main__':    
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel, height, width = 1, 16, 32, 32   
    inputs = torch.randn((batch_size, channel, height, width)).to(device)

    module = KSFA(channel).to(device)     
    module.eval()  

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    # print(ORANGE)    
    # flops, macs, _ = calculate_flops(model=module,  
    #                                  input_shape=(batch_size, channel, height, width),    
    #                                  output_as_string=True,    
    #                                  output_precision=4,
    #                                  print_detailed=True)     
    # print(RESET)  
