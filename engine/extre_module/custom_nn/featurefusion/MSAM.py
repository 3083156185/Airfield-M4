'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/TGRS2025-MSAM.md
engine/extre_module/module_images/TGRS2025-MSAM.png
论文链接：https://ieeexplore.ieee.org/abstract/document/10969832  
'''
  
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')    
  
import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops   
    
import torch
import torch.nn as nn     
from einops import rearrange

from engine.extre_module.ultralytics_nn.conv import Conv, DSConv
    
class IndentityBlock(nn.Module):
    def __init__(self, in_channel, kernel_size, filters, rate=1):
        super(IndentityBlock, self).__init__() 
        F1, F2, F3 = filters 
        self.stage = nn.Sequential(
            Conv(in_channel, F1, act=nn.ReLU),
            DSConv(F1, F2, kernel_size, d=rate),    
            Conv(F2, F3, act=nn.ReLU)
        ) 
        self.relu_1 = nn.ReLU(True)   

    def forward(self, X):   
        X_shortcut = X
        X = self.stage(X)     
        X = X + X_shortcut
        X = self.relu_1(X)
        return X   

class SelfAttention(nn.Module):
    def __init__(self, dim_out):    
        super(SelfAttention, self).__init__()
        self.conv = DSConv(dim_out * 2, (dim_out // 2) * 3, 3)     
        self.att_dim = dim_out // 2     
 
    def forward(self, x, y):
        b, c, h, w = x.shape     
        fm = self.conv(torch.concat([x, y], dim=1))
    
        Q, K, V = rearrange(fm, 'b (qkv c) h w -> qkv b h c w'     
                            , qkv=3, b=b, c=self.att_dim, h=h, w=w) 

        dots = (Q @ K.transpose(-2, -1))  
        attn = dots.softmax(dim=-1)
        attn = attn @ V 
        attn = attn.view(b, -1, h, w)  

        return attn
    
class MSAM_P4(nn.Module):    
    def __init__(self, dim_in, dim_out):
        super(MSAM_P4, self).__init__()  
        self.branch1 = nn.Sequential(     
            DSConv(dim_in[0], dim_out, 3, s=2),
            DSConv(dim_out, dim_out, 3, s=2),
        )    
        self.branch2 = DSConv(dim_in[1], dim_out, 3, s=2)
        self.branch3 = Conv(dim_in[2], dim_out, 1)
        self.branch4 = nn.Sequential(
            nn.Upsample(scale_factor=2),   
            Conv(dim_in[3], dim_out)  
        )
        self.merge = Conv(4 * dim_out, dim_out)
        self.resblock = nn.Sequential(     
            IndentityBlock(in_channel=dim_out, kernel_size=3, filters=[dim_out, dim_out, dim_out]), 
            IndentityBlock(in_channel=dim_out, kernel_size=3, filters=[dim_out, dim_out, dim_out])  
        )
        self.transformer = SelfAttention(dim_out)     
        self.conv = nn.Conv2d(dim_out // 2 * 10, dim_out, 1)
        self.dim_out = dim_out 

    def forward(self, input):
        b, c, h, w = input[2].shape   
        list1 = []
        list2 = []

        x1 = self.branch1(input[0])     
        x2 = self.branch2(input[1])
        x3 = self.branch3(input[2])  
        x = self.branch4(input[3])  
   
        # CNN
        merge = self.merge(torch.cat([x, x1, x2, x3], dim=1)) 
        merge = self.resblock(merge)  

        # Transformer
        list1.append(x)
        list1.append(x3)  
        list1.append(x2) 
        list1.append(x1)     
  
        for i in range(len(list1)):    
            for j in range(len(list1)):
                if i <= j:
                    att = self.transformer(list1[i], list1[j])  
                    list2.append(att)  

        for j in range(len(list2)):  
            list2[j] = list2[j].view(b, self.dim_out // 2, h, w)
     
        out = self.conv(torch.concat(list2, dim=1))

        return out + merge

class MSAM_P5(nn.Module):     
    def __init__(self, dim_in, dim_out): 
        super(MSAM_P5, self).__init__()
        self.branch1 = nn.Sequential(
            DSConv(dim_in[0], dim_out, 3, s=2),   
            DSConv(dim_out, dim_out, 3, s=2),  
            DSConv(dim_out, dim_out, 3, s=2),
        )   
        self.branch2 = nn.Sequential(DSConv(dim_in[1], dim_out, 3, s=2), DSConv(dim_out, dim_out, 3, s=2))
        self.branch3 = DSConv(dim_in[2], dim_out, 3, s=2)   
        self.branch4 = Conv(dim_in[3], dim_out)
        self.merge = Conv(4 * dim_out, dim_out)    
        self.resblock = nn.Sequential(
            IndentityBlock(in_channel=dim_out, kernel_size=3, filters=[dim_out, dim_out, dim_out]),
            IndentityBlock(in_channel=dim_out, kernel_size=3, filters=[dim_out, dim_out, dim_out])
        )    
        self.transformer = SelfAttention(dim_out)
        self.conv = nn.Conv2d(dim_out // 2 * 10, dim_out, 1)
        self.dim_out = dim_out
  
    def forward(self, input):   
        b, c, h, w = input[3].shape     
        list1 = []
        list2 = []    
    
        x1 = self.branch1(input[0]) 
        x2 = self.branch2(input[1])   
        x3 = self.branch3(input[2])     
        x = self.branch4(input[3])
  
        # CNN
        merge = self.merge(torch.cat([x, x1, x2, x3], dim=1))     
        merge = self.resblock(merge)
     
        # Transformer
        list1.append(x)
        list1.append(x3)
        list1.append(x2)
        list1.append(x1)
     
        for i in range(len(list1)):
            for j in range(len(list1)):     
                if i <= j:     
                    att = self.transformer(list1[i], list1[j])  
                    list2.append(att)  
  
        for j in range(len(list2)):
            list2[j] = list2[j].view(b, self.dim_out // 2, h, w)   

        out = self.conv(torch.concat(list2, dim=1))
    
        return out + merge    

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel, height, width = 1, 128, 20, 20
    inputs_P2 = torch.randn((batch_size, channel // 8, height * 8, width * 8)).to(device)
    inputs_P3 = torch.randn((batch_size, channel // 4, height * 4, width * 4)).to(device)   
    inputs_P4 = torch.randn((batch_size, channel // 2, height * 2, width * 2)).to(device)
    inputs_P5 = torch.randn((batch_size, channel, height, width)).to(device)
   
    print(RED + '-'*20 + " MSAM_P4 " + '-'*20 + RESET)

    feats = [inputs_P2, inputs_P3, inputs_P4, inputs_P5]
    module = MSAM_P4([channel // 8, channel // 4, channel // 2, channel], channel).to(device)

    outputs = module(feats)    
    print(GREEN + f'inputs_P2.size:{inputs_P2.size()} inputs_P3.size:{inputs_P3.size()} inputs_P4.size:{inputs_P4.size()} inputs_P5.size:{inputs_P5.size()} outputs.size:{outputs.size()}' + RESET)    
    
    print(ORANGE)    
    flops, macs, _ = calculate_flops(model=module,  
                                     args=[feats],
                                     output_as_string=True,     
                                     output_precision=4,     
                                     print_detailed=True)  
    print(RESET)  

    print(RED + '-'*20 + " MSAM_P5 " + '-'*20 + RESET)   
  
    feats = [inputs_P2, inputs_P3, inputs_P4, inputs_P5]    
    module = MSAM_P5([channel // 8, channel // 4, channel // 2, channel], channel).to(device)    
 
    outputs = module(feats)
    print(GREEN + f'inputs_P2.size:{inputs_P2.size()} inputs_P3.size:{inputs_P3.size()} inputs_P4.size:{inputs_P4.size()} inputs_P5.size:{inputs_P5.size()} outputs.size:{outputs.size()}' + RESET)   

    print(ORANGE) 
    flops, macs, _ = calculate_flops(model=module,
                                     args=[feats],
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)