'''  
本文件由BiliBili：魔傀面具整理 
engine/extre_module/module_images/ASF-Neck.png
engine/extre_module/module_images/ASF-Neck.md     
论文链接：https://arxiv.org/pdf/2312.06458
'''
   
import os, sys    
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')   
     
import warnings   
warnings.filterwarnings('ignore')  
from calflops import calculate_flops 

import torch, math
import torch.nn as nn   
import torch.nn.functional as F     
   
from engine.extre_module.ultralytics_nn.conv import Conv    

__all__ = ("Add", "ScalSeq", "Zoom_cat", "asf_attention_model")
 
class Zoom_cat(nn.Module):     
    def __init__(self):
        super().__init__()
     
    def forward(self, x):  
        large, medium, small = x
        target_size = medium.shape[2:]
        large = F.adaptive_max_pool2d(large, target_size) + F.adaptive_avg_pool2d(large, target_size)     
        small = F.interpolate(small, target_size, mode="nearest")     
        return torch.cat([large, medium, small], dim=1)  
  
 
class ScalSeq(nn.Module):   
    def __init__(self, inc, channel):
        super().__init__()
        if channel != inc[0]:    
            self.conv0 = Conv(inc[0], channel, 1)     
        self.conv1 = Conv(inc[1], channel, 1)
        self.conv2 = Conv(inc[2], channel, 1)
        self.conv3d = nn.Conv3d(channel, channel, kernel_size=(1, 1, 1))
        self.bn = nn.BatchNorm3d(channel)     
        self.act = nn.LeakyReLU(0.1)    
        self.pool_3d = nn.MaxPool3d(kernel_size=(3, 1, 1))   
    
    def forward(self, x):
        p3, p4, p5 = x
        if hasattr(self, "conv0"):    
            p3 = self.conv0(p3)
        p4 = F.interpolate(self.conv1(p4), p3.shape[2:], mode="nearest") 
        p5 = F.interpolate(self.conv2(p5), p3.shape[2:], mode="nearest")
        combine = torch.cat([p3.unsqueeze(2), p4.unsqueeze(2), p5.unsqueeze(2)], dim=2)  
        fused = self.act(self.bn(self.conv3d(combine)))
        return self.pool_3d(fused).squeeze(2)   
   
    
class Add(nn.Module):
    def __init__(self):
        super().__init__()     

    def forward(self, x):     
        return torch.sum(torch.stack(x, dim=0), dim=0)

  
class asf_channel_att(nn.Module):
    def __init__(self, channel, b=1, gamma=2):     
        super(asf_channel_att, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
   
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1)     
        y = y.transpose(-1, -2)    
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)   
        y = self.sigmoid(y)
        return x * y.expand_as(x)
 

class asf_local_att(nn.Module): 
    def __init__(self, channel, reduction=16):
        super(asf_local_att, self).__init__()     

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)
  
        self.relu   = nn.ReLU()    
        self.bn     = nn.BatchNorm2d(channel//reduction) 

        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)    
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)   
     
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()   

    def forward(self, x):     
        _, _, h, w = x.size()     

        x_h = torch.mean(x, dim = 3, keepdim = True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim = 2, keepdim = True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))
    
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)     

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2))) 
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
     
        out = x * s_h.expand_as(x) * s_w.expand_as(x)    
        return out 
   
  
class asf_attention_model(nn.Module):    
    # Concatenate a list of tensors along dimension 
    def __init__(self, ch=256):
        super().__init__()
        self.channel_att = asf_channel_att(ch)  
        self.local_att = asf_local_att(ch) 

    def forward(self, x):    
        input1, input2 = x[0], x[1]  
        input1 = self.channel_att(input1)
        x = input1 + input2     
        x = self.local_att(x)     
        return x
    
 
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel, height, width = 1, 128, 20, 20
     
    inputs_P3 = torch.randn((batch_size, channel, height * 4, width * 4)).to(device)
    inputs_P4 = torch.randn((batch_size, channel, height * 2, width * 2)).to(device)
    inputs_P5 = torch.randn((batch_size, channel, height, width)).to(device)
  
    print(RED + '-' * 20 + " Zoom_cat " + '-' * 20 + RESET)    
    feats = [inputs_P3, inputs_P4, inputs_P5]
    module = Zoom_cat().to(device)    
 
    outputs = module(feats)
    print(GREEN + f'inputs_P3.size:{inputs_P3.size()} inputs_P4.size:{inputs_P4.size()} inputs_P5.size:{inputs_P5.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     args=[feats],
                                     output_as_string=True,    
                                     output_precision=4,    
                                     print_detailed=True)    
    print(RESET)    

    print(RED + '-' * 20 + " ScalSeq " + '-' * 20 + RESET)
    feats = [inputs_P3, inputs_P4, inputs_P5]
    module = ScalSeq([channel, channel, channel], channel).to(device)   
   
    outputs = module(feats)
    print(GREEN + f'inputs_P3.size:{inputs_P3.size()} inputs_P4.size:{inputs_P4.size()} inputs_P5.size:{inputs_P5.size()} outputs.size:{outputs.size()}' + RESET)   
 
    print(ORANGE)  
    flops, macs, _ = calculate_flops(model=module,  
                                     args=[feats],
                                     output_as_string=True,
                                     output_precision=4, 
                                     print_detailed=True)
    print(RESET)    

    print(RED + '-' * 20 + " Add " + '-' * 20 + RESET)  
    add_input1 = torch.randn((batch_size, channel, height * 2, width * 2)).to(device)
    add_input2 = torch.randn((batch_size, channel, height * 2, width * 2)).to(device)    
    feats = [add_input1, add_input2]
    module = Add().to(device)   
   
    outputs = module(feats)
    print(GREEN + f'add_input1.size:{add_input1.size()} add_input2.size:{add_input2.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     args=[feats],     
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)

    print(RED + '-' * 20 + " asf_attention_model " + '-' * 20 + RESET)
    attention_input1 = torch.randn((batch_size, channel, height * 2, width * 2)).to(device)
    attention_input2 = torch.randn((batch_size, channel, height * 2, width * 2)).to(device)     
    feats = [attention_input1, attention_input2]
    module = asf_attention_model(channel).to(device)

    outputs = module(feats)    
    print(GREEN + f'attention_input1.size:{attention_input1.size()} attention_input2.size:{attention_input2.size()} outputs.size:{outputs.size()}' + RESET)     
   
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     args=[feats],   
                                     output_as_string=True, 
                                     output_precision=4,  
                                     print_detailed=True)  
    print(RESET)  
