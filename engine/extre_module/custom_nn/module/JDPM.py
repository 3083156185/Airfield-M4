''' 
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/ECCV2024-JDPM.png   
论文链接：https://arxiv.org/pdf/2409.01686  
'''   
  
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')
   
import warnings     
warnings.filterwarnings('ignore')    
from calflops import calculate_flops   

import torch   
import torch.nn as nn   
   
from engine.extre_module.ultralytics_nn.conv import Conv    
  
class JDPM(nn.Module): # JDPM (Joint Domain Perception Module)     
    def __init__(self, inc, channels):   
        super(JDPM, self).__init__()
  
        self.conv1 = nn.Sequential(     
            # nn.Conv2d(channels, channels, 1), nn.BatchNorm2d(channels), nn.ReLU(True)
            Conv(inc, channels)
        )

        self.Dconv3 = nn.Sequential(  
            # nn.Conv2d(channels, channels, 1), nn.BatchNorm2d(channels), 
            # nn.Conv2d(channels, channels, 3, padding=3,dilation=3), nn.BatchNorm2d(channels), nn.ReLU(True)     
            Conv(channels, channels, act=False),
            Conv(channels, channels, k=3, d=3)
        )    

        self.Dconv5 = nn.Sequential(
            # nn.Conv2d(channels, channels, 1), nn.BatchNorm2d(channels),   
            # nn.Conv2d(channels, channels, 3, padding=5,dilation=5), nn.BatchNorm2d(channels), nn.ReLU(True)   
            Conv(channels, channels, act=False),
            Conv(channels, channels, k=3, d=5)  
        )
        self.Dconv7 = nn.Sequential(
            # nn.Conv2d(channels, channels, 1), nn.BatchNorm2d(channels),
            # nn.Conv2d(channels, channels, 3, padding=7,dilation=7), nn.BatchNorm2d(channels), nn.ReLU(True)
            Conv(channels, channels, act=False),  
            Conv(channels, channels, k=3, d=7)  
        )     
        self.Dconv9 = nn.Sequential(  
            # nn.Conv2d(channels, channels, 1), nn.BatchNorm2d(channels),   
            # nn.Conv2d(channels, channels, 3, padding=9,dilation=9), nn.BatchNorm2d(channels),nn.ReLU(True)     
            Conv(channels, channels, act=False),
            Conv(channels, channels, k=3, d=9)
        )

        self.reduce = nn.Sequential(  
            # nn.Conv2d(channels * 5, channels, 1), nn.BatchNorm2d(channels),nn.ReLU(True)    
            Conv(channels * 5, channels)   
        )

        self.weight = nn.Sequential(
            nn.Conv2d(channels, channels // 16, 1, bias=True),  
            nn.BatchNorm2d(channels // 16),
            nn.ReLU(True),
            nn.Conv2d(channels // 16, channels, 1, bias=True),
            nn.Sigmoid())    

        self.norm = nn.BatchNorm2d(channels)    
        self.relu = nn.ReLU(True)     

    def forward(self, F1):   

       F1_input  = self.conv1(F1) 

       F1_3_s = self.Dconv3(F1_input) 
       F1_3_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(F1_3_s.float()).real)*torch.fft.fft2(F1_3_s.float())))))
       F1_3 = torch.add(F1_3_s,F1_3_f)
  
       F1_5_s = self.Dconv5(F1_input + F1_3)  
       F1_5_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(F1_5_s.float()).real)*torch.fft.fft2(F1_5_s.float())))))  
       F1_5 = torch.add(F1_5_s, F1_5_f)  
  
       F1_7_s = self.Dconv7(F1_input + F1_5)
       F1_7_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(F1_7_s.float()).real)*torch.fft.fft2(F1_7_s.float())))))    
       F1_7 = torch.add(F1_7_s, F1_7_f)

       F1_9_s = self.Dconv9(F1_input + F1_7)
       F1_9_f = self.relu(self.norm(torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(F1_9_s.float()).real)*torch.fft.fft2(F1_9_s.float())))))
       F1_9 = torch.add(F1_9_s, F1_9_f)    

       return self.reduce(torch.cat((F1_3,F1_5,F1_7,F1_9,F1_input),1)) + F1_input
     
if __name__ == '__main__': 
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32    
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)   
   
    module = JDPM(in_channel, out_channel).to(device)

    outputs = module(inputs) 
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)   

    print(ORANGE)   
    flops, macs, _ = calculate_flops(model=module,     
                                     input_shape=(batch_size, in_channel, height, width),  
                                     output_as_string=True, 
                                     output_precision=4,
                                     print_detailed=True)    
    print(RESET)     
