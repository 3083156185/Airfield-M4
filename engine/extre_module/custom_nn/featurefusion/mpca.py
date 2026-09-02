'''     
本文件由BiliBili：魔傀面具整理     
engine/extre_module/module_images/BIBM2024-MultiScalePCA.png  
论文链接：https://arxiv.org/pdf/2406.07952
'''     
  
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')  
    
import warnings   
warnings.filterwarnings('ignore')   
from calflops import calculate_flops
   
import torch, math
import torch.nn as nn   

from engine.extre_module.ultralytics_nn.conv import Conv   

class MultiScalePCA(nn.Module):
    def __init__(self, input_channel, output_channel, gamma=2, bias=1):
        super(MultiScalePCA, self).__init__() 
        input_channel1, input_channel2 = input_channel   
        self.input_channel1 = input_channel1    
        self.input_channel2 = input_channel2   

        self.avg1 = nn.AdaptiveAvgPool2d(1) 
        self.avg2 = nn.AdaptiveAvgPool2d(1)     
   
        kernel_size1 = int(abs((math.log(input_channel1, 2) + bias) / gamma))
        kernel_size1 = kernel_size1 if kernel_size1 % 2 else kernel_size1 + 1  
    
        kernel_size2 = int(abs((math.log(input_channel2, 2) + bias) / gamma)) 
        kernel_size2 = kernel_size2 if kernel_size2 % 2 else kernel_size2 + 1 
   
        kernel_size3 = int(abs((math.log(input_channel1 + input_channel2, 2) + bias) / gamma))
        kernel_size3 = kernel_size3 if kernel_size3 % 2 else kernel_size3 + 1  
  
        self.conv1 = nn.Conv1d(1, 1, kernel_size=kernel_size1, padding=(kernel_size1 - 1) // 2, bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=kernel_size2, padding=(kernel_size2 - 1) // 2, bias=False)     
        self.conv3 = nn.Conv1d(1, 1, kernel_size=kernel_size3, padding=(kernel_size3 - 1) // 2, bias=False)
   
        self.sigmoid = nn.Sigmoid()    
        self.up = nn.ConvTranspose2d(in_channels=input_channel2, out_channels=input_channel1, kernel_size=3, stride=2, 
                                     padding=1, output_padding=1) 
  
        self.conv1x1 = Conv(input_channel1, output_channel) if input_channel1 != output_channel else nn.Identity()
  
    def forward(self, x):
        x1, x2 = x
        x1_ = self.avg1(x1)
        x2_ = self.avg2(x2)
  
        x1_ = self.conv1(x1_.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        x2_ = self.conv2(x2_.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
 
        x_middle = torch.cat((x1_, x2_), dim=1)    
        x_middle = self.conv3(x_middle.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        x_middle = self.sigmoid(x_middle)
 
        x_1, x_2 = torch.split(x_middle, [self.input_channel1, self.input_channel2], dim=1)   

        x1_out = x1 * x_1     
        x2_out = x2 * x_2   
     
        x2_out = self.up(x2_out)   
 
        result = x1_out + x2_out
        return self.conv1x1(result)    

class MultiScalePCA_Down(nn.Module):
    def __init__(self, input_channel, output_channel, gamma=2, bias=1):   
        super(MultiScalePCA_Down, self).__init__()  
        input_channel1, input_channel2 = input_channel
        self.input_channel1 = input_channel1
        self.input_channel2 = input_channel2

        self.avg1 = nn.AdaptiveAvgPool2d(1)  
        self.avg2 = nn.AdaptiveAvgPool2d(1)     

        kernel_size1 = int(abs((math.log(input_channel1, 2) + bias) / gamma))   
        kernel_size1 = kernel_size1 if kernel_size1 % 2 else kernel_size1 + 1
     
        kernel_size2 = int(abs((math.log(input_channel2, 2) + bias) / gamma))  
        kernel_size2 = kernel_size2 if kernel_size2 % 2 else kernel_size2 + 1   

        kernel_size3 = int(abs((math.log(input_channel1 + input_channel2, 2) + bias) / gamma))
        kernel_size3 = kernel_size3 if kernel_size3 % 2 else kernel_size3 + 1  

        self.conv1 = nn.Conv1d(1, 1, kernel_size=kernel_size1, padding=(kernel_size1 - 1) // 2, bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=kernel_size2, padding=(kernel_size2 - 1) // 2, bias=False)   
        self.conv3 = nn.Conv1d(1, 1, kernel_size=kernel_size3, padding=(kernel_size3 - 1) // 2, bias=False)    

        self.sigmoid = nn.Sigmoid()
        self.down = nn.Conv2d(in_channels=input_channel2, out_channels=input_channel1, kernel_size=3, stride=2, padding=1)     
    
        self.conv1x1 = Conv(input_channel1, output_channel) if input_channel1 != output_channel else nn.Identity()
  
    def forward(self, x):
        x1, x2 = x     
        x1_ = self.avg1(x1) 
        x2_ = self.avg2(x2)
     
        x1_ = self.conv1(x1_.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1) 
        x2_ = self.conv2(x2_.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)    

        x_middle = torch.cat((x1_, x2_), dim=1)
        x_middle = self.conv3(x_middle.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        x_middle = self.sigmoid(x_middle)   

        x_1, x_2 = torch.split(x_middle, [self.input_channel1, self.input_channel2], dim=1)

        x1_out = x1 * x_1 
        x2_out = x2 * x_2   

        x2_out = self.down(x2_out)

        result = x1_out + x2_out   
        return self.conv1x1(result)
     
if __name__ == '__main__':    
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  
    batch_size, channel_1, height_1, width_1 = 1, 32, 40, 40
    batch_size, channel_2, height_2, width_2 = 1, 64, 20, 20  
    ouc_channel = 64
    inputs_1 = torch.randn((batch_size, channel_1, height_1, width_1)).to(device)    
    inputs_2 = torch.randn((batch_size, channel_2, height_2, width_2)).to(device)  

    # 此模块有使用教程在VideoBaiduYun.txt内
    
    print(RED + '-'*20 + " MultiScalePCA " + '-'*20 + RESET)     

    module = MultiScalePCA([channel_1, channel_2], ouc_channel).to(device)  

    outputs = module([inputs_1, inputs_2])
    print(GREEN + f'inputs1.size:{inputs_1.size()} inputs2.size:{inputs_2.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)   
    flops, macs, _ = calculate_flops(model=module,    
                                     args=[[inputs_1, inputs_2]],   
                                     output_as_string=True,    
                                     output_precision=4,     
                                     print_detailed=True)
    print(RESET)

    print(RED + '-'*20 + " MultiScalePCA_Down " + '-'*20 + RESET)

    module = MultiScalePCA_Down([channel_2, channel_1], ouc_channel).to(device)    
 
    outputs = module([inputs_2, inputs_1])
    print(GREEN + f'inputs1.size:{inputs_1.size()} inputs2.size:{inputs_2.size()} outputs.size:{outputs.size()}' + RESET)   
   
    print(ORANGE)     
    flops, macs, _ = calculate_flops(model=module,
                                     args=[[inputs_2, inputs_1]],   
                                     output_as_string=True,
                                     output_precision=4,  
                                     print_detailed=True) 
    print(RESET)