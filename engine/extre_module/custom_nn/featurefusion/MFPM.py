'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/TGRS2025-MFPM.png   
engine/extre_module/module_images/TGRS2025-MFPM.md     
论文链接：https://ieeexplore.ieee.org/document/11232501  
'''     
 
import os, sys     
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')  

import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops    
   
import math
import torch     
import torch.nn as nn
import torch.nn.functional as F
   
from engine.extre_module.ultralytics_nn.conv import Conv    
  
class ExternalAttention(nn.Module):
    def __init__(self, in_planes, S=8):
        super().__init__() 

        self.mk = nn.Linear(in_planes, S, bias=False)     
        self.mv = nn.Linear(S, in_planes, bias=False)  
        self.softmax = nn.Softmax(dim=1)     
        self.init_weights()    
  
    def init_weights(self):
        for m in self.modules():  
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))   
            elif isinstance(m, nn.Conv1d):  
                n = m.kernel_size[0] * m.out_channels    
                m.weight.data.normal_(0, math.sqrt(2. / n))    
            elif isinstance(m, nn.BatchNorm2d):     
                m.weight.data.fill_(1)   
                if m.bias is not None:    
                    m.bias.data.zero_()  

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w
        queries = x.view(b, c, n)  # 即bs,n,d_model
        queries = queries.permute(0, 2, 1) 
        attn = self.mk(queries)  # bs,n,S
        attn = self.softmax(attn)  # bs,n,S  
        attn = attn / (1e-9 + torch.sum(attn, dim=2, keepdim=True))  # bs,n,S
        attn = self.mv(attn)  # bs,n,d_model
        attn = attn.permute(0, 2, 1)
        x_attn = attn.view(b, c, h, w)
        x = x + x_attn   
        x = F.relu(x)     
        return x  
  
def get_freq_indices(method):  
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32', 
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method: 
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]  
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,  
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq] 
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:   
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5, 
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3, 
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:    
        raise NotImplementedError     
    return mapper_x, mapper_y

class MultiFrequencyChannelAttention(nn.Module):
    def __init__(self,  
                 in_channels,  
                 dct_h, dct_w,
                 frequency_branches=16,
                 frequency_selection='top',     
                 reduction=16): 
        super(MultiFrequencyChannelAttention, self).__init__()

        assert frequency_branches in [1, 2, 4, 8, 16, 32]
        frequency_selection = frequency_selection + str(frequency_branches)

        self.num_freq = frequency_branches   
        self.dct_h = dct_h
        self.dct_w = dct_w
     
        mapper_x, mapper_y = get_freq_indices(frequency_selection)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]     
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]

        assert len(mapper_x) == len(mapper_y)

        for freq_idx in range(frequency_branches):
            self.register_buffer('dct_weight_{}'.format(freq_idx),   
                                 self.get_dct_filter(dct_h, dct_w, mapper_x[freq_idx], mapper_y[freq_idx], in_channels))   
   
        self.fc = nn.Sequential(    
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=False))     
    
        self.average_channel_pooling = nn.AdaptiveAvgPool2d(1)    
        self.max_channel_pooling = nn.AdaptiveMaxPool2d(1)     

    def forward(self, x):
        batch_size, C, H, W = x.shape  
    
        x_pooled = x    
     
        if H != self.dct_h or W != self.dct_w:    
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))  

        multi_spectral_feature_avg, multi_spectral_feature_max, multi_spectral_feature_min = 0, 0, 0    
        for name, params in self.state_dict().items():
            if 'dct_weight' in name:    
                x_pooled_spectral = x_pooled * params 
                multi_spectral_feature_avg += self.average_channel_pooling(x_pooled_spectral)
                multi_spectral_feature_max += self.max_channel_pooling(x_pooled_spectral)    
                multi_spectral_feature_min += -self.max_channel_pooling(-x_pooled_spectral) 
        multi_spectral_feature_avg = multi_spectral_feature_avg / self.num_freq     
        multi_spectral_feature_max = multi_spectral_feature_max / self.num_freq  
        multi_spectral_feature_min = multi_spectral_feature_min / self.num_freq

        multi_spectral_avg_map = self.fc(multi_spectral_feature_avg).view(batch_size, C, 1, 1)     
        multi_spectral_max_map = self.fc(multi_spectral_feature_max).view(batch_size, C, 1, 1)
        multi_spectral_min_map = self.fc(multi_spectral_feature_min).view(batch_size, C, 1, 1)    
   
        multi_spectral_attention_map = F.sigmoid(   
            multi_spectral_avg_map + multi_spectral_max_map + multi_spectral_min_map) 

        return x * multi_spectral_attention_map.expand_as(x)     

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, in_channels):  
        dct_filter = torch.zeros(in_channels, tile_size_x, tile_size_y)

        for t_x in range(tile_size_x): 
            for t_y in range(tile_size_y):     
                dct_filter[:, t_x, t_y] = self.build_filter(t_x, mapper_x, tile_size_x) * self.build_filter(t_y,
                                                                                                            mapper_y,
                                                                                                            tile_size_y)

        return dct_filter
    
    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0: 
            return result     
        else:
            return result * math.sqrt(2)

 
class EA_MF(nn.Module):
    def __init__(self, out_channel, hw, frequency_branches=16, frequency_selection='top'):
        super(EA_MF, self).__init__()   
        self.frequency_branches = frequency_branches 

        self.relu = nn.ReLU(True)  
     
        self.ea = nn.Sequential(*[
            Conv(out_channel, out_channel, k=3, act=nn.ReLU),
            ExternalAttention(out_channel)     
        ])

        self.multi_frequency_branches = MultiFrequencyChannelAttention(out_channel, hw[0],
                                                                       hw[1], frequency_branches,    
                                                                       frequency_selection)
        self.multi_frequency_branches_conv = Conv(out_channel, out_channel, k=3, act=nn.ReLU) 
     
    def forward(self, x):
        x = self.multi_frequency_branches_conv(self.multi_frequency_branches(self.ea(x)))    
    
        return x     

class MFPM(nn.Module):   
    def __init__(self, in_channel, out_channel, hw=[20, 20]):
        super(MFPM, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.ea_mf = EA_MF(out_channel, hw) 
 
        self.conv1x1 = nn.ModuleList([])
        for i in in_channel:
            if i != out_channel:    
                self.conv1x1.append(Conv(i, out_channel, 1))
            else: 
                self.conv1x1.append(nn.Identity())

    def forward(self, inputs):
        g, x = inputs

        g = self.conv1x1[0](g)
        x = self.conv1x1[1](x)

        fusion = g + x    
        fusion = self.ea_mf(fusion)
        scale = torch.sigmoid(fusion)
        out = self.relu(x * scale)    
     
        return out    
 
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    batch_size, channel_1, channel_2, height, width = 1, 32, 16, 32, 32
    ouc_channel = 32
    inputs_1 = torch.randn((batch_size, channel_1, height, width)).to(device)
    inputs_2 = torch.randn((batch_size, channel_2, height, width)).to(device)

    module = MFPM([channel_1, channel_2], ouc_channel, hw=(height, width)).to(device)   
    
    outputs = module([inputs_1, inputs_2])
    print(GREEN + f'inputs1.size:{inputs_1.size()} inputs2.size:{inputs_2.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     args=[[inputs_1, inputs_2]],
                                     output_as_string=True, 
                                     output_precision=4,     
                                     print_detailed=True) 
    print(RESET)   
