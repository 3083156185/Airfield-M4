'''  
本文件由BiliBili：魔傀面具整理  
engine/extre_module/module_images/AAAI2026-SPJFB.png
engine/extre_module/module_images/AAAI2026-SPJFB.md  
论文链接：https://arxiv.org/pdf/2508.04041
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')
 
import warnings    
warnings.filterwarnings('ignore')
from calflops import calculate_flops     
   
import torch
import torch.nn as nn
import torch.nn.functional as F
   
from engine.extre_module.ultralytics_nn.conv import Conv     

# -----------------------------
# Basic utilities (self-contained) 
# -----------------------------
def compute_gradient(image):
    sobel_x = torch.tensor(     
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=image.device
    ).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=image.device     
    ).unsqueeze(0).unsqueeze(0)
  
    sobel_x = sobel_x.repeat(image.shape[1], 1, 1, 1)
    sobel_y = sobel_y.repeat(image.shape[1], 1, 1, 1)
  
    grad_x = F.conv2d(image, sobel_x, padding=1, groups=image.shape[1])
    grad_y = F.conv2d(image, sobel_y, padding=1, groups=image.shape[1])
    return torch.sqrt(grad_x ** 2 + grad_y ** 2)


def dwt_init(x):     
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2     
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]     
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_ll = x1 + x2 + x3 + x4
    x_hl = -x1 - x2 + x3 + x4  
    x_lh = -x1 + x2 - x3 + x4
    x_hh = x1 - x2 - x3 + x4     
    return x_ll, x_hl, x_lh, x_hh 

   
def iwt_init(x):   
    r = 2   
    in_batch, in_channel, in_height, in_width = x.size()    
    out_batch = in_batch
    out_channel = int(in_channel / (r ** 2))
    out_height = r * in_height
    out_width = r * in_width   
   
    x1 = x[:, :out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2   
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2   
    
    h = torch.zeros(
        [out_batch, out_channel, out_height, out_width],     
        dtype=x.dtype,
        device=x.device,
    )    
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4 
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4     
    return h 
 
     
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False     

    def forward(self, x):
        return dwt_init(x)

    
class IWT(nn.Module):
    def __init__(self):   
        super(IWT, self).__init__()    
        self.requires_grad = False
 
    def forward(self, x): 
        return iwt_init(x)
 

class DepthConv(nn.Module):
    def __init__(self, in_ch, out_ch):     
        super(DepthConv, self).__init__()    
        self.depth_conv = nn.Conv2d(    
            in_channels=in_ch,  
            out_channels=in_ch, 
            kernel_size=3,
            stride=1,
            padding=1, 
            groups=in_ch, 
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,     
            padding=0,  
            groups=1,   
        )
  
    def forward(self, x):
        return self.point_conv(self.depth_conv(x))
    

# -----------------------------
# FFC blocks (inlined from ffc.py)
# -----------------------------
class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, fft_norm='ortho'):
        super(FourierUnit, self).__init__()     
        self.groups = groups
        self.fft_norm = fft_norm    
        self.conv_layer = nn.Conv2d(    
            in_channels=in_channels * 2, 
            out_channels=out_channels * 2,     
            kernel_size=1, 
            stride=1,
            padding=0,    
            groups=self.groups,
            bias=False,  
        )
        self.bn = nn.BatchNorm2d(out_channels * 2) 
        self.relu = nn.ReLU(inplace=True) 
  
    def forward(self, x):
        x = x.to(torch.float32)    
        batch = x.shape[0]   
        ffted = torch.fft.rfftn(x, dim=(-2, -1), norm=self.fft_norm)     
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)    
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()    
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])  
  
        ffted = self.conv_layer(ffted)    
        ffted = self.relu(self.bn(ffted.to(torch.float32)))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        return torch.fft.irfftn(ffted, s=x.shape[-2:], dim=(-2, -1), norm=self.fft_norm)


class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True):
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu  
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2) if stride == 2 else nn.Identity()   
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, groups=groups, bias=False), 
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups)     
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)  
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)
   
        if self.enable_lfu:     
            n, c, h, w = x.shape   
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()  
            xs = torch.cat(torch.split(xs, split_s, dim=-1), dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous() 
        else:
            xs = 0   

        return self.conv2(x + output + xs)     

 
class FFC(nn.Module):
    def __init__(    
        self, 
        in_channels,     
        out_channels,    
        kernel_size,  
        ratio_gin,
        ratio_gout,     
        stride=1,    
        padding=0,
        dilation=1,  
        groups=1,  
        bias=False, 
        enable_lfu=True,   
        padding_type='reflect',
    ):
        super(FFC, self).__init__()
        assert stride in (1, 2), "Stride should be 1 or 2."  

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)    
        out_cl = out_channels - out_cg
    
        self.ratio_gout = ratio_gout     
     
        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_type)    

        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d  
        self.convl2g = module(in_cl, out_cg, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_type)     
    
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size, stride, padding, dilation, groups, bias, padding_mode=padding_type)
   
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu)  

    def forward(self, x):
        x_l, x_g = x if isinstance(x, tuple) else (x, 0)  
        out_xl, out_xg = 0, 0    
 
        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:  
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)     

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):  
    def __init__(
        self,
        in_channels, 
        out_channels,
        kernel_size, 
        ratio_gin,
        ratio_gout,
        stride=1,
        padding=0,
        dilation=1,
        groups=1, 
        bias=False,    
        norm_layer=nn.BatchNorm2d,
        activation_layer=nn.Identity,
        padding_type='reflect',   
        enable_lfu=True,
    ):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(
            in_channels,
            out_channels,   
            kernel_size,
            ratio_gin,
            ratio_gout,     
            stride,
            padding,
            dilation, 
            groups,   
            bias,
            enable_lfu,   
            padding_type=padding_type,
        )
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)
 
        lact = nn.Identity if ratio_gout == 1 else activation_layer 
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)     

    def forward(self, x):  
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l.to(torch.float32)))    
        x_g = self.act_g(self.bn_g(x_g.to(torch.float32)))   
        return x_l, x_g


class FFCResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, activation_layer=nn.ReLU):
        super(FFCResnetBlock, self).__init__()
        self.ffc1 = FFC_BN_ACT(     
            dim, 
            dim,
            3,
            0.75,
            0.75,
            stride=1,
            padding=1,   
            dilation=dilation,     
            groups=1,    
            bias=False,  
            norm_layer=nn.BatchNorm2d,
            activation_layer=activation_layer,
            enable_lfu=False,  
        )  
        self.ffc2 = FFC_BN_ACT( 
            dim,    
            dim,   
            3,  
            0.75, 
            0.75,  
            stride=1,
            padding=1,
            dilation=1,
            groups=1,  
            bias=False,
            norm_layer=nn.BatchNorm2d,   
            activation_layer=activation_layer, 
            enable_lfu=False,     
        )
     
    def forward(self, x): 
        output = x
        _, c, _, _ = output.shape     
        output = torch.split(output, [c - int(c * 0.75), int(c * 0.75)], dim=1)    
        x_l, x_g = self.ffc1(output)    
        output = self.ffc2((x_l, x_g))
        output = torch.cat(output, dim=1)
        return x + output     


# -----------------------------     
# Core SPJ frequency components   
# -----------------------------  
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()  
        self.fc = nn.Sequential(   
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),     
            nn.Conv2d(channels // reduction, channels, kernel_size=1),   
            nn.Sigmoid(),    
        )   
  
    def forward(self, x):     
        scale = self.fc(x)
        return x * scale + x
  
  
class ChannelAttentionFusion(nn.Module):   
    def __init__(self, nf):    
        super(ChannelAttentionFusion, self).__init__()    
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.fc = nn.Sequential(   
            nn.Conv2d(nf * 2, nf // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf // 4, nf * 2, 1, bias=False), 
            nn.Sigmoid(),  
        ) 
   
    def forward(self, fft_features, multi_features):   
        combined = torch.cat([fft_features, multi_features], dim=1)
        attn = self.fc(self.global_avg_pool(combined))
        fft_w, multi_w = torch.split(attn, fft_features.size(1), dim=1)
        return fft_w * fft_features + multi_w * multi_features  

     
class MultiConvBlock(nn.Module):    
    def __init__(self, dim, num_heads=4):
        super(MultiConvBlock, self).__init__()
        self.conv_reduction = nn.Conv2d(dim, dim // 4, kernel_size=1, stride=1, bias=True)    
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)
        self.local_convs = nn.ModuleList([  
            nn.Conv2d(
                dim // 4,
                dim // 4,
                kernel_size=(3 + i * 2),
                padding=(1 + i), 
                stride=1,    
                groups=dim // 4,
            )
            for i in range(num_heads)    
        ])  
        self.conv_fusion = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=True) 
        self.se_block = SEBlock(dim)

    def forward(self, x):     
        x_reduced = self.leakyrelu(self.conv_reduction(x)) 
        multi_scale = []    
        for conv in self.local_convs:
            x_scale = self.leakyrelu(conv(x_reduced))
            x_scale = x_scale * torch.sigmoid(x_reduced)
            multi_scale.append(x_scale)
        x_concat = torch.cat(multi_scale, dim=1)
        x_fused = self.se_block(self.conv_fusion(x_concat))
        return x + x_fused

  
class FrequencyFusion(nn.Module):    
    def __init__(self, channels):
        super(FrequencyFusion, self).__init__()    
        self.channel_attention = nn.Sequential(     
            nn.Conv2d(channels * 2, channels // 2, 1),     
            nn.ReLU(inplace=True),     
            nn.Conv2d(channels // 2, channels, 1), 
            nn.Sigmoid(),    
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(channels, channels, 1), 
        )    
        self.gate = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid())    

    def forward(self, d, g):
        cat_feature = torch.cat([d, g], dim=1)
        channel_weight = self.channel_attention(cat_feature)  
        weighted_d = d * channel_weight   
        fused_feature = self.fusion_conv(cat_feature)    
        gate_weight = self.gate(fused_feature)
        return weighted_d + g * gate_weight    
   
     
class FFTProcess(nn.Module):
    def __init__(self, nf):  
        super(FFTProcess, self).__init__() 
        self.freq_preprocess = nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0)   
        self.process_amp = self._make_process_block(nf)
        self.process_pha = self._make_process_block(nf)  
        self.process_fr = self._make_process_block(nf) 
        self.process_sigmoid_amp = FrequencyFusion(nf)   
        self.process_amp_post = self._make_process_block(nf)
        self.process_pha_post = self._make_process_block_pha(nf)

    @staticmethod
    def _make_process_block(nf):
        return nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0),  
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0),
        )

    @staticmethod
    def _make_process_block_pha(nf):     
        return nn.Sequential(   
            nn.Conv2d(nf * 2, nf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0),    
        )
   
    def forward(self, x, fr, y_map):
        _, _, h, w = x.shape     
        x_freq = torch.fft.rfft2(self.freq_preprocess(x), norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)  
        mag = self.process_amp(mag)  
        pha = self.process_pha(pha)  
   
        fr = self.process_fr(fr)
        pha = torch.cat([pha, fr], dim=1)
    
        mag = self.process_sigmoid_amp(mag, y_map)
        pha = self.process_pha_post(pha)    
        mag = self.process_amp_post(mag)     

        real = mag * torch.cos(pha)     
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(h, w), norm='backward')
        return x_out + x, fr, y_map

    
class LowFrequencyProcessing(nn.Module):
    def __init__(self, nf=64, num_blocks=6, input_channels=3, output_channels=None):     
        super(LowFrequencyProcessing, self).__init__()
        if output_channels is None:   
            output_channels = input_channels 
   
        self.initial_conv = nn.Conv2d(input_channels, nf, kernel_size=1, stride=1, padding=0)
        self.fft_blocks = nn.ModuleList([FFTProcess(nf) for _ in range(6)])  
        self.ffc_blocks = nn.ModuleList([FFCResnetBlock(nf) for _ in range(num_blocks)])  
        self.multi_blocks = nn.ModuleList([MultiConvBlock(nf) for _ in range(num_blocks)])
        self.fusion_block = ChannelAttentionFusion(nf)

        self.concat_layers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(nf * 2, nf, kernel_size=1, stride=1, padding=0), SEBlock(nf))
            for _ in range(3)  
        ]) 

        self.upconv_last = nn.Conv2d(nf, output_channels, 3, 1, 1, bias=True)

    def forward(self, x, fr, y_map): 
        xori = x
        x0 = self.initial_conv(x)

        x, fr, y_map = self.fft_blocks[0](x0, fr, y_map)     
        x1, fr, y_map = self.fft_blocks[1](x, fr, y_map)     
        x2, fr, y_map = self.fft_blocks[2](x1, fr, y_map) 
   
        x3_input = self.concat_layers[0](torch.cat((x2, x1), dim=1))
        x3, fr, y_map = self.fft_blocks[3](x3_input, fr, y_map)    

        x4_input = self.concat_layers[1](torch.cat((x3, x), dim=1))     
        x4, fr, y_map = self.fft_blocks[4](x4_input, fr, y_map)    

        x5_input = self.concat_layers[1](torch.cat((x4, x0), dim=1))    
        x5, fr, y_map = self.fft_blocks[5](x5_input, fr, y_map) 

        fft_features = x5 
        multi_features = x5
        for ffc_block, multi_block in zip(self.ffc_blocks, self.multi_blocks):  
            fft_features = ffc_block(fft_features)
            multi_features = multi_block(multi_features)     
   
        fused = self.fusion_block(fft_features, multi_features)   
        return self.upconv_last(fused) + xori     

  
class DownFRG(nn.Module):
    def __init__(self):
        super().__init__()
        self.dwt = DWT()
    
    def forward(self, x):
        x_ll, x_hl, x_lh, x_hh = self.dwt(x)
        return x_ll, (x_hl, x_lh, x_hh)
    

class UpFRG(nn.Module):     
    def __init__(self):
        super().__init__()  
        self.iwt = IWT()   
     
    def forward(self, x_ll, x_h):    
        x_hl, x_lh, x_hh = x_h  
        return self.iwt(torch.cat([x_ll, x_hl, x_lh, x_hh], dim=1))     
 

class GammaNet(nn.Module):  
    def __init__(self, input_channels=3, feature_channels=16):
        super(GammaNet, self).__init__()
        self.features = nn.Sequential(   
            nn.Conv2d(input_channels, feature_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.LeakyReLU(0.2, inplace=True),   
            nn.Conv2d(feature_channels, feature_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),  
        ) 
        self.gamma_pred = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_channels * 2, 1),     
            nn.Sigmoid(), 
        )
        self._initialize_weights()
  
    def _initialize_weights(self):    
        for m in self.modules():   
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)   
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)   
                nn.init.constant_(m.bias, 0)
 
    def forward(self, x): 
        features = self.features(x)
        gamma = self.gamma_pred(features)
        return gamma.view(-1, 1, 1, 1)


class DFGFLow(nn.Module): 
    def __init__(self, nf=16, numblocks=6, in_channels=3):
        super(DFGFLow, self).__init__()
        self.s_nf = nf 
        self.in_channels = in_channels   
        self.processblock = LowFrequencyProcessing(   
            nf=self.s_nf,
            num_blocks=numblocks,     
            input_channels=self.in_channels,  
            output_channels=self.in_channels,    
        )   
        self.conv_first_fr = nn.Conv2d(self.in_channels, self.s_nf, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_first_map = nn.Conv2d(self.in_channels, self.s_nf, kernel_size=1, stride=1, padding=0, bias=True)   
  
    def forward(self, x, gradient, x_light):     
        x_light_fre = torch.fft.rfft2(x_light, norm='backward')    
        x_light_mag = torch.abs(x_light_fre)
        x_light_pha = torch.angle(x_light_fre)     
        x_light_mag = self.conv_first_fr(x_light_mag)
        x_light_pha = self.conv_first_map(x_light_pha) 

        x_amplitude = self.processblock(x, x_light_pha, x_light_mag)     
        return x_amplitude, x, x_light_pha  

    
class DFGFHigh(nn.Module):  
    def __init__(self, in_nf=3, out_if=3, nf=16):
        super(DFGFHigh, self).__init__() 

        self.conv1_first = nn.Conv2d(in_nf, nf, kernel_size=1, stride=1, padding=0, bias=True)   
        self.conv2_first = nn.Conv2d(in_nf, nf, kernel_size=1, stride=1, padding=0, bias=True)    

        self.conv1 = nn.Sequential(nn.Conv2d(nf, nf, 3, 1, 1, groups=4), nn.LeakyReLU(0.1)) 
        self.conv2 = nn.Sequential(nn.Conv2d(nf, nf, 3, 1, 1, groups=4), nn.LeakyReLU(0.1))
        self.conv3 = nn.Sequential(nn.Conv2d(nf, nf, 3, 1, 1, groups=4), nn.LeakyReLU(0.1)) 

        self.convf1_first = nn.Sequential(nn.Conv2d(in_nf, nf, 1, 1, 0, bias=True), DepthConv(in_ch=nf, out_ch=nf))
        self.convf2_first = nn.Sequential(nn.Conv2d(in_nf, nf, 1, 1, 0, bias=True), DepthConv(in_ch=nf, out_ch=nf))
        self.convf3_first = nn.Sequential(nn.Conv2d(in_nf, nf, 1, 1, 0, bias=True), DepthConv(in_ch=nf, out_ch=nf))
  
        self.convf1 = nn.Sequential(DepthConv(in_ch=nf, out_ch=nf), nn.LeakyReLU(0.1), nn.Conv2d(nf, nf, 3, 1, 1, groups=4))
        self.convf2 = nn.Sequential(DepthConv(in_ch=nf, out_ch=nf), nn.LeakyReLU(0.1), nn.Conv2d(nf, nf, 3, 1, 1, groups=4))   
  
        self.conv1_out = nn.Conv2d(nf, out_if, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2_out = nn.Conv2d(nf, out_if, kernel_size=1, stride=1, padding=0, bias=True)    
        self.conv3_out = nn.Conv2d(nf, out_if, kernel_size=1, stride=1, padding=0, bias=True)
    
        self.sigm = nn.Sigmoid()
   
    def forward(self, f, fg):    
        f1, f2, f3 = f
        f_add = f1 + f2 + f3
        fg = self.conv1_first(fg)
        f_add = self.conv2_first(f_add)     

        attention1 = self.sigm(self.conv1(fg))  
        attention2 = self.sigm(self.conv2(f_add))  
        attention = fg * attention1 + f_add * attention2  
        attention = self.sigm(self.conv3(attention))  
    
        # Keep the same behavior as the source implementation. 
        f1 = self.convf1_first(f1)  
        f1 = f1 + attention * f1
        f1 = self.conv1_out(self.convf1(f1)) 
 
        f2 = self.convf1_first(f2)  
        f2 = f2 + attention * f2  
        f2 = self.conv2_out(self.convf2(f2))  
 
        f3 = self.convf1_first(f3)   
        f3 = f3 + attention * f3
        f3 = self.conv3_out(self.convf1(f3))
 
        return (f1, f2, f3)

   
class SPJFrequencyBlock(nn.Module):    
    """Plug-and-play dual-frequency block for arbitrary channel feature maps."""    

    def __init__(
        self,
        in_channels,   
        out_channels,
        nf=32,
        num_low_blocks=2,
        pad_mode="reflect",
        safe_gamma_pow=True,
        gamma_eps=1e-6,
    ):
        super(SPJFrequencyBlock, self).__init__()   
        if nf % 4 != 0:
            raise ValueError(f"`nf` must be divisible by 4 for grouped convs in DFGFHigh, got {nf}.")
    
        self.in_channels = in_channels
        self.pad_mode = pad_mode     
        self.safe_gamma_pow = safe_gamma_pow
        self.gamma_eps = gamma_eps
  
        self.down_group = DownFRG()    
        self.up_group = UpFRG()
    
        self.gammanet = GammaNet(input_channels=in_channels, feature_channels=nf)
        self.low_process = DFGFLow(nf=nf, numblocks=num_low_blocks, in_channels=in_channels)  
        self.high_process = DFGFHigh(in_nf=in_channels, out_if=in_channels, nf=nf)

        self.final_conv = Conv(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()    

    def _pad_to_even(self, x):
        _, _, h, w = x.shape
        pad_h = (2 - h % 2) % 2
        pad_w = (2 - w % 2) % 2
        if pad_h != 0 or pad_w != 0:   
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=self.pad_mode)   
        return x, h, w 
   
    def _apply_gamma(self, x_ll, gamma):
        if not self.safe_gamma_pow:
            return torch.pow(x_ll, gamma)     
        x_abs = x_ll.abs().clamp_min(self.gamma_eps)
        return torch.sign(x_ll) * torch.pow(x_abs, gamma)   
    
    def forward(self, x):    
        x, orig_h, orig_w = self._pad_to_even(x)  
        x_ll, x_h = self.down_group(x)     
 
        gamma = self.gammanet(x_ll)    
        x_ll_gamma = self._apply_gamma(x_ll, gamma)   
        gradient = compute_gradient(x_ll_gamma)     

        x_ll_out, _, _ = self.low_process(x_ll, gradient, x_ll_gamma)
        x_h_out = self.high_process(x_h, gradient)    

        out = self.up_group(x_ll_out, x_h_out)
        out = out[:, :, :orig_h, :orig_w]
        return self.final_conv(out) 
   
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = SPJFrequencyBlock(in_channel, out_channel).to(device)

    outputs = module(inputs)  
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)     
    
    print(ORANGE)    
    flops, macs, _ = calculate_flops(model=module,     
                                     input_shape=(batch_size, in_channel, height, width),     
                                     output_as_string=True,     
                                     output_precision=4,
                                     print_detailed=True)    
    print(RESET)