'''
本文件由BiliBili：魔傀面具整理   
engine/extre_module/module_images/自研模块-LowFrequencyFeatureFusion.png    
engine/extre_module/module_images/自研模块-LowFrequencyFeatureFusion.md
'''    
 
import os, sys   
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')    
    
import warnings
warnings.filterwarnings('ignore')     
from calflops import calculate_flops  
    
import math    
import torch    
import torch.nn as nn

from engine.extre_module.ultralytics_nn.conv import Conv

class LowFrequencyFeatureFusion(nn.Module):
    """Symmetric same-scale feature fusion with shared low-frequency enhancement. 

    The module treats all input branches symmetrically:
    - a shared 1x1 projection aligns channel statistics 
    - only masked low-frequency components are enhanced
    - high-frequency residuals are preserved    
    - branch aggregation is permutation-invariant
    """
    
    def __init__(
        self,     
        in_channels,    
        out_channels,
        lowfreq_ratio: float = 0.25,
        reduction: int = 4,     
        alpha: float = 1.0,
        bias: bool = True,   
    ) -> None:   
        super().__init__() 

        if not 0.0 < lowfreq_ratio <= 1.0: 
            raise ValueError(f'lowfreq_ratio must be in (0, 1], got {lowfreq_ratio}.')   
  
        self.conv_adjust = nn.ModuleList([Conv(i, out_channels) for i in in_channels]) 

        hidden_channels = max(out_channels // reduction, 1)

        self.channels = out_channels
        self.lowfreq_ratio = lowfreq_ratio  
        self.alpha = alpha   
     
        self.pre_project = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=bias)
        self.gate = nn.Sequential(
            nn.Conv2d(out_channels, hidden_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.out_project = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, groups=out_channels, stride=1, padding=1, bias=bias)  
   
    def _normalize_inputs(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise TypeError('inputs must be provided as a list or tuple of 4D tensors.')

        features = list(inputs)   
        if len(features) < 2: 
            raise ValueError(f'Expected at least 2 inputs, got {len(features)}.')  

        for idx, feature in enumerate(features):
            features[idx] = self.conv_adjust[idx](feature)  
        return features
   
    def _build_lowfreq_mask(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        fy = torch.fft.fftfreq(height, device=device).unsqueeze(1)  
        fx = torch.fft.rfftfreq(width, device=device).unsqueeze(0)

        max_radius = math.sqrt(0.5 ** 2 + 0.5 ** 2)
        radius = torch.sqrt(fy.square() + fx.square()) / max_radius
        mask = (radius <= self.lowfreq_ratio).to(dtype=dtype)     
        return mask.unsqueeze(0).unsqueeze(0) 

    def forward(self, inputs):    
        features = self._normalize_inputs(inputs)  

        spatial_size = features[0].shape[-2:]   
        original_dtype = features[0].dtype 

        projected = [self.pre_project(feature) for feature in features] 
        spectra = [torch.fft.rfft2(feature.float(), dim=(-2, -1)) for feature in projected]

        lowfreq_mask = self._build_lowfreq_mask(    
            height=spatial_size[0],
            width=spatial_size[1],
            device=spectra[0].device, 
            dtype=spectra[0].real.dtype,  
        )

        low_components = [spectrum * lowfreq_mask for spectrum in spectra] 
        high_components = [spectrum - low for spectrum, low in zip(spectra, low_components)]

        shared_low_energy = torch.stack( 
            [torch.log1p(torch.abs(low_component)) for low_component in low_components],
            dim=1,
        ).mean(dim=1)     
        shared_low_energy = shared_low_energy.mean(dim=(-2, -1), keepdim=True)
        gate = self.gate(shared_low_energy).to(dtype=spectra[0].real.dtype)  
 
        enhanced_low = [low_component * (1.0 + self.alpha * gate) for low_component in low_components]
 
        reconstructed = [
            torch.fft.irfft2(low_component + high_component, s=spatial_size, dim=(-2, -1)).to(dtype=original_dtype)
            for low_component, high_component in zip(enhanced_low, high_components)
        ] 
    
        stacked = torch.stack(reconstructed, dim=1)  
        mean_feature = stacked.mean(dim=1)    
        var_feature = stacked.var(dim=1, unbiased=False)

        return self.out_project(torch.cat([mean_feature, var_feature], dim=1))
 
   
if __name__ == '__main__':  
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"   
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   
    batch_size, channel_1, channel_2, height, width = 1, 32, 16, 32, 32
    ouc_channel = 32
    inputs_1 = torch.randn((batch_size, channel_1, height, width)).to(device)
    inputs_2 = torch.randn((batch_size, channel_2, height, width)).to(device)    

    module = LowFrequencyFeatureFusion([channel_1, channel_2], ouc_channel).to(device)

    outputs = module([inputs_1, inputs_2])  
    print(GREEN + f'inputs1.size:{inputs_1.size()} inputs2.size:{inputs_2.size()} outputs.size:{outputs.size()}' + RESET)  
 
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module, 
                                     args=[[inputs_1, inputs_2]],
                                     output_as_string=True,   
                                     output_precision=4,
                                     print_detailed=True)   
    print(RESET)