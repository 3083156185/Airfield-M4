'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/TGRS2025-DHPF.png    
engine/extre_module/module_images/TGRS2025-DHPF.md   
论文链接：https://ieeexplore.ieee.org/document/11017756
'''
   
import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops
 
import torch, math 
import torch.nn as nn   
import torch.nn.functional as F  

class DHPF(nn.Module):
    def __init__(self, chn, energy=0.4, reduction=None):
        super(DHPF, self).__init__()
        self.energy = energy
    
    def _determine_cutoff_frequency(self, f_transform, target_ratio):
        total_energy = self._calculate_total_energy(f_transform)
        target_low_freq_energy = total_energy * target_ratio

        for cutoff_frequency in range(1, min(f_transform.shape[0], f_transform.shape[1]) // 2):
            low_freq_energy = self._calculate_low_freq_energy(f_transform, cutoff_frequency)
            if low_freq_energy >= target_low_freq_energy:     
                return cutoff_frequency
        return 5     
    
    def _calculate_total_energy(self, f_transform):  
        magnitude_spectrum = torch.abs(f_transform)   
        total_energy = torch.sum(magnitude_spectrum ** 2)  
        return total_energy  
  
    def _calculate_low_freq_energy(self, f_transform, cutoff_frequency):
        magnitude_spectrum = torch.abs(f_transform)  
        height, width = magnitude_spectrum.shape
   
        low_freq_energy = torch.sum(magnitude_spectrum[
            height // 2 - cutoff_frequency:height // 2 + cutoff_frequency,
            width // 2 - cutoff_frequency:width // 2 + cutoff_frequency
        ] ** 2)   
    
        return low_freq_energy

    def forward(self, x):
        B, C, H, W = x.shape   
        f = torch.fft.fft2(x)     
        fshift = torch.fft.fftshift(f)     
        crow, ccol = H // 2, W // 2     
        for i in range(B):
            cutoff_frequency = self._determine_cutoff_frequency(fshift[i, 0], self.energy) 
            fshift[i, :, crow - cutoff_frequency:crow + cutoff_frequency, ccol - cutoff_frequency:ccol + cutoff_frequency] = 0
        ishift = torch.fft.ifftshift(fshift)   
        ideal_high_pass = torch.abs(torch.fft.ifft2(ishift))     
        return ideal_high_pass 
    
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"     
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel, height, width = 1, 16, 32, 32     
    inputs = torch.randn((batch_size, channel, height, width)).to(device)

    module = DHPF(channel, reduction=32).to(device)    
 
    outputs = module(inputs)   
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
 
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width),   
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)
