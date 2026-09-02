'''    
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/CVPR2025-CASAB.png
engine/extre_module/module_images/CVPR2025-CASAB.md  
论文链接：https://openaccess.thecvf.com/content/CVPR2025/papers/Wazir_Rethinking_Decoder_Design_Improving_Biomarker_Segmentation_Using_Depth-to-Space_Restoration_and_CVPR_2025_paper.pdf
''' 
 
import os, sys 
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')   

import warnings
warnings.filterwarnings('ignore')  
from calflops import calculate_flops     

import torch     
import torch.nn as nn 
import torch.nn.functional as F    

from engine.extre_module.ultralytics_nn.conv import DSConv
 
class CASAB(nn.Module): 
  """ 
  Channel and Spatial Attention Block (CASAB) 
   
  This module combines channel attention and spatial attention to selectively   
  emphasize the most important channels and spatial regions in feature maps.  
  """
  
  def __init__(self, in_channels, reduction_ratio=16):  
      """ 
      Args:   
          in_channels (int): Number of input channels    
          reduction_ratio (int): Reduction ratio for channel attention FC layers 
      """
      super(CASAB, self).__init__()
     
      # Channel Attention Module (CAM)
      self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
      self.global_max_pool = nn.AdaptiveMaxPool2d(1)     
 
      # FC layers for channel attention
      self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
      self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
    
      # Spatial Attention Module (SAM)
      # 7x7 depthwise convolution for spatial attention    
      self.spatial_conv = nn.Conv2d(4, 1, kernel_size=7, padding=3, groups=1)
 
      # Activations
      self.swish = nn.SiLU()  # SiLU is equivalent to Swish
      self.sigmoid = nn.Sigmoid()
      
      # Feature refinement (mentioned in the paper)
      self.feature_refine = DSConv(in_channels, in_channels, act=nn.LeakyReLU)   
      
  def channel_attention(self, x): 
      """
      Channel Attention Module (CAM)
      
      Args:  
          x (torch.Tensor): Input feature map [B, C, H, W]     
          
      Returns:
          torch.Tensor: Channel attention weighted feature map
      """
      b, c, h, w = x.size()
  
      # Global Average Pooling and Global Max Pooling
      gap = self.global_avg_pool(x).view(b, c)  # [B, C]     
      gmp = self.global_max_pool(x).view(b, c)  # [B, C] 
      
      # Combine GAP and GMP
      combined = gap + gmp  # [B, C] 
    
      # Pass through FC layers with Swish activation  
      channel_att = self.fc1(combined)  # [B, C//r]
      channel_att = self.swish(channel_att)     
      channel_att = self.fc2(channel_att)  # [B, C]
      channel_att = self.sigmoid(channel_att)  # [B, C]
      
      # Reshape and apply attention weights   
      channel_att = channel_att.view(b, c, 1, 1)  # [B, C, 1, 1] 
    
      return x * channel_att
    
  def spatial_attention(self, x):    
      """
      Spatial Attention Module (SAM)
      
      Args: 
          x (torch.Tensor): Input feature map [B, C, H, W]
     
      Returns: 
          torch.Tensor: Spatial attention weighted feature map    
      """  
      # Multiple pooling operations along channel dimension  
      mean_pool = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W] 
      max_pool, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]     
      min_pool, _ = torch.min(x, dim=1, keepdim=True)  # [B, 1, H, W]   
      sum_pool = torch.sum(x, dim=1, keepdim=True)    # [B, 1, H, W]    
      
      # Concatenate all pooling results
      pooled_features = torch.cat([mean_pool, max_pool, min_pool, sum_pool], dim=1)  # [B, 4, H, W]    
     
      # 7x7 convolution for broader contextual information 
      spatial_att = self.spatial_conv(pooled_features)  # [B, 1, H, W]
      spatial_att = self.swish(spatial_att)    
      spatial_att = self.sigmoid(spatial_att)
   
      return x * spatial_att
  
  def forward(self, x):
      """
      Forward pass of CASAB module
      
      Args:  
          x (torch.Tensor): Input feature map [B, C, H, W]    
 
      Returns:
          torch.Tensor: Output feature map with channel and spatial attention applied
      """
      # Feature refinement (as mentioned in the paper)
      x_refined = self.feature_refine(x)   
      
      # Apply channel attention  
      x_channel = self.channel_attention(x_refined)
 
      # Apply spatial attention  
      x_spatial = self.spatial_attention(x_refined)  
     
      # Combine channel and spatial attention using element-wise addition
      output = x_channel + x_spatial 
    
      return output  

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    batch_size, channel, height, width = 1, 16, 32, 32
    inputs = torch.randn((batch_size, channel, height, width)).to(device)

    module = CASAB(channel).to(device)

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
    
    print(ORANGE)  
    flops, macs, _ = calculate_flops(model=module, 
                                     input_shape=(batch_size, channel, height, width),  
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)     
