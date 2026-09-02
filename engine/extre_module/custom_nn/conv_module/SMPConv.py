'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/CVPR2023-SMPConv.md    
论文链接：https://arxiv.org/pdf/2304.02330
'''    
  
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings    
warnings.filterwarnings('ignore')     
from calflops import calculate_flops

import torch  
import torch.nn as nn  
import torch.nn.functional as F     
from timm.layers import trunc_normal_, DropPath
    
try: 
    from depthwise_conv2d_implicit_gemm import _DepthWiseConv2dImplicitGEMMFP16, _DepthWiseConv2dImplicitGEMMFP32
except ImportError as e:     
    pass    
  
from engine.extre_module.ultralytics_nn.conv import Conv

def rel_pos(kernel_size):     
    tensors = [torch.linspace(-1, 1, steps=kernel_size) for _ in range(2)]
    kernel_coord = torch.stack(torch.meshgrid(*tensors), dim=-0)   
    kernel_coord = kernel_coord.unsqueeze(0)  
    return kernel_coord  

class SMPConv(nn.Module):
    def __init__(self, inc, planes, kernel_size=11, n_points=4):
        super().__init__()

        self.planes = planes
        self.kernel_size = kernel_size
        self.n_points = n_points
        self.init_radius = 2 * (2/kernel_size)
  
        # kernel_coord    
        kernel_coord = rel_pos(kernel_size)
        self.register_buffer('kernel_coord', kernel_coord)
     
        # weight_coord
        weight_coord = torch.empty(1, n_points, 2)     
        nn.init.trunc_normal_(weight_coord, std=0.2, a=-1., b=1.) 
        self.weight_coord = nn.Parameter(weight_coord)

        self.radius = nn.Parameter(torch.empty(1, n_points).unsqueeze(-1).unsqueeze(-1))    
        self.radius.data.fill_(value=self.init_radius)  
  
        # weight
        weights = torch.empty(1, planes, n_points)    
        trunc_normal_(weights, std=.02)   
        self.weights = nn.Parameter(weights)
    
        self.conv1x1 = Conv(inc, planes, 1) if inc != planes else nn.Identity()
 
    def forward(self, x):   
        x = self.conv1x1(x) 
        kernels = self.make_kernels().unsqueeze(1)
        x = x.contiguous()
        kernels = kernels.contiguous()  
    
        if x.dtype == torch.float32:
            x = _DepthWiseConv2dImplicitGEMMFP32.apply(x, kernels)
        elif x.dtype == torch.float16:
            x = _DepthWiseConv2dImplicitGEMMFP16.apply(x, kernels)
        else:
            raise TypeError("Only support fp32 and fp16, get {}".format(x.dtype))
        return x    
  
    def make_kernels(self):
        diff = self.weight_coord.unsqueeze(-2) - self.kernel_coord.reshape(1,2,-1).transpose(1,2)  # [1, n_points, kernel_size^2, 2]     
        diff = diff.transpose(2,3).reshape(1, self.n_points, 2, self.kernel_size, self.kernel_size)    
        diff = F.relu(1 - torch.sum(torch.abs(diff), dim=2) / self.radius)  # [1, n_points, kernel_size, kernel_size]    
 
        # Apply weighted diff for average weighted kernel    
        # non_zero = (diff != 0) # [1, n_points, kernel_size, kernel_size]  
        # count_weight = 1 / (torch.sum(non_zero, dim=1, keepdim=True) + 1e-6)  # [1, 1, kernel_size, kernel_size]
        # weighted_diff = count_weight * diff  # [1, n_points, kernel_size, kernel_size]    

        kernels = torch.matmul(self.weights, diff.reshape(1, self.n_points, -1)) # [1, planes, kernel_size*kernel_size]
        kernels = kernels.reshape(1, self.planes, *self.kernel_coord.shape[2:]) # [1, planes, kernel_size, kernel_size]
        kernels = kernels.squeeze(0)    
        kernels = torch.flip(kernels.permute(0,2,1), dims=(1,))
        return kernels  
    
    def radius_clip(self, min_radius=1e-3, max_radius=1.):
        r = self.radius.data  
        r = r.clamp(min_radius, max_radius)    
        self.radius.data = r

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"   
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)  
    
    # 此模块需要编译,详细编译命令在 compile_module/cutlass/examples/19_large_depthwise_conv2d_torch_extension
    # 对于Linux用户可以直接运行上述的make.sh文件    
    # 对于Windows用户需要逐行执行make.sh文件里的内容
    module = SMPConv(in_channel, out_channel, kernel_size=11).to(device)

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)  
     
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)   
