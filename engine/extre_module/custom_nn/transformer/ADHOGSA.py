'''    
本文件由BiliBili：魔傀面具整理   
engine/extre_module/module_images/自研模块-ADHOGSA.png   
engine/extre_module/module_images/自研模块-ADHOGSA.md
'''
  
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..') 
     
import math     
import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops

import torch    
import torch.nn as nn    
import torch.nn.functional as F   
from einops import rearrange


class LearnableGradientExtractor(nn.Module):
    def __init__(self, channels): 
        super().__init__()
        self.channels = channels
        self.grad = nn.Conv2d(
            channels,
            channels * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels,     
            bias=False,
        )
        self._init_sobel()     
    
    def _init_sobel(self):
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],    
            dtype=torch.float32,
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],    
            dtype=torch.float32,
        )
        weight = torch.zeros(self.channels * 2, 1, 3, 3, dtype=torch.float32)
        for i in range(self.channels): 
            weight[2 * i, 0] = sobel_x
            weight[2 * i + 1, 0] = sobel_y
        with torch.no_grad():     
            self.grad.weight.copy_(weight)

    def forward(self, x):    
        grad = self.grad(x)  
        b, _, h, w = grad.shape    
        grad = grad.view(b, self.channels, 2, h, w)
        gx = grad[:, :, 0]     
        gy = grad[:, :, 1]   
        return gx, gy

     
class AdaptiveSoftHOGPrior(nn.Module):    
    def __init__(self, channels, n_bins=9, patch_sizes=(4, 8), prior_channels=None, bias=False):     
        super().__init__()
        if n_bins < 2:
            raise ValueError(f"`n_bins` must be >= 2, but got {n_bins}.")     
        if not patch_sizes: 
            raise ValueError("`patch_sizes` must contain at least one positive integer.")    
        if any(int(size) <= 0 for size in patch_sizes):     
            raise ValueError(f"`patch_sizes` must be positive, but got {patch_sizes}.")
   
        self.channels = channels
        self.n_bins = n_bins     
        self.patch_sizes = tuple(int(size) for size in patch_sizes)
        self.prior_channels = prior_channels or max(8, channels // 2)
     
        self.reduce = nn.Conv2d(channels, self.prior_channels, kernel_size=1, bias=bias)
        self.gradient = LearnableGradientExtractor(self.prior_channels)
        centers = torch.linspace(-math.pi, math.pi, steps=n_bins + 1)[:-1]
        self.bin_centers = nn.Parameter(centers)    
        self.log_bandwidth = nn.Parameter(torch.log(torch.tensor(0.5)))
        self.scale_logits = nn.Parameter(torch.zeros(len(self.patch_sizes)))
        self.scale_projs = nn.ModuleList( 
            nn.Conv2d(n_bins, channels, kernel_size=1, bias=bias)
            for _ in self.patch_sizes
        )
   
    def _soft_assign(self, orientation):    
        centers = self.bin_centers.view(1, 1, self.n_bins, 1, 1)     
        delta = orientation.unsqueeze(2) - centers  
        delta = torch.atan2(torch.sin(delta), torch.cos(delta)).abs()
        bandwidth = F.softplus(self.log_bandwidth) + 1e-4
        logits = -(delta / bandwidth).pow(2)
        return logits.softmax(dim=2)   

    def _histogram_confidence(self, histogram):
        entropy = -(histogram * (histogram + 1e-6).log()).sum(dim=1, keepdim=True)    
        confidence = 1.0 - entropy / math.log(self.n_bins)
        return confidence.clamp_(0.0, 1.0)

    def forward(self, x): 
        _, _, h, w = x.shape    
        reduced = self.reduce(x)
        gx, gy = self.gradient(reduced)     
        magnitude = torch.sqrt(gx.square() + gy.square() + 1e-6)    
        orientation = torch.atan2(gy, gx)

        assign = self._soft_assign(orientation)   
        pixel_hist = (magnitude.unsqueeze(2) * assign).mean(dim=1)  
        pixel_hist = pixel_hist / (pixel_hist.sum(dim=1, keepdim=True) + 1e-6)   
        pixel_confidence = self._histogram_confidence(pixel_hist)
    
        bin_positions = torch.linspace(   
            0.0,
            1.0,
            steps=self.n_bins, 
            device=x.device,   
            dtype=x.dtype,
        ).view(1, self.n_bins, 1, 1)
        orientation_expectation = (pixel_hist * bin_positions).sum(dim=1, keepdim=True) 
        token_score = magnitude.mean(dim=1, keepdim=True) * (1.0 + pixel_confidence + orientation_expectation)

        scale_weights = torch.softmax(self.scale_logits, dim=0)     
        prior = x.new_zeros(x.shape)
        confidence = x.new_zeros((x.shape[0], 1, h, w))
        for weight, patch_size, projector in zip(scale_weights, self.patch_sizes, self.scale_projs):
            kernel_h = min(patch_size, h)   
            kernel_w = min(patch_size, w)
            scale_hist = F.avg_pool2d(
                pixel_hist,
                kernel_size=(kernel_h, kernel_w),
                stride=(kernel_h, kernel_w),  
                ceil_mode=True,
            )     
            scale_hist = scale_hist / (scale_hist.sum(dim=1, keepdim=True) + 1e-6)
            scale_confidence = self._histogram_confidence(scale_hist)  
            scale_prior = projector(scale_hist) * scale_confidence   
            scale_prior = F.interpolate(scale_prior, size=(h, w), mode='bilinear', align_corners=False)     
            scale_confidence = F.interpolate(scale_confidence, size=(h, w), mode='bilinear', align_corners=False)
            prior = prior + weight * scale_prior
            confidence = confidence + weight * scale_confidence

        return prior, token_score, confidence, pixel_hist 
 

class ADHOGSA(nn.Module):  
    def __init__(     
        self,  
        dim,     
        num_heads=8,
        bias=False,     
        factor=None,  
        patch_sizes=(4, 8),
        n_bins=9,     
        prior_channels=None,
    ): 
        super().__init__()     
        if dim % num_heads != 0: 
            raise ValueError(f"`dim` must be divisible by num_heads, but got dim={dim}, num_heads={num_heads}.")    

        self.dim = dim     
        self.num_heads = num_heads
        self.factor = factor or num_heads   
        self.temperature = nn.Parameter(torch.ones(2, num_heads, 1, 1))

        self.prior = AdaptiveSoftHOGPrior(
            channels=dim,    
            n_bins=n_bins,
            patch_sizes=patch_sizes, 
            prior_channels=prior_channels, 
            bias=bias,   
        )
        self.qkv = nn.Conv2d(dim, dim * 5, kernel_size=1, bias=bias)    
        self.qkv_dwconv = nn.Conv2d(  
            dim * 5,
            dim * 5,
            kernel_size=3,
            stride=1,
            padding=1,   
            groups=dim * 5, 
            bias=bias,  
        )     
        self.prior_qkv = nn.Conv2d(dim, dim * 5, kernel_size=1, bias=bias)
        self.prior_qkv_scale = nn.Parameter(torch.tensor(0.1)) 
   
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(dim * 3 + 1, dim, kernel_size=1, bias=bias),     
            nn.Sigmoid(),
        )
        self.prior_residual = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.prior_residual_scale = nn.Parameter(torch.tensor(0.1))   
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    
    def _pad_tokens(self, x):     
        tokens = x.shape[-1] 
        pad = (self.factor - tokens % self.factor) % self.factor     
        if pad > 0:  
            x = F.pad(x, (0, pad), mode='constant', value=0)
        return x, pad
    
    def _unpad_tokens(self, x, pad):
        if pad == 0:    
            return x   
        return x[:, :, :-pad]     

    def _reshape_attn(self, q, k, v, branch_index, box_rearrange):
        b, _ = q.shape[:2]   
        q, pad = self._pad_tokens(q)
        k, _ = self._pad_tokens(k)
        v, _ = self._pad_tokens(v)
        hw = q.shape[-1] // self.factor    

        if box_rearrange:
            q = rearrange(q, 'b (head c) (factor hw) -> b head (c factor) hw', head=self.num_heads, factor=self.factor, hw=hw)     
            k = rearrange(k, 'b (head c) (factor hw) -> b head (c factor) hw', head=self.num_heads, factor=self.factor, hw=hw)
            v = rearrange(v, 'b (head c) (factor hw) -> b head (c factor) hw', head=self.num_heads, factor=self.factor, hw=hw)  
        else:
            q = rearrange(q, 'b (head c) (hw factor) -> b head (c factor) hw', head=self.num_heads, factor=self.factor, hw=hw)
            k = rearrange(k, 'b (head c) (hw factor) -> b head (c factor) hw', head=self.num_heads, factor=self.factor, hw=hw)     
            v = rearrange(v, 'b (head c) (hw factor) -> b head (c factor) hw', head=self.num_heads, factor=self.factor, hw=hw)
   
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature[branch_index]
        attn = attn.softmax(dim=-1)    
        out = attn @ v 

        if box_rearrange:     
            out = rearrange(out, 'b head (c factor) hw -> b (head c) (factor hw)', head=self.num_heads, factor=self.factor, hw=hw, b=b)
        else:   
            out = rearrange(out, 'b head (c factor) hw -> b (head c) (hw factor)', head=self.num_heads, factor=self.factor, hw=hw, b=b)
        return self._unpad_tokens(out, pad)
    
    def _scatter_back(self, x, index, h, w):
        out = x.new_zeros(x.shape)
        out.scatter_(2, index, x)    
        return out.view(x.shape[0], x.shape[1], h, w)

    def forward(self, x):
        b, c, h, w = x.shape
        prior, token_score, confidence, _ = self.prior(x)  
  
        qkv = self.qkv_dwconv(self.qkv(x))     
        qkv = qkv + self.prior_qkv_scale * self.prior_qkv(prior)
        q1, k1, q2, k2, v = qkv.chunk(5, dim=1)    
        v = v * (1.0 + confidence)
     
        index = token_score.flatten(2).argsort(dim=-1, descending=True)
        index = index.expand(b, c, h * w)

        q1 = torch.gather(q1.flatten(2), dim=2, index=index)    
        k1 = torch.gather(k1.flatten(2), dim=2, index=index)
        q2 = torch.gather(q2.flatten(2), dim=2, index=index)    
        k2 = torch.gather(k2.flatten(2), dim=2, index=index) 
        v = torch.gather(v.flatten(2), dim=2, index=index) 

        out1 = self._reshape_attn(q1, k1, v, branch_index=0, box_rearrange=True)
        out2 = self._reshape_attn(q2, k2, v, branch_index=1, box_rearrange=False)   

        out1 = self._scatter_back(out1, index, h, w)
        out2 = self._scatter_back(out2, index, h, w)
 
        gate = self.fusion_gate(torch.cat((out1, out2, prior, confidence), dim=1))    
        out = gate * out1 + (1.0 - gate) * out2
        out = out + self.prior_residual_scale * self.prior_residual(prior)
        return self.project_out(out)
 

if __name__ == '__main__':   
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m" 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel, height, width = 1, 32, 21, 19    
    inputs = torch.randn((batch_size, channel, height, width)).to(device)    

    module = ADHOGSA(channel, num_heads=8, patch_sizes=(3, 7)).to(device)
    
    outputs = module(inputs)   
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(
        model=module,   
        input_shape=(batch_size, channel, height, width),
        output_as_string=True,   
        output_precision=4,
        print_detailed=True,     
    )
    print(RESET)  
