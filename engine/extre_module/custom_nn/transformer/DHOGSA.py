'''
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/AAAI2026-DHOGSA.png     
engine/extre_module/module_images/AAAI2026-DHOGSA.md
论文链接：https://arxiv.org/pdf/2504.09377
'''

import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops 

import torch   
import torch.nn as nn
import torch.nn.functional as F  
from einops import rearrange    

class DHOGSA(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False, ifBox=True, patch_size=8, n_bins=9):    
        super(DHOGSA, self).__init__()     
        self.factor = num_heads
        self.ifBox = ifBox
        self.num_heads = num_heads     
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))   
        self.qkv = nn.Conv2d(dim, dim*5, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*5, dim*5, kernel_size=3, stride=1, padding=1, groups=dim*5, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)    
        self.bin_proj = nn.Conv2d(n_bins, dim // 2, kernel_size=1, bias=bias)
        self.patch_size = patch_size
        self.n_bins = n_bins
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3) 
        self.register_buffer('sobel_x', sobel_x.repeat(dim, 1, 1, 1))    
        self.register_buffer('sobel_y', sobel_y.repeat(dim, 1, 1, 1)) 
    
    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw//factor+1)*factor-hw]
        x = F.pad(x, t_pad, 'constant', 0)     
        return x, t_pad    
    
    def unpad(self, x, t_pad):
        *_, hw = x.shape     
        return x[:,:,t_pad[0]:hw-t_pad[1]]
    
    def softmax_1(self, x, dim=-1):    
        logit = x.exp()
        logit  = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit
     
    def normalize(self, x):
        mu = x.mean(-2, keepdim=True)    
        sigma = x.var(-2, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5)

    def reshape_attn(self, q, k, v, ifBox):    
        b, c = q.shape[:2]   
        q, t_pad = self.pad(q, self.factor)
        k, t_pad = self.pad(k, self.factor)
        v, t_pad = self.pad(v, self.factor)  
        hw = q.shape[-1] // self.factor  
        shape_ori = "b (head c) (factor hw)" if ifBox else "b (head c) (hw factor)"
        shape_tar = "b head (c factor) hw"
        q = rearrange(q, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        k = rearrange(k, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)  
        v = rearrange(v, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)     
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature  
        attn = self.softmax_1(attn, dim=-1)  
        out = (attn @ v)
        out = rearrange(out, '{} -> {}'.format(shape_tar, shape_ori), factor=self.factor, hw=hw, b=b, head=self.num_heads)
        out = self.unpad(out, t_pad)
        return out     
 
    def split_into_patches(self, x):
        b, c, h, w = x.shape     
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size    
        if pad_h > 0 or pad_w > 0:  
            x = F.pad(x, (0, pad_w, 0, pad_h))
        patches = rearrange(x, 'b c (h p1) (w p2) -> b (h w) c (p1 p2)', p1=self.patch_size, p2=self.patch_size)
        n_h, n_w = (h + pad_h)//self.patch_size, (w + pad_w)//self.patch_size  
        return patches, (b, c, h, w, pad_h, pad_w, n_h, n_w)   

    def merge_patches(self, patches, shape_info):     
        b, c, h, w, pad_h, pad_w, n_h, n_w = shape_info 
        patches = rearrange(patches, 'b (h w) c (p1 p2) -> b c (h p1) (w p2)', h=n_h, w=n_w, p1=self.patch_size, p2=self.patch_size)
        if pad_h > 0 or pad_w > 0:
            patches = patches[:, :, :h, :w]
        return patches  
  
    def apply_hog_to_patch(self, x_half):
        b, c, h, w = x_half.shape  
        gx = F.conv2d(x_half, self.sobel_x[:c], padding=1, groups=c)     
        gy = F.conv2d(x_half, self.sobel_y[:c], padding=1, groups=c)
        magnitude = torch.sqrt(gx**2 + gy**2 + 1e-6) 
        orientation = torch.atan2(gy, gx)  # [-pi, pi]    
        orientation_bin = ((orientation + torch.pi) / (2 * torch.pi) * self.n_bins).long() % self.n_bins
        patches_x, shape_info = self.split_into_patches(x_half)  
        patches_mag, _ = self.split_into_patches(magnitude)
        patches_ori, _ = self.split_into_patches(orientation_bin.float())
        b, n_patches, c, patch_pixels = patches_x.shape 
        sort_values = torch.zeros_like(patches_x)  
        hog_features = torch.zeros(b, n_patches, self.n_bins, device=x_half.device)
        for i in range(self.n_bins):
            bin_mask = (patches_ori == i).float()   
            bin_magnitude = patches_mag * bin_mask   
            sort_values += bin_magnitude * (i + 1)
            hog_features[..., i] = bin_magnitude.mean(dim=[-1,-2])  
 
        hog_features = hog_features / (hog_features.sum(dim=-1, keepdim=True) + 1e-8)  
        _, sort_indices = sort_values.sum(dim=2, keepdim=True).expand_as(patches_x).sort(dim=-1)
        patches_x_sorted = torch.gather(patches_x, -1, sort_indices)
        x_half_processed = self.merge_patches(patches_x_sorted, shape_info)   
        return x_half_processed, sort_indices, hog_features, shape_info   
  
    def forward(self, x): 
        b, c, h, w = x.shape
        half_c = c // 2
        x_half = x[:, :half_c]
        x_half_processed, idx_patch, hog_features, shape_info = self.apply_hog_to_patch(x_half)     
        b, n_patches, n_bins = hog_features.shape
        n_h = shape_info[-2] #int(math.sqrt(n_patches))
        n_w = shape_info[-1]  
        hog_map = rearrange(hog_features, 'b (nh nw) bins -> b bins nh nw', nh=n_h, nw=n_w).contiguous()
        hog_map = self.bin_proj(hog_map)   
        hog_map = F.interpolate(hog_map, size=(h, w), mode='bilinear')   
        x = torch.cat((x_half_processed + hog_map, x[:, half_c:]), dim=1)    

        qkv = self.qkv_dwconv(self.qkv(x))  
        q1, k1, q2, k2, v = qkv.chunk(5, dim=1)  # b,c,x,x  
        gx = F.conv2d(v, self.sobel_x[:c], padding=1, groups=c)
        gy = F.conv2d(v, self.sobel_y[:c], padding=1, groups=c)  
        magnitude = torch.sqrt(gx**2 + gy**2 + 1e-6).view(b, c, -1)    
        orientation = torch.atan2(gy, gx).view(b, c, -1)  # [-pi, pi]  
   
        orientation_norm = ((orientation + torch.pi) / (2 * torch.pi))
        weighted_magnitude = magnitude * orientation_norm
        _, idx = weighted_magnitude.sum(dim=1).sort(dim=-1)
        idx = idx.unsqueeze(1).expand(b, c, -1) 
        v = torch.gather(v.view(b, c, -1), dim=2, index=idx) 
        q1 = torch.gather(q1.view(b, c, -1), dim=2, index=idx) 
        k1 = torch.gather(k1.view(b, c, -1), dim=2, index=idx)
        q2 = torch.gather(q2.view(b, c, -1), dim=2, index=idx)   
        k2 = torch.gather(k2.view(b, c, -1), dim=2, index=idx)

        out1 = self.reshape_attn(q1, k1, v, True)    
        out2 = self.reshape_attn(q2, k2, v, False)   

        out1 = torch.scatter(out1, 2, idx, out1).view(b, c, h, w)  
        out2 = torch.scatter(out2, 2, idx, out2).view(b, c, h, w)     
        out = out1 * out2  
        out = self.project_out(out)

        out_replace = out[:,:half_c]  
        patches_out, shape_info = self.split_into_patches(out_replace)
        patches_out = torch.scatter(patches_out, -1, idx_patch, patches_out)   
        out_replace = self.merge_patches(patches_out, shape_info)
        out[:,:half_c] = out_replace   
        return out 

if __name__ == '__main__':  
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    batch_size, channel, height, width = 1, 16, 20, 20
    inputs = torch.randn((batch_size, channel, height, width)).to(device)   
   
    module = DHOGSA(channel, num_heads=8).to(device)  
   
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
 
    print(ORANGE)     
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width),
                                     output_as_string=True,
                                     output_precision=4, 
                                     print_detailed=True) 
    print(RESET)