'''  
本文件由BiliBili：魔傀面具整理    
engine/extre_module/module_images/CVPR2025-TAB.md   
engine/extre_module/module_images/CVPR2025-TAB.png
论文链接：https://arxiv.org/pdf/2503.06896    
'''
  
import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops   
    
import torch  
import torch.nn as nn
import torch.nn.functional as F  
from einops import rearrange   
from inspect import isfunction    

def exists(val):   
    return val is not None
  
def is_empty(t):
    return t.nelement() == 0

def expand_dim(t, dim, k):   
    t = t.unsqueeze(dim)  
    expand_shape = [-1] * len(t.shape) 
    expand_shape[dim] = k 
    return t.expand(*expand_shape)
  
def default(x, d):
    if not exists(x):
        return d if not isfunction(d) else d()
    return x
    
def ema(old, new, decay):
    if not exists(old):   
        return new
    return old * decay + new * (1 - decay)  
    
def ema_inplace(moving_avg, new, decay):
    if is_empty(moving_avg):     
        moving_avg.data.copy_(new)
        return   
    moving_avg.data.mul_(decay).add_(new, alpha= (1 - decay))    
    
  
def similarity(x, means):   
    return torch.einsum('bld,cd->blc', x, means)

def dists_and_buckets(x, means):
    dists = similarity(x, means)
    _, buckets = torch.max(dists, dim=-1)
    return dists, buckets

def batched_bincount(index, num_classes, dim=-1):
    shape = list(index.shape)   
    shape[dim] = num_classes
    out = index.new_zeros(shape)   
    out.scatter_add_(dim, index, torch.ones_like(index, dtype=index.dtype))     
    return out
    
def center_iter(x, means, buckets = None): 
    b, l, d, dtype, num_tokens = *x.shape, x.dtype, means.shape[0]  
 
    if not exists(buckets):  
        _, buckets = dists_and_buckets(x, means)     
 
    bins = batched_bincount(buckets, num_tokens).sum(0, keepdim=True)     
    zero_mask = bins.long() == 0

    means_ = buckets.new_zeros(b, num_tokens, d, dtype=dtype)    
    means_.scatter_add_(-2, expand_dim(buckets, -1, d), x)  
    means_ = F.normalize(means_.sum(0, keepdim=True), dim=-1).type(dtype)    
    means = torch.where(zero_mask.unsqueeze(-1), means, means_)  
    means = means.squeeze(0)    
    return means   
    
class IASA(nn.Module): 
    def __init__(self, dim, qk_dim, heads, group_size): 
        super().__init__()    
        self.heads = heads
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False) 
        self.group_size = group_size
        
 
    def forward(self, normed_x, idx_last, k_global, v_global):    
        x = normed_x   
        B, N, _ = x.shape
       
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)   
        q = torch.gather(q, dim=-2, index=idx_last.expand(q.shape))
        k = torch.gather(k, dim=-2, index=idx_last.expand(k.shape))    
        v = torch.gather(v, dim=-2, index=idx_last.expand(v.shape))
   
        gs = min(N, self.group_size)  # group size
        ng = (N + gs - 1) // gs
        pad_n = ng * gs - N
     
        paded_q = torch.cat((q, torch.flip(q[:,N-pad_n:N, :], dims=[-2])), dim=-2)
        paded_q = rearrange(paded_q, "b (ng gs) (h d) -> b ng h gs d",ng=ng,h=self.heads)
        paded_k = torch.cat((k, torch.flip(k[:,N-pad_n-gs:N, :], dims=[-2])), dim=-2)
        paded_k = paded_k.unfold(-2,2*gs,gs)
        paded_k = rearrange(paded_k, "b ng (h d) gs -> b ng h gs d",h=self.heads)
        paded_v = torch.cat((v, torch.flip(v[:,N-pad_n-gs:N, :], dims=[-2])), dim=-2)  
        paded_v = paded_v.unfold(-2,2*gs,gs)
        paded_v = rearrange(paded_v, "b ng (h d) gs -> b ng h gs d",h=self.heads)  
        out1 = F.scaled_dot_product_attention(paded_q,paded_k,paded_v)     
        
     
        k_global = k_global.reshape(1,1,*k_global.shape).expand(B,ng,-1,-1,-1)   
        v_global = v_global.reshape(1,1,*v_global.shape).expand(B,ng,-1,-1,-1)    
       
        out2 = F.scaled_dot_product_attention(paded_q,k_global,v_global)
        out = out1 + out2    
        out = rearrange(out, "b ng h gs d -> b (ng gs) (h d)")[:, :N, :]
 
        out = out.scatter(dim=-2, index=idx_last.expand(out.shape), src=out)
        out = self.proj(out)
    
        return out   
    
class IRCA(nn.Module): 
    def __init__(self, dim, qk_dim, heads):
        super().__init__() 
        self.heads = heads
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
      
    def forward(self, normed_x, x_means): 
        x = normed_x   
        if self.training:
            x_global = center_iter(F.normalize(x,dim=-1), F.normalize(x_means,dim=-1))    
        else:
            x_global = x_means
     
        k, v = self.to_k(x_global), self.to_v(x_global)
        k = rearrange(k, 'n (h dim_head)->h n dim_head', h=self.heads)    
        v = rearrange(v, 'n (h dim_head)->h n dim_head', h=self.heads)    

        return k,v, x_global.detach() 
    
class TAB(nn.Module):
    def __init__(self, dim, heads=8, n_iter=3,
                 num_tokens=8, group_size=128,
                 ema_decay = 0.999):
        super().__init__()    
  
        qk_dim = dim // 2
        self.n_iter = n_iter    
        self.ema_decay = ema_decay 
        self.num_tokens = num_tokens   
        
        self.irca_attn = IRCA(dim, qk_dim, heads) 
        self.iasa_attn = IASA(dim, qk_dim, heads, group_size)    
        self.register_buffer('means', torch.randn(num_tokens, dim))
        self.register_buffer('initted', torch.tensor(False))
        self.conv1x1 = nn.Conv2d(dim,dim,1, bias=False)     
 
    def forward(self, x):
        _,_,h, w = x.shape
        x = rearrange(x, 'b c h w->b (h w) c')
        B, N, _ = x.shape     
        
        idx_last = torch.arange(N, device=x.device).reshape(1,N).expand(B,-1)
        if not self.initted:    
            pad_n = self.num_tokens - N % self.num_tokens
            paded_x = torch.cat((x, torch.flip(x[:,N-pad_n:N, :], dims=[-2])), dim=-2)     
            x_means=torch.mean(rearrange(paded_x, 'b (cnt n) c->cnt (b n) c',cnt=self.num_tokens),dim=-2).detach()   
        else:  
            x_means = self.means.detach() 

        if self.training:     
            with torch.no_grad(): 
                for _ in range(self.n_iter-1):    
                    x_means = center_iter(F.normalize(x,dim=-1), F.normalize(x_means,dim=-1))    
                
        k_global, v_global, x_means = self.irca_attn(x, x_means)     
        
        with torch.no_grad():  
            x_scores = torch.einsum('b i c,j c->b i j', 
                                        F.normalize(x, dim=-1),     
                                        F.normalize(x_means, dim=-1)) 
            x_belong_idx = torch.argmax(x_scores, dim=-1)
    
            idx = torch.argsort(x_belong_idx, dim=-1)   
            idx_last = torch.gather(idx_last, dim=-1, index=idx).unsqueeze(-1) 
  
        y = self.iasa_attn(x, idx_last,k_global,v_global)
        y = rearrange(y,'b (h w) c->b c h w',h=h).contiguous()     
        y = self.conv1x1(y)     
     
        if self.training:  
            with torch.no_grad():   
                new_means = x_means
                if not self.initted:
                    self.means.data.copy_(new_means)
                    self.initted.data.copy_(torch.tensor(True))
                else:    
                    ema_inplace(self.means, new_means, self.ema_decay)     
            
     
        return rearrange(x, 'b (h w) c->b c h w',h=h)    
   
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  
    batch_size, channel, height, width = 1, 32, 32, 32    
    inputs = torch.randn((batch_size, channel, height, width)).to(device)
     
    module = TAB(channel).to(device)   
 
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module, 
                                     input_shape=(batch_size, channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,     
                                     print_detailed=True)
    print(RESET)    
