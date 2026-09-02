''' 
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/TGRS2025-DWM_MSA.png
engine/extre_module/module_images/TGRS2025-DWM_MSA.md    
论文链接：https://ieeexplore.ieee.org/document/10955125
'''
   
import warnings  
warnings.filterwarnings('ignore')     
from calflops import calculate_flops   

import torch
import torch.nn as nn
from torch import einsum    
from einops import rearrange     

class DWM_MSA(nn.Module):     
    def __init__(   
            self,
            dim,
            window_size1=(4, 4),
            window_size2=(10, 10),   
            dim_head=32,
            heads=2,
    ):   
        super().__init__()
        self.dim = dim    
        self.heads = heads  
        self.scale = dim_head ** -0.5
        self.window_size1 = window_size1
        self.window_size2 = window_size2
  
        # position embedding
        # seq_l1 = window_size1[0] * window_size1[1]   
        # self.pos_emb1 = nn.Parameter(torch.Tensor(1, 1, heads, seq_l1, seq_l1))    
        # h, w = 128 // self.heads, 128 // self.heads    
        # seq_l2 = h * w * 4 // seq_l1 
        # self.pos_emb2 = nn.Parameter(torch.Tensor(1, 1, heads, seq_l2, seq_l2))
        # seq_l3 = window_size2[0] * window_size2[1]   
        # self.pos_emb3 = nn.Parameter(torch.Tensor(1, 1, heads, seq_l3, seq_l3))    
        # h, w = 128 // self.heads, 128 // self.heads
        # seq_l4 = h * w * 4 // seq_l3     
        # self.pos_emb4 = nn.Parameter(torch.Tensor(1, 1, heads, seq_l4, seq_l4))     

        # trunc_normal_(self.pos_emb1)
        # trunc_normal_(self.pos_emb2)
        # trunc_normal_(self.pos_emb3)
        # trunc_normal_(self.pos_emb4)  
    
        inner_dim = dim_head * heads   
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)     
        self.to_v = nn.Linear(dim, inner_dim, bias=False)   
        self.to_out = nn.Linear(inner_dim, dim)     
     
    def forward(self, x):    
        """   
        x: [b,h,w,c]     
        return out: [b,h,w,c]     
        """     
        x = x.permute(0, 2, 3, 1)     
        b, h, w, _ = x.shape
        w_size1 = self.window_size1
        w_size2 = self.window_size2
        assert h % w_size1[0] == 0 and w % w_size1[1] == 0, 'fmap dimensions must be divisible by the window size 1'
        assert h % w_size2[0] == 0 and w % w_size2[1] == 0, 'fmap dimensions must be divisible by the window size 2'

        q = self.to_q(x)
        k = self.to_k(x)   
        v = self.to_v(x)
        _, _, _, c = q.shape
        q1, q2, q3, q4 = q[:, :, :, :c // 4], q[:, :, :, c // 4:c // 2], \
                         q[:, :, :, c // 2:c // 4 * 3], q[:, :, :, c // 4 * 3:]  
        k1, k2, k3, k4 = k[:, :, :, :c // 4], k[:, :, :, c // 4:c // 2], \
                         k[:, :, :, c // 2:c // 4 * 3], k[:, :, :, c // 4 * 3:]
        v1, v2, v3, v4 = v[:, :, :, :c // 4], v[:, :, :, c // 4:c // 2], \
                         v[:, :, :, c // 2:c // 4 * 3], v[:, :, :, c // 4 * 3:]
        # local branch of window size 1
        q1, k1, v1 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c', b0=w_size1[0], b1=w_size1[1]),
                         (q1, k1, v1))
        q1, k1, v1 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads), (q1, k1, v1)) 
        q1 *= self.scale
        sim1 = einsum('b n h i d, b n h j d -> b n h i j', q1, k1)
        # sim1 = sim1 + self.pos_emb1
        attn1 = sim1.softmax(dim=-1)
        out1 = einsum('b n h i j, b n h j d -> b n h i d', attn1, v1)
        out1 = rearrange(out1, 'b n h mm d -> b n mm (h d)')  
  
        # non-local branch of window size 1
        q2, k2, v2 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c', b0=w_size1[0], b1=w_size1[1]),
                         (q2, k2, v2))     
        q2, k2, v2 = map(lambda t: t.permute(0, 2, 1, 3), (q2.clone(), k2.clone(), v2.clone()))
        q2, k2, v2 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads), (q2, k2, v2))  
        q2 *= self.scale
        sim2 = einsum('b n h i d, b n h j d -> b n h i j', q2, k2)
        # sim2 = sim2 + self.pos_emb2
        attn2 = sim2.softmax(dim=-1)
        out2 = einsum('b n h i j, b n h j d -> b n h i d', attn2, v2)    
        out2 = rearrange(out2, 'b n h mm d -> b n mm (h d)')   
        out2 = out2.permute(0, 2, 1, 3)

        out_1 = torch.cat([out1, out2], dim=-1).contiguous()
        out_1 = rearrange(out_1, 'b (h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // w_size1[0], w=w // w_size1[1],
                          b0=w_size1[0])   
   
        # local branch of window size 2    
        q3, k3, v3 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c', b0=w_size2[0], b1=w_size2[1]),  
                         (q3, k3, v3))
        q3, k3, v3 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads), (q3, k3, v3))
        q3 *= self.scale
        sim3 = einsum('b n h i d, b n h j d -> b n h i j', q3, k3)
        # sim3 = sim3 + self.pos_emb3  
        attn3 = sim3.softmax(dim=-1)
        out3 = einsum('b n h i j, b n h j d -> b n h i d', attn3, v3)
        out3 = rearrange(out3, 'b n h mm d -> b n mm (h d)')

        # non-local of window size 2 
        q4, k4, v4 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c', b0=w_size2[0], b1=w_size2[1]),   
                         (q4, k4, v4))
        q4, k4, v4 = map(lambda t: t.permute(0, 2, 1, 3), (q4.clone(), k4.clone(), v4.clone()))
        q4, k4, v4 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads), (q4, k4, v4))
        q4 *= self.scale   
        sim4 = einsum('b n h i d, b n h j d -> b n h i j', q4, k4)   
        # sim4 = sim4 + self.pos_emb4
        attn4 = sim4.softmax(dim=-1)  
        out4 = einsum('b n h i j, b n h j d -> b n h i d', attn4, v4)
        out4 = rearrange(out4, 'b n h mm d -> b n mm (h d)') 
        out4 = out4.permute(0, 2, 1, 3)
    
        out_2 = torch.cat([out3, out4], dim=-1).contiguous()  
        out_2 = rearrange(out_2, 'b (h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // w_size2[0], w=w // w_size2[1],
                          b0=w_size2[0])     

        out = torch.cat([out_1, out_2], dim=-1).contiguous()    
        out = self.to_out(out)  
  
        return out.permute(0, 3, 1, 2)     
     
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel, height, width = 1, 512, 20, 20
    inputs = torch.randn((batch_size, channel, height, width)).to(device)    
  
    module = DWM_MSA(channel).to(device)  
   
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
     
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width),   
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)    
