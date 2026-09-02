'''
本文件由BiliBili：魔傀面具整理 
engine/extre_module/module_images/CVPR2025-ASSM.png 
MambaIRV2中的ASSM模块
论文链接：https://arxiv.org/pdf/2411.15269
'''

import warnings
warnings.filterwarnings('ignore')   
from calflops import calculate_flops
    
import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
from einops import repeat

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  
except ImportError as e:
    # print(f'from mamba_ssm.ops.selective_scan_interface import selective_scan_fn Failure. message:{e}') 
    pass 
 
def index_reverse(index):   
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device) 
    for i in range(index.shape[0]):   
        index_r[i, index[i, :]] = ind
    return index_r    
   

def semantic_neighbor(x, index):   
    dim = index.dim()   
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)
   
    for _ in range(x.dim() - index.dim()):   
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim - 1, index=index)   
    return shuffled_x

class Selective_Scan(nn.Module):
    def __init__( 
            self,     
            d_model,
            d_state=16,
            expand=2.,   
            dt_rank="auto",    
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",  
            dt_scale=1.0,
            dt_init_floor=1e-4,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state  
        self.expand = expand 
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank  

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        ) 
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)    
        del self.x_proj

        self.dt_projs = (     
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)  
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs 
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=4, D, N)     
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.selective_scan = selective_scan_fn

    @staticmethod 
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs): 
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)    

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale  
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random": 
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)     
        else:
            raise NotImplementedError 

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max    
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))     
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759    
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt) 
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj
 
    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):  
        # S4D real initialization     
        A = repeat(     
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,   
        ).contiguous() 
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)   
            if merge: 
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log) 
        A_log._no_weight_decay = True
        return A_log

    @staticmethod     
    def D_init(d_inner, copies=1, device=None, merge=True):     
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)  
        if copies > 1:    
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)     
        D = nn.Parameter(D)  # Keep in fp32    
        D._no_weight_decay = True
        return D    
 
    def forward_core(self, x: torch.Tensor, prompt):     
        B, L, C = x.shape
        K = 1  # mambairV2 needs noly 1 scan 
        xs = x.permute(0, 2, 1).view(B, 1, C, L).contiguous()  # B, 1, C ,L 
    
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight) 
        xs = xs.float().view(B, -1, L)  
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l) 
        Bs = Bs.float().view(B, K, -1, L)    
        Cs = Cs.float().view(B, K, -1, L) + prompt  # (b, k, d_state, l)  our ASE here!
        Ds = self.Ds.float().view(-1)    
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(  
            xs, dts,     
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,    
            return_last_state=False,    
        ).view(B, K, -1, L)  
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, prompt, **kwargs):
        b, l, c = prompt.shape
        prompt = prompt.permute(0, 2, 1).contiguous().view(b, 1, c, l)
        y = self.forward_core(x, prompt)  # [B, L, C] 
        y = y.permute(0, 2, 1).contiguous()
        return y     
     
class ASSM(nn.Module):
    def __init__(self, dim, d_state=16, num_tokens=64, inner_rank=128, mlp_ratio=2.):  
        super().__init__()   
        self.dim = dim
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank    

        # Mamba params   
        self.expand = mlp_ratio
        hidden = int(self.dim * self.expand) 
        self.d_state = d_state  
        self.selectiveScan = Selective_Scan(d_model=hidden, d_state=self.d_state, expand=1)
        self.out_norm = nn.LayerNorm(hidden)
        self.act = nn.SiLU()     
        self.out_proj = nn.Linear(hidden, dim, bias=True)
 
        self.in_proj = nn.Sequential(
            nn.Conv2d(self.dim, hidden, 1, 1, 0),
        )
  
        self.CPE = nn.Sequential( 
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden),     
        )    
    
        self.embeddingB = nn.Embedding(self.num_tokens, self.inner_rank)  # [64,32] [32, 48] = [64,48] 
        self.embeddingB.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)     

        self.route = nn.Sequential(
            nn.Linear(self.dim, self.dim // 3),
            nn.GELU(),
            nn.Linear(self.dim // 3, self.num_tokens),    
            nn.LogSoftmax(dim=-1) 
        )

        self.embeddingA = nn.Embedding(self.inner_rank, d_state)
        self.embeddingA.weight.data.uniform_(-1 / self.inner_rank, 1 / self.inner_rank)

    def forward(self, x): 
        B, C, H, W = x.shape   
        n = H * W 
        x = x.flatten(2).permute(0, 2, 1) # B C H W -> N L C

        full_embedding = self.embeddingB.weight @ self.embeddingA.weight  # [128, C]  

        pred_route = self.route(x)  # [B, HW, num_token]     
        cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)  # [B, HW, num_token]

        prompt = torch.matmul(cls_policy, full_embedding).view(B, n, self.d_state)

        detached_index = torch.argmax(cls_policy.detach(), dim=-1, keepdim=False).view(B, n)  # [B, HW]
        x_sort_values, x_sort_indices = torch.sort(detached_index, dim=-1, stable=False)
        x_sort_indices_reverse = index_reverse(x_sort_indices)
   
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()   
        x = self.in_proj(x)
        x = x * torch.sigmoid(self.CPE(x))
        cc = x.shape[1] 
        x = x.view(B, cc, -1).contiguous().permute(0, 2, 1)  # b,n,c
 
        semantic_x = semantic_neighbor(x, x_sort_indices)    
        y = self.selectiveScan(semantic_x, prompt).to(x.dtype)
        y = self.out_proj(self.out_norm(y))
        x = semantic_neighbor(y, x_sort_indices_reverse)  
   
        return x.permute(0, 2, 1).view([B, C, H, W]).contiguous() # N L C -> B C H W
   
if __name__ == '__main__':     
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"  
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel, height, width = 1, 16, 20, 20
    inputs = torch.randn((batch_size, channel, height, width)).to(device)  
  
    # 此模块需要编译,详细编译命令在 compile_module/mamba
    # 对于Linux用户可以直接运行上述的make.sh文件
    # 对于Windows用户需要逐行执行make.sh文件里的内容
    module = ASSM(channel).to(device)
  
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
     
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width),
                                     output_as_string=True, 
                                     output_precision=4,
                                     print_detailed=True)  
    print(RESET)    
