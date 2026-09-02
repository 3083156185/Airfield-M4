'''  
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/AAAI2026-PatchMamba.md     
engine/extre_module/module_images/AAAI2026-PatchMamba.png 
论文链接：https://arxiv.org/pdf/2508.03336
'''
     
import warnings 
warnings.filterwarnings('ignore')   
from calflops import calculate_flops     
    
import math
import numbers
 
import torch 
import torch.nn as nn  
import torch.nn.functional as F
from einops import rearrange, repeat  
  
try:   
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except ImportError as e:   
    pass    
   
def selective_scan(u, delta, a, b, c, d, z=None, delta_bias=None, delta_softplus=True, return_last_state=False): 
    scan_impl = selective_scan_fn if u.is_cuda else selective_scan_ref    
    return scan_impl(   
        u,     
        delta,  
        a,  
        b, 
        c,    
        d,
        z=z,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
        return_last_state=return_last_state,
    )    

 
class ImageBlockProcessor:
    def __init__(self, num_blocks=None): 
        self.num_blocks = num_blocks or 4    
   
    def _get_block_layout(self, height, width):   
        if isinstance(self.num_blocks, tuple):  
            num_rows, num_cols = self.num_blocks
        else:
            total_blocks = self.num_blocks
            aspect_ratio = width / height
            num_rows = int(math.sqrt(total_blocks / aspect_ratio))
            num_rows = max(1, num_rows)
            num_cols = total_blocks // num_rows

            while num_rows * num_cols < total_blocks: 
                if num_cols * aspect_ratio < num_rows:
                    num_cols += 1
                else:
                    num_rows += 1
    
        return num_rows, num_cols   

    def split_and_stack_blocks(self, image):     
        batch_size, channels, height, width = image.shape     
        device = image.device    

        num_rows, num_cols = self._get_block_layout(height, width)
     
        block_height = height // num_rows
        block_width = width // num_cols   
  
        pad_height = 0 if block_height % 8 == 0 else 8 - (block_height % 8)
        pad_width = 0 if block_width % 8 == 0 else 8 - (block_width % 8) 
     
        if pad_height > 0 or pad_width > 0:  
            image = F.pad(image, (0, pad_width, 0, pad_height), mode="reflect")    

        _, _, padded_height, padded_width = image.shape
        block_height = padded_height // num_rows   
        block_width = padded_width // num_cols   

        blocks = []
        for i in range(num_rows): 
            for j in range(num_cols):
                start_h = i * block_height
                end_h = (i + 1) * block_height
                start_w = j * block_width   
                end_w = (j + 1) * block_width  
                blocks.append(image[:, :, start_h:end_h, start_w:end_w]) 
  
        stacked_blocks = torch.cat(blocks, dim=1).to(device)
        block_info = {
            "original_shape": (batch_size, channels, height, width),
            "padded_shape": (batch_size, channels, padded_height, padded_width), 
            "num_blocks": (num_rows, num_cols),   
            "block_size": (block_height, block_width), 
            "pad_size": (pad_height, pad_width),
        }  
        return stacked_blocks, block_info
  
    def unstack_and_merge_blocks(self, stacked_blocks, block_info):
        batch_size, channels, height, width = block_info["original_shape"]
        _, _, padded_height, padded_width = block_info["padded_shape"]
        num_rows, num_cols = block_info["num_blocks"]     
        block_height, block_width = block_info["block_size"]
     
        split_blocks = torch.split(stacked_blocks, channels, dim=1) 
        merged_image = torch.zeros(  
            (batch_size, channels, padded_height, padded_width),  
            device=stacked_blocks.device,
            dtype=stacked_blocks.dtype,
        )  
    
        idx = 0
        for i in range(num_rows):
            for j in range(num_cols):  
                start_h = i * block_height   
                end_h = (i + 1) * block_height    
                start_w = j * block_width
                end_w = (j + 1) * block_width
                merged_image[:, :, start_h:end_h, start_w:end_w] = split_blocks[idx]    
                idx += 1
     
        return merged_image[:, :, :height, :width]


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)  
    

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()     
        self.net = nn.Sequential( 
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),    
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        ) 
   
    def forward(self, x):
        out = self.net(x.permute(0, 3, 1, 2).contiguous()) 
        return out.permute(0, 2, 3, 1) 


class PreNorm(nn.Module):  
    def __init__(self, dim, fn):
        super().__init__()   
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, *args, **kwargs):
        x = self.norm(x)  
        return self.fn(x, *args, **kwargs)

 
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):    
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,) 
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
  
    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False) 
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):   
        super().__init__() 
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)    
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1   
   
        self.weight = nn.Parameter(torch.ones(normalized_shape))     
        self.bias = nn.Parameter(torch.zeros(normalized_shape))  
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)   
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

    
def to_4d(x, height, width):   
    return rearrange(x, "b (h w) c -> b c h w", h=height, w=width) 
   
   
def to_3d(x):    
    return rearrange(x, "b c h w -> b (h w) c")
  

class LayerNorm(nn.Module):  
    def __init__(self, dim, layer_norm_type):  
        super().__init__()     
        if layer_norm_type == "BiasFree":   
            self.body = BiasFree_LayerNorm(dim)    
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        height, width = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), height, width)   


class SS2D6(nn.Module):
    def __init__(    
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,     
        dt_rank="auto",  
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",    
        dt_scale=1.0,  
        dt_init_floor=1e-4,     
        dropout=0.0,
        conv_bias=True,  
        bias=False,
        device=None,
        dtype=None,   
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}     
        super().__init__()
        self.d_model = d_model   
        self.d_state = d_state  
        self.d_conv = d_conv   
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model) 
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
     
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)    
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner, 
            groups=self.d_inner,
            bias=conv_bias,  
            kernel_size=d_conv,   
            padding=(d_conv - 1) // 2,     
            **factory_kwargs,     
        )
        self.act = nn.SiLU()  

        self.x_proj = (  
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),    
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),   
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),     
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),     
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))    
        del self.x_proj     

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),  
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),    
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs), 
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs   

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=6, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=6, merge=True)
   
        self.out_norm = nn.LayerNorm(self.d_inner) 
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)   
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
 
    @staticmethod  
    def dt_init(     
        dt_rank,    
        d_inner,   
        dt_scale=1.0,
        dt_init="random",
        dt_min=0.001,    
        dt_max=0.1,
        dt_init_floor=1e-4, 
        **factory_kwargs,
    ):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale  
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":   
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:   
            raise NotImplementedError     

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)     
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)  
        dt_proj.bias._no_reinit = True    
        return dt_proj  

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):    
        a = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=d_inner)
        a_log = torch.log(a).contiguous()  
        if copies > 1:  
            a_log = repeat(a_log, "d n -> r d n", r=copies)
            if merge:
                a_log = a_log.flatten(0, 1)    
        a_log = nn.Parameter(a_log)  
        a_log._no_weight_decay = True  
        return a_log
     
    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        d = torch.ones(d_inner, device=device)  
        if copies > 1:   
            d = repeat(d, "n -> r n", r=copies)
            if merge:
                d = d.flatten(0, 1)
        d = nn.Parameter(d)     
        d._no_weight_decay = True
        return d
    
    def diagonal_trans(self, x, height, width):
        batch_size, num_routes, _, seq_len = x.shape
        assert seq_len == height * width, "last dimension must equal H*W"
        idx = torch.arange(height * width, device=x.device).reshape(height, width)
        i_idx = torch.arange(height, device=x.device).reshape(-1, 1).expand(height, width)
        j_idx = torch.arange(width, device=x.device).reshape(1, -1).expand(height, width) 
        diag_mask = i_idx + j_idx   
        sorted_idx = torch.argsort(diag_mask.reshape(-1))  
        diag_indices = torch.index_select(idx.reshape(-1), 0, sorted_idx)
        x_flat = x.view(batch_size, num_routes, -1, height * width)
        diag_flat = torch.index_select(x_flat, dim=3, index=diag_indices)
        return diag_flat, diag_indices

    def reverse_diagonal_trans(self, x, diag_indices):    
        batch_size, num_routes, _, seq_len = x.shape 
        x_flat = x.view(batch_size, num_routes, -1, seq_len)    
        reverse_indices = torch.argsort(diag_indices)
        return torch.index_select(x_flat, dim=3, index=reverse_indices) 

    def forward_corev0(self, x):
        batch_size, _, height, width = x.shape  
        seq_len = height * width   
        num_routes = 6 

        x_hwwh = torch.stack(
            [x.view(batch_size, -1, seq_len), torch.transpose(x, dim0=2, dim1=3).contiguous().view(batch_size, -1, seq_len)],
            dim=1,  
        ).view(batch_size, 2, -1, seq_len)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_invhh = torch.flip(x, dims=[-1])
        x_hh = torch.stack([x.view(batch_size, -1, seq_len), x_invhh.view(batch_size, -1, seq_len)], dim=1).view(
            batch_size, 2, -1, seq_len  
        )
        x_hh, diag_indices = self.diagonal_trans(x_hh, height, width)
        xs = torch.cat([xs, x_hh], dim=1)   
   
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(batch_size, num_routes, -1, seq_len), self.x_proj_weight)
        dts, bs, cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)    
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(batch_size, num_routes, -1, seq_len), self.dt_projs_weight)

        xs = xs.float().view(batch_size, -1, seq_len)
        dts = dts.contiguous().float().view(batch_size, -1, seq_len)  
        bs = bs.float().view(batch_size, num_routes, -1, seq_len)   
        cs = cs.float().view(batch_size, num_routes, -1, seq_len)
        ds = self.Ds.float().view(-1)
        a_s = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) 

        out_y = selective_scan(
            xs,
            dts,  
            a_s,    
            bs,
            cs,
            ds,
            z=None, 
            delta_bias=dt_projs_bias,  
            delta_softplus=True,
            return_last_state=False,
        ).view(batch_size, num_routes, -1, seq_len)     
        assert out_y.dtype == torch.float    

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(batch_size, 2, -1, seq_len)   
        wh_y = torch.transpose(out_y[:, 1].view(batch_size, -1, width, height), dim0=2, dim1=3).contiguous().view( 
            batch_size, -1, seq_len     
        )    
        invwh_y = torch.transpose(inv_y[:, 1].view(batch_size, -1, width, height), dim0=2, dim1=3).contiguous().view(
            batch_size, -1, seq_len    
        )

        transhh_y = self.reverse_diagonal_trans(out_y[:, 4:6], diag_indices)
        invhh_y = torch.flip(transhh_y[:, 1], dims=[-1])  
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y, transhh_y[:, 0], invhh_y

    def forward(self, x, **kwargs):
        batch_size, height, width, _ = x.shape     
        xz = self.in_proj(x)   
        x, z = xz.chunk(2, dim=-1) 

        x = x.permute(0, 3, 1, 2).contiguous()     
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4, y5, y6 = self.forward_corev0(x)
        y = y1 + y2 + y3 + y4 + y5 + y6    
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(batch_size, height, width, -1)
        y = self.out_norm(y) 
        y = y * F.silu(z)  
        out = self.out_proj(y)
        if self.dropout is not None: 
            out = self.dropout(out)
        return out
    
  
class PatchMamba(nn.Module):
    def __init__(self, input_channels, num_blocks=2, num_mamba_layers=2, LayerNorm_type="WithBias"):    
        super().__init__()   
        self.block_processor = ImageBlockProcessor(num_blocks)
        self.num_mamba_layers = num_mamba_layers
     
        total_blocks = num_blocks[0] * num_blocks[1] if isinstance(num_blocks, tuple) else num_blocks 
        total_channels = input_channels * total_blocks  

        self.init_ccnv = nn.Conv2d(input_channels, input_channels, 3, 1, 1, bias=True)  
        self.conv1 = nn.Conv2d(total_channels, total_channels, 1, 1)  
        self.norm1 = LayerNorm(total_channels, LayerNorm_type)  
        self.conv_out = nn.Conv2d(input_channels, input_channels, 3, 1, 1, bias=True)
     
        self.hhmamba = nn.ModuleList(
            [     
                nn.ModuleList(  
                    [
                        SS2D6(d_model=total_channels),  
                        PreNorm(total_channels, FeedForward(dim=total_channels)), 
                    ]
                )
                for _ in range(self.num_mamba_layers)
            ]   
        ) 

    def forward(self, x):     
        x = self.init_ccnv(x) 
        stacked_blocks, block_info = self.block_processor.split_and_stack_blocks(x)  
        processed_blocks = self.conv1(stacked_blocks)  
        processed_blocks = self.norm1(processed_blocks)   

        for ss2d, ff in self.hhmamba:
            y = processed_blocks.permute(0, 2, 3, 1) 
            processed_blocks = ss2d(y) + processed_blocks.permute(0, 2, 3, 1)
            processed_blocks = ff(processed_blocks) + processed_blocks    
            processed_blocks = processed_blocks.permute(0, 3, 1, 2) 
 
        merged_image = self.block_processor.unstack_and_merge_blocks(processed_blocks, block_info)
        return self.conv_out(merged_image)


if __name__ == "__main__":
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    batch_size, channel, height, width = 1, 16, 20, 20
    inputs = torch.randn((batch_size, channel, height, width)).to(device)  
  
    # 此模块需要编译,详细编译命令在 compile_module/mamba
    # 对于Linux用户可以直接运行上述的make.sh文件
    # 对于Windows用户需要逐行执行make.sh文件里的内容
    module = PatchMamba(channel).to(device)
 
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)    

    print(ORANGE)    
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width),
                                     output_as_string=True,
                                     output_precision=4, 
                                     print_detailed=True)
    print(RESET)     
