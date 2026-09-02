'''    
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/CVPR2026-SFSFusion.png     
engine/extre_module/module_images/CVPR2026-SFSFusion.md
论文链接：https://arxiv.org/pdf/2508.06878 
'''

import warnings     
warnings.filterwarnings('ignore') 
from calflops import calculate_flops

import torch  
import torch.nn as nn
import torch.nn.functional as F  
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd
from torch import Tensor
from torch.nn.init import constant_, xavier_uniform_
import math

try:
    import MultiScaleDeformableAttention as MSDA  
except Exception as e:
    pass

class MSDeformAttnFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)   
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step   
        output = MSDA.ms_deform_attn_forward(value, value_spatial_shapes,
                                             value_level_start_index,
                                             sampling_locations,     
                                             attention_weights,
                                             ctx.im2col_step)  
        ctx.save_for_backward(value, value_spatial_shapes, 
                              value_level_start_index, sampling_locations,     
                              attention_weights)
        return output
    
    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, \
        sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, grad_output, ctx.im2col_step)
 
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None 
 

def ms_deform_attn_core_pytorch(value, value_spatial_shapes,
                                sampling_locations, attention_weights): 
    # for debug and test only,
    # need to use cuda version instead   
    N_, S_, M_, D_ = value.shape 
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1) 
    sampling_grids = 2 * sampling_locations - 1    
    sampling_value_list = []     
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):    
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2     
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1) 
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode='bilinear',
                                          padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)  
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).view(N_, M_ * D_, Lq_)   
    return output.transpose(1, 2).contiguous()

def generate_structured_grid(n_heads, n_points, n_levels=1, base_radius=1.0, radius_step=1.0):
    """
    Initialization of spiral-aware sampling pattern.

    parameters:
    - n_heads: number of attention heads
    - n_points: number of sampling points of each head  
    - n_levels: number of feature levels, default=1   
    - base_radius: initial radius of sampling point  
    - radius_step: radial step between consecutive points of each head
  
    return:
    - grid: Tensor, [n_heads, n_levels, n_points, 2]
    """   
    offsets = []   
    for h in range(n_heads):    
        head_offsets = []
        delta_theta = 2 * math.pi * h / n_heads  # initial angle of each head  
        for i in range(n_points):
            theta = 2 * math.pi * i / n_points + delta_theta
            r = base_radius + i * radius_step
            dx = r * math.cos(theta)    
            dy = r * math.sin(theta)    
            head_offsets.append([dx, dy])     
        offsets.append(head_offsets)
  
    grid = torch.tensor(offsets, dtype=torch.float32)
    grid = grid.unsqueeze(1).repeat(1, n_levels, 1, 1)  # [n_heads, n_levels, n_points, 2]
    return grid

def _is_power_of_2(n):  
    if (not isinstance(n, int)) or (n < 0):  
        raise ValueError('invalid input for _is_power_of_2: {} (type: {})'.format(n, type(n)))   
    return (n & (n - 1) == 0) and n != 0  
   
class MSDeformAttn_for_sfs(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, ratio=1.0):     
        """Multi-Scale Deformable Attention Module.  
  
        :param d_model      hidden dimension     
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per head 
        """   
        super().__init__()
        if d_model % n_heads != 0: 
            raise ValueError('d_model must be divisible by n_heads, ' 
                             'but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 
        # which is more efficient in our CUDA implementation 
        if not _is_power_of_2(_d_per_head):     
            warnings.warn(   
                "You'd better set d_model in MSDeformAttn_for_sfs to make "
                'the dimension of each attention head a power of 2 '   
                'which is more efficient in our CUDA implementation.')  

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.ratio = ratio
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points) 
        self.value_proj = nn.Linear(d_model, int(d_model * ratio))
        self.output_proj = nn.Linear(int(d_model * ratio), d_model)

        self._reset_parameters()   

    def _reset_parameters(self):    
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)    
        constant_(self.value_proj.bias.data, 0.)   
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)   
  
    def forward(self, query, reference_points, input_flatten, input_spatial_shapes,   
                input_level_start_index, sampling_offsets,     
                input_padding_mask=None):    
        """
        :param query                       (N, Length_{query}, C)     
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area 
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes     
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]    
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements     
 
        :return output                     (N, Length_{query}, C)  
        """
  
        N, Len_q, _ = query.shape  
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] *     
                input_spatial_shapes[:, 1]).sum() == Len_in
    
        value = self.value_proj(input_flatten)  
        if input_padding_mask is not None:  
            value = value.masked_fill(input_padding_mask[..., None], float(0))    

        value = value.view(N, Len_in, self.n_heads,
                           int(self.ratio * self.d_model) // self.n_heads)     
        attention_weights = self.attention_weights(query).view(  
            N, Len_q, self.n_heads, self.n_levels * self.n_points)   
        attention_weights = F.softmax(attention_weights, -1).\
            view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
   
        if reference_points.shape[-1] == 2:   
            offset_normalizer = torch.stack(    
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)  
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = sampling_locations.contiguous()    
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            sampling_locations = sampling_locations.contiguous()
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'
                .format(reference_points.shape[-1]))

        output = MSDeformAttnFunction.apply(value, input_spatial_shapes, input_level_start_index,  
                                            sampling_locations, attention_weights, self.im2col_step) 
        output = self.output_proj(output)  
        return output   
     
# SFS Module:
class SFSFusion(nn.Module):
    """ 
    Spiral-Aware MSDeformAttn.    

    Inputs:
        - query_feat: [B, C, H1, W1], larger scale feature maps
        - key_feat:   [B, C, H2, W2], smaller scale feature maps
    Output:  
        - out:   [B, C, H1, W1]    
    """
    def __init__(self, in_dim, out_dim, n_heads=8, n_points=4):     
        super().__init__()
        dim = out_dim     
        self.dim = dim  
        self.n_heads = n_heads
        self.n_points = n_points     
     
        self.query_Conv = nn.Sequential(    
            nn.Conv2d(in_dim[0], dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),    
            nn.ReLU(inplace=True)  
        )   
        self.key_Conv = nn.Sequential(   
            nn.Conv2d(in_dim[1], dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),     
            nn.ReLU(inplace=True)     
        )  
  
        self.shared_offsets_residual = nn.Parameter(torch.zeros(n_heads, n_points, 2))
        # generate uniform spiral-aware sampling pattern, and register it as buffer
        fixed_bias = generate_structured_grid(n_heads, n_points, n_levels=1, base_radius=1.0, radius_step=1.0)   
        self.register_buffer("offset_base", fixed_bias.view(1, 1, n_heads, 1, n_points, 2))  

        # LayerNorm on flattened features
        self.query_norm = nn.LayerNorm(dim) 
        self.key_norm = nn.LayerNorm(dim) 
        self.out_norm = nn.LayerNorm(dim)
 
        self.attn = MSDeformAttn_for_sfs(
            d_model=dim,  
            n_levels=1,     
            n_heads=n_heads,
            n_points=n_points    
        )
     
    def forward(self, x) -> Tensor:    
        query_feat, key_feat = x

        query_feat = self.query_Conv(query_feat)
        key_feat = self.key_Conv(key_feat)    
 
        B, C, H1, W1 = query_feat.shape
        _, _, H2, W2 = key_feat.shape     
    
        offsets_residual = self.shared_offsets_residual   

        shared_offsets = self.offset_base.view(self.n_heads, 1, self.n_points, 2) + offsets_residual.view(self.n_heads, 1, self.n_points, 2)
        offsets = shared_offsets.view(1, 1, self.n_heads, 1, self.n_points, 2).expand(B, H1 * W1, -1, -1, -1, -1)

        # flatten & transpose to [B, HW, C] 
        query = query_feat.flatten(2).transpose(1, 2)       # [B, H1*W1, C]
        kv = key_feat.flatten(2).transpose(1, 2)            # [B, H2*W2, C]    
        query = self.query_norm(query)     
        kv = self.key_norm(kv)     
     
        spatial_shapes = torch.tensor([[H2, W2]], device=key_feat.device, dtype=torch.long)
        level_start_index = torch.tensor([0], device=key_feat.device, dtype=torch.long)     
   
        # generate normalized reference points for each query position, both shapes: [H1, W1]    
        grid_y, grid_x = torch.meshgrid(     
            torch.linspace(0.5 / H1, 1 - 0.5 / H1, H1, device=query_feat.device),
            torch.linspace(0.5 / W1, 1 - 0.5 / W1, W1, device=query_feat.device),
            indexing='ij'  
        )
        reference_points = torch.stack((grid_x, grid_y), -1)  # [H1, W1, 2]
        reference_points = reference_points.view(1, H1 * W1, 1, 2).repeat(B, 1, 1, 1)  # [B, H1*W1, 1, 2] 

        # run deformable attention
        attn = self.attn(
            query=query,
            reference_points=reference_points,  
            input_flatten=kv,
            input_spatial_shapes=spatial_shapes,     
            input_level_start_index=level_start_index,
            sampling_offsets=offsets  
        )  # [B, H1*W1, C]
   
        out = query + query * attn     
        out = self.out_norm(out).transpose(1, 2).reshape(B, C, H1, W1)     
        return out     

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"   
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel_1, channel_2, height, width = 1, 32, 16, 32, 32
    ouc_channel = 32
    inputs_1 = torch.randn((batch_size, channel_1, height * 2, width * 2)).to(device)     
    inputs_2 = torch.randn((batch_size, channel_2, height, width)).to(device)

    # 此模块需要编译,详细编译命令在 compile_module/SFS_MSDeformAttn   
    # 对于Linux用户可以直接运行上述的make.sh文件
    # 对于Windows用户需要逐行执行make.sh文件里的内容
    module = SFSFusion([channel_1, channel_2], ouc_channel).to(device)   
     
    # inputs_1 为浅层特征图, inputs_2 为深层特征图, 举个例子：inputs_1为P3,inputs_2为P4
    # 可以参考engine/extre_module/custom_nn/featurefusion/mpca.py的教程     
    outputs = module([inputs_1, inputs_2])
    print(GREEN + f'inputs1.size:{inputs_1.size()} inputs2.size:{inputs_2.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)     
    flops, macs, _ = calculate_flops(model=module,  
                                     args=[[inputs_1, inputs_2]],
                                     output_as_string=True,   
                                     output_precision=4,
                                     print_detailed=True)    
    print(RESET)