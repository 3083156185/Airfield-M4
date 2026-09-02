""" 
DEIM: DETR with Improved Matching for Fast Convergence  
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.     
---------------------------------------------------------------------------------   
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2023 . All Rights Reserved.     
"""

import math    
from typing import List, Tuple, Union
from pathlib import Path

import cv2    
import numpy as np   
import matplotlib.pyplot as plt   
from prettytable import PrettyTable
from scipy.ndimage import gaussian_filter   
from scipy.spatial import KDTree   
    
import torch     
import torch.nn as nn
import torch.nn.functional as F     

  
def inverse_sigmoid(x: torch.Tensor, eps: float=1e-5) -> torch.Tensor:  
    x = x.clip(min=0., max=1.)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))

     
def bias_init_with_prob(prior_prob=0.01):     
    """initialize conv/fc bias value according to a given probability value."""    
    bias_init = float(-math.log((1 - prior_prob) / prior_prob)) 
    return bias_init

     
def deformable_attention_core_func(value, value_spatial_shapes, sampling_locations, attention_weights):   
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]  
        value_spatial_shapes (Tensor|List): [n_levels, 2]
        value_level_start_index (Tensor|List): [n_levels] 
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels, n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels, n_points]
   
    Returns:
        output (Tensor): [bs, Length_{query}, C]    
    """
    bs, _, n_head, c = value.shape
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape    
  
    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.split(split_shape, dim=1) 
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_ 
        value_l_ = value_list[level].flatten(2).permute(
            0, 2, 1).reshape(bs * n_head, c, h, w)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2     
        sampling_grid_l_ = sampling_grids[:, :, :, level].permute(
            0, 2, 1, 3, 4).flatten(0, 1) 
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(     
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros', 
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_) 
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(   
        bs * n_head, 1, Len_q, n_levels * n_points)
    output = (torch.stack(    
        sampling_value_list, dim=-2).flatten(-2) *   
              attention_weights).sum(-1).reshape(bs, n_head * c, Len_q)  

    return output.permute(0, 2, 1) 
  


def deformable_attention_core_func_v2(\
    value: torch.Tensor,    
    value_spatial_shapes,     
    sampling_locations: torch.Tensor,  
    attention_weights: torch.Tensor,  
    num_points_list: List[int],
    method='default',  
    value_shape='default',    
    ):   
    """     
    Args:    
        value (Tensor): [bs, value_length, n_head, c]  
        value_spatial_shapes (Tensor|List): [n_levels, 2]    
        value_level_start_index (Tensor|List): [n_levels]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels * n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels * n_points]

    Returns:   
        output (Tensor): [bs, Length_{query}, C]
    """
    # TODO find the version
    if value_shape == 'default':
        bs, n_head, c, _ = value[0].shape
    elif value_shape == 'reshape':   # reshape following RT-DETR
        bs, _, n_head, c = value.shape
        split_shape = [h * w for h, w in value_spatial_shapes]     
        value = value.permute(0, 2, 3, 1).flatten(0, 1).split(split_shape, dim=-1)
    _, Len_q, _, _, _ = sampling_locations.shape

    # sampling_offsets [8, 480, 8, 12, 2]
    if method == 'default':
        sampling_grids = 2 * sampling_locations - 1

    elif method == 'discrete':
        sampling_grids = sampling_locations

    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1) 
    sampling_locations_list = sampling_grids.split(num_points_list, dim=-2)     

    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):    
        value_l = value[level].reshape(bs * n_head, c, h, w)
        sampling_grid_l: torch.Tensor = sampling_locations_list[level]   
    
        if method == 'default':   
            sampling_value_l = F.grid_sample(     
                value_l,    
                sampling_grid_l,     
                mode='bilinear',     
                padding_mode='zeros',   
                align_corners=False)
     
        elif method == 'discrete':     
            # n * m, seq, n, 2
            sampling_coord = (sampling_grid_l * torch.tensor([[w, h]], device=value_l.device) + 0.5).to(torch.int64)
    
            # FIX ME? for rectangle input    
            sampling_coord = sampling_coord.clamp(0, h - 1)
            sampling_coord = sampling_coord.reshape(bs * n_head, Len_q * num_points_list[level], 2) 

            s_idx = torch.arange(sampling_coord.shape[0], device=value_l.device).unsqueeze(-1).repeat(1, sampling_coord.shape[1])   
            sampling_value_l: torch.Tensor = value_l[s_idx, :, sampling_coord[..., 1], sampling_coord[..., 0]] # n l c   
     
            sampling_value_l = sampling_value_l.permute(0, 2, 1).reshape(bs * n_head, c, Len_q, num_points_list[level])    

        sampling_value_list.append(sampling_value_l) 
 
    attn_weights = attention_weights.permute(0, 2, 1, 3).reshape(bs * n_head, 1, Len_q, sum(num_points_list))   
    weighted_sample_locs = torch.concat(sampling_value_list, dim=-1) * attn_weights
    output = weighted_sample_locs.sum(-1).reshape(bs, n_head * c, Len_q)
 
    return output.permute(0, 2, 1)  


def get_activation(act: str, inpace: bool=True):
    """get activation
    """
    if act is None:
        return nn.Identity()

    elif isinstance(act, nn.Module): 
        return act   
 
    act = act.lower()    

    if act == 'silu' or act == 'swish':  
        m = nn.SiLU()

    elif act == 'relu':
        m = nn.ReLU()

    elif act == 'leaky_relu':
        m = nn.LeakyReLU()

    elif act == 'silu':
        m = nn.SiLU()

    elif act == 'gelu':
        m = nn.GELU()
   
    elif act == 'hardsigmoid':
        m = nn.Hardsigmoid()

    else:  
        raise RuntimeError('')
   
    if hasattr(m, 'inplace'):
        m.inplace = inpace
     
    return m

class DensityMapGenerator:
    """密度图生成器 - 支持多种生成策略""" 
     
    def __init__(self, image_size: Tuple[int, int]):
        """
        Args:   
            image_size: (height, width) 图像尺寸    
        """
        self.height, self.width = image_size
    
    def generate_from_boxes(self, 
                           boxes: Union[List, np.ndarray],
                           method: str = 'gaussian', 
                           sigma: float = None,
                           normalize: bool = True) -> np.ndarray:   
        """
        从检测框生成密度图
        
        Args:  
            boxes: 检测框列表，格式是：
                   - [[x_center, y_center, w, h], ...] (xywh格式)
            method: 生成方法 
                   - 'gaussian': 高斯核密度图（推荐）  
                   - 'adaptive_gaussian': 自适应高斯核    
                   - 'point': 点标注密度图    
                   - 'box_mask': 框区域密度图     
            sigma: 高斯核标准差，None时自动计算  
            normalize: 是否归一化使积分等于目标数量
            
        Returns:
            density_map: (H, W) 密度图
        """
        if len(boxes) == 0:
            return np.zeros((self.height, self.width), dtype=np.float32)
        
        boxes = np.array(boxes)     
  
        centers = self._boxes_to_centers(boxes)    
        
        if method == 'gaussian':
            return self._generate_gaussian(centers, sigma, normalize)
        elif method == 'adaptive_gaussian':
            return self._generate_adaptive_gaussian(centers, boxes, normalize)    
        elif method == 'point':   
            return self._generate_point(centers, normalize)  
        elif method == 'box_mask':
            return self._generate_box_mask(boxes, normalize)
        else:
            raise ValueError(f"Unknown method: {method}")
   
    def _boxes_to_centers(self, boxes: np.ndarray) -> np.ndarray:
        """将检测框转换为中心点坐标"""  
        if boxes.shape[1] == 4:
            centers = boxes[:, :2].copy()     
        else: 
            raise ValueError("Boxes should have 4 columns")
        
        return centers
    
    def _generate_gaussian(self, 
                          centers: np.ndarray,    
                          sigma: float = None,
                          normalize: bool = True) -> np.ndarray:    
        """    
        生成固定高斯核密度图（最常用方法）
        
        Args:
            centers: (N, 2) 中心点坐标 [x, y]
            sigma: 高斯核标准差，None时自动设置为15     
            normalize: 是否归一化   
        """
        if sigma is None:     
            sigma = 15  # 默认值，适用于大多数场景   
     
        density_map = np.zeros((self.height, self.width), dtype=np.float32)   
   
        # 为每个中心点生成高斯分布
        for center in centers:
            x, y = int(center[0]), int(center[1])
  
            # 边界检查
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                continue    
    
            # 在该点位置添加一个点
            density_map[y, x] += 1     
 
        # 应用高斯滤波     
        density_map = gaussian_filter(density_map, sigma=sigma, mode='constant')   
        
        # 归一化：使积分等于目标数量
        if normalize and density_map.sum() > 0:  
            density_map = density_map * len(centers) / density_map.sum()
        
        return density_map   
    
    def _generate_adaptive_gaussian(self, 
                                   centers: np.ndarray,
                                   boxes: np.ndarray,     
                                   normalize: bool = True) -> np.ndarray:  
        """   
        生成自适应高斯核密度图（根据目标大小和密度调整sigma）
        这是MCNN、CSRNet等论文中使用的方法   
     
        Args:
            centers: (N, 2) 中心点坐标  
            boxes: (N, 4) 检测框   
            normalize: 是否归一化
        """
        density_map = np.zeros((self.height, self.width), dtype=np.float32)   
     
        if len(centers) == 0:
            return density_map    
     
        # 使用KD树找到每个点的k近邻
        k = min(3, len(centers))  # 使用3个最近邻 
        tree = KDTree(centers)   
     
        for idx, center in enumerate(centers): 
            x, y = int(center[0]), int(center[1])
            
            # 边界检查
            if x < 0 or x >= self.width or y < 0 or y >= self.height: 
                continue
  
            # 计算自适应sigma 
            if len(centers) > 1:
                distances, _ = tree.query(center, k=k+1)  # +1因为包含自己 
                avg_distance = np.mean(distances[1:])  # 排除自己
                sigma = avg_distance * 0.3  # 经验系数
            else:    
                sigma = 15  # 只有一个目标时使用默认值    
            
            # 也可以考虑目标框的大小
            box = boxes[idx]
            if boxes.shape[1] == 4:  
                # 计算框的面积作为参考
                # xywh格式     
                box_size = np.sqrt(box[2] * box[3])
  
                # 结合距离和框大小
                sigma = max(sigma, box_size * 0.5)    
            
            # 限制sigma范围     
            sigma = np.clip(sigma, 3, 50)
            
            # 生成高斯核     
            size = int(6 * sigma)  # 3-sigma原则
            if size % 2 == 0:
                size += 1
    
            # 创建高斯核 
            gaussian_kernel = self._create_gaussian_kernel(size, sigma)    
            
            # 计算粘贴位置
            half_size = size // 2
            y_start = max(0, y - half_size)  
            y_end = min(self.height, y + half_size + 1)
            x_start = max(0, x - half_size)     
            x_end = min(self.width, x + half_size + 1)  
   
            # 计算核的对应区域
            ky_start = half_size - (y - y_start) 
            ky_end = half_size + (y_end - y)
            kx_start = half_size - (x - x_start)
            kx_end = half_size + (x_end - x)     
            
            # 添加到密度图    
            density_map[y_start:y_end, x_start:x_end] += \
                gaussian_kernel[ky_start:ky_end, kx_start:kx_end]
    
        # 归一化
        if normalize and density_map.sum() > 0:
            density_map = density_map * len(centers) / density_map.sum()
  
        return density_map
    
    def _create_gaussian_kernel(self, size: int, sigma: float) -> np.ndarray:
        """创建2D高斯核"""
        ax = np.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        return kernel / kernel.sum()  
    
    def _generate_point(self, centers: np.ndarray, normalize: bool = True) -> np.ndarray:     
        """生成点标注密度图（简单方法，不推荐用于训练）""" 
        density_map = np.zeros((self.height, self.width), dtype=np.float32)
  
        for center in centers:
            x, y = int(center[0]), int(center[1])    
            if 0 <= x < self.width and 0 <= y < self.height:
                density_map[y, x] = 1.0
        
        return density_map   
    
    def _generate_box_mask(self, boxes: np.ndarray, normalize: bool = True) -> np.ndarray:
        """生成框区域密度图（将密度均匀分布在框内）"""
        density_map = np.zeros((self.height, self.width), dtype=np.float32)
        
        for box in boxes:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[0] + box[2])     
            y2 = int(box[1] + box[3])     
   
            # 边界裁剪  
            x1 = max(0, x1)    
            y1 = max(0, y1)
            x2 = min(self.width, x2)
            y2 = min(self.height, y2) 
            
            # 在框内均匀分布密度     
            area = (x2 - x1) * (y2 - y1)
            if area > 0:
                density_map[y1:y2, x1:x2] += 1.0 / area   
        
        if normalize and density_map.sum() > 0: 
            density_map = density_map * len(boxes) / density_map.sum() 
 
        return density_map
    

def visualize_density_map(image: np.ndarray,  
                         density_map: np.ndarray, 
                         boxes: np.ndarray = None,
                         save_path: str = None):   
    """  
    可视化密度图
    
    Args:
        image: 原始图像 (H, W, 3)    
        density_map: 密度图 (H, W)  
        boxes: 检测框（可选）  
        save_path: 保存路径（可选）  
    """   
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
     
    # 原始图像 
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')     
    axes[0].axis('off')
 
    # 带框的图像   
    img_with_boxes = image.copy()
    if boxes is not None:
        for box in boxes:   
            if len(box) == 4:
                # xywh格式     
                x1 = int(box[0] - box[2] / 2) 
                y1 = int(box[1] - box[3] / 2) 
                x2 = int(box[0] + box[2] / 2)
                y2 = int(box[1] + box[3] / 2)
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)     
    
    axes[1].imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Bounding Boxes (Count: {len(boxes) if boxes is not None else 0})')
    axes[1].axis('off')
    
    # 密度图   
    im = axes[2].imshow(density_map, cmap='jet')     
    axes[2].set_title(f'Density Map (Sum: {density_map.sum():.2f})')  
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:   
        plt.savefig(save_path, dpi=150, bbox_inches='tight')  
        print(f"Saved to {save_path}")

def visualize_density_map_only(density_map: np.ndarray, save_path: str = None):
    """
    可视化密度图（仅显示密度图）   
    
    Args:    
        density_map: 密度图 (H, W)
        save_path: 保存路径（可选）    
    """
    plt.figure(figsize=(8, 6))
    im = plt.imshow(density_map, cmap='jet')    
    plt.title(f'Density Map (Sum: {density_map.sum():.2f})')   
    plt.axis('off')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    plt.tight_layout()  
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")   

def point_sample(input, point_coords, **kwargs):     
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.   
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args: 
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid. 
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.  
 
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains     
            features for points in `point_coords`. The features are obtained via bilinear 
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.     
    """ 
    add_dim = False     
    if point_coords.dim() == 3:    
        add_dim = True 
        point_coords = point_coords.unsqueeze(2)    
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:  
        output = output.squeeze(3) 
    return output 

def get_uncertain_point_coords_with_randomness(  
    coarse_logits, uncertainty_func, num_points, oversample_ratio=3, importance_sample_ratio=0.75
):     
    """    
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties   
        are calculated for each point using 'uncertainty_func' function that takes point's logit    
        prediction as input.
    See PointRend paper for details.
  
    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.    
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.    
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.     

    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points. 
    """  
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0     
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False) 
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between 
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,  
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)    
    num_uncertain_points = int(importance_sample_ratio * num_points)    
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]     
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view( 
        num_boxes, num_uncertain_points, 2     
    ) 
    if num_random_points > 0: 
        point_coords = torch.cat(
            [  
                point_coords, 
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,   
        ) 
    return point_coords 

def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the    
        foreground class in `classes`. 
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or  
            class-agnostic, where R is the total number of predicted masks in all images and C is   
            the number of foreground classes. The values are logits.     
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1  
    gt_class_logits = logits.clone()  
    return -(torch.abs(gt_class_logits)) 
   
def dice_loss(  
        inputs: torch.Tensor,     
        targets: torch.Tensor, 
        num_masks: float,   
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:    
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.  
        targets: A float tensor with the same shape as inputs. Stores the binary    
                 classification label for each element in inputs     
                (0 for the negative class and 1 for the positive class).     
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)     
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)   
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks 

  
dice_loss_jit = torch.jit.script(
    dice_loss  
)  # type: torch.jit.ScriptModule

 
def sigmoid_ce_loss(
        inputs: torch.Tensor,    
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:     
        inputs: A float tensor of arbitrary shape.     
                The predictions for each example. 
        targets: A float tensor with the same shape as inputs. Stores the binary 
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).   
    Returns:
        Loss tensor  
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")    
     
    return loss.mean(1).sum() / num_masks

 
sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule
   
class MaskVisualizer: 
    """独立的mask可视化工具"""
    
    def __init__(self, save_dir='./vis_masks', auto_increment=True, clear_existing=True):
        """   
        Args:   
            save_dir: 保存目录  
            auto_increment: 是否自动递增文件名   
        """   
        self.save_dir = Path(save_dir)    
        # 如果目录存在且需要清空
        # if self.save_dir.exists() and clear_existing:    
        #     import shutil    
        #     shutil.rmtree(self.save_dir)     
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.auto_increment = auto_increment    
        self.counter = self._get_current_counter()
  
    def _get_current_counter(self):
        """获取当前计数器值"""
        existing_files = list(self.save_dir.glob('mask_comparison_*.png'))    
        if not existing_files:
            return 0 
  
        numbers = []  
        for f in existing_files:
            try:
                num_str = f.stem.split('_')[-1]  
                numbers.append(int(num_str))     
            except (ValueError, IndexError):   
                continue  
        
        return max(numbers) + 1 if numbers else 0 
    
    def get_next_filename(self, prefix='mask_comparison', ext='.png'):  
        """获取下一个文件名"""     
        if self.auto_increment:    
            filename = f'{prefix}_{self.counter:04d}{ext}'    
            self.counter += 1   
        else:
            filename = f'{prefix}{ext}'
        return self.save_dir / filename   
     
    def visualize_single_sample(self, pred_mask, target_mask, box_mask, 
                                target_box=None, save_path=None):   
        """
        可视化单个样本
        
        Args:
            pred_mask: 预测mask [H, W] (logits)
            target_mask: 真实mask [H, W] (0/1)
            box_mask: 边界框mask [H, W] (0/1)    
            target_box: 边界框坐标 [4] (cx, cy, w, h)
            save_path: 保存路径，如果为None则自动生成    
        """
        # 转换为numpy     
        if torch.is_tensor(pred_mask):
            pred_mask = torch.sigmoid(pred_mask).detach().cpu().numpy()
        if torch.is_tensor(target_mask):  
            target_mask = target_mask.detach().cpu().numpy()
        if torch.is_tensor(box_mask):
            box_mask = box_mask.detach().cpu().numpy()
 
        pred_binary = (pred_mask > 0.5).astype(np.float32)
   
        # 创建图形
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        
        # 1. 预测概率 
        im0 = axes[0].imshow(pred_mask, cmap='jet', vmin=0, vmax=1)  
        axes[0].set_title('Prediction (Probability)', fontsize=12)
        axes[0].axis('off')     
        plt.colorbar(im0, ax=axes[0], fraction=0.046)
        
        # 2. 二值化预测
        axes[1].imshow(pred_binary, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Prediction (Binary)', fontsize=12)
        axes[1].axis('off')    
        
        # 3. 真实mask   
        axes[2].imshow(target_mask, cmap='gray', vmin=0, vmax=1)     
        axes[2].set_title('Ground Truth', fontsize=12)
        axes[2].axis('off')
 
        # 4. 边界框区域
        axes[3].imshow(box_mask, cmap='gray', vmin=0, vmax=1)
        axes[3].set_title('Box Region', fontsize=12)    
        axes[3].axis('off')
        
        # 5. 叠加对比
        overlay = self._create_overlay(pred_binary, target_mask, box_mask) 
        axes[4].imshow(overlay)
        
        # 计算指标    
        metrics = self._calculate_metrics(pred_binary, target_mask, box_mask)  
        title = f"Overlay (Green=TP, Red=FP, Blue=FN)\n"
        title += f"IoU: {metrics['iou']:.3f} | "  
        title += f"Precision: {metrics['precision']:.3f} | "   
        title += f"Recall: {metrics['recall']:.3f}"     
        axes[4].set_title(title, fontsize=10)   
        axes[4].axis('off')
   
        # 绘制边界框
        if target_box is not None:
            self._draw_box_on_axes(axes, target_box, target_mask.shape) 
        
        plt.tight_layout()
 
        # 保存
        if save_path is None:     
            save_path = self.get_next_filename()     
        plt.savefig(save_path, dpi=150, bbox_inches='tight') 
        plt.close()
        
        return save_path, metrics
 
    def visualize_batch(self, pred_masks, target_masks, box_masks, 
                        target_boxes=None, max_samples=8, save_path=None):
        """
        可视化一个batch的样本
     
        Args: 
            pred_masks: [N, H, W] 
            target_masks: [N, H, W]
            box_masks: [N, H, W]
            target_boxes: [N, 4]     
            max_samples: 最多显示的样本数
            save_path: 保存路径    
        """  
        N = min(pred_masks.shape[0], max_samples)   
  
        # 转换为numpy
        pred_masks_np = torch.sigmoid(pred_masks[:N]).detach().cpu().numpy() 
        target_masks_np = target_masks[:N].detach().cpu().numpy()   
        box_masks_np = box_masks[:N].detach().cpu().numpy()
     
        pred_binary = (pred_masks_np > 0.5).astype(np.float32)
        
        # 创建图形
        fig, axes = plt.subplots(N, 5, figsize=(20, 4 * N)) 
        if N == 1:     
            axes = axes[np.newaxis, :]
  
        all_metrics = []
  
        for i in range(N):
            # 1. 预测概率
            im = axes[i, 0].imshow(pred_masks_np[i], cmap='jet', vmin=0, vmax=1)
            axes[i, 0].set_title(f'Sample {i}\nPrediction (Prob)', fontsize=10)
            axes[i, 0].axis('off')  
   
            # 2. 二值化预测
            axes[i, 1].imshow(pred_binary[i], cmap='gray', vmin=0, vmax=1) 
            axes[i, 1].set_title('Prediction (Binary)', fontsize=10)
            axes[i, 1].axis('off')
            
            # 3. 真实mask
            axes[i, 2].imshow(target_masks_np[i], cmap='gray', vmin=0, vmax=1)   
            axes[i, 2].set_title('Ground Truth', fontsize=10)  
            axes[i, 2].axis('off')
            
            # 4. 边界框区域
            axes[i, 3].imshow(box_masks_np[i], cmap='gray', vmin=0, vmax=1)
            axes[i, 3].set_title('Box Region', fontsize=10)  
            axes[i, 3].axis('off')
            
            # 5. 叠加对比
            overlay = self._create_overlay(pred_binary[i], target_masks_np[i], box_masks_np[i]) 
            axes[i, 4].imshow(overlay)
  
            # 计算指标
            metrics = self._calculate_metrics(pred_binary[i], target_masks_np[i], box_masks_np[i])  
            all_metrics.append(metrics)    
 
            title = f"IoU: {metrics['iou']:.3f}\n"
            title += f"P: {metrics['precision']:.3f} R: {metrics['recall']:.3f}"     
            axes[i, 4].set_title(title, fontsize=10)
            axes[i, 4].axis('off')   
            
            # 绘制边界框
            if target_boxes is not None:  
                box = target_boxes[i].cpu().numpy()    
                self._draw_box_on_row(axes[i], box, target_masks_np[i].shape)
     
        # 添加总体统计
        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}   
        fig.suptitle(   
            f"Average - IoU: {avg_metrics['iou']:.3f} | "    
            f"Precision: {avg_metrics['precision']:.3f} | "
            f"Recall: {avg_metrics['recall']:.3f}",   
            fontsize=14, y=0.995
        )
     
        plt.tight_layout()    
        
        # 保存
        if save_path is None:    
            save_path = self.get_next_filename()  
        plt.savefig(save_path, dpi=150, bbox_inches='tight')   
        plt.close()  
        
        print(f"✅ Saved visualization to: {save_path}") 
        return save_path, all_metrics   
    
    def _create_overlay(self, pred_binary, target_mask, box_mask):
        """创建叠加图像"""
        overlay = np.zeros((*pred_binary.shape, 3)) 
        
        box_region = box_mask > 0.5
        pred_region = pred_binary > 0.5
        target_region = target_mask > 0.5
    
        # True Positive (绿色)
        tp = box_region & pred_region & target_region 
        overlay[tp] = [0, 1, 0]
   
        # False Positive (红色)    
        fp = box_region & pred_region & (~target_region)
        overlay[fp] = [1, 0, 0]
  
        # False Negative (蓝色) 
        fn = box_region & (~pred_region) & target_region    
        overlay[fn] = [0, 0, 1]
        
        # True Negative (深灰色)
        tn = box_region & (~pred_region) & (~target_region)    
        overlay[tn] = [0.3, 0.3, 0.3]     
        
        # Box外区域 (浅灰色) 
        overlay[~box_region] = [0.8, 0.8, 0.8]
        
        return overlay
    
    def _calculate_metrics(self, pred_binary, target_mask, box_mask):   
        """计算评估指标"""
        box_region = box_mask > 0.5
        pred_region = pred_binary > 0.5  
        target_region = target_mask > 0.5
        
        # 只在box区域内计算
        pred_in_box = pred_region & box_region 
        target_in_box = target_region & box_region 
  
        tp = (pred_in_box & target_in_box).sum()   
        fp = (pred_in_box & (~target_in_box)).sum()
        fn = ((~pred_in_box) & target_in_box).sum()  
        tn = ((~pred_in_box) & (~target_in_box) & box_region).sum()    
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0   
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0 
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {   
            'precision': float(precision),    
            'recall': float(recall),     
            'iou': float(iou),     
            'f1': float(f1),
            'tp': int(tp),   
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)  
        }
  
    def _draw_box_on_axes(self, axes, box, shape): 
        """在所有子图上绘制边界框"""
        from matplotlib.patches import Rectangle
    
        cx, cy, w, h = box     
        H, W = shape 
    
        x1 = (cx - w / 2) * W   
        y1 = (cy - h / 2) * H
        width = w * W
        height = h * H
        
        for ax in axes:
            rect = Rectangle((x1, y1), width, height,
                           linewidth=2, edgecolor='yellow',   
                           facecolor='none', linestyle='--')     
            ax.add_patch(rect)
     
    def _draw_box_on_row(self, axes_row, box, shape):    
        """在一行子图上绘制边界框""" 
        from matplotlib.patches import Rectangle
     
        cx, cy, w, h = box
        H, W = shape
    
        x1 = (cx - w / 2) * W     
        y1 = (cy - h / 2) * H    
        width = w * W  
        height = h * H 
        
        for ax in axes_row:  
            rect = Rectangle((x1, y1), width, height,  
                           linewidth=2, edgecolor='yellow',  
                           facecolor='none', linestyle='--')
            ax.add_patch(rect)   
  
def coco_evaluator_per_class(coco_evaluator, iouType):
    results_per_category = coco_evaluator.coco_eval[iouType].summarize_per_class()     

    model_metrice_table = PrettyTable()
    model_metrice_table.title = f"COCO Metrice({iouType})"   
    if getattr(coco_evaluator, 'using_vtiny_metrice', False):    
        model_metrice_table.field_names = ['category', 'AP', 'AP_50', 'AP_75', 'AP_vt', 'AP_t', 'AP_s', 'AP_m', 'AP_l']
    elif getattr(coco_evaluator, 'using_tiny_metrice', False):     
        model_metrice_table.field_names = ['category', 'AP', 'AP_50', 'AP_75', 'AP_t', 'AP_s', 'AP_m', 'AP_l']
    else:  
        model_metrice_table.field_names = ['category', 'AP', 'AP_50', 'AP_75', 'AP_s', 'AP_m', 'AP_l']
    for data in results_per_category: 
        model_metrice_table.add_row(list(data))
 
    # 提取数值列（跳过第一列的类别名称）
    numeric_data = [list(data)[1:] for data in results_per_category]
  
    # 计算每列的平均值，排除 -1 的值
    avg_values = []
    for col_idx in range(len(numeric_data[0])): 
        col_values = [float(row[col_idx]) for row in numeric_data if float(row[col_idx]) != -1]  # 排除 -1
        if col_values:     
            avg_value = sum(col_values) / len(col_values)
            avg_values.append(round(avg_value, 3))  # 保留3位小数
        else:
            avg_values.append(-1)  # 如果所有值都是 -1，则平均值也为 -1    
    
    # 添加平均值行     
    all_row = ['all'] + avg_values     
    model_metrice_table.add_row(all_row)

    return model_metrice_table
