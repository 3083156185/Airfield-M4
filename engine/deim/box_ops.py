"""   
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/util/box_ops.py
"""  

import math 
import numpy as np 
import torch
from torch import Tensor  
from torchvision.ops.boxes import box_area     


def box_cxcywh_to_xyxy(x):  
    x_c, y_c, w, h = x.unbind(-1) 
    b = [(x_c - 0.5 * w.clamp(min=0.0)), (y_c - 0.5 * h.clamp(min=0.0)),     
         (x_c + 0.5 * w.clamp(min=0.0)), (y_c + 0.5 * h.clamp(min=0.0))]  
    return torch.stack(b, dim=-1)  
    
 
def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    x0, y0, x1, y1 = x.unbind(-1)  
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)   

    
# modified from torchvision to also return the union     
def box_iou(boxes1: Tensor, boxes2: Tensor): 
    area1 = box_area(boxes1) 
    area2 = box_area(boxes2) 
 
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]     
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]    

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter  

    iou = inter / union 
    return iou, union  
   
def shrink_boxes(boxes, ratio=1.5):
    """
    按比例缩小边界框（保持中心点不变） 
    
    Args:
        boxes (torch.Tensor | np.ndarray): 边界框坐标，形状为 [n, 4]，格式为 (x1, y1, x2, y2)
        ratio (float): 缩放比例，默认 0.7 表示缩小到原来的 70%
  
    Returns:    
        torch.Tensor | np.ndarray: 缩小后的边界框，形状为 [n, 4]，格式为 (x1, y1, x2, y2)  
   
    Example: 
        >>> boxes = torch.tensor([[10, 20, 50, 60], [100, 100, 200, 200]])    
        >>> shrunk = shrink_boxes(boxes, ratio=0.5)     
        >>> print(shrunk)
        tensor([[ 20.,  30.,  40.,  50.],
                [125., 125., 175., 175.]])     
    """ 
    assert boxes.shape[-1] == 4, f"Expected last dimension to be 4, but got shape {boxes.shape}"   
     
    # 计算中心点坐标
    cx = (boxes[..., 0] + boxes[..., 2]) / 2  # x center   
    cy = (boxes[..., 1] + boxes[..., 3]) / 2  # y center
    
    # 计算宽高
    w = boxes[..., 2] - boxes[..., 0]  # width     
    h = boxes[..., 3] - boxes[..., 1]  # height
  
    # 计算缩小后的宽高 
    new_w = w * ratio     
    new_h = h * ratio
    
    # 计算缩小后的边界框坐标（保持中心不变）
    if isinstance(boxes, torch.Tensor):     
        shrunk_boxes = torch.empty_like(boxes)
    else:     
        shrunk_boxes = np.empty_like(boxes)
 
    shrunk_boxes[..., 0] = cx - new_w / 2  # x1    
    shrunk_boxes[..., 1] = cy - new_h / 2  # y1
    shrunk_boxes[..., 2] = cx + new_w / 2  # x2    
    shrunk_boxes[..., 3] = cy + new_h / 2  # y2     
  
    return shrunk_boxes     
 
def generalized_box_iou(boxes1, boxes2):   
    """ 
    Generalized IoU from https://giou.stanford.edu/     

    The boxes should be in [x0, y0, x1, y1] format   
  
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)  
    """
    # degenerate boxes gives inf / nan results   
    # so do an early check    
    # assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    # assert (boxes2[:, 2:] >= boxes2[:, :2]).all()    
    iou, union = box_iou(boxes1, boxes2)
  
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1] 

    return iou - (area - union) / area
    
def generalized_box_inner_iou(boxes1, boxes2, ratio=2.0):
    """
    Generalized IoU from https://giou.stanford.edu/    

    The boxes should be in [x0, y0, x1, y1] format
     
    Returns a [N, M] pairwise matrix, where N = len(boxes1)     
    and M = len(boxes2)  
    """
    # degenerate boxes gives inf / nan results   
    # so do an early check
    # assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    # assert (boxes2[:, 2:] >= boxes2[:, :2]).all()   
    # boxes1 = shrink_boxes(boxes1, ratio)
    boxes2 = shrink_boxes(boxes2, ratio)    

    iou, union = box_iou(boxes1, boxes2)
 
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])   

    wh = (rb - lt).clamp(min=0)  # [N,M,2]   
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area
  
    
BBOX_IOU_TYPES = (
    'iou',   
    'giou',
    'diou',
    'ciou',    
    'eiou', 
    'siou',
    'shapeiou',
    'piou', 
    'piou2',    
)  
     
  
def normalize_bbox_iou_type(iou_type: str) -> str:    
    if not isinstance(iou_type, str):
        raise TypeError(f'iou_type must be a string, but got {type(iou_type)!r}.')
    iou_type = iou_type.lower()
    if iou_type not in BBOX_IOU_TYPES: 
        raise ValueError(f'Unsupported iou_type "{iou_type}". Supported: {BBOX_IOU_TYPES}.')
    return iou_type
  

def bbox_iou_aligned_xyxy( 
    boxes1: Tensor,
    boxes2: Tensor,
    iou_type: str='giou',   
    eps: float=1e-7,
    scale: float=0.0,    
    Lambda: float=1.3,
) -> Tensor:   
    """  
    Compute aligned IoU variants between ``boxes1`` and ``boxes2`` in xyxy format.   

    ``boxes1`` and ``boxes2`` must both be tensors with shape [N, 4]. 
    The returned tensor has shape [N]. 
    """ 
    if boxes1.ndim != 2 or boxes2.ndim != 2 or boxes1.shape != boxes2.shape or boxes1.shape[-1] != 4:
        raise ValueError(f'boxes1/boxes2 must both be [N,4] with same shape, got {boxes1.shape=} and {boxes2.shape=}.') 

    iou_type = normalize_bbox_iou_type(iou_type)

    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1.unbind(-1)    
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2.unbind(-1)   
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    inter = (torch.minimum(b1_x2, b2_x2) - torch.maximum(b1_x1, b2_x1)).clamp(min=0) * \
            (torch.minimum(b1_y2, b2_y2) - torch.maximum(b1_y1, b2_y1)).clamp(min=0)  
    union = w1 * h1 + w2 * h2 - inter + eps    
    iou = inter / union
  
    if iou_type == 'iou':  
        return iou

    cw = torch.maximum(b1_x2, b2_x2) - torch.minimum(b1_x1, b2_x1)
    ch = torch.maximum(b1_y2, b2_y2) - torch.minimum(b1_y1, b2_y1)

    if iou_type == 'giou':
        c_area = cw * ch + eps 
        return iou - (c_area - union) / c_area     
     
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
 
    if iou_type == 'diou':
        return iou - rho2 / c2
    
    if iou_type == 'ciou':
        v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)     
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - (rho2 / c2 + v * alpha)

    if iou_type == 'eiou':     
        rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2    
        rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2    
        cw2 = cw ** 2 + eps 
        ch2 = ch ** 2 + eps
        return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2)
    
    if iou_type == 'siou':   
        s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps
        s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps
        sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)     
        sin_alpha_1 = torch.abs(s_cw) / sigma
        sin_alpha_2 = torch.abs(s_ch) / sigma   
        threshold = math.sqrt(2) / 2   
        sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)  
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
        rho_x = (s_cw / cw) ** 2
        rho_y = (s_ch / ch) ** 2     
        gamma = angle_cost - 2
        distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)   
        omiga_w = torch.abs(w1 - w2) / torch.maximum(w1, w2)
        omiga_h = torch.abs(h1 - h2) / torch.maximum(h1, h2)     
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)     
        return iou - 0.5 * (distance_cost + shape_cost) + eps

    if iou_type == 'shapeiou':
        ww = 2 * torch.pow(w2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
        hh = 2 * torch.pow(h2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
        c2 = cw ** 2 + ch ** 2 + eps 
        center_distance_x = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2) / 4
        center_distance_y = ((b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4   
        center_distance = hh * center_distance_x + ww * center_distance_y
        distance = center_distance / c2

        omiga_w = hh * torch.abs(w1 - w2) / torch.maximum(w1, w2) 
        omiga_h = ww * torch.abs(h1 - h2) / torch.maximum(h1, h2)
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)     
        return iou - distance - 0.5 * shape_cost    

    dw1 = torch.abs(torch.minimum(b1_x2, b1_x1) - torch.minimum(b2_x2, b2_x1))   
    dw2 = torch.abs(torch.maximum(b1_x2, b1_x1) - torch.maximum(b2_x2, b2_x1))
    dh1 = torch.abs(torch.minimum(b1_y2, b1_y1) - torch.minimum(b2_y2, b2_y1))
    dh2 = torch.abs(torch.maximum(b1_y2, b1_y1) - torch.maximum(b2_y2, b2_y1))
    P = ((dw1 + dw2) / (torch.abs(w2) + eps) + (dh1 + dh2) / (torch.abs(h2) + eps)) / 4  
    piou_v1 = 2 - iou - torch.exp(-P ** 2)
    if iou_type == 'piou':   
        return 1 - piou_v1
    
    q = torch.exp(-P)    
    x = q * Lambda 
    return 1 - 3 * x * torch.exp(-x ** 2) * piou_v1
     

def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.
   
    Returns a [N, 4] tensors, with the boxes in xyxy format 
    """   
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)    
    
    h, w = masks.shape[-2:] 
     
    y = torch.arange(0, h, dtype=torch.float)  
    x = torch.arange(0, w, dtype=torch.float) 
    y, x = torch.meshgrid(y, x)
     
    x_mask = (masks * x.unsqueeze(0)) 
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]    
  
    y_mask = (masks * y.unsqueeze(0))    
    y_max = y_mask.flatten(1).max(-1)[0]     
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]
     
    return torch.stack([x_min, y_min, x_max, y_max], 1)    
