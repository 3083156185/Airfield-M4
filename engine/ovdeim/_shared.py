import torch
import torch.nn.functional as F
from torchvision.ops.boxes import box_area     

     
def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clip(min=0.0, max=1.0)    
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))   
   

def bias_init_with_prob(prior_prob: float = 0.01) -> float: 
    return float(-torch.log(torch.tensor((1 - prior_prob) / prior_prob)))


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    return torch.stack(    
        [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)],     
        dim=-1,
    )    


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)], dim=-1) 

  
def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):     
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)   

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])   
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
  
    wh = (rb - lt).clamp(min=0) 
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter    
    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    # assert (boxes1[:, 2:] >= boxes1[:, :2]).all(), f"boxes1: {boxes1}"
    # assert (boxes2[:, 2:] >= boxes2[:, :2]).all(), f"boxes2: {boxes2}"
    iou, union = box_iou(boxes1, boxes2)   

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]  
    return iou - (area - union) / area
