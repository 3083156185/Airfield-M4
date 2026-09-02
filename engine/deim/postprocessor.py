"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F     

import torchvision     
import numpy as np   

from ..core import register

 
__all__ = ['PostProcessor', 'SegPostProcessor', 'DQPostProcessor']
  
 
def mod(a, b):
    out = a - a // b * b     
    return out
 

def scale_boxes_to_original_image(  
    boxes: torch.Tensor,
    orig_target_sizes: torch.Tensor,  
    resize_pad: dict | None = None,
) -> torch.Tensor: 
    bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy') 
  
    if resize_pad is None: 
        return bbox_pred * orig_target_sizes.repeat(1, 2).unsqueeze(1)

    input_sizes = resize_pad['size'].to(device=boxes.device, dtype=boxes.dtype) 
    padding = resize_pad['padding'].to(device=boxes.device, dtype=boxes.dtype)
  
    padded_boxes = bbox_pred * input_sizes.repeat(1, 2).unsqueeze(1)
    left_top = padding[:, [0, 1]].unsqueeze(1)   
    resized_sizes = input_sizes - padding[:, [0, 1]] - padding[:, [2, 3]]     
    scales = resized_sizes / orig_target_sizes.to(device=boxes.device, dtype=boxes.dtype)   
 
    original_boxes = (padded_boxes - left_top.repeat(1, 1, 2)) / scales.repeat(1, 2).unsqueeze(1)
    max_xyxy = orig_target_sizes.to(device=boxes.device, dtype=boxes.dtype).repeat(1, 2).unsqueeze(1) 
    return original_boxes.clamp(min=0).minimum(max_xyxy)    
    
 
@register()
class PostProcessor(nn.Module):
    # 共享的参数列表
    __share__ = [  
        'num_classes',
        'use_focal_loss',
        'num_top_queries',     
        'remap_mscoco_category'
    ]  

    def __init__(
        self,
        num_classes=80,  # 目标类别数，默认 80（如 COCO 数据集）
        use_focal_loss=True,  # 是否使用 Focal Loss    
        num_top_queries=300,  # 选择前 num_top_queries 个目标   
        remap_mscoco_category=False  # 是否对 COCO 数据集的类别进行重新映射   
    ) -> None:  
        super().__init__() 
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries     
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category    
        self.deploy_mode = False  # 部署模式标志位，默认关闭
  
    def extra_repr(self) -> str:  
        """
        额外的表示方法，在打印模型结构时显示  
        """    
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}' 

    def forward(self, outputs, orig_target_sizes: torch.Tensor, for_eval=False, resize_pad=None):
        """ 
        前向传播函数，处理检测头的输出，将其转换为最终检测结果。
        :param outputs: 预测的输出字典，包含 'pred_logits'（类别分数）和 'pred_boxes'（边界框）
        :param orig_target_sizes: 原始图像的尺寸，用于恢复边界框到原始图像尺度
        :param for_eval: 是否用于验证结果
        :return: 处理后的检测结果，包含 labels、boxes、scores
        """
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']

        bbox_pred = scale_boxes_to_original_image(boxes, orig_target_sizes, resize_pad)
    
        if self.use_focal_loss:
            # 使用 Focal Loss 计算类别分数
            scores = F.sigmoid(logits)
            # 选择最高的 num_top_queries 个目标
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1) 
            # 计算类别标签
            labels = index % self.num_classes  # 取模计算类别索引
            index = index // self.num_classes  # 计算 box 索引
            # 根据索引选择对应的边界框
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
        else: 
            # 计算 softmax 归一化后的类别分数    
            scores = F.softmax(logits, dim=-1)[:, :, :-1]  # 去掉背景类别   
            scores, labels = scores.max(dim=-1)  # 选择最大分数对应的类别  
            if scores.shape[1] > self.num_top_queries:   
                # 选择最高的 num_top_queries 个目标  
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)    
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1])) 
   
        # 部署模式下，直接返回 labels、boxes、scores   
        if self.deploy_mode:
            return labels, boxes, scores 

        # 如果需要映射 COCO 类别索引   
        if self.remap_mscoco_category: 
            from ..data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)    
    
        # 组织输出结果
        results = []     
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)  
  
        return results

    def deploy(self):  
        """
        设置模型为部署模式（推理模式）
        """   
        self.eval()
        self.deploy_mode = True
        return self
 
@register()
class SegPostProcessor(nn.Module):   
    # 共享的参数列表
    __share__ = [   
        'num_classes',
        'use_focal_loss',
        'num_top_queries',     
        'remap_mscoco_category',
        'eval_spatial_size',     
        'eval_mask_ratio', 
        'score_threshold'     
    ]   

    def __init__(
        self,    
        num_classes=80,  # 目标类别数，默认 80（如 COCO 数据集）
        use_focal_loss=True,  # 是否使用 Focal Loss
        num_top_queries=300,  # 选择前 num_top_queries 个目标
        remap_mscoco_category=False,  # 是否对 COCO 数据集的类别进行重新映射  
        eval_spatial_size=None,   
        eval_mask_ratio=1,
        score_threshold=0.001     
    ) -> None:   
        super().__init__()    
        self.use_focal_loss = use_focal_loss   
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)  
        self.remap_mscoco_category = remap_mscoco_category
        self.eval_spatial_size = eval_spatial_size # h, w  
        self.eval_mask_ratio = eval_mask_ratio
        self.score_threshold = score_threshold
        self.deploy_mode = False  # 部署模式标志位，默认关闭    

    def extra_repr(self) -> str:  
        """
        额外的表示方法，在打印模型结构时显示   
        """
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}, score_threshold={self.score_threshold}'
    
    def forward(self, outputs, orig_target_sizes: torch.Tensor, for_eval=False, resize_pad=None):
        """     
        前向传播函数，处理检测头的输出，将其转换为最终检测结果。   
        :param outputs: 预测的输出字典，包含 'pred_logits'（类别分数）和 'pred_boxes'（边界框）  
        :param orig_target_sizes: 原始图像的尺寸，用于恢复边界框到原始图像尺度
        :param for_eval: 是否用于验证结果  
        :return: 处理后的检测结果，包含 labels、boxes、scores、masks     
        """
        logits, boxes, pred_masks = outputs['pred_logits'], outputs['pred_boxes'], outputs['pred_masks']
 
        bbox_pred = scale_boxes_to_original_image(boxes, orig_target_sizes, resize_pad)  
    
        if self.use_focal_loss:
            # 使用 Focal Loss 计算类别分数 
            scores = F.sigmoid(logits)
            # 选择最高的 num_top_queries 个目标 
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)    
            # 计算类别标签
            labels = index % self.num_classes  # 取模计算类别索引
            index = index // self.num_classes  # 计算 box 索引
            # 根据索引选择对应的边界框
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))

            # 根据索引选择对应的masks  
            # pred_masks: [B, Q, H, W], index: [B, num_top_queries]    
            pred_masks = pred_masks.gather( 
                dim=1, 
                index=index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, pred_masks.shape[-2], pred_masks.shape[-1])
            )  # [B, num_top_queries, H, W]    
        else:   
            # 计算 softmax 归一化后的类别分数
            scores = F.softmax(logits, dim=-1)[:, :, :-1]  # 去掉背景类别
            scores, labels = scores.max(dim=-1)  # 选择最大分数对应的类别
            boxes = bbox_pred
            if scores.shape[1] > self.num_top_queries:
                # 选择最高的 num_top_queries 个目标
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)  
                labels = torch.gather(labels, dim=1, index=index) 
                boxes = torch.gather(bbox_pred, dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
     
                # 根据索引选择对应的masks
                pred_masks = pred_masks.gather(   
                    dim=1,  
                    index=index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, pred_masks.shape[-2], pred_masks.shape[-1])  
                )  # [B, num_top_queries, H, W]    
   
        # 部署模式下，直接返回 labels、boxes、scores
        if self.deploy_mode:
            masks = []   
            for i in range(pred_masks.size(0)): # batch_size dim 
                masks_i = self.process_mask(pred_masks[i], boxes[i], orig_target_sizes[i].tolist(), for_eval)     
                masks.append(masks_i)
            return labels, boxes, scores, masks  

        # 组织输出结果
        results = []
        for i, (lab, box, sco, mask_logits) in enumerate(zip(labels, boxes, scores, pred_masks)):   
            keep = sco > self.score_threshold 
            lab = lab[keep]
            box = box[keep]
            sco = sco[keep]
            mask_logits = mask_logits[keep] 

            mask = self.process_mask(mask_logits, box, orig_target_sizes[i].tolist(), for_eval)   

            # 如果需要映射 COCO 类别索引    
            if self.remap_mscoco_category:  
                from ..data.dataset import mscoco_label2category 
                lab = torch.tensor([mscoco_label2category[int(x.item())] for x in lab])\
                    .to(boxes.device)
     
            result = dict(labels=lab, boxes=box, scores=sco, masks=mask)
            results.append(result)
 
        return results
  
    def deploy(self):
        """     
        设置模型为部署模式（推理模式）
        """
        self.eval()
        self.deploy_mode = True    
        return self
     
    def process_mask(self, pred_masks, boxes, orig_target_sizes, for_eval=False):   
        """
        处理掩码：插值到原始图像尺寸并将边界框外的区域置为0   
        :param pred_masks: 预测的掩码 [num_top_queries, H_mask, W_mask]
        :param boxes: 边界框坐标 [num_top_queries, 4] (xyxy格式)   
        :param orig_target_sizes: 原始图像尺寸 [w, h]
        :return: 处理后的掩码 [num_top_queries, H, W]  
        """ 

        w, h = orig_target_sizes   
        if for_eval:  
            boxes = boxes / torch.from_numpy(np.array([w / (self.eval_spatial_size[1] / self.eval_mask_ratio), h / (self.eval_spatial_size[0] / self.eval_mask_ratio)])).unsqueeze(0).repeat(1, 2).to(boxes.device)   
            w = int(self.eval_spatial_size[1] / self.eval_mask_ratio)    
            h = int(self.eval_spatial_size[0] / self.eval_mask_ratio)

        if pred_masks.shape[0] == 0: 
            return torch.zeros((0, int(h), int(w)), dtype=torch.bool, device=pred_masks.device)   

        # 获取当前 mask 的尺寸
        H_mask, W_mask = pred_masks.shape[-2:]

        # 判断尺寸是否相同
        if H_mask != int(h) or W_mask != int(w):     
            masks_i = F.interpolate(   
                pred_masks.unsqueeze(1),  # [num_top_queries, 1, H_mask, W_mask]     
                size=(int(h), int(w)),     
                mode="bilinear", 
                align_corners=False  
            ).squeeze(1)  # [num_top_queries, H, W]  
        else:  
            masks_i = pred_masks  # 尺寸相同，直接使用
        
        # 二值化掩码     
        masks_i = masks_i > 0.0  
     
        # 将边界框外的区域置为0
        if not self.deploy_mode:
            boxes = boxes.to(device=masks_i.device, dtype=torch.int64)
            x1, y1, x2, y2 = boxes.unbind(dim=1)
            x1 = x1.clamp(0, int(w))  
            y1 = y1.clamp(0, int(h))
            x2 = x2.clamp(0, int(w))
            y2 = y2.clamp(0, int(h))
   
            rows = torch.arange(int(h), device=masks_i.device).view(1, int(h), 1)   
            cols = torch.arange(int(w), device=masks_i.device).view(1, 1, int(w)) 
            crop = (  
                (cols >= x1.view(-1, 1, 1))
                & (cols < x2.view(-1, 1, 1))  
                & (rows >= y1.view(-1, 1, 1))
                & (rows < y2.view(-1, 1, 1))
            )  
            masks_i = masks_i & crop
        
        return masks_i
     
@register()
class DQPostProcessor(nn.Module):
    # 共享的参数列表
    __share__ = [ 
        'num_classes',
        'use_focal_loss', 
        'num_top_queries',
        'remap_mscoco_category'    
    ]     

    def __init__(  
        self,    
        num_classes=80,  # 目标类别数，默认 80（如 COCO 数据集）
        use_focal_loss=True,  # 是否使用 Focal Loss
        num_top_queries=300,  # 选择前 num_top_queries 个目标  
        remap_mscoco_category=False  # 是否对 COCO 数据集的类别进行重新映射 
    ) -> None: 
        super().__init__()     
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False  # 部署模式标志位，默认关闭

    def extra_repr(self) -> str:
        """   
        额外的表示方法，在打印模型结构时显示
        """   
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'

    def forward(self, outputs, orig_target_sizes: torch.Tensor, for_eval=False, resize_pad=None):
        """   
        前向传播函数，处理检测头的输出，将其转换为最终检测结果。
        :param outputs: 预测的输出字典，包含 'pred_logits'（类别分数）和 'pred_boxes'（边界框）   
        :param orig_target_sizes: 原始图像的尺寸，用于恢复边界框到原始图像尺度
        :return: 处理后的检测结果，包含 labels、boxes、scores   
        """
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
     
        if 'num_queries_list' in outputs:
            num_queries_list = outputs['num_queries_list']     
            
            # 创建 mask
            B, N, C = logits.shape
            device = logits.device
            
            # 为每个 batch 创建 mask 
            mask = torch.zeros(B, N, dtype=torch.bool, device=device) 
            for i, qn in enumerate(num_queries_list):
                mask[i, :qn] = True  # 保留前 qn 个 queries
 
            # 扩展 mask 到所有类别
            mask = mask.unsqueeze(-1).expand(B, N, C)  # [B, N, C]   
  
            # 使用 where 替代切片赋值     
            if self.use_focal_loss:     
                logits = torch.where(mask, logits, torch.tensor(-10000.0, device=device))    
            else:     
                logits = torch.where(mask, logits, torch.tensor(0.0, device=device))
   
        bbox_pred = scale_boxes_to_original_image(boxes, orig_target_sizes, resize_pad)
     
        if self.use_focal_loss:
            # 使用 Focal Loss 计算类别分数
            scores = F.sigmoid(logits)
            # 选择最高的 max(num_queries_list) 个目标
            scores, index = torch.topk(scores.flatten(1), max(num_queries_list), dim=-1)
            # 计算类别标签
            labels = index % self.num_classes  # 取模计算类别索引     
            index = index // self.num_classes  # 计算 box 索引  
            # 根据索引选择对应的边界框
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
        else:   
            # 计算 softmax 归一化后的类别分数     
            scores = F.softmax(logits, dim=-1)[:, :, :-1]  # 去掉背景类别 
            scores, labels = scores.max(dim=-1)  # 选择最大分数对应的类别   
            if scores.shape[1] > max(num_queries_list):    
                # 选择最高的 num_top_queries 个目标  
                scores, index = torch.topk(scores, max(num_queries_list), dim=-1)
                labels = torch.gather(labels, dim=1, index=index)     
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))    

        # 部署模式下，直接返回 labels、boxes、scores
        if self.deploy_mode: 
            return labels, boxes, scores
    
        # 如果需要映射 COCO 类别索引
        if self.remap_mscoco_category:
            from ..data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        # 组织输出结果
        results = []
        for lab, box, sco, qn in zip(labels, boxes, scores, num_queries_list):
            # print(qn, lab.size(0))
            result = dict(labels=lab[:qn], boxes=box[:qn], scores=sco[:qn])
            results.append(result)     

        return results
  
    def deploy(self):
        """    
        设置模型为部署模式（推理模式）
        """     
        self.eval()
        self.deploy_mode = True
        return self  
