"""
DEIM: DETR with Improved Matching for Fast Convergence  
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

import torch     
import torch.nn as nn 
import torch.distributed
import torch.nn.functional as F  
import torchvision
    
import copy    
import numpy as np

from .utils import DensityMapGenerator, visualize_density_map, point_sample, \
    get_uncertain_point_coords_with_randomness, calculate_uncertainty, sigmoid_ce_loss_jit, dice_loss_jit, \
    MaskVisualizer   
from .dfine_utils import bbox2distance
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou, bbox_iou_aligned_xyxy, normalize_bbox_iou_type
from ..misc.dist_utils import get_world_size, is_dist_available_and_initialized  
from ..core import register 

from ..logger_module import get_logger   
logger = get_logger(__name__)

RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
SEG_MASK_LOSS_VIS = False
SEG_MASK_STEP_VIS = 200     
   
@register()
class DEIMCriterion(nn.Module):    
    """ This class computes the loss for DEIM.
    """ 
    __share__ = ['num_classes', ]    
    __inject__ = ['matcher', ]    
  
    def __init__(self, \
        matcher,
        weight_dict,
        losses,
        alpha=0.2,     
        gamma=2.0,
        num_classes=80,     
        reg_max=32,   
        boxes_weight_format=None,
        boxes_iou_type='giou',   
        share_matched_indices=False,   
        mal_alpha=None, 
        use_uni_set=True,
        no_weight_vfl_epoch=-1,
        no_weight_vfl_warmup_iter=500,
        ccm_params=None,
        density_recall_penalty=1.1,    
        density_precision_penalty=1.3,
        mask_point_sample_ratio=8
        ):     
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals.
            weight_dict: dict containing as key the names of the losses and as values their relative weight. 
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            num_classes: number of object categories, omitting the special no-object category.
            reg_max (int): Max number of the discrete bins in D-FINE. 
            boxes_weight_format: format for boxes weight (iou, giou, diou, ciou, eiou, siou, shapeiou, piou, piou2).
            boxes_iou_type: IoU type for loss_boxes (iou, giou, diou, ciou, eiou, siou, shapeiou, piou, piou2).
        """     
        super().__init__()
        self.num_classes = num_classes     
        self.matcher = matcher  
        self.weight_dict = weight_dict     
        self.losses = losses  
        self.boxes_weight_format = None if boxes_weight_format is None else normalize_bbox_iou_type(boxes_weight_format)    
        self.boxes_iou_type = normalize_bbox_iou_type(boxes_iou_type)    
        self.share_matched_indices = share_matched_indices    
        self.alpha = alpha
        self.gamma = gamma 
        self.fgl_targets, self.fgl_targets_dn = None, None 
        self.own_targets, self.own_targets_dn = None, None
        self.reg_max = reg_max 
        self.num_pos, self.num_neg = None, None
        self.mal_alpha = mal_alpha
        self.use_uni_set = use_uni_set
        self.no_weight_vfl_epoch = no_weight_vfl_epoch
        self.no_weight_vfl_warmup_iter = no_weight_vfl_warmup_iter
        if hasattr(self.matcher, 'no_weight_vfl_epoch'):
            self.matcher.no_weight_vfl_epoch = no_weight_vfl_epoch   
        self.epoch = 0     
        self.ccm_params = ccm_params     
        self.density_recall_penalty = density_recall_penalty 
        self.density_precision_penalty = density_precision_penalty   
        self.mask_point_sample_ratio = mask_point_sample_ratio
        self.model = None 
        self._loss_config_logged = False  
  
        if SEG_MASK_LOSS_VIS:    
            self.visualizer = MaskVisualizer()
        else:    
            self.visualizer = None   
        self.vis_counter = 0

        if self.no_weight_vfl_epoch != -1:    
            logger.info(RED + f"no_weight_vfl_epoch set {self.no_weight_vfl_epoch}" + RESET)   
            logger.info(RED + f"no_weight_vfl_warmup_iter set {self.no_weight_vfl_warmup_iter}" + RESET)

    def _log_loss_config_once(self):     
        if self._loss_config_logged:  
            return

        logger.info(
            ORANGE
            + (
                f"Loss Config | losses={self.losses}, "     
                f"weight_dict={self.weight_dict}, "   
                f"boxes_iou_type={self.boxes_iou_type}"    
            )
            + RESET
        )
        self._loss_config_logged = True

    def loss_moe(self, outputs, targets, indices, num_boxes):
        assert self.model is not None
        assert 'pred_logits' in outputs
        device = outputs['pred_logits'].device    
        loss_moe = torch.tensor(0.0, device=device)    
        for m in self.model.modules():     
            if hasattr(m, 'aux_loss'):     
                loss_moe += m.aux_loss
        return {'loss_moe': loss_moe}

    @torch.no_grad() 
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes 
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients     
        """  
        assert 'pred_logits' in outputs     
        pred_logits = outputs['pred_logits']  
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)    
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)   
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())     
  
        return {'loss_cardinality': card_err}
    
    def loss_densitymap(self, outputs, targets, indices, num_boxes): 
        assert 'pred_densitymap' in outputs     
        pred_densitymap = outputs['pred_densitymap'] # bs, 1, h, w
        bs, c, h, w = pred_densitymap.size()  
        dmg = DensityMapGenerator([h, w])     
        density_loss = []     
        for b in range(bs):
            boxes = targets[b]["boxes"].cpu().detach().numpy()
            boxes[:, [0, 2]] *= w    
            boxes[:, [1, 3]] *= h   
            
            gt_densitymap = torch.from_numpy(dmg.generate_from_boxes(boxes, method='gaussian')).to(pred_densitymap.device)
            gt_densitymap /= (gt_densitymap.max() + 1e-6) # 归一化策略修正

            density_map = pred_densitymap[b].squeeze() / (pred_densitymap.max() + 1e-6) # 归一化策略修正
            diff = density_map - gt_densitymap
            underestimation_mask = (density_map < gt_densitymap).float() 
            overestimation_mask = (density_map > gt_densitymap).float()    
            penalty_weight = 1 + self.density_recall_penalty * gt_densitymap * underestimation_mask + self.density_precision_penalty * gt_densitymap * overestimation_mask  
            density_loss.append((penalty_weight * (diff ** 2)).mean())

            # image = np.zeros((h, w, 3), dtype=np.uint8) 
            # visualize_density_map(image, gt_densitymap.cpu().detach().numpy(), boxes, save_path=f'result_{b}.png')
            
        return {'loss_densitymap': sum(density_loss) / len(density_loss)}    
    
    def loss_densitycount(self, outputs, targets, indices, num_boxes):   
        assert 'pred_densitymap' in outputs    
        pred_densitymap = outputs['pred_densitymap'].sum(dim=[1,2,3]) / 300 # bs  

        targets_count = [targets[i]['labels'].shape[0] / 300 for i in range(len(targets))]   
        targets_count = torch.tensor(targets_count, dtype=torch.float).to(pred_densitymap.device)  
   
        return {'loss_densitycount': F.smooth_l1_loss(pred_densitymap, targets_count)}   
     
    def loss_ccm(self, outputs, targets, indices, num_boxes):  
        assert 'pred_bbox_number' in outputs
        ccm_targets = []    
        for i in range(len(targets)):     
            tgt_num = targets[i]['labels'].shape[0]
            t = 0   
            for j in range(len(self.ccm_params)): 
                if tgt_num >= self.ccm_params[j]:
                    t = j + 1
            ccm_targets.append(t)   
        ccm_targets = torch.tensor(ccm_targets, dtype=torch.int64).to(outputs['pred_bbox_number'].device)
   
        loss = torch.nn.CrossEntropyLoss()(outputs['pred_bbox_number'], ccm_targets) 
  
        return {'loss_ccm': loss}
   
    def _get_vfl_weight_warmup_factor(self):  
        if self.no_weight_vfl_epoch == -1:
            return 1.0   
        if self.epoch < self.no_weight_vfl_epoch:
            return 0.0     
        if self.no_weight_vfl_warmup_iter <= 0:
            return 1.0

        step = getattr(self, 'step', None)
        epoch_step = getattr(self, 'epoch_step', None) 
        if step is None or epoch_step is None:
            return 1.0
  
        epoch_step = max(int(epoch_step), 1)
        transition_iter = (self.epoch - self.no_weight_vfl_epoch) * epoch_step + int(step)   
        transition_iter = max(transition_iter, 0)     
        return min(transition_iter / float(self.no_weight_vfl_warmup_iter), 1.0)    
   
    def _vfl_weighted_bce_loss(self, src_logits, target_score, weight):     
        warmup_factor = self._get_vfl_weight_warmup_factor() 
        if warmup_factor <= 0:  
            return F.binary_cross_entropy_with_logits(src_logits, target_score, reduction='none')
        loss_weighted = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        if warmup_factor >= 1:
            return loss_weighted   

        loss_no_weight = F.binary_cross_entropy_with_logits(src_logits, target_score, reduction='none')  
        return loss_no_weight * (1 - warmup_factor) + loss_weighted * warmup_factor

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs   
        src_logits = outputs['pred_logits']    
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])   
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)  
        target_classes[idx] = target_classes_o 
        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1] 
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')   
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
     
        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):   
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:  
            src_boxes = outputs['pred_boxes'][idx]    
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach() 
        else:   
            ious = values     
 
        src_logits = outputs['pred_logits'] 
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) 
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]    
  
        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)     
        target_score = target_score_o.unsqueeze(-1) * target 

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score     
 
        loss = self._vfl_weighted_bce_loss(src_logits, target_score, weight)  
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}   
   
    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None): 
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)     
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))  
            ious = torch.diag(ious).detach()     
        else:   
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) 
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o     
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)    
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        target_score = target_score.pow(self.gamma)
        if self.mal_alpha != None:
            weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target   
        else: 
            weight = pred_score.pow(self.gamma) * (1 - target) + target

        # print(" ### DEIM-gamma{}-alpha{} ### ".format(self.gamma, self.mal_alpha))   
        loss = self._vfl_weighted_bce_loss(src_logits, target_score, weight)
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes  
        return {'loss_mal': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):    
        """Compute the losses related to the bounding boxes, the L1 regression loss and IoU-type loss  
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """    
        assert 'pred_boxes' in outputs   
        idx = self._get_src_permutation_idx(indices)    
        src_boxes = outputs['pred_boxes'][idx]    
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}    
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')    
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes   

        src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)  
        target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes) 
        if self.boxes_iou_type == 'iou':
            iou, _ = box_iou(src_boxes_xyxy, target_boxes_xyxy)  
            iou = torch.diag(iou)
        elif self.boxes_iou_type == 'giou':
            # Keep historical numeric behavior for default path.
            iou = torch.diag(generalized_box_iou(src_boxes_xyxy, target_boxes_xyxy))
        else: 
            iou = bbox_iou_aligned_xyxy(src_boxes_xyxy, target_boxes_xyxy, iou_type=self.boxes_iou_type)  
        loss_iou = 1 - iou
        loss_iou = loss_iou if boxes_weight is None else loss_iou * boxes_weight
        losses[f'loss_{self.boxes_iou_type}'] = loss_iou.sum() / num_boxes
  
        return losses  
  
    def loss_local(self, outputs, targets, indices, num_boxes, T=5):     
        """Compute Fine-Grained Localization (FGL) Loss
            and Decoupled Distillation Focal (DDF) Loss. """ 
    
        losses = {} 
        if 'pred_corners' in outputs:
            idx = self._get_src_permutation_idx(indices)
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            pred_corners = outputs['pred_corners'][idx].reshape(-1, (self.reg_max+1))    
            ref_points = outputs['ref_points'][idx].detach()  
            with torch.no_grad():
                if self.fgl_targets_dn is None and 'is_dn' in outputs:   
                        self.fgl_targets_dn= bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),  
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])
                if self.fgl_targets is None and 'is_dn' not in outputs:
                        self.fgl_targets = bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),  
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])   
 
            target_corners, weight_right, weight_left = self.fgl_targets_dn if 'is_dn' in outputs else self.fgl_targets
     
            ious = torch.diag(box_iou(\
                        box_cxcywh_to_xyxy(outputs['pred_boxes'][idx]), box_cxcywh_to_xyxy(target_boxes))[0])
            weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()   
    
            losses['loss_fgl'] = self.unimodal_distribution_focal_loss(  
                pred_corners, target_corners, weight_right, weight_left, weight_targets, avg_factor=num_boxes)

            if 'teacher_corners' in outputs:    
                pred_corners = outputs['pred_corners'].reshape(-1, (self.reg_max+1))   
                target_corners = outputs['teacher_corners'].reshape(-1, (self.reg_max+1))   
                if not torch.equal(pred_corners, target_corners):   
                    weight_targets_local = outputs['teacher_logits'].sigmoid().max(dim=-1)[0]    

                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                    mask[idx] = True
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)  
     
                    weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(weight_targets_local.dtype)
                    weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()  

                    loss_match_local = weight_targets_local * (T ** 2) * (nn.KLDivLoss(reduction='none')
                    (F.log_softmax(pred_corners / T, dim=1), F.softmax(target_corners.detach() / T, dim=1))).sum(-1)   
                    if 'is_dn' not in outputs:
                        batch_scale = 8 / outputs['pred_boxes'].shape[0]  # Avoid the influence of batch size per GPU  
                        self.num_pos, self.num_neg = (mask.sum() * batch_scale) ** 0.5, ((~mask).sum() * batch_scale) ** 0.5
                    loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                    loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0 
                    losses['loss_ddf'] = (loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg) / (self.num_pos + self.num_neg)     

        return losses   
  
    def loss_masks_ps(self, outputs, targets, indices, num_boxes):
        """Compute BCE-with-logits and Dice losses for segmentation masks on matched pairs.     
        Expects outputs to contain 'pred_masks' of shape [B, Q, H, W] and targets with key 'masks'. 
        """
        assert 'pred_masks' in outputs, "pred_masks missing in model outputs"  
        pred_masks = outputs['pred_masks']  # [B, Q, H, W]
        # gather matched prediction masks
        idx = self._get_src_permutation_idx(indices)     
        src_masks = pred_masks[idx]  # [N, H, W]   
        # handle no matches
        if src_masks.numel() == 0:   
            return {     
                'loss_mask_ps_ce': src_masks.sum(),     
                'loss_mask_ps_dice': src_masks.sum(),
            }
        # gather matched target masks  
        target_masks = torch.cat([t['masks'][j] for t, (_, j) in zip(targets, indices)], dim=0)  # [N, Ht, Wt]   
  
        # No need to upsample predictions as we are using normalized coordinates :)  
        # N x 1 x H x W 
        src_masks = src_masks.unsqueeze(1)
        target_masks = target_masks.unsqueeze(1).float()

        num_points = max(src_masks.shape[-2], src_masks.shape[-2] * src_masks.shape[-1] // self.mask_point_sample_ratio)

        with torch.no_grad():
            # sample point_coords     
            point_coords = get_uncertain_point_coords_with_randomness( 
                src_masks,   
                lambda logits: calculate_uncertainty(logits),  
                num_points,
                3,     
                0.75,   
            )
            # get gt labels   
            point_labels = point_sample(   
                target_masks,  
                point_coords,
                align_corners=False,     
                mode="nearest",
            ).squeeze(1)     
    
        point_logits = point_sample(     
            src_masks,
            point_coords,     
            align_corners=False,     
        ).squeeze(1)    
     
        if self.visualizer and self.vis_counter % SEG_MASK_STEP_VIS == 0:    
            target_masks = target_masks.squeeze(1)
            N, H, W = target_masks.shape
     
            # 上采样target masks
            src_masks = F.interpolate(
                src_masks,  # [N, 1, Ht, Wt]
                size=(H, W),  # 下采样到预测mask的尺寸
                mode='nearest'  
            ).squeeze(1)  # [N, H, W]
 
            # 创建边界框mask，只保留box内的区域
            box_masks = torch.zeros_like(target_masks)  # [N, Ht, Wt]
 
            target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)  # [N, 4] (cx, cy, w, h)     
            for i in range(N):
                cx, cy, w_box, h_box = target_boxes[i]
                # 转换归一化坐标到像素坐标（在预测mask的尺寸上）  
                x1 = int((cx - w_box / 2) * W)  
                y1 = int((cy - h_box / 2) * H)
                x2 = int((cx + w_box / 2) * W)
                y2 = int((cy + h_box / 2) * H)
    
                # 确保坐标在有效范围内   
                x1 = max(0, min(x1, W - 1))     
                y1 = max(0, min(y1, H - 1))
                x2 = max(0, min(x2, W))    
                y2 = max(0, min(y2, H))
                
                # 在box区域内设置为1  
                box_masks[i, y1:y2, x1:x2] = 1.0

            with torch.no_grad():
                self.visualizer.visualize_batch(
                    src_masks, 
                    target_masks,     
                    box_masks,     
                    target_boxes,
                    max_samples=16
                )     
        self.vis_counter += 1
  
        losses = {
            "loss_mask_ps_ce": sigmoid_ce_loss_jit(point_logits, point_labels, num_boxes),
            "loss_mask_ps_dice": dice_loss_jit(point_logits, point_labels, num_boxes), 
        }

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):  
        # permute predictions following indices   
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx 

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]) 
        tgt_idx = torch.cat([tgt for (_, tgt) in indices]) 
        return batch_idx, tgt_idx  
   
    def _get_go_indices(self, indices, indices_aux_list):     
        """Get a matching union set across all decoder layers. """
        results = []
        for indices_aux in indices_aux_list:
            indices = [(torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))    
                        for idx1, idx2 in zip(indices.copy(), indices_aux.copy())]
 
        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)   
            count_sort_indices = torch.argsort(counts, descending=True)     
            unique_sorted = unique[count_sort_indices]  
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item() 
                if row_idx not in column_to_row:   
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)     
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)   
            results.append((final_rows.long(), final_cols.long()))    
        return results
  
    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None   
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None
     
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):    
        loss_map = { 
            'boxes': self.loss_boxes,    
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,  
            'mal': self.loss_labels_mal, 
            'local': self.loss_local,   
            'cardinality': self.loss_cardinality,    
            'ccm': self.loss_ccm, 
            'densitymap': self.loss_densitymap,
            'densitycount': self.loss_densitycount,
            'mask_ps': self.loss_masks_ps,  
            'moe': self.loss_moe    
        }   
        assert loss in loss_map, f'do you really want to compute {loss} loss?'  
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
  
    def forward(self, outputs, targets, **kwargs): 
        """ This performs the loss computation. 
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format  
             targets: list of dicts, such that len(targets) == batch_size.  
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        epoch = kwargs.get('epoch', 0)  
        self.step = kwargs.get('step', 0)
        self.global_step = kwargs.get('global_step', None)   
        self.epoch_step = kwargs.get('epoch_step', 1) 

        # 处理Dynamic-Query的多batch问题     
        num_queries_list = outputs.get('num_queries_list', None)
        loss_filter = ['ccm', 'densitymap', 'densitycount', 'moe']
 
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}
     
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, epoch=epoch, num_queries_list=num_queries_list)['indices']   
        self._clear_cache()   
  
        # Get the matching union set across all decoder layers.  
        if 'aux_outputs' in outputs:
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []
            aux_outputs_list = outputs['aux_outputs']
            if 'pre_outputs' in outputs:   
                aux_outputs_list = outputs['aux_outputs'] + [outputs['pre_outputs']]  
            for i, aux_outputs in enumerate(aux_outputs_list): 
                indices_aux = self.matcher(aux_outputs, targets, epoch=epoch, num_queries_list=num_queries_list)['indices']     
                cached_indices.append(indices_aux)    
                indices_aux_list.append(indices_aux)
            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):    
                indices_enc = self.matcher(aux_outputs, targets, epoch=epoch, num_queries_list=num_queries_list)['indices']     
                cached_indices_enc.append(indices_enc)  
                indices_aux_list.append(indices_enc)     
            indices_go = self._get_go_indices(indices, indices_aux_list)
  
            num_boxes_go = sum(len(x[0]) for x in indices_go)  
            num_boxes_go = torch.as_tensor([num_boxes_go], dtype=torch.float, device=next(iter(outputs.values())).device)
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()
        else: 
            assert 'aux_outputs' in outputs, ''    

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)   
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device) 
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses, main loss
        losses = {} 
        for loss in self.losses: 
            # TODO, indices and num_box are different from RT-DETRv2 
            use_uni_set = self.use_uni_set and (loss in ['boxes', 'local']) 
            indices_in = indices_go if use_uni_set else indices     
            num_boxes_in = num_boxes_go if use_uni_set else num_boxes 
            meta = self.get_loss_meta_info(loss, outputs, targets, indices_in)
            l_dict = self.get_loss(loss, outputs, targets, indices_in, num_boxes_in, **meta)  
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict} 
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):    
                if 'local' in self.losses:      # only work for local loss 
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    if loss in loss_filter and 'mask' not in loss:  
                        continue
                    # TODO, indices and num_box are different from RT-DETRv2
                    use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
                    indices_in = indices_go if use_uni_set else cached_indices[i] 
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)  
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)    

                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}    
                    losses.update(l_dict)
    
        # In case of auxiliary traditional head output at first decoder layer. just for dfine
        if 'pre_outputs' in outputs:   
            aux_outputs = outputs['pre_outputs']
            for loss in self.losses:
                if loss in loss_filter or 'mask' in loss:    
                    continue
                # TODO, indices and num_box are different from RT-DETRv2   
                use_uni_set = self.use_uni_set and (loss in ['boxes', 'local']) 
                indices_in = indices_go if use_uni_set else cached_indices[-1]   
                num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)  

                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                l_dict = {k + '_pre': v for k, v in l_dict.items()}  
                losses.update(l_dict)

        # In case of encoder auxiliary losses.
        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs, ''
            class_agnostic = outputs['enc_meta']['class_agnostic']
            if class_agnostic: 
                orig_num_classes = self.num_classes
                self.num_classes = 1 
                enc_targets = copy.deepcopy(targets)   
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:   
                enc_targets = targets   
     
            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                for loss in self.losses:    
                    if loss in loss_filter:     
                        continue  
                    # TODO, indices and num_box are different from RT-DETRv2 
                    use_uni_set = self.use_uni_set and (loss == 'boxes')
                    indices_in = indices_go if use_uni_set else cached_indices_enc[i] 
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes  
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in)  
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices_in, num_boxes_in, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)    
    
            if class_agnostic:
                self.num_classes = orig_num_classes   

        # In case of cdn auxiliary losses.
        if 'dn_outputs' in outputs: 
            assert 'dn_meta' in outputs, ''
            indices_dn = self.get_cdn_matched_indices(outputs['dn_meta'], targets)    
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']    

            for i, aux_outputs in enumerate(outputs['dn_outputs']):
                if 'local' in self.losses:      # only work for local loss
                    aux_outputs['is_dn'] = True
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale'] 
                for loss in self.losses:
                    if loss in loss_filter:     
                        continue
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)  
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)  
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
 
            # In case of auxiliary traditional head output at first decoder layer, just for dfine  
            if 'dn_pre_outputs' in outputs:   
                aux_outputs = outputs['dn_pre_outputs']  
                for loss in self.losses:   
                    if loss in loss_filter or 'mask' in loss:   
                        continue
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta) 
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict} 
                    l_dict = {k + '_dn_pre': v for k, v in l_dict.items()}
                    losses.update(l_dict) 

        # For debugging Objects365 pre-train.   
        losses = {k:torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}    
        return losses
  
    def get_loss_meta_info(self, loss, outputs, targets, indices):  
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)
        src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes.detach())   
        target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(src_boxes_xyxy, target_boxes_xyxy)
            iou = torch.diag(iou) 
        elif self.boxes_weight_format == 'giou':
            # Keep historical numeric behavior for existing configs.
            iou = torch.diag(generalized_box_iou(src_boxes_xyxy, target_boxes_xyxy))
        else:   
            iou = bbox_iou_aligned_xyxy(  
                src_boxes_xyxy,
                target_boxes_xyxy,   
                iou_type=self.boxes_weight_format) 

        if loss in ('boxes', ): 
            meta = {'boxes_weight': iou}
        elif loss in ('vfl', 'mal'):
            meta = {'values': iou}    
        else:
            meta = {}     
  
        return meta    
 
    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices  
        """    
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]   
        device = targets[0]['labels'].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:     
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)   
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx)) 
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))

        return dn_match_indices  


    def feature_loss_function(self, fea, target_fea):
        loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float() 
        return torch.abs(loss) 
    

    def unimodal_distribution_focal_loss(self, pred, label, weight_right, weight_left, weight=None, reduction='sum', avg_factor=None):
        dis_left = label.long()
        dis_right = dis_left + 1  

        loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left.reshape(-1) \
             + F.cross_entropy(pred, dis_right, reduction='none') * weight_right.reshape(-1)
    
        if weight is not None: 
            weight = weight.float()     
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor 
        elif reduction == 'mean':
            loss = loss.mean()  
        elif reduction == 'sum':
            loss = loss.sum()

        return loss 
 
    def get_gradual_steps(self, outputs):     
        num_layers = len(outputs['aux_outputs']) + 1 if 'aux_outputs' in outputs else 1  
        step = .5 / (num_layers - 1)  
        opt_list = [.5  + step * i for i in range(num_layers)] if num_layers > 1 else [1]
        return opt_list   
    
    def set_epoch(self, epoch):   
        self.matcher.set_epoch(epoch)
        self.epoch = epoch    
        self._log_loss_config_once()
        if self.no_weight_vfl_epoch != -1 and self.epoch == self.no_weight_vfl_epoch:  
            logger.info(
                RED 
                + (
                    f"Epoch:[{self.epoch}]>=[{self.no_weight_vfl_epoch}] "
                    f"warming up weight in vfl/mal for {self.no_weight_vfl_warmup_iter} iters."
                )
                + RESET     
            )
  
    def set_model(self, model):
        self.model = model   
    
    def reset_model(self):
        self.model = None  
