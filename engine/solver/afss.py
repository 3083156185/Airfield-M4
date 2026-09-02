import copy  
import random

import torch   
import torch.nn.functional as F
 
from engine.extre_module.yolo_metrice import process_batch_det, process_batch_mask

  
class AFSSController:   
    def __init__(
        self,
        easy_threshold=0.85,
        moderate_threshold=0.55,   
        easy_ratio=0.02,
        moderate_ratio=0.4,
        easy_review_gap=10,
        moderate_gap=3,    
        update_interval=5,
        warmup_epochs=5,   
        review_force_cap_ratio=0.5,
        iou_threshold=0.5,
        conf_threshold=0.25,
        random_seed=0,   
    ):
        self.easy_threshold = easy_threshold     
        self.moderate_threshold = moderate_threshold  
        self.easy_ratio = easy_ratio 
        self.moderate_ratio = moderate_ratio  
        self.easy_review_gap = easy_review_gap    
        self.moderate_gap = moderate_gap     
        self.update_interval = update_interval
        self.warmup_epochs = warmup_epochs    
        self.review_force_cap_ratio = review_force_cap_ratio
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.random_seed = random_seed
        self.states = {}     

    @staticmethod  
    def _default_state():
        return {
            "box_precision": 0.0, 
            "box_recall": 0.0,     
            "mask_precision": 0.0,
            "mask_recall": 0.0,
            "precision": 0.0,    
            "recall": 0.0, 
            "score": 0.0,
            "has_mask_metrics": False,  
            "joint_updated": False,
            "last_eval_epoch": -1, 
            "last_used_epoch": -1,
        }
  
    def initialize_states(self, total_samples):
        self.states = {
            idx: self._default_state()
            for idx in range(total_samples)
        }

    def state_dict(self):    
        return {"states": copy.deepcopy(self.states)}  

    def load_state_dict(self, state):   
        loaded_states = copy.deepcopy(state["states"])
        self.states = {}
        for raw_idx, raw_state in loaded_states.items():
            idx = int(raw_idx)
            normalized = self._default_state()
            normalized.update(raw_state)    
            if not normalized.get("joint_updated", False): 
                normalized["box_precision"] = float(normalized.get("precision", 0.0))
                normalized["box_recall"] = float(normalized.get("recall", 0.0))    
                normalized["mask_precision"] = float(normalized.get("precision", 0.0))
                normalized["mask_recall"] = float(normalized.get("recall", 0.0)) 
                normalized["score"] = min(normalized["precision"], normalized["recall"])
            self.states[idx] = normalized
    
    def update_image_metrics(self, idx, box_precision, box_recall, mask_precision=None, mask_recall=None, eval_epoch=None): 
        has_mask_metrics = mask_precision is not None and mask_recall is not None
        if mask_precision is None:
            mask_precision = box_precision  
        if mask_recall is None:     
            mask_recall = box_recall     
  
        score = min(box_precision, box_recall, mask_precision, mask_recall)    
        state = self.states[int(idx)]
        state["box_precision"] = float(box_precision)
        state["box_recall"] = float(box_recall)
        state["mask_precision"] = float(mask_precision)
        state["mask_recall"] = float(mask_recall)   
        state["precision"] = float(score) 
        state["recall"] = float(score)  
        state["score"] = float(score)
        state["has_mask_metrics"] = bool(has_mask_metrics)   
        state["joint_updated"] = True  
        if eval_epoch is not None:
            state["last_eval_epoch"] = int(eval_epoch) 
   
    def level_for_index(self, idx):    
        state = self.states[idx]
        if state.get("joint_updated", False): 
            score = float(state.get("score", 0.0))
        else:     
            score = min(float(state.get("precision", 0.0)), float(state.get("recall", 0.0)))
        if score > self.easy_threshold:   
            return "easy"  
        if score >= self.moderate_threshold:
            return "moderate"   
        return "hard"

    def level_counts(self):
        counts = {"easy": 0, "moderate": 0, "hard": 0} 
        for idx in self.states:    
            counts[self.level_for_index(idx)] += 1
        return counts

    def select_indices_for_epoch(self, epoch):   
        rng = random.Random(self.random_seed + epoch)
        easy, moderate, hard = [], [], []
        for idx in self.states:  
            level = self.level_for_index(idx)
            if level == "easy":   
                easy.append(idx)
            elif level == "moderate":
                moderate.append(idx) 
            else:
                hard.append(idx)

        forced_easy = [idx for idx in easy if epoch - self.states[idx]["last_used_epoch"] >= self.easy_review_gap]     
        easy_budget = max(1, int(len(easy) * self.easy_ratio)) if easy else 0
        forced_easy_budget = min(len(forced_easy), int(easy_budget * self.review_force_cap_ratio))     
        selected_easy = rng.sample(forced_easy, forced_easy_budget) if forced_easy_budget else []
        remaining_easy = [idx for idx in easy if idx not in selected_easy]     
        random_easy_budget = max(easy_budget - len(selected_easy), 0)
        if random_easy_budget:    
            selected_easy += rng.sample(remaining_easy, min(len(remaining_easy), random_easy_budget))

        forced_moderate = [idx for idx in moderate if epoch - self.states[idx]["last_used_epoch"] >= self.moderate_gap]
        moderate_budget = int(len(moderate) * self.moderate_ratio)   
        selected_moderate = forced_moderate[:]  
        remaining_moderate = [idx for idx in moderate if idx not in selected_moderate]
        random_moderate_budget = max(moderate_budget - len(selected_moderate), 0)   
        if random_moderate_budget:
            selected_moderate += rng.sample(remaining_moderate, min(len(remaining_moderate), random_moderate_budget))
 
        selected = sorted(set(hard + selected_moderate + selected_easy))
        if not selected: 
            selected = sorted(self.states.keys()) 
        return selected
  
    def mark_selected_indices(self, indices, epoch):    
        for idx in indices:
            self.states[int(idx)]["last_used_epoch"] = epoch
    
    def should_refresh_scores(self, epoch):  
        if epoch < self.warmup_epochs:   
            return False
        return (epoch - self.warmup_epochs) % self.update_interval == 0
  
    @staticmethod
    def compute_image_precision_recall(result, target, iou_threshold=0.5, conf_threshold=0.25): 
        labels = target["labels"].to(torch.float32).unsqueeze(1)
        boxes = target["boxes"].to(torch.float32)
        gt = torch.cat([labels, boxes], dim=1)
  
        if result["scores"].numel():
            keep = result["scores"] >= conf_threshold  
        else:
            keep = result["scores"].new_zeros((0,), dtype=torch.bool)

        boxes_pred = result["boxes"][keep]
        scores_pred = result["scores"][keep]    
        labels_pred = result["labels"][keep]
  
        if boxes_pred.numel() == 0:   
            pred = torch.zeros((0, 6), dtype=torch.float32, device=boxes.device)  
        else:     
            pred = torch.cat(
                [
                    boxes_pred.to(torch.float32),  
                    scores_pred.to(torch.float32).unsqueeze(1),  
                    labels_pred.to(torch.float32).unsqueeze(1),
                ],
                dim=1,     
            )
     
        iouv = torch.tensor([iou_threshold], device=boxes.device)
        correct = process_batch_det(pred, gt, iouv)
        tp = float(correct[:, 0].sum().item()) if correct.numel() else 0.0     
        fp = float(pred.shape[0] - tp)   
        fn = float(gt.shape[0] - tp)   
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0   
        return precision, recall

    @staticmethod
    def compute_image_mask_precision_recall(result, target, iou_threshold=0.5, conf_threshold=0.25):
        if "masks" not in result or "masks" not in target:  
            return None, None
    
        if result["scores"].numel():     
            keep = result["scores"] >= conf_threshold
        else:
            keep = result["scores"].new_zeros((0,), dtype=torch.bool)
     
        pred_masks = result["masks"][keep].to(torch.float32) 
        pred_cls = result["labels"][keep].to(torch.float32) 
        label_masks = target["masks"].to(torch.float32)
        label_cls = target["labels"].to(torch.float32)

        if pred_masks.ndim == 4 and pred_masks.shape[1] == 1:
            pred_masks = pred_masks[:, 0]     
        if label_masks.ndim == 4 and label_masks.shape[1] == 1:
            label_masks = label_masks[:, 0]
    
        if pred_masks.numel() and label_masks.shape[-2:] != pred_masks.shape[-2:]:
            label_masks = F.interpolate(
                label_masks.unsqueeze(1),
                size=pred_masks.shape[-2:],     
                mode='nearest',
            ).squeeze(1)  

        if pred_masks.numel() == 0:    
            tp = 0.0
            fp = 0.0
            fn = float(label_masks.shape[0])
        else:
            iouv = torch.tensor([iou_threshold], device=label_masks.device)
            correct = process_batch_mask(pred_masks, pred_cls, label_masks, label_cls, iouv)   
            tp = float(correct[:, 0].sum().item()) if correct.numel() else 0.0
            fp = float(pred_masks.shape[0] - tp)  
            fn = float(label_masks.shape[0] - tp)   
 
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0 
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        return precision, recall     
