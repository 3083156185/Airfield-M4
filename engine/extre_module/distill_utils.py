import torch    
import torch.nn as nn
import torch.nn.functional as F

import math, functools
import numpy as np    

def bce_loss_pairwise(A, B):
    """     
    计算A中每个元素与B中每个元素的BCEWithLogitsLoss
    
    Args:
        A: torch.Tensor, shape [m, 10] - 作为logits
        B: torch.Tensor, shape [n, 10] - 作为target（会被sigmoid转换为概率）   
 
    Returns:    
        torch.Tensor, shape [m, n, 10] - 每个类别的BCE loss
    """    
    m, num_classes = A.shape
    n, _ = B.shape   
    
    # 方法1: 使用expand（推荐）
    A_expanded = A.unsqueeze(1).expand(-1, n, -1)  # [m, n, 10]     
    B_expanded = B.unsqueeze(0).expand(m, -1, -1)  # [m, n, 10]   
   
    # 将B转换为目标概率   
    B_probs = torch.sigmoid(B_expanded)   
    
    # 计算BCE loss，保持所有维度   
    loss_matrix = F.binary_cross_entropy_with_logits(    
        A_expanded, B_probs, reduction='none'
    )  # [m, n, 10]     
    
    return loss_matrix   

def box_iou(box1, box2, GIOU=False, eps=1e-7):
    # 拆分，增加维度以便广播
    x1, y1, w1, h1 = box1[..., 0:1], box1[..., 1:2], box1[..., 2:3], box1[..., 3:4]  # [B,S,1]
    x2, y2, w2, h2 = box2[..., 0:1], box2[..., 1:2], box2[..., 2:3], box2[..., 3:4]  # [B,S,1] 

    # 扩展为 [B, S, S]
    x1, y1, w1, h1 = x1.unsqueeze(2), y1.unsqueeze(2), w1.unsqueeze(2), h1.unsqueeze(2)  # [B,S,1]   
    x2, y2, w2, h2 = x2.unsqueeze(1), y2.unsqueeze(1), w2.unsqueeze(1), h2.unsqueeze(1)  # [B,1,S] 
     
    w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
    b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_  
    b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    
    # Intersection area  
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)   
     
    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps    

    # IoU
    iou = inter / union
 
    if GIOU:     
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height   
        c_area = cw * ch + eps  # convex area   
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf

    return iou    

# ------------------------------BiliBili 魔鬼面具up主------------------------------
class DETRLogicLoss(nn.Module):
    def __init__(self, device):   
        super().__init__()
    
        self.device = device    
        self.iou_thresh = 0.1 
        self.teacher_conf = 0.2   
 
    def get_loss(self, t_pred, s_pred, target_nums, loss_ratio = 1.0):    
        t_pred_logits, t_pred_boxes = t_pred  
        s_pred_logits, s_pred_boxes = s_pred

        # 对教师网络的logits应用sigmoid得到概率
        # 取每个查询的最大类别概率作为该查询的置信度
        t_obj_scale = t_pred_logits.sigmoid().max(-1)[0].unsqueeze(-1).detach() 
        t_obj_conf_flag = (t_obj_scale > self.teacher_conf).squeeze(-1)[:, None, :].repeat((1, t_pred_boxes.size(1), 1)) # 只使用教师网络高置信度的预测框来指导学生。    
    
        if t_obj_conf_flag.sum() != 0: # 判断从教师中提取符合置信度要求的数量是否为0
            boxes_iou = box_iou(s_pred_boxes, t_pred_boxes).squeeze() # 计算学生的预测框与符合置信度要求的教师预测框的iou
            max_iou_value, max_idx = boxes_iou.max(-1) # 提取学生的预测框跟教师预测框的匹配关系
            # 需要 max_iou_value大于一定阈值 和 教师的目标置信度要满足一定阈值<学生的每个目标要向哪些教师框学习。>
            flag = torch.gather((max_iou_value > self.iou_thresh).unsqueeze(-1).repeat((1, 1, t_pred_boxes.size(1))) & t_obj_conf_flag, 1, max_idx.unsqueeze(-1))

            # 关于预测框只算positive部分
            lbox_l1 = torch.cdist(s_pred_boxes, t_pred_boxes, p=1)
            lbox_l1 = torch.gather(lbox_l1, 1, max_idx.unsqueeze(-1))[flag] # 学生框对应的匹配教师框的 L1 误差。    
            lbox_iou = (1 - box_iou(s_pred_boxes, t_pred_boxes, GIOU=True)).squeeze()  
            lbox_iou = torch.gather(lbox_iou, 1, max_idx.unsqueeze(-1))[flag] # 学生框对应的匹配教师框的 GIOU-Loss。     

            # 分类部分   
            s_pred_logits = s_pred_logits.reshape((s_pred_logits.size(0) * s_pred_logits.size(1), s_pred_logits.size(2)))   
            t_pred_logits = t_pred_logits.reshape((t_pred_logits.size(0) * t_pred_logits.size(1), t_pred_logits.size(2)))
            lcls = bce_loss_pairwise(s_pred_logits, t_pred_logits).mean(-1)
            lcls = torch.gather(lcls, 1, max_idx.reshape((-1)).unsqueeze(-1)) # 只取学生匹配的教师框对应的损失。  
  
            lbox_l1 = lbox_l1.sum() / target_nums * 5 # L1损失，权重5 
            lbox_iou = lbox_iou.sum() / target_nums * 2 # IoU损失，权重2   
            lcls = lcls.sum() / target_nums
        else:   
            lbox_l1 = torch.zeros(1, device=self.device)
            lbox_iou = torch.zeros(1, device=self.device)
            lcls = torch.zeros(1, device=self.device)
     
        return (lbox_l1 + lbox_iou + lcls) * loss_ratio

    def forward(self, s_p, t_p, batch):
        target_nums = sum([d['boxes'].size(0) for d in batch]) # object counts    

        distill_loss = self.get_loss([t_p['pred_logits'], t_p['pred_boxes']], [s_p['pred_logits'], s_p['pred_boxes']], target_nums) 
     
        return distill_loss

class DETRMutilDecoderLogicLoss(nn.Module):
    def __init__(self, pre_outputs_distill_ratio, aux_outputs_distill_ratio, aux_outputs_len, pred_outputs_distill_ratio, device):  
        super().__init__()  
  
        self.device = device   

        self.iou_thresh = 0.1
        self.teacher_conf = 0.2

        self.pre_outputs_distill_ratio = pre_outputs_distill_ratio     
        self.aux_outputs_distill_ratio = aux_outputs_distill_ratio / aux_outputs_len 
        self.pred_outputs_distill_ratio = pred_outputs_distill_ratio
 
    def get_loss(self, t_pred, s_pred, target_nums, loss_ratio = 1.0):
        t_pred_logits, t_pred_boxes = t_pred    
        s_pred_logits, s_pred_boxes = s_pred     

        # 对教师网络的logits应用sigmoid得到概率   
        # 取每个查询的最大类别概率作为该查询的置信度   
        t_obj_scale = t_pred_logits.sigmoid().max(-1)[0].unsqueeze(-1).detach()
        t_obj_conf_flag = (t_obj_scale > self.teacher_conf).squeeze(-1)[:, None, :].repeat((1, t_pred_boxes.size(1), 1))
    
        if t_obj_conf_flag.sum() != 0: # 判断从教师中提取符合置信度要求的数量是否为0
            boxes_iou = box_iou(s_pred_boxes, t_pred_boxes).squeeze() # 计算学生的预测框与符合置信度要求的教师预测框的iou
            max_iou_value, max_idx = boxes_iou.max(-1) # 提取学生的预测框跟教师预测框的匹配关系
            # 需要 max_iou_value大于一定阈值 和 教师的目标置信度要满足一定阈值  
            flag = torch.gather((max_iou_value > self.iou_thresh).unsqueeze(-1).repeat((1, 1, t_pred_boxes.size(1))) & t_obj_conf_flag, 1, max_idx.unsqueeze(-1))    

            # 关于预测框只算positive部分   
            lbox_l1 = torch.cdist(s_pred_boxes, t_pred_boxes, p=1)
            lbox_l1 = torch.gather(lbox_l1, 1, max_idx.unsqueeze(-1))[flag]    
            lbox_iou = (1 - box_iou(s_pred_boxes, t_pred_boxes, GIOU=True)).squeeze()  
            lbox_iou = torch.gather(lbox_iou, 1, max_idx.unsqueeze(-1))[flag]  
     
            # 分类部分    
            s_pred_logits = s_pred_logits.reshape((s_pred_logits.size(0) * s_pred_logits.size(1), s_pred_logits.size(2)))  
            t_pred_logits = t_pred_logits.reshape((t_pred_logits.size(0) * t_pred_logits.size(1), t_pred_logits.size(2)))
            lcls = bce_loss_pairwise(s_pred_logits, t_pred_logits).mean(-1)
            lcls = torch.gather(lcls, 1, max_idx.reshape((-1)).unsqueeze(-1))     
 
            lbox_l1 = lbox_l1.sum() / target_nums * 5 # L1损失，权重5     
            lbox_iou = lbox_iou.sum() / target_nums * 2 # IoU损失，权重2 
            lcls = lcls.sum() / target_nums     
        else:
            lbox_l1 = torch.zeros(1, device=self.device)
            lbox_iou = torch.zeros(1, device=self.device)   
            lcls = torch.zeros(1, device=self.device)    

        return (lbox_l1 + lbox_iou + lcls) * loss_ratio 
  
    def forward(self, s_p, t_p, batch):
        target_nums = sum([d['boxes'].size(0) for d in batch]) # object counts     

        l_distill = []   
     
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.     
        if 'aux_outputs' in s_p and 'aux_outputs' in t_p:   
            s_aux_logits, s_aux_boxes = [p['pred_logits'] for p in s_p['aux_outputs']], [p['pred_boxes'] for p in s_p['aux_outputs']] 
            t_aux_logits, t_aux_boxes = [p['pred_logits'] for p in t_p['aux_outputs']], [p['pred_boxes'] for p in t_p['aux_outputs']]
   
            aux_length = len(s_aux_logits)    
            for aux_index in range(aux_length): 
                l_distill.append(self.get_loss([t_aux_logits[aux_index], t_aux_boxes[aux_index]], [s_aux_logits[aux_index], s_aux_boxes[aux_index]], target_nums, self.aux_outputs_distill_ratio))
 
        l_distill.append(self.get_loss([t_p['pred_logits'], t_p['pred_boxes']], [s_p['pred_logits'], s_p['pred_boxes']], target_nums, self.pred_outputs_distill_ratio))

        # In case of auxiliary traditional head output at first decoder layer. just for dfine
        if 'pre_outputs' in s_p and 'pre_outputs' in t_p:  
            l_distill.append(self.get_loss([t_p['pre_outputs']['pred_logits'], t_p['pre_outputs']['pred_boxes']], [s_p['pre_outputs']['pred_logits'], s_p['pre_outputs']['pred_boxes']], target_nums, self.pre_outputs_distill_ratio))
 
        return sum(l_distill)
 
# ------------------------------BiliBili 魔鬼面具up主------------------------------   
class FeatureLoss(nn.Module):
    def __init__(self,
                 channels_s,
                 channels_t,   
                 distiller='cwd'):
        super(FeatureLoss, self).__init__()
  
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.align_module = nn.ModuleList([
            nn.Conv2d(channel, tea_channel, kernel_size=1, stride=1,     
                      padding=0).to(device) if channel != tea_channel else nn.Identity()   
            for channel, tea_channel in zip(channels_s, channels_t)
        ])
        self.norm = nn.ModuleList([
            nn.BatchNorm2d(tea_channel, affine=False)     
            for tea_channel in channels_t
        ])    
   
        if (distiller == 'mimic'):
            self.feature_loss = MimicLoss(channels_s, channels_t)     
        elif (distiller == 'mgd'):
            self.feature_loss = MGDLoss(channels_s, channels_t)
        elif (distiller == 'cwd'):
            self.feature_loss = CWDLoss(channels_s, channels_t)     
        elif (distiller == 'chsim'):
            self.feature_loss = ChSimLoss(channels_s, channels_t)
        elif (distiller == 'sp'):
            self.feature_loss = SPKDLoss(channels_s, channels_t)
        else:   
            raise NotImplementedError   

    def forward(self, y_s, y_t):
        assert len(y_s) == len(y_t)
        tea_feats = []    
        stu_feats = []    
 
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            s = self.align_module[idx](s)   
            s = self.norm[idx](s)   
            t = self.norm[idx](t)
            tea_feats.append(t)
            stu_feats.append(s)

        loss = self.feature_loss(stu_feats, tea_feats)
        return loss
  
# ------------------------------BiliBili 魔鬼面具up主------------------------------ 
class MimicLoss(nn.Module): 
    def __init__(self, channels_s, channels_t):
        super(MimicLoss, self).__init__()  
        device = 'cuda' if torch.cuda.is_available() else 'cpu'     
        self.mse = nn.MSELoss()

    def forward(self, y_s, y_t):  
        """Forward computation.
        Args:
            y_s (list): The student model prediction with 
                shape (N, C, H, W) in list. 
            y_t (list): The teacher model prediction with    
                shape (N, C, H, W) in list.  
        Return:     
            torch.Tensor: The calculated loss value of all stages.   
        """     
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)): 
            assert s.shape == t.shape   
            losses.append(self.mse(s, t))  
        loss = sum(losses)     
        return loss    
     
# ------------------------------BiliBili 魔鬼面具up主------------------------------  
class MGDLoss(nn.Module):   
    def __init__(self, 
                 channels_s,  
                 channels_t,
                 alpha_mgd=0.00002,
                 lambda_mgd=0.65):
        super(MGDLoss, self).__init__()  
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd

        self.generation = nn.ModuleList([     
            nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, kernel_size=3,
                          padding=1)).to(device) for channel in channels_t
        ])

    def forward(self, y_s, y_t):
        """Forward computation.   
        Args:
            y_s (list): The student model prediction with    
                shape (N, C, H, W) in list.   
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.    
        """
        assert len(y_s) == len(y_t)   
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):   
            assert s.shape == t.shape
            losses.append(self.get_dis_loss(s, t, idx) * self.alpha_mgd)  
        loss = sum(losses)
        return loss
    
    def get_dis_loss(self, preds_S, preds_T, idx):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        device = preds_S.device   
        mat = torch.rand((N, 1, H, W)).to(device)    
        mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation[idx](masked_fea)     

        dis_loss = loss_mse(new_fea, preds_T) / N     
    
        return dis_loss

# ------------------------------BiliBili 魔鬼面具up主------------------------------
class CWDLoss(nn.Module):     
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.
    <https://arxiv.org/abs/2011.13256>`_.
    """
    def __init__(self, channels_s, channels_t, tau=1.0):
        super(CWDLoss, self).__init__()
        self.tau = tau
   
    def forward(self, y_s, y_t):
        """Forward computation.  
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.   
            y_t (list): The teacher model prediction with     
                shape (N, C, H, W) in list.  
        Return:
            torch.Tensor: The calculated loss value of all stages.     
        """     
        assert len(y_s) == len(y_t)     
        losses = [] 
    
        for idx, (s, t) in enumerate(zip(y_s, y_t)):   
            assert s.shape == t.shape
            N, C, H, W = s.shape
            # normalize in channel diemension 
            softmax_pred_T = F.softmax(t.view(-1, W * H) / self.tau,
                                       dim=1)  # [N*C, H*W] 

            logsoftmax = torch.nn.LogSoftmax(dim=1)     
            cost = torch.sum(    
                softmax_pred_T * logsoftmax(t.view(-1, W * H) / self.tau) -     
                softmax_pred_T * logsoftmax(s.view(-1, W * H) / self.tau)) * (
                    self.tau**2)    
            # cost = torch.sum(-softmax_pred_T * logsoftmax(s.view(-1, W * H)/self.tau)) * (self.tau ** 2)
  
            losses.append(cost / (C * N))
        loss = sum(losses)
  
        return loss
 
# ------------------------------BiliBili 魔鬼面具up主------------------------------
class ChSimLoss(nn.Module):   
    '''   
    A loss module for Inter-Channel Correlation for Knowledge Distillation (ICKD).
    https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Exploring_Inter-Channel_Correlation_for_Diversity-Preserved_Knowledge_Distillation_ICCV_2021_paper.html     
    '''
    def __init__(self, channels_s, channels_t) -> None:
        super().__init__()
     
    @staticmethod   
    def batch_loss(f_s, f_t):
        bsz, ch = f_s.shape[0], f_s.shape[1]
        f_s = f_s.view(bsz, ch, -1)
        f_t = f_t.view(bsz, ch, -1)
        emd_s = torch.bmm(f_s, f_s.permute(0, 2, 1))  
        emd_s = torch.nn.functional.normalize(emd_s, dim=2)
 
        emd_t = torch.bmm(f_t, f_t.permute(0, 2, 1))
        emd_t = torch.nn.functional.normalize(emd_t, dim=2)   
 
        g_diff = emd_s - emd_t
        loss = (g_diff * g_diff).view(bsz, -1).sum() / (ch * bsz * bsz)
        return loss   
 
    def forward(self, y_s, y_t):  
        assert len(y_s) == len(y_t)    
        losses = []
        
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            losses.append(self.batch_loss(s, t) / s.size(0))  
        loss = sum(losses) 
        return loss
   
# ------------------------------BiliBili 魔鬼面具up主------------------------------
class SPKDLoss(nn.Module):   
    '''   
    Similarity-Preserving Knowledge Distillation  
    https://arxiv.org/pdf/1907.09682.pdf
    '''    
 
    def __init__(self, channels_s, channels_t):   
        super(SPKDLoss, self).__init__()
  
    def matmul_and_normalize(self, z):  
        z = torch.flatten(z, 1)    
        return F.normalize(torch.matmul(z, torch.t(z)), 1)

    def forward(self, y_s, y_t):
        assert len(y_s) == len(y_t) 
        losses = []
     
        for idx, (s, t) in enumerate(zip(y_s, y_t)): 
            assert s.shape == t.shape
            g_t = self.matmul_and_normalize(t)     
            g_s = self.matmul_and_normalize(s)   
   
            sp_loss = torch.norm(g_t - g_s) ** 2    
            sp_loss = sp_loss.sum() / s.size(0)     
            losses.append(sp_loss) 
        loss = sum(losses)  
        return loss