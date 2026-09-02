import os, torch, cv2, math, tqdm, time, shutil, argparse, json, pickle    
import numpy as np
from pathlib import Path    
from faster_coco_eval import mask as mask_utils
from prettytable import PrettyTable    
from ..logger_module import get_logger     
from .utils import visualize_masks, create_comparison_plot    

RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"    
logger = get_logger(__name__)
DEBUG = False

def clip_boxes(boxes, shape):   
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2     
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)  
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):    
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape  
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:   
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding  
    boxes[..., [1, 3]] -= pad[1]  # y padding 
    boxes[..., :4] /= gain 
    clip_boxes(boxes, img0_shape)   
    return boxes

def box_iou(box1, box2, eps=1e-7):   
    """
    Calculate intersection-over-union (IoU) of boxes. Both sets of boxes are expected to be in (x1, y1, x2, y2) format.    
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py  

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.     
    """     

    # NOTE: Need .float() to get accurate iou values   
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)   
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)     

    # IoU = inter / (area1 + area2 - inter)     
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)   
   
def mask_iou(mask1: torch.Tensor, mask2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Calculate masks IoU. 

    Args:
        mask1 (torch.Tensor): A tensor of shape (N, n) where N is the number of ground truth objects and n is the
            product of image width and height.
        mask2 (torch.Tensor): A tensor of shape (M, n) where M is the number of predicted objects and n is the product   
            of image width and height.
        eps (float, optional): A small value to avoid division by zero.
    
    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing masks IoU.     
    """
    intersection = torch.matmul(mask1, mask2.T).clamp_(0)
    union = (mask1.sum(1)[:, None] + mask2.sum(1)[None]) - intersection  # (area1 + area2) - intersection   
    return intersection / (union + eps)  
    
def process_batch_det(detections, labels, iouv):
    """    
    Return correct prediction matrix
    Arguments:     
        detections (array[N, 6]), x1, y1, x2, y2, conf, class     
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns: 
        correct (array[N, 10]), for 10 IoU levels    
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool) 
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match 
        if x[0].shape[0]: 
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:  
                matches = matches[matches[:, 2].argsort()[::-1]]    
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  
                # matches = matches[matches[:, 2].argsort()[::-1]]     
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

def process_batch_mask(pred_masks, pred_cls, label_mask, label_cls, iouv):
    correct = np.zeros((pred_masks.shape[0], iouv.shape[0]), dtype=bool)     
    iou = mask_iou(label_mask.flatten(1), pred_masks.flatten(1).float())    
    correct_class = label_cls[:, np.newaxis] == pred_cls     
    for i in range(len(iouv)):    
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match    
        if x[0].shape[0]: 
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]    
            if x[0].shape[0] > 1:     
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]    
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)    

def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed    

     
def ap_per_class(tp, conf, pred_cls, target_cls, eps=1e-16):
    """ Compute the average precision, given the recall and precision curves.     
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).  
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.   
    """  

    # Sort by objectness    
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes  
    unique_classes, nt = np.unique(target_cls, return_counts=True)     
    nc = unique_classes.shape[0]  # number of classes, number of detections  
   
    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting  
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))  
    for ci, c in enumerate(unique_classes):    
        i = pred_cls == c
        n_l = nt[ci]  # number of labels    
        n_p = i.sum()  # number of predictions 
        if n_p == 0 or n_l == 0:    
            continue     

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)    

        # Recall  
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases  

        # Precision   
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score
 
        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])    
  
    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)   
  
    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives    
    fp = (tp / (p + eps) - tp).round()  # false positives  
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)    

  
def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves     
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns  
        Average precision, precision curve, recall curve
    """    

    # Append sentinel values to beginning and end    
    mrec = np.concatenate(([0.0], recall, [1.0]))    
    mpre = np.concatenate(([1.0], precision, [0.0]))    
   
    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))     
   
    # Integrate area under curve 
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes   
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve
   
    return ap, mpre, mrec

class ConfusionMatrix:
    """ 
    目标检测混淆矩阵类(包含背景类)
    """     
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        """
        Args:
            nc: 类别数量(不包含背景)
            conf: 置信度阈值
            iou_thres: IoU阈值 
        """    
        self.matrix = np.zeros((nc + 1, nc + 1))  # +1 for background class
        self.nc = nc  # 目标类别数量
        self.conf = conf
        self.iou_thres = iou_thres  
  
    def process_batch(self, detections, labels):
        """
        更新混淆矩阵    
        Args: 
            detections: (N, 6) [x1, y1, x2, y2, conf, class]
            labels: (M, 5) [class, x1, y1, x2, y2]     
        """
        if detections is None:     
            detections = np.zeros((0, 6))
        if labels is None:
            labels = np.zeros((0, 5))  
            
        # 过滤低置信度预测
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].astype(int)
        detection_classes = detections[:, 5].astype(int) 
        
        # 计算IoU矩阵     
        iou = self.box_iou(labels[:, 1:], detections[:, :4])   
  
        # 匹配预测和真实标签   
        x = np.where(iou > self.iou_thres)
  
        if x[0].shape[0]:  # 如果存在匹配
            matches = np.concatenate((np.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
            if x[0].shape[0] > 1:  
                matches = matches[matches[:, 2].argsort()[::-1]]  # 按IoU降序排列 
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # 每个预测只匹配一次    
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # 每个GT只匹配一次   
        else:     
            matches = np.zeros((0, 3))     
    
        n = matches.shape[0] > 0 
        m0, m1, _ = matches.transpose().astype(int)
    
        # 更新混淆矩阵   
        for i, gc in enumerate(gt_classes):
            j = m0 == i   
            if n and sum(j) == 1:  
                # 正确匹配  
                self.matrix[gc, detection_classes[m1[j]]] += 1  # TP
            else:   
                # 漏检 (FN) - GT被分类为背景 
                self.matrix[gc, self.nc] += 1  # background FN

        if n:    
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i): 
                    # 误检 (FP) - 预测被分类为背景    
                    self.matrix[self.nc, dc] += 1  # background FP     
        else: 
            # 所有预测都是误检     
            for dc in detection_classes:    
                self.matrix[self.nc, dc] += 1
     
    def matrix_to_metrics(self):
        """  
        从混淆矩阵计算精确率、召回率、F1分数   
        Returns:
            precision: 每个类别的精确率  
            recall: 每个类别的召回率   
            f1: 每个类别的F1分数
        """
        tp = np.diag(self.matrix)[:-1]  # 不包含背景类    
        fp = self.matrix[:-1, :-1].sum(1) - tp  # 每行的和减去对角线   
        fn = self.matrix[:-1, :-1].sum(0) - tp  # 每列的和减去对角线     
  
        precision = tp / (tp + fp + 1e-16)     
        recall = tp / (tp + fn + 1e-16)
        f1 = 2 * precision * recall / (precision + recall + 1e-16)
  
        return precision, recall, f1
    
    def print(self):
        """打印混淆矩阵"""
        logger.info(RED + "------------------------ Confusion Matrix ------------------------" + RESET)
        for i in range(self.nc + 1):
            logger.info(' '.join(map(str, self.matrix[i])))
    
    def plot(self, save_dir='', names=None):   
        """ 
        绘制混淆矩阵
        Args:     
            save_dir: 保存路径
            names: 类别名称列表(包含背景类)   
        """     
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns     
            from pathlib import Path 
            
            array = self.matrix / (self.matrix.sum(0).reshape(1, -1) + 1E-9)  # 归一化
            array[array < 0.005] = np.nan  # 不显示小于0.5%的值   
 
            fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
            nc, nn = self.nc, len(names) if names else self.nc + 1  
            
            # 设置类别名称 
            if names is None:     
                names = [f'Class {i}' for i in range(nc)] + ['Background']
            
            # 绘制热力图
            sns.set(font_scale=1.0)
            labels = (0 < nn < 99) and (nn == nc + 1)  # 是否显示标签
            ticklabels = names if labels else "auto"
            
            sns.heatmap(array,
                       ax=ax,
                       annot=nc < 30,
                       annot_kws={"size": 8},
                       cmap='rocket',
                       fmt='.2f',
                       square=True,
                       vmin=0.0,  
                       xticklabels=ticklabels,
                       yticklabels=ticklabels).set_facecolor((1, 1, 1))   
  
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')    
            ax.set_title('Confusion Matrix (Normalized)')   
            
            # 保存图片 
            fig.savefig(save_dir / 'confusion_matrix.png', dpi=250) 
            plt.close()
            logger.info(f"Confusion matrix saved to {save_dir / 'confusion_matrix.png'}")
        except Exception as e:    
            logger.warning(f"Failed to plot confusion matrix: {e}")  

    @staticmethod
    def box_iou(box1, box2):     
        """ 
        计算两组框之间的IoU
        Args:    
            box1: (N, 4) [x1, y1, x2, y2]     
            box2: (M, 4) [x1, y1, x2, y2]
        Returns: 
            iou: (N, M) IoU矩阵
        """  
        def box_area(box):  
            return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]) 

        area1 = box_area(box1)
        area2 = box_area(box2)

        # 计算交集
        inter = (np.minimum(box1[:, None, 2:], box2[:, 2:]) -  
                np.maximum(box1[:, None, :2], box2[:, :2])).clip(0).prod(2)  
 
        # 计算IoU   
        return inter / (area1[:, None] + area2 - inter)     

def get_yolo_det_metrice(coco_evaluator, coco_pred_json, save_dir=None):
    logger.info(RED + "------------------------ YOLO Metrics[bbox] Start ------------------------" + RESET)     
    logger.info(RED + "------------------------ YOLO指标和COCO指标有差异属于正常情况 ------------------------" + RESET)
    iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    stats = []
   
    cocoGt = coco_evaluator.coco_eval['bbox'].cocoGt

    classes = []     
    for data in cocoGt.dataset['categories']:   
        classes.append(data['name']) 

    image_id = []
    for data in cocoGt.anns:
        image_id.append(cocoGt.anns[data]['image_id'])   
    image_id = sorted(list(set(image_id)))    
    image_id_dict = {i:{'pred':[], 'label':[]} for i in image_id}
   
    for key in cocoGt.anns: 
        data = cocoGt.anns[key]
        category_id = data['category_id']     
        x_min, y_min, w, h = data['bbox'][0], data['bbox'][1], data['bbox'][2], data['bbox'][3]    
        x_max, y_max = x_min + w, y_min + h
    
        image_id_dict[data['image_id']]['label'].append(np.array([int(category_id), x_min, y_min, x_max, y_max]))
 
    for data in coco_pred_json:
        score = data['score']
        category_id = data['category_id']
        x_min, y_min, w, h = data['bbox'][0], data['bbox'][1], data['bbox'][2], data['bbox'][3]    
        x_max, y_max = x_min + w, y_min + h  

        if data['image_id'] not in image_id_dict: 
            image_id_dict[data['image_id']] = {'pred':[], 'label':[]}

        image_id_dict[data['image_id']]['pred'].append(np.array([x_min, y_min, x_max, y_max, float(score), int(category_id)]))
    
    confusion_matrix = ConfusionMatrix(nc=len(classes), conf=0.25, iou_thres=0.5)
    for idx, image_id in enumerate(tqdm.tqdm(list(image_id_dict.keys()), desc="Cal YOLO(bbox) mAP...")):
        label = np.array(image_id_dict[image_id]['label'])
        pred = np.array(image_id_dict[image_id]['pred'])  
   
        if len(pred) == 0:
            pred = np.empty((0, 6))
        else:   
            pred = torch.from_numpy(pred[pred[:, 4].argsort()[::-1]])     
        
        if len(label) == 0:  
            label = np.empty((0, 5))  
   
        nl, npr = label.shape[0], pred.shape[0] 
        correct = torch.zeros(npr, niou, dtype=torch.bool)     
        if npr == 0:
            if nl:
                stats.append((correct, *torch.zeros((2, 0)), torch.from_numpy(label[:, 0])))
            continue     
  
        if nl: 
            correct = process_batch_det(pred, torch.from_numpy(label), iouv)     
        stats.append((correct, pred[:, 4], pred[:, 5], torch.from_numpy(label[:, 0])))

        if save_dir is not None:
            # 更新混淆矩阵   
            confusion_matrix.process_batch(pred.numpy(), label)
    
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  
    tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats)

    if save_dir is not None: 
        confusion_matrix.plot(save_dir=save_dir, names=classes + ['background'])

    table = PrettyTable()
    table.title = f"YOLO Metrics(bbox)"
    table.field_names = ["Classes", 'Precision', 'Recall', 'F1-Score', 'mAP50', 'mAP75', 'mAP50-95']
    table.add_row(['all', f'{np.mean(p):.3f}', f'{np.mean(r):.3f}', f'{np.mean(f1):.3f}', f'{np.mean(ap[:, 0]):.3f}', f'{np.mean(ap[:, 5]):.3f}', f'{np.mean(ap):.3f}'])
    for cls_idx, classes in enumerate(classes):
        table.add_row([classes, f'{p[cls_idx]:.3f}', f'{r[cls_idx]:.3f}', f'{f1[cls_idx]:.3f}', f'{ap[cls_idx, 0]:.3f}', f'{ap[cls_idx, 5]:.3f}', f'{ap[cls_idx, :].mean():.3f}'])     
    print(table)
 
def get_yolo_seg_metrice(coco_evaluator, coco_pred_json, save_vis=False, vis_dir='./yolo_mask_visualization'): 
    logger.info(RED + "------------------------ YOLO Metrics[segm] Start ------------------------" + RESET)
    logger.info(RED + "------------------------ YOLO指标和COCO指标有差异属于正常情况 ------------------------" + RESET) 
    iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95   
    niou = iouv.numel()
    stats = []  
 
    cocoGt = coco_evaluator.coco_gt['segm'] 

    classes = []
    for data in cocoGt.dataset['categories']:    
        classes.append(data['name'])
     
    image_size_dict = {}
    for data in cocoGt.imgs: 
        image_size_dict[cocoGt.imgs[data]['id']] = (cocoGt.imgs[data]['height'], cocoGt.imgs[data]['width'])   
     
    image_id = []     
    for data in cocoGt.anns:   
        image_id.append(cocoGt.anns[data]['image_id'])
    image_id = sorted(list(set(image_id)))   
    image_id_dict = {i:{'pred_mask':[], 'pred_cls':[], 'pred_score':[], 'label_mask':[], 'labek_cls':[]} for i in image_id}

    for key in cocoGt.anns:
        data = cocoGt.anns[key]
        category_id = data['category_id']    
        mask = mask_utils.decode(data['segmentation']) 

        image_id_dict[data['image_id']]['labek_cls'].append(category_id)
        image_id_dict[data['image_id']]['label_mask'].append(mask)   
    
    for data in coco_pred_json:     
        score = data['score']
        category_id = data['category_id']   
        mask = mask_utils.decode(data['segmentation']) 

        if data['image_id'] not in image_id_dict:    
            continue
    
        image_id_dict[data['image_id']]['pred_score'].append(score)     
        image_id_dict[data['image_id']]['pred_cls'].append(category_id)     
        image_id_dict[data['image_id']]['pred_mask'].append(mask)    
    
    # 创建可视化目录  
    if save_vis:    
        vis_path = Path(vis_dir)
        vis_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Saving mask visualizations to: {vis_path}")  
  
    for idx, image_id in enumerate(tqdm.tqdm(list(image_id_dict.keys()), desc="Cal YOLO(segm) mAP...")):
        label_cls = np.array(image_id_dict[image_id]['labek_cls'])   
        label_mask = np.array(image_id_dict[image_id]['label_mask'])
        pred_conf = np.array(image_id_dict[image_id]['pred_score'])
        pred_cls = np.array(image_id_dict[image_id]['pred_cls'])    
        pred_mask = np.array(image_id_dict[image_id]['pred_mask'])   
     
        # ========== 可视化部分 ==========    
        if save_vis:
            # 为每个image_id创建文件夹
            img_vis_dir = vis_path / f"image_{image_id}"
            img_vis_dir.mkdir(exist_ok=True)
     
            # 保存label masks
            label_dir = img_vis_dir / "labels" 
            label_dir.mkdir(exist_ok=True)
 
            for i, (mask, cls) in enumerate(zip(label_mask, label_cls)):
                # 单独保存每个mask
                mask_img = (mask * 255).astype(np.uint8) 
                cv2.imwrite(str(label_dir / f"label_{i}_cls_{cls}.png"), mask_img) 
 
            # 保存合并的label mask（不同颜色）
            if len(label_mask) > 0:     
                combined_label = visualize_masks(label_mask, label_cls, classes, "Ground Truth")   
                cv2.imwrite(str(img_vis_dir / "labels_combined.png"), combined_label)
            
            # 保存prediction masks
            pred_dir = img_vis_dir / "predictions" 
            pred_dir.mkdir(exist_ok=True)     
     
            for i, (mask, cls, conf) in enumerate(zip(pred_mask, pred_cls, pred_conf)):
                # 单独保存每个mask
                mask_img = (mask * 255).astype(np.uint8)
                cv2.imwrite(str(pred_dir / f"pred_{i}_cls_{cls}_conf_{conf:.3f}.png"), mask_img)     
            
            # 保存合并的prediction mask（不同颜色）  
            if len(pred_mask) > 0:   
                combined_pred = visualize_masks(pred_mask, pred_cls, classes, "Predictions", pred_conf)
                cv2.imwrite(str(img_vis_dir / "predictions_combined.png"), combined_pred)
   
            # 创建对比图  
            create_comparison_plot(img_vis_dir, label_mask, label_cls, pred_mask, pred_cls, pred_conf, classes, image_id)
        # ========== 可视化部分结束 ==========
        
        if len(pred_conf) == 0:
            pred_conf = np.empty((0, 1))    
            pred_cls = np.empty((0, 1))
            pred_mask = np.empty((0, 1))     
        else:
            pred_conf = torch.from_numpy(np.array(pred_conf))
            pred_cls = torch.from_numpy(np.array(pred_cls))  
            pred_mask = torch.from_numpy(np.array(pred_mask))
        
        nl, npr = label_cls.shape[0], pred_conf.shape[0]
        correct = torch.zeros(npr, niou, dtype=torch.bool)  
        if npr == 0:    
            if nl:
                stats.append((correct, *torch.zeros((2, 0)), torch.from_numpy(label_cls)))     
            continue    
  
        if nl:   
            correct = process_batch_mask(pred_mask.float(), pred_cls, torch.from_numpy(label_mask).float(), torch.from_numpy(label_cls), iouv)  
        stats.append((correct, pred_conf, pred_cls, torch.from_numpy(label_cls)))     
  
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
    tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats)

    table = PrettyTable() 
    table.title = f"YOLO Metrics(segm)"
    table.field_names = ["Classes", 'Precision', 'Recall', 'F1-Score', 'mAP50', 'mAP75', 'mAP50-95']  
    table.add_row(['all', f'{np.mean(p):.3f}', f'{np.mean(r):.3f}', f'{np.mean(f1):.3f}', f'{np.mean(ap[:, 0]):.3f}', f'{np.mean(ap[:, 5]):.3f}', f'{np.mean(ap):.3f}'])   
    for cls_idx, classes in enumerate(classes):   
        if cls_idx in list(ap_class):
            table.add_row([classes, f'{p[cls_idx]:.3f}', f'{r[cls_idx]:.3f}', f'{f1[cls_idx]:.3f}', f'{ap[cls_idx, 0]:.3f}', f'{ap[cls_idx, 5]:.3f}', f'{ap[cls_idx, :].mean():.3f}'])   
    print(table)   
