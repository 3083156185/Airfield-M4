import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLOR_LIST = [
    (255, 0, 0),         # 红色 (person)
    (0, 255, 0),         # 绿色 (car)
    (0, 0, 255),         # 蓝色 (bike)
    (255, 165, 0),       # 橙色 (motorcycle)
    (255, 255, 0),       # 黄色 (truck)
    (0, 255, 255),       # 青色 (bus)
    (255, 0, 255),       # 品红 (train)
    (128, 0, 0),         # 棕色 (dog)
    (0, 128, 0),         # 深绿色 (cat)
    (0, 0, 128),         # 深蓝色 (horse)
    (128, 128, 0),       # 橄榄色 (sheep)
    (0, 128, 128),       # 蓝绿色 (cow)
    (128, 0, 128),       # 紫色 (elephant)
    (192, 192, 192),     # 银色 (giraffe)
    (255, 99, 71),       # 番茄色 (zebra)
    (0, 255, 127),       # 春绿色 (monkey)
    (255, 105, 180),     # 深粉色 (bird)
    (70, 130, 180),      # 钢蓝色 (fish)
]

def get_color_by_class(class_id):
    # 根据类别的索引返回固定颜色
    return COLOR_LIST[class_id % len(COLOR_LIST)]  # 确保索引不越界

# 获取动态调整的字体
def get_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)  # 你可以指定字体路径
    except IOError:
        return ImageFont.load_default()  # 如果加载失败，使用默认字体

# font_scale 越大 绘制字体越大 反之亦言
# box_thickness 越大 绘制框越大 反之亦言
def draw(images, labels, boxes, scores, masks=None, tracks=None, thrh=0.4, font_scale=0.6, box_thickness=2, class_name=None, mask_alpha=0.5):
    """
    绘制检测框和实例分割mask
    
    Args:
        images: 图像列表
        labels: 标签列表
        boxes: 边界框列表
        scores: 置信度分数列表
        masks: mask列表 (可选，用于实例分割)
        tracks: 跟踪id列别 (可选)
        thrh: 置信度阈值
        font_scale: 字体大小因子
        box_thickness: 框线粗细因子
        class_name: 类别名称列表
        mask_alpha: mask透明度 (0-1之间)
    """
    for i, im in enumerate(images):
        im_cv = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        # 创建一个副本用于绘制mask
        if masks is not None:
            im_array = np.array(im)
            mask_overlay = np.zeros_like(im_array, dtype=np.uint8)

        if tracks is None:
            scr = scores[i]
            lab = labels[i][scr > thrh]
            box = boxes[i][scr > thrh]
            scrs = scr[scr > thrh]
        else:
            scrs = tracks[i][:, 5]
            lab = tracks[i][:, 6]
            box = tracks[i][:, :4]
            track_id = tracks[i][:, 4]
        
        # 过滤mask
        if masks is not None and tracks is None:
            msk = masks[i][scr > thrh]

        w, h = im.size  # 获取图像宽高

        for j, b in enumerate(box):
            lab_id = int(lab[j].item())
            color = get_color_by_class(lab_id)
            color_bgr = (color[2], color[1], color[0])  # RGB to BGR

            # 绘制框
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(im_cv, (x1, y1), (x2, y2), color_bgr, box_thickness)

            # 绘制类别名称和分数
            text = f"{class_name[lab_id] if class_name else lab_id} {round(scrs[j].item(), 2)}"
            if tracks is not None:
                text = f"{int(track_id[j].item())} {text}"

            # 获取文本大小
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
            )
            
            # 绘制文本背景
            cv2.rectangle(im_cv, 
                         (x1, y1 - text_height - 10),
                         (x1 + text_width, y1),
                         color_bgr, -1)
            
            # 绘制文本
            cv2.putText(im_cv, text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                       (255, 255, 255), 2, cv2.LINE_AA)  # LINE_AA 抗锯齿
            
            # 绘制mask
            if masks is not None:
                mask = msk[j]
                
                # 如果mask是tensor，转换为numpy
                if torch.is_tensor(mask):
                    mask = mask.cpu().detach().numpy()
                
                # 获取box坐标并转换为整数
                x1, y1, x2, y2 = map(int, b)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # 创建一个只在box区域内的mask
                box_mask = np.zeros_like(mask, dtype=bool)
                box_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
                
                for c in range(3):
                    mask_overlay[:, :, c][box_mask] = color_bgr[c]
        
        # 将mask叠加到原图上
        if masks is not None:
            # 使用alpha混合
            im_cv = (im_cv * (1 - mask_alpha) + mask_overlay * mask_alpha).astype(np.uint8)

    return Image.fromarray(cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB))