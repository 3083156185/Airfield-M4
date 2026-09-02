import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')

import warnings
warnings.filterwarnings('ignore')
import json, argparse, tqdm, cv2, shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from collections import defaultdict

from engine.logger_module import get_logger

logger = get_logger(__name__)

def iou(box1, box2):
    x11, y11, x12, y12 = np.split(box1, 4, axis=1)
    x21, y21, x22, y22 = np.split(box2, 4, axis=1)
 
    xa = np.maximum(x11, np.transpose(x21))
    xb = np.minimum(x12, np.transpose(x22))
    ya = np.maximum(y11, np.transpose(y21))
    yb = np.minimum(y12, np.transpose(y22))
 
    area_inter = np.maximum(0, (xb - xa + 1)) * np.maximum(0, (yb - ya + 1))
 
    area_1 = (x12 - x11 + 1) * (y12 - y11 + 1)
    area_2 = (x22 - x21 + 1) * (y22 - y21 + 1)
    area_union = area_1 + np.transpose(area_2) - area_inter
 
    iou = area_inter / area_union
    return iou

def get_box_area(box):
    """计算框的面积"""
    return (box[2] - box[0]) * (box[3] - box[1])

def classify_object_size(area, image_area):
    """
    根据目标面积占图像面积的比例分类目标尺寸
    小目标: < 0.12% (32x32 in 1024x1024)
    中目标: 0.12% - 1% 
    大目标: > 1%
    """
    ratio = area / image_area
    if ratio < 0.0012:
        return 'small'
    elif ratio < 0.01:
        return 'medium'
    else:
        return 'large'

def classify_fp_type(pred_box, pred_class, all_labels, iou_threshold):
    """
    分类FP错误类型:
    - duplicate: 重复检测（与已匹配的GT重叠）
    - confusion: 类别混淆（与GT重叠但类别不同）
    - localization: 定位错误（IoU在0.1-0.5之间）
    - background: 背景误检（与所有GT的IoU都很小）
    """
    if len(all_labels) == 0:
        return 'background'
    
    max_iou = 0
    same_class_max_iou = 0
    diff_class_max_iou = 0
    
    for label in all_labels:
        label_class = label[0]
        label_box = label[1:5]
        
        current_iou = iou(pred_box.reshape(1, -1), label_box.reshape(1, -1))[0][0]
        max_iou = max(max_iou, current_iou)
        
        if label_class == pred_class:
            same_class_max_iou = max(same_class_max_iou, current_iou)
        else:
            diff_class_max_iou = max(diff_class_max_iou, current_iou)
    
    # 判断错误类型
    if same_class_max_iou >= iou_threshold:
        return 'duplicate'
    elif diff_class_max_iou >= iou_threshold:
        return 'confusion'
    elif max_iou >= 0.1:
        return 'localization'
    else:
        return 'background'

def classify_fn_type(label_box, label_class, all_preds, iou_threshold):
    """
    分类FN错误类型:
    - missed: 完全漏检（与所有预测的IoU都很小）
    - low_iou: IoU不足（有预测但IoU < threshold）
    - confusion: 类别混淆（被检测为其他类别）
    """
    if len(all_preds) == 0:
        return 'missed'
    
    max_iou = 0
    same_class_max_iou = 0
    diff_class_max_iou = 0
    
    for pred in all_preds:
        pred_class = pred[0]
        pred_box = pred[1:5]
        
        current_iou = iou(label_box.reshape(1, -1), pred_box.reshape(1, -1))[0][0]
        max_iou = max(max_iou, current_iou)
        
        if pred_class == label_class:
            same_class_max_iou = max(same_class_max_iou, current_iou)
        else:
            diff_class_max_iou = max(diff_class_max_iou, current_iou)
    
    # 判断错误类型
    if diff_class_max_iou >= iou_threshold:
        return 'confusion'
    elif same_class_max_iou >= 0.1:
        return 'low_iou'
    else:
        return 'missed'

def draw_box(img, box, color):
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness=2)
    return img

# def draw_legend(img, detect_color, missing_color, error_color, right_num, error_num, missing_num):
#     """在图像上绘制图例"""
#     height, width = img.shape[:2]
    
#     # 图例位置和尺寸
#     legend_height = 120
#     legend_width = 250
#     margin = 10
#     x_start = width - legend_width - margin
#     y_start = margin
    
#     # 创建半透明背景
#     overlay = img.copy()
#     cv2.rectangle(overlay, (x_start, y_start), 
#                   (x_start + legend_width, y_start + legend_height), 
#                   (255, 255, 255), -1)
#     img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    
#     # 绘制边框
#     cv2.rectangle(img, (x_start, y_start), 
#                   (x_start + legend_width, y_start + legend_height), 
#                   (0, 0, 0), 2)
    
#     # 绘制图例项
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 0.6
#     thickness = 2
#     line_height = 35
    
#     # TP (绿色)
#     y_pos = y_start + 30
#     cv2.rectangle(img, (x_start + 10, y_pos - 10), 
#                   (x_start + 40, y_pos + 10), detect_color, -1)
#     cv2.putText(img, f'TP (Correct) {right_num}', (x_start + 50, y_pos + 5), 
#                 font, font_scale, (0, 0, 0), thickness)
    
#     # FN (蓝色)
#     y_pos += line_height
#     cv2.rectangle(img, (x_start + 10, y_pos - 10), 
#                   (x_start + 40, y_pos + 10), missing_color, -1)
#     cv2.putText(img, f'FN (Missing) {missing_num}', (x_start + 50, y_pos + 5), 
#                 font, font_scale, (0, 0, 0), thickness)
    
#     # FP (红色)
#     y_pos += line_height
#     cv2.rectangle(img, (x_start + 10, y_pos - 10), 
#                   (x_start + 40, y_pos + 10), error_color, -1)
#     cv2.putText(img, f'FP (Error) {error_num}', (x_start + 50, y_pos + 5), 
#                 font, font_scale, (0, 0, 0), thickness)
    
#     return img

def draw_legend(img, detect_color, missing_color, error_color, right_num, error_num, missing_num):
    """在图像上绘制图例（自适应尺寸）"""
    height, width = img.shape[:2]
    
    # 根据图像尺寸自适应调整参数
    # 基准尺寸：1920x1080
    scale_factor = min(width / 1920, height / 1080)
    scale_factor = max(0.5, min(scale_factor, 2.0))  # 限制缩放范围在0.5-2.0之间
    
    # 自适应尺寸参数
    legend_height = int(120 * scale_factor)
    legend_width = int(200 * scale_factor)
    margin = int(10 * scale_factor)
    box_size = int(15 * scale_factor)  # 色块大小
    font_scale = 0.6 + 0.2 * scale_factor  # 字体大小
    thickness = max(2, int(2 * scale_factor))  # 线条粗细
    line_height = int(35 * scale_factor)
    padding = int(10 * scale_factor)
    
    # 计算位置（右上角）
    x_start = width - legend_width - margin
    y_start = margin
    
    # 确保不会超出图像边界
    if x_start < 0:
        x_start = margin
    if y_start + legend_height > height:
        y_start = height - legend_height - margin
    
    # 创建半透明背景
    overlay = img.copy()
    cv2.rectangle(overlay, (x_start, y_start), 
                  (x_start + legend_width, y_start + legend_height), 
                  (255, 255, 255), -1)
    img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    
    # 绘制边框
    border_thickness = max(1, int(2 * scale_factor))
    cv2.rectangle(img, (x_start, y_start), 
                  (x_start + legend_width, y_start + legend_height), 
                  (0, 0, 0), border_thickness)
    
    # 绘制图例项
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # 计算文本和色块的位置
    text_x = x_start + padding + box_size * 2 + 5
    box_x = x_start + padding
    
    # TP (绿色)
    y_pos = y_start + int(30 * scale_factor)
    cv2.rectangle(img, (box_x, y_pos - box_size), 
                  (box_x + box_size * 2, y_pos + box_size), detect_color, -1)
    
    # 动态调整文本，如果数字太大就换行或缩小
    text = f'TP: {right_num}'
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    if text_size[0] > legend_width - text_x - padding:
        font_scale_adjusted = font_scale * 0.8
    else:
        font_scale_adjusted = font_scale
    
    cv2.putText(img, text, (text_x, y_pos + int(5 * scale_factor)), 
                font, font_scale_adjusted, (0, 150, 0), thickness, cv2.LINE_AA)
    
    # FN (蓝色)
    y_pos += line_height
    cv2.rectangle(img, (box_x, y_pos - box_size), 
                  (box_x + box_size * 2, y_pos + box_size), missing_color, -1)
    
    text = f'FN: {missing_num}'
    cv2.putText(img, text, (text_x, y_pos + int(5 * scale_factor)), 
                font, font_scale_adjusted, (0, 0, 200), thickness, cv2.LINE_AA)
    
    # FP (红色)
    y_pos += line_height
    cv2.rectangle(img, (box_x, y_pos - box_size), 
                  (box_x + box_size * 2, y_pos + box_size), error_color, -1)
    
    text = f'FP: {error_num}'
    cv2.putText(img, text, (text_x, y_pos + int(5 * scale_factor)), 
                font, font_scale_adjusted, (200, 0, 0), thickness, cv2.LINE_AA)
    
    return img

def plot_size_performance(size_stats, output_path):
    """绘制不同尺寸目标的检测性能对比图"""
    sizes = ['small', 'medium', 'large']
    size_labels = ['Small', 'Medium', 'Large']
    
    tp_values = [size_stats[s]['tp'] for s in sizes]
    fp_values = [size_stats[s]['fp'] for s in sizes]
    fn_values = [size_stats[s]['fn'] for s in sizes]
    
    # 计算召回率和精确率
    recalls = []
    precisions = []
    for s in sizes:
        tp = size_stats[s]['tp']
        fp = size_stats[s]['fp']
        fn = size_stats[s]['fn']
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recalls.append(recall * 100)
        precisions.append(precision * 100)
    
    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. TP/FP/FN 柱状图
    x = np.arange(len(sizes))
    width = 0.25
    
    axes[0].bar(x - width, tp_values, width, label='TP', color='green', alpha=0.8)
    axes[0].bar(x, fp_values, width, label='FP', color='red', alpha=0.8)
    axes[0].bar(x + width, fn_values, width, label='FN', color='blue', alpha=0.8)
    
    axes[0].set_xlabel('Object Size', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[0].set_title('TP/FP/FN by Object Size', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(size_labels)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, (tp, fp, fn) in enumerate(zip(tp_values, fp_values, fn_values)):
        axes[0].text(i - width, tp + 1, str(tp), ha='center', va='bottom', fontsize=9)
        axes[0].text(i, fp + 1, str(fp), ha='center', va='bottom', fontsize=9)
        axes[0].text(i + width, fn + 1, str(fn), ha='center', va='bottom', fontsize=9)
    
    # 2. Precision & Recall 对比图
    x2 = np.arange(len(sizes))
    width2 = 0.35
    
    axes[1].bar(x2 - width2/2, precisions, width2, label='Precision', color='orange', alpha=0.8)
    axes[1].bar(x2 + width2/2, recalls, width2, label='Recall', color='cyan', alpha=0.8)
    
    axes[1].set_xlabel('Object Size', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Precision & Recall by Object Size', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(size_labels)
    axes[1].set_ylim([0, 105])
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, (p, r) in enumerate(zip(precisions, recalls)):
        axes[1].text(i - width2/2, p + 1, f'{p:.1f}%', ha='center', va='bottom', fontsize=9)
        axes[1].text(i + width2/2, r + 1, f'{r:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 3. 目标数量分布饼图
    total_objects = [size_stats[s]['tp'] + size_stats[s]['fn'] for s in sizes]
    colors_pie = ['#ff9999', '#66b3ff', '#99ff99']
    
    axes[2].pie(total_objects, labels=size_labels, autopct='%1.1f%%', 
                colors=colors_pie, startangle=90, textprops={'fontsize': 11})
    axes[2].set_title('Object Distribution by Size', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'size_performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f'Size performance comparison plot saved to {output_path}')

def plot_error_distribution(fp_types, fn_types, output_path):
    """绘制错误类型分布图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # FP 错误类型分布
    fp_labels = list(fp_types.keys())
    fp_values = list(fp_types.values())
    fp_colors = ['#ff6b6b', '#ee5a6f', '#c44569', '#f8b500']
    
    if sum(fp_values) > 0:
        wedges, texts, autotexts = axes[0].pie(fp_values, labels=fp_labels, autopct='%1.1f%%',
                                                 colors=fp_colors, startangle=90,
                                                 textprops={'fontsize': 10})
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    axes[0].set_title('False Positive Error Types', fontsize=14, fontweight='bold', pad=20)
    
    # 添加图例说明
    fp_legend_labels = [
        f'Background: {fp_types.get("background", 0)}',
        f'Confusion: {fp_types.get("confusion", 0)}',
        f'Localization: {fp_types.get("localization", 0)}',
        f'Duplicate: {fp_types.get("duplicate", 0)}'
    ]
    axes[0].legend(fp_legend_labels, loc='upper left', bbox_to_anchor=(0, 0, 0.5, 1), fontsize=9)
    
    # FN 错误类型分布
    fn_labels = list(fn_types.keys())
    fn_values = list(fn_types.values())
    fn_colors = ['#4834d4', '#686de0', '#30336b']
    
    if sum(fn_values) > 0:
        wedges, texts, autotexts = axes[1].pie(fn_values, labels=fn_labels, autopct='%1.1f%%',
                                                 colors=fn_colors, startangle=90,
                                                 textprops={'fontsize': 10})
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    axes[1].set_title('False Negative Error Types', fontsize=14, fontweight='bold', pad=20)
    
    # 添加图例说明
    fn_legend_labels = [
        f'Missed: {fn_types.get("missed", 0)}',
        f'Low IoU: {fn_types.get("low_iou", 0)}',
        f'Confusion: {fn_types.get("confusion", 0)}'
    ]
    axes[1].legend(fn_legend_labels, loc='upper left', bbox_to_anchor=(0, 0, 0.5, 1), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'error_type_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f'Error type distribution plot saved to {output_path}')

def plot_combined_error_analysis(fp_types, fn_types, output_path):
    """绘制综合错误分析图（柱状图）"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 准备数据
    fp_categories = ['Background', 'Confusion', 'Localization', 'Duplicate']
    fn_categories = ['Missed', 'Low IoU', 'Confusion']
    
    fp_values = [
        fp_types.get('background', 0),
        fp_types.get('confusion', 0),
        fp_types.get('localization', 0),
        fp_types.get('duplicate', 0)
    ]
    
    fn_values = [
        fn_types.get('missed', 0),
        fn_types.get('low_iou', 0),
        fn_types.get('confusion', 0)
    ]
    
    # 绘制FP柱状图
    x_fp = np.arange(len(fp_categories))
    bars1 = ax.bar(x_fp, fp_values, 0.4, label='False Positive', color='#ff6b6b', alpha=0.8)
    
    # 绘制FN柱状图
    x_fn = np.arange(len(fn_categories)) + len(fp_categories) + 0.5
    bars2 = ax.bar(x_fn, fn_values, 0.4, label='False Negative', color='#4834d4', alpha=0.8)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 设置标签
    all_categories = fp_categories + [''] + fn_categories
    all_positions = list(x_fp) + [len(fp_categories) + 0.25] + list(x_fn)
    ax.set_xticks(all_positions)
    ax.set_xticklabels(all_categories, rotation=15, ha='right')
    
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Detailed Error Type Analysis', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加分隔线
    ax.axvline(x=len(fp_categories) + 0.25, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'error_type_analysis_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f'Combined error analysis plot saved to {output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', '-i', required=True, type=str)
    parser.add_argument('--label_json', '-l', required=True, type=str)
    parser.add_argument('--pred_json', '-p', default="outputs/deim_dinov3_s_visdrone_eval/pred_bbox.json", type=str)
    parser.add_argument('--output_path', '-o', default='tp_fp_fn_visual', type=str)
    parser.add_argument('--iou', default=0.5, type=float)
    parser.add_argument('--score', default=0.25, type=float)
    args = parser.parse_args()

    image_path, output_path = args.image_path, args.output_path
    label_json_path, pred_json_path = args.label_json, args.pred_json
    iou_threshold, score_threshold = args.iou, args.score

    logger.info(f'image_path: {image_path} | output_path:{output_path}')
    logger.info(f'label_json_path: {label_json_path} | pred_json_path: {pred_json_path}')
    logger.info(f'iou_threshold: {iou_threshold} score_threshold:{score_threshold}')

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    with open(label_json_path, 'r') as f:
        label_json_data = json.load(f)
    
    with open(pred_json_path, 'r') as f:
        pred_json_data = json.load(f)
    
    classes = []
    for data in label_json_data['categories']:
        classes.append(data['name'])
    logger.info(f'classes:{classes}')

    image_id, imageID_path_dict = [], {}
    for data in label_json_data['images']:
        image_id.append(data['id'])
        imageID_path_dict[data['id']] = os.path.join(image_path, data['file_name'])
    image_id = sorted(list(set(image_id)))
    image_id_dict = {i:{'pred':[], 'label':[]} for i in image_id}

    for data in tqdm.tqdm(label_json_data['annotations'], desc='process label...'):
        category_id = data['category_id']
        x_min, y_min, w, h = data['bbox'][0], data['bbox'][1], data['bbox'][2], data['bbox'][3]
        x_max, y_max = x_min + w, y_min + h

        image_id_dict[data['image_id']]['label'].append(np.array([int(category_id), x_min, y_min, x_max, y_max]))
    
    for data in tqdm.tqdm(pred_json_data, desc='process pred...'):
        score = data['score']
        if score < score_threshold:
            continue
        category_id = data['category_id']
        x_min, y_min, w, h = data['bbox'][0], data['bbox'][1], data['bbox'][2], data['bbox'][3]
        x_max, y_max = x_min + w, y_min + h

        if data['image_id'] not in image_id_dict:
            image_id_dict[data['image_id']] = {'pred':[], 'label':[]}

        image_id_dict[data['image_id']]['pred'].append(np.array([int(category_id), x_min, y_min, x_max, y_max, float(score)]))

    detect_color, missing_color, error_color = (0, 255, 0), (0, 0, 255), (255, 0, 0)    # Opencv -> BGR
    
    # 统计变量
    all_right_num, all_missing_num, all_error_num = 0, 0, 0
    
    # 目标尺寸统计
    size_stats = {
        'small': {'tp': 0, 'fp': 0, 'fn': 0},
        'medium': {'tp': 0, 'fp': 0, 'fn': 0},
        'large': {'tp': 0, 'fp': 0, 'fn': 0}
    }
    
    # 错误类型统计
    fp_error_types = defaultdict(int)  # FP错误类型
    fn_error_types = defaultdict(int)  # FN错误类型
    
    with open(os.path.join(output_path, 'result.txt'), 'w') as f_r:
        for img_id_index, img_id in enumerate(image_id_dict):
            img_path = imageID_path_dict[img_id]
            label = np.array(image_id_dict[img_id]['label'])
            pred = sorted(image_id_dict[img_id]['pred'], key=lambda x:x[-1], reverse=True)

            if not os.path.exists(img_path):
                logger.warning(f'image:{img_path} is not found, skip...')
                continue

            image = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
            h, w = image.shape[:2]
            image_area = h * w

            right_num, missing_num, error_num = 0, 0, 0
            matched_labels = []  # 记录已匹配的标签索引
            matched_preds = []   # 记录已匹配的预测索引
            
            # 处理TP和FN
            for i in range(label.shape[0]):
                label_box = label[i, 1:5]
                label_class = label[i, 0]
                label_area = get_box_area(label_box)
                size_category = classify_object_size(label_area, image_area)
                
                if len(pred) == 0:
                    # 没有预测，全部为FN
                    image = draw_box(image, label_box, missing_color)
                    missing_num += 1
                    size_stats[size_category]['fn'] += 1
                    
                    # FN错误类型分析
                    fn_type = classify_fn_type(label_box, label_class, 
                                               np.array(pred) if len(pred) > 0 else np.array([]), 
                                               iou_threshold)
                    fn_error_types[fn_type] += 1
                    continue
                
                ious = iou(label[i:i+1, 1:5], np.array(pred)[:, 1:5])[0]
                ious_argsort = ious.argsort()[::-1]
                is_matched = False
                
                for j in ious_argsort:
                    if j in matched_preds:  # 跳过已匹配的预测
                        continue
                    if ious[j] < iou_threshold:
                        break
                    if label[i, 0] == pred[j][0]:
                        # TP
                        image = draw_box(image, pred[j][1:5], detect_color)
                        matched_preds.append(j)
                        matched_labels.append(i)
                        is_matched = True
                        right_num += 1
                        size_stats[size_category]['tp'] += 1
                        break
                
                if not is_matched:
                    # FN
                    image = draw_box(image, label_box, missing_color)
                    missing_num += 1
                    size_stats[size_category]['fn'] += 1
                    
                    # FN错误类型分析
                    fn_type = classify_fn_type(label_box, label_class, 
                                               np.array(pred), iou_threshold)
                    fn_error_types[fn_type] += 1
            
            # 处理FP（未匹配的预测）
            for j in range(len(pred)):
                if j not in matched_preds:
                    pred_box = pred[j][1:5]
                    pred_class = pred[j][0]
                    pred_area = get_box_area(pred_box)
                    size_category = classify_object_size(pred_area, image_area)
                    
                    image = draw_box(image, pred_box, error_color)
                    error_num += 1
                    size_stats[size_category]['fp'] += 1
                    
                    # FP错误类型分析
                    fp_type = classify_fp_type(pred_box, pred_class, label, iou_threshold)
                    fp_error_types[fp_type] += 1
            
            # 添加图例
            image = draw_legend(image, detect_color, missing_color, error_color, right_num, error_num, missing_num)
            
            all_right_num += right_num
            all_missing_num += missing_num
            all_error_num += error_num
            
            cv2.imwrite(os.path.join(output_path, os.path.basename(img_path)), image)
            logger.info(f'{img_id_index + 1}/{len(image_id_dict)} name:{os.path.basename(img_path)} right:{right_num} missing:{missing_num} error:{error_num}')
            f_r.write(f'name:{os.path.basename(img_path)} right:{right_num} missing:{missing_num} error:{error_num}\n')
        
        # 写入总体统计
        logger.info(f'all_result: right:{all_right_num} missing:{all_missing_num} error:{all_error_num}')
        f_r.write(f'\n=== Overall Statistics ===\n')
        f_r.write(f'all_result: right:{all_right_num} missing:{all_missing_num} error:{all_error_num}\n')
        
        # 写入尺寸统计
        f_r.write(f'\n=== Object Size Statistics ===\n')
        for size in ['small', 'medium', 'large']:
            stats = size_stats[size]
            total = stats['tp'] + stats['fn']
            recall = stats['tp'] / total * 100 if total > 0 else 0
            precision = stats['tp'] / (stats['tp'] + stats['fp']) * 100 if (stats['tp'] + stats['fp']) > 0 else 0
            f_r.write(f'{size.capitalize()} objects: TP={stats["tp"]}, FP={stats["fp"]}, FN={stats["fn"]}, '
                     f'Recall={recall:.2f}%, Precision={precision:.2f}%\n')
        
        # 写入错误类型统计
        f_r.write(f'\n=== False Positive Error Types ===\n')
        for error_type, count in fp_error_types.items():
            f_r.write(f'{error_type}: {count}\n')
        
        f_r.write(f'\n=== False Negative Error Types ===\n')
        for error_type, count in fn_error_types.items():
            f_r.write(f'{error_type}: {count}\n')
    
    # 生成可视化图表
    logger.info('Generating visualization plots...')
    
    # 1. 目标尺寸性能对比图
    plot_size_performance(size_stats, output_path)
    
    # 2. 错误类型分布图（饼图）
    plot_error_distribution(dict(fp_error_types), dict(fn_error_types), output_path)
    
    # 3. 错误类型分析图（柱状图）
    plot_combined_error_analysis(dict(fp_error_types), dict(fn_error_types), output_path)
    
    logger.info(f'All visualizations completed! Results saved to {output_path}')
