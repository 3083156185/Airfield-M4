""""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
""" 
import matplotlib.pyplot as plt
import numpy as np
import PIL   
import torch
import torch.utils.data
import torchvision  
from typing import Any, BinaryIO, List, Optional, Tuple, Union
from PIL import Image, ImageColor, ImageDraw, ImageFont
from ..extre_module.utils import plt_settings, TryExcept     
torchvision.disable_beta_transforms_warning()     

__all__ = ['plot_sample']    
 
DINOV3_MEAN = torch.from_numpy(np.array([0.485, 0.456, 0.406]))   
DINOV3_STD = torch.from_numpy(np.array([0.229, 0.224, 0.225]))
    
@torch.no_grad()   
def draw_bounding_boxes(   
    image: torch.Tensor,
    boxes: torch.Tensor, 
    labels_id: Optional[List[str]] = None, 
    labels_names: Optional[List[str]] = None,   
    colors: Optional[Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]] = None,     
    width: int = 1,     
    font: Optional[str] = None,
    font_size: Optional[int] = None,  
) -> torch.Tensor:   
    num_boxes = boxes.shape[0]    

    if num_boxes == 0:
        return image  

    if font is None:    
        txt_font = ImageFont.load_default()
    else:
        txt_font = ImageFont.truetype(font=font, size=font_size or 10)

    # Handle Grayscale images    
    if image.size(0) == 1:
        image = torch.tile(image, (3, 1, 1))    
     
    ndarr = image.permute(1, 2, 0).cpu().numpy()
    img_to_draw = Image.fromarray(ndarr)    
    img_boxes = boxes.to(torch.int64).tolist() 

    draw = ImageDraw.Draw(img_to_draw) 
     
    for bbox, label_id, label_name in zip(img_boxes, labels_id, labels_names):  # type: ignore[arg-type]
        draw.rectangle(bbox, width=width, outline=colors[label_id])
    
        margin = width + 1
        draw.text((bbox[0] + margin, bbox[1] + margin), label_name, fill=colors[label_id], font=txt_font) 
 
    return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8) 
    
colors_hex = [
    "042AFF",  
    "0BDBEB",
    "F3F3F3",
    "00DFB7",
    "111F68",  
    "FF6FDD",     
    "FF444F",
    "CCED00",   
    "00F344",
    "BD00FF",   
    "00B4FF", 
    "DD00BA",
    "00FFFF",
    "26C000",  
    "01FFB3",
    "7D24FF",
    "7B0068",
    "FF1B6C", 
    "FC6D2F",   
    "A2FF0B",     
]
 
def hex2rgb(h):     
    """Converts hex color codes to RGB values (i.e. default PIL order)."""
    return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))    

colors = [hex2rgb(i) for i in colors_hex] * 5

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w.clamp(min=0.0)), (y_c - 0.5 * h.clamp(min=0.0)),     
         (x_c + 0.5 * w.clamp(min=0.0)), (y_c + 0.5 * h.clamp(min=0.0))]  
    return torch.stack(b, dim=-1)
    
@TryExcept('WARNING ⚠️ plot_sample failed.')  # known issue https://github.com/ultralytics/yolov5/issues/5395   
@plt_settings()
def plot_sample(data, category2name, output_dir, coco2category=None):     
    """for coco dataset/dataloader
    """
    from torchvision.transforms.v2 import functional as F
    images, targets = data   
    sample_num = images.size(0)
   
    if sample_num > 9:   
        plot_num = 16
        figsize = (20, 20)  
    elif sample_num > 4:
        plot_num = 9  
        figsize = (15, 15)
    else:  
        plot_num = 4
        figsize = (12, 12)

    plot_num_hw = int(plot_num ** 0.5)    
    fig, ax = plt.subplots(plot_num_hw, plot_num_hw, figsize=figsize)
    fig.subplots_adjust(hspace=0, wspace=0)
    for i in range(plot_num_hw ** 2):
        if i >= sample_num:     
            ax[i // plot_num_hw, i % plot_num_hw].axis('off')  
            continue 
        image, target = images[i], targets[i]   

        if image.min() < 0:    
            image = (image * DINOV3_STD.to(image.device).reshape((3, 1, 1))) + DINOV3_MEAN.to(image.device).reshape((3, 1, 1))   
        image = (image.clamp(0, 1) * 255.0).to(torch.uint8)
        boxes = box_cxcywh_to_xyxy(target["boxes"])
        boxes[:, [0, 2]] *= image.size(2) 
        boxes[:, [1, 3]] *= image.size(1)    
        labels_id = [int(cls_id) for cls_id in target['labels']]
        if coco2category:   
            labels_name = [category2name[coco2category[int(cls_id)]] for cls_id in target['labels']] 
        else:     
            labels_name = [category2name[int(cls_id)] for cls_id in target['labels']] 
     
        # --- draw masks on image first ---
        if "masks" in target:
            masks = target["masks"]  # (N, H, W)   
            masks = masks.to(image.device)

            image_np = image.clone().permute(1, 2, 0).cpu().numpy() 
  
            # overlay each mask
            for m in masks:
                mask = m.cpu().numpy().astype(np.uint8)

                # random color
                color = np.array([255, 0, 0], dtype=np.uint8)     

                # overlay (alpha blending)   
                alpha = 0.45
                colored_mask = color.reshape(1,1,3) 
                image_np = np.where(mask[:,:,None] == 1,   
                                    image_np * (1-alpha) + colored_mask * alpha,    
                                    image_np)  

            # convert back to tensor
            image = torch.from_numpy(image_np.astype(np.uint8)).permute(2,0,1)

        annotated_image = draw_bounding_boxes(image, boxes, labels_id, labels_name, colors=colors, width=3, font_size=3)   
    
        ax[i // plot_num_hw, i % plot_num_hw].imshow(annotated_image.permute(1, 2, 0).numpy())   
        ax[i // plot_num_hw, i % plot_num_hw].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])    
    fig.tight_layout(pad=0.1)    
    plt.savefig(output_dir, dpi=300)
    plt.close('all')
