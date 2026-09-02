import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, yaml, cv2, os, shutil, tqdm
import torchvision.transforms as T
import numpy as np
np.random.seed(0)
from engine.core import YAMLConfig
from PIL import Image

# 1. COLORMAP_AUTUMN:
# 	•	色调：从红色到黄色，模拟秋天的色彩。
# 	•	常用于突出显示数据的高值区域。

# 2. COLORMAP_BONE:
# 	•	色调：浅灰色至白色，带有冷色调，类似骨骼的颜色。
# 	•	用于表现较为柔和的效果，适合医学图像等领域。

# 3. COLORMAP_JET:
# 	•	色调：从蓝色到红色，常用于科学可视化。
# 	•	是 OpenCV 中最常用的颜色映射之一，具有明显的彩虹渐变，能够清晰显示不同的数值范围。

# 4. COLORMAP_WINTER:
# 	•	色调：从蓝色到绿色，模拟冬天的色彩。
# 	•	常用于较冷色调的数据可视化。

# 5. COLORMAP_RAINBOW:
# 	•	色调：彩虹色调（红、橙、黄、绿、蓝、紫）。
# 	•	用于产生鲜艳、全彩的效果，可以高亮显示不同的数值范围。

# 6. COLORMAP_OCEAN:
# 	•	色调：从蓝色到绿色，模拟海洋的颜色。
# 	•	用于表示较低至中等的数值，通常用于科学、环境领域的可视化。

# 7. COLORMAP_SUMMER:
# 	•	色调：绿色到黄色，类似夏天的色彩。
# 	•	用于表现较为温暖的数值，适合气候或生物学领域的可视化。

# 8. COLORMAP_SPRING:
# 	•	色调：从红色到绿色，模拟春天的颜色。
# 	•	适用于展示较为生动的变化，常用于数据的逐渐变化。

# 9. COLORMAP_COOL:
# 	•	色调：从紫色到青色。
# 	•	用于显示较为清凉的色调，适合表示冷色调的数值分布。

# 10. COLORMAP_HSV:
# 	•	色调：使用 HSV（色相、饱和度、亮度）空间的渐变。
# 	•	提供了一种平滑的色彩渐变方式，常用于展示不同数值的广泛变化。

# 11. COLORMAP_PINK:
# 	•	色调：粉红色。
# 	•	用于突出显示某些区域，色彩柔和。

# 12. COLORMAP_HOT:
# 	•	色调：从黑色到红色，再到黄色，最终到白色，模拟热源的颜色（像火焰的颜色）。
# 	•	用于强调数据的热度或强度，适合热力图和温度分布图的可视化。

# 13. COLORMAP_PARULA:
# 	•	色调：一种平滑的渐变色，蓝色到绿色，类似于 MATLAB 的默认 colormap。
# 	•	用于较为平滑的数值渐变，避免过于鲜艳的效果。

# 14. COLORMAP_MAGMA:
# 	•	色调：黑色到紫色，再到橙色，最终到黄色，适合表示深色至高亮区域的变化。
# 	•	适用于深度图或温度图，具有良好的视觉效果。

# 15. COLORMAP_INFERNO:
# 	•	色调：类似于 MAGMA，但更偏向红色至黄色的渐变，提供较高的对比度。
# 	•	用于强烈的数值变化，可用来强调热力图中的关键区域。

# 16. COLORMAP_PLASMA:
# 	•	色调：从深紫色到亮黄色，具有较强的对比度。
# 	•	通常用于对比度较高的可视化，突出显示变化。

# 17. COLORMAP_VIRIDIS:
# 	•	色调：从蓝绿色到黄色，平滑的渐变。
# 	•	适用于需要避免过多视觉干扰的情况，适合表现连续的变化。

# 18. COLORMAP_CIVIDIS:
# 	•	色调：一种设计用于色盲友好的色彩映射，颜色由浅蓝色到绿色。
# 	•	专为色盲用户设计，具有较高的可访问性。

# 19. COLORMAP_TWILIGHT:
# 	•	色调：紫色到黄色，平滑的渐变。
# 	•	适用于对比度较强的视觉效果。

# 20. COLORMAP_TWILIGHT_SHIFTED:
# 	•	色调：与 TWILIGHT 相似，但颜色映射进行了“偏移”。
# 	•	适合需要更强烈的对比度并突出数据变化的场景。

# 21. COLORMAP_TURBO:
# 	•	色调：一种鲜艳的颜色渐变，旨在提供更高的对比度和视觉效果。
# 	•	适用于需要鲜艳显示数值变化的情境。

# 22. COLORMAP_DEEPGREEN:
# 	•	色调：深绿色，适用于展示生物或环境数据中的渐变。

RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"

def get_param_by_string(model, param_str):
    # 分割字符串，按 '.' 进行分割，得到各个层次
    keys = param_str.split('.')
    
    # 从模型开始，逐步获取每一层
    param = model
    for key in keys:  # 逐层访问，直到最后一层
        if key.isdigit():  # 如果是数字，说明是一个列表的索引
            key = int(key)  # 将字符串转换为整数索引
            param = param[key]
        else:
            param = getattr(param, key)  # 动态访问属性

    return param


def normalize_feature_map_to_uint8(feature_map):
    feature_map = feature_map.detach().cpu()
    min_val = feature_map.min()
    max_val = feature_map.max()
    scale = max_val - min_val

    if float(scale) == 0.0:
        return np.zeros(feature_map.shape, dtype=np.uint8)

    feature_map = (feature_map - min_val) / scale * 255
    return feature_map.numpy().astype(np.uint8)


def save_feature_map_image(feature_map, save_path, color_map=cv2.COLORMAP_AUTUMN):
    feature_map = normalize_feature_map_to_uint8(feature_map)
    feature_map = cv2.applyColorMap(feature_map, color_map)
    Image.fromarray(feature_map).save(save_path)


def save_layer_feature_maps(featmap, feature_save_path, color_map=cv2.COLORMAP_AUTUMN):
    os.makedirs(feature_save_path, exist_ok=True)

    for channel_idx in range(featmap.size(1)):
        save_feature_map_image(
            featmap[0, channel_idx],
            os.path.join(feature_save_path, f'{channel_idx}.png'),
            color_map=color_map
        )

    save_feature_map_image(
        featmap[0].sum(dim=0),
        os.path.join(feature_save_path, 'sum.png'),
        color_map=color_map
    )

class deim_featuremap:
    handles = []
    activations = []
    def __init__(self, config, weight, layer, device, isUltralytics) -> None:
        device = torch.device(device)

        model = self.init_model(config, weight)
        model.to(device)
        model.eval()

        if isUltralytics:
            target_layers = []
            for l in layer:
                target_layers.append(eval(l))
        else:
            target_layers = [get_param_by_string(model, l) for l in layer]
        for l in target_layers:
            self.handles.append(l.register_forward_hook(self.save_activation))
        
        self.__dict__.update(locals())
    
    def init_model(self, config, weight):
        cfg = YAMLConfig(config, resume=weight)
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False
        
        checkpoint = torch.load(weight, map_location='cpu')
        if checkpoint.get('name', None) != None:
            CLASS_NAME = checkpoint['name']
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
        
        # Load train mode state and convert to deploy mode
        cfg.model.load_state_dict(state)
        return cfg.model

    def save_activation(self, module, input, output):
        self.activations.append(output.cpu().detach())
    
    def show_layer(self):
        for name, module in self.model.named_modules():
            print(BLUE + f"Layer Name: " + ORANGE + name + BLUE + ", Layer Type: ", ORANGE, module.__class__.__name__, RESET)

    def process(self, img_path, save_path):
        file_name = os.path.basename(img_path)
        file_name_without_extension = os.path.splitext(file_name)[0]
        save_path = os.path.join(save_path, file_name_without_extension)
        os.makedirs(save_path, exist_ok=True)

        # img process
        im_pil = Image.open(img_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).to(self.device)

        transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        im_data = transforms(im_pil).unsqueeze(0).to(self.device)

        self.model(im_data)

        for i, featmap in enumerate(self.activations):
            if len(featmap.size()) != 4:
                print(RED + f"Warning... {featmap.size()} size illegal... skip." + RESET)
                continue       
            feature_save_path = os.path.join(save_path, self.layer[i])
            save_layer_feature_maps(featmap, feature_save_path)

        self.activations.clear()

    def __call__(self, img_path, save_path):
        # remove dir if exist
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        # make dir if not exist
        os.makedirs(save_path, exist_ok=True)

        if os.path.isdir(img_path):
            for img_path_ in tqdm.tqdm(os.listdir(img_path)):
                if os.path.splitext(img_path_)[-1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.process(f'{img_path}/{img_path_}', save_path)
        else:
            self.process(img_path, save_path)

def get_params():
    params = {
        'config': 'configs/test/deim_hgnetv2_s_visdrone.yml',
        'weight': '../best_stg2.pth',
        'device': 'cuda:0',
        'layer': ['backbone.stem', 'backbone.stages.0'],
        # 'layer': ['model.backbone[0]', 'model.encoder[0]'],
        'isUltralytics': False # 是否是Ultralytics配置文件的方式
    }
    return params

if __name__ == '__main__':
    model = deim_featuremap(**get_params())
    model.show_layer() # 如果是Ultralytics版本配置文件的话，不需要执行此函数
    # model(r'/home/dataset/dataset_visdrone/VisDrone2019-DET-test-challenge/images', 'featuremap_result')