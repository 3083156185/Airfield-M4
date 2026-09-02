import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, yaml, cv2, os, shutil
import torchvision.transforms as T
import numpy as np
np.random.seed(0)
from tqdm import trange
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from engine.core import YAMLConfig
from tools.inference.utils import draw

RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
CLASS_NAME = None

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def post_process(self, result):
        boxes, logits = result['pred_boxes'], result['pred_logits']
        sorted, indices = torch.sort(logits.max(2)[0], descending=True)
        return logits[0, indices][0], boxes[0, indices][0]
  
    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)
        logits, boxes = self.post_process(model_output)
        return [[logits, boxes]]

    def release(self):
        for handle in self.handles:
            handle.remove()

class deim_target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio) -> None:
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio
    
    def forward(self, data):
        logits, boxes = data
        result = []
        for i in trange(int(logits.size(0) * self.ratio)):
            if float(logits[i].max()) < self.conf:
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(logits[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(boxes[i, j])
        return sum(result)

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

class deim_heatmap:
    def __init__(self, config, weight, device, method, layer, backward_type, conf_threshold, ratio, show_box, renormalize, isUltralytics):
        device = torch.device(device)

        model, postprocessor = self.init_model(config, weight)
        model.to(device)
        model.eval()

        target = deim_target(backward_type, conf_threshold, ratio)
        if isUltralytics:
            target_layers = []
            for l in layer:
                target_layers.append(eval(l))
        else:
            target_layers = [get_param_by_string(model, l) for l in layer]
        method = eval(method)(model, target_layers)
        method.activations_and_grads = ActivationsAndGradients(model, target_layers, None)

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
        return cfg.model, cfg.postprocessor.deploy()

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        """Normalize the CAM to be in the range [0, 1] 
        inside every bounding boxes, and zero outside of the bounding boxes. """
        h, w, _ = image_float_np.shape
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            try:
                x1, y1 = max(x1 , 0) , max(y1, 0) 
                x2, y2 = min(grayscale_cam.shape[1] - 1, x2) , min(grayscale_cam.shape[0] - 1, y2) 
                renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy()) 
            except Exception as e:
                continue  
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam)
        return eigencam_image_renormalized
    
    def post_process(self, pred, orig_size):
        labels, boxes, scores = self.postprocessor(pred, orig_size)
        return boxes[scores > self.conf_threshold]

    def process(self, img_path, save_path):
        # img process
        im_pil = Image.open(img_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).to(self.device)

        transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        im_data = transforms(im_pil).unsqueeze(0).to(self.device)
        
        try:
            grayscale_cam = self.method(im_data, [self.target])
        except AttributeError as e:
            print(f"Warning... self.method(tensor, [self.target]) failure.")
            return
        
        grayscale_cam = grayscale_cam[0, :]
        grayscale_cam = cv2.resize(grayscale_cam, (w, h))
        pred = self.model(im_data)
        if self.renormalize:
            boxes = self.post_process(pred, orig_size)
            cam_image = self.renormalize_cam_in_bounding_boxes(boxes.cpu().detach().numpy().astype(np.int32), np.array(im_pil) / 255.0, grayscale_cam)
        else:
            cam_image = show_cam_on_image(np.array(im_pil) / 255.0, grayscale_cam)
        cam_image = Image.fromarray(cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB))
        if self.show_box:
            cam_image = draw([cam_image], *self.postprocessor(pred, orig_size), thrh=self.conf_threshold, class_name=CLASS_NAME)
        cam_image.save(save_path)
    
    def show_layer(self):
        for name, module in self.model.named_modules():
            if module.__class__.__name__ == 'ModuleList':
                continue
            print(BLUE + f"Layer Name: " + ORANGE + name + BLUE + ", Layer Type: ", ORANGE, module.__class__.__name__, RESET)
    
    def __call__(self, img_path, save_path):
        # remove dir if exist
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        # make dir if not exist
        os.makedirs(save_path, exist_ok=True)

        if os.path.isdir(img_path):
            for img_path_ in os.listdir(img_path):
                if os.path.splitext(img_path_)[-1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.process(f'{img_path}/{img_path_}', f'{save_path}/{img_path_}')
        else:
            self.process(img_path, f'{save_path}/result.png')

def get_params():
    params = {
        'config': 'configs/test/deim_hgnetv2_s_visdrone.yml',
        'weight': '../best_stg2.pth',
        'device': 'cuda:0',
        'method': 'GradCAM', # GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
        'layer': ['backbone.stem.stem1.conv', 'encoder.fpn_blocks.0.cv3.1', 'encoder.input_proj.0.conv', 'encoder.input_proj.1.conv'],
        # 'layer': ['model.backbone[0]', 'model.encoder[0]'],
        'backward_type': 'all', # class, box, all
        'conf_threshold': 0.2, # 0.2
        'ratio': 1.0, # 0.02-0.1
        'show_box': True, # 不需要绘制框请设置为False
        'renormalize': False, # 需要把热力图限制在框内请设置为True
        'isUltralytics': False # 是否是Ultralytics配置文件的方式
    }
    return params

# 需要安装grad-cam==1.5.4

if __name__ == '__main__':
    model = deim_heatmap(**get_params())
    model.show_layer() # B C H W 如果是Ultralytics版本配置文件的话，不需要执行此函数
    # model('/path/to/images', 'heatmap_result')
