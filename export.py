import warnings
warnings.filterwarnings('ignore')

import os, sys, importlib, argparse, json

import torch
import torch.nn as nn

from engine.core import YAMLConfig
from engine.extre_module.utils import check_version, arange_patch
from engine.extre_module.export import export_onnx, export_engine
from engine.logger_module import get_logger

RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
logger = get_logger(__name__)

CLASS_NAME = ['train', 'diningtable', 'person', 'bus', 'pottedplant', 'chair', 'cat', 'tvmonitor', 'motorbike', 'sofa']
TORCH_VERSION = str(torch.__version__)  # Normalize torch.__version__ (PyTorch>1.9 returns TorchVersion objects)
TORCHVISION_VERSION = importlib.metadata.version("torchvision")  # faster than importing torchvision
TORCH_1_9 = check_version(TORCH_VERSION, "1.9.0")
TORCH_1_13 = check_version(TORCH_VERSION, "1.13.0")
TORCH_2_0 = check_version(TORCH_VERSION, "2.0.0")
TORCH_2_4 = check_version(TORCH_VERSION, "2.4.0")
TORCHVISION_0_10 = check_version(TORCHVISION_VERSION, "0.10.0")
TORCHVISION_0_11 = check_version(TORCHVISION_VERSION, "0.11.0")
TORCHVISION_0_13 = check_version(TORCHVISION_VERSION, "0.13.0")
TORCHVISION_0_18 = check_version(TORCHVISION_VERSION, "0.18.0")

def get_latest_opset():
    """
    Return the second-most recent ONNX opset version supported by this version of PyTorch, adjusted for maturity.

    Returns:
        (int): The ONNX opset version.
    """
    if TORCH_1_13:
        # If the PyTorch>=1.13, dynamically compute the latest opset minus one using 'symbolic_opset'
        return max(int(k[14:]) for k in vars(torch.onnx) if "symbolic_opset" in k) - 1
    # Otherwise for PyTorch<=1.12 return the corresponding predefined opset
    version = torch.onnx.producer_version.rsplit(".", 1)[0]  # i.e. '2.3'
    return {"1.12": 15, "1.11": 14, "1.10": 13, "1.9": 12, "1.8": 12}.get(version, 12)

def main(args):
    import onnx
    global CLASS_NAME
    
    cfg = YAMLConfig(args.config, resume=args.resume)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    
    if 'DEIM_MG' in cfg.yaml_cfg:
        cfg.yaml_cfg['DEIM_MG']['pretrained'] = False
    if 'OVDEIM_MG' in cfg.yaml_cfg:
        cfg.yaml_cfg['OVDEIM_MG']['pretrained'] = False
    
    if args.resume:
        logger.info(ORANGE + f"Loading checkpoint from {args.resume}..." + RESET)
        checkpoint = torch.load(args.resume, map_location='cpu')
        if checkpoint.get('name', None) != None:
            CLASS_NAME = checkpoint['name']
            if type(CLASS_NAME) is dict:
                CLASS_NAME = [value for key, value in CLASS_NAME.items()]
            logger.info(f'classes: {CLASS_NAME} found in {args.resume}')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # NOTE load train mode state -> convert to deploy mode
        try:
            cfg.model.load_state_dict(state)
            logger.info(ORANGE + f"Loading checkpoint from {args.resume} Success!" + RESET)
        except Exception as e:
            logger.info(RED + f"Loading checkpoint from {args.resume} Failure! error:{e} \n export exit." + RESET)
            exit(-1)
    
    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    model = Model()
    im_shape = (args.batch, 3, *args.img_size)
    data = torch.rand(*im_shape)
    size = torch.tensor([args.img_size])
    output = model(data, size)
    if len(output) == 3:
        output_names = ['labels', 'boxes', 'scores']
        logger.info(f'judge model type is detect...')
    elif len(output) == 4:
        output_names = ['labels', 'boxes', 'scores', 'masks']
        logger.info(f'judge model type is instance segment...')
    else:
        raise Exception(f'model output length[{len(output)}] illegal...')

    dynamic_axes = None
    if args.dynamic and args.format == 'onnx':
        dynamic_axes = {
            'images': {0: 'N', },
            'orig_target_sizes': {0: 'N'},
        }
    
    opset = args.opset if args.opset else get_latest_opset()
    onnx_output_file = args.resume.replace('.pth', '.onnx') if args.resume else 'model.onnx'

    with arange_patch(args):
        export_onnx(
            model,
            (data, size),
            onnx_output_file,
            opset=opset,
            input_names=['images', 'orig_target_sizes'],
            output_names=output_names,
            dynamic=dynamic_axes,
        )
    logger.info(ORANGE + f"export_onnx success! save in path[{onnx_output_file}]" + RESET)
    
    # Checks
    model_onnx = onnx.load(onnx_output_file)  # load onnx model
    onnx.checker.check_model(model_onnx)

    if CLASS_NAME:
        metadata = model_onnx.metadata_props.add()
        metadata.key = "names"
        metadata.value = ",".join(CLASS_NAME)

        onnx.save(model_onnx, onnx_output_file)
        logger.info(ORANGE + f'{CLASS_NAME} info save in {onnx_output_file} success.' + RESET)

    if args.simplify:
        try:
            import onnxslim

            logger.info(ORANGE + f"slimming with onnxslim {onnxslim.__version__}..." + RESET)
            model_onnx = onnxslim.slim(model_onnx)
            onnx.save(model_onnx, onnx_output_file)
            logger.info(ORANGE + f"simplifier success! save in path[{onnx_output_file}]" + RESET)

        except Exception as e:
            logger.error(f"simplifier failure: {e}")
    
    if args.format == 'engine':
        try:
            import tensorrt as trt
        except ImportError as e:
            logger.error('package tensorrt is not install...')
            exit(-1)

        dla = None
        if args.device is None:
            logger.info("TensorRT requires GPU export, automatically assigning device=0")
            args.device = '0'
        elif 'dla' in str(args.device):
            dla = args.device.rsplit(":", 1)[-1]
            args.device = "0"  # update device to "0"
            assert dla in {"0", "1"}, f"Expected self.args.device='dla:0' or 'dla:1, but got {args.device}."
        
        engine_output_file = args.resume.replace('.pth', '.engine') if args.resume else 'model.engine'
        
        logger.info(ORANGE + f"starting export with TensorRT {trt.__version__}..." + RESET)
        export_engine(
            onnx_output_file,
            engine_output_file,
            args.workspace,
            args.half,
            args.dynamic,
            im_shape,
            dla=dla,
            verbose=True,
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='configs/yaml/dfine_hgnetv2_n_mg.yml', type=str, help='model yml path')
    parser.add_argument('--resume', '-r', type=str, help='model wight path')
    parser.add_argument('--opset',  default=17, type=int, help='onnx opset version')
    parser.add_argument('--simplify', action='store_true', default=False, help='onnx simplift model')
    parser.add_argument('--dynamic', action='store_true', default=False, help='onnx dynamic size inference')

    # engine param
    parser.add_argument('--half', action='store_true', default=False, help='export half precision model')
    parser.add_argument('--device', default='0', type=str, help='device')
    parser.add_argument('--workspace', default=None, type=float, help='trt workspace')

    parser.add_argument('--batch', default=1, type=int, help='batch (如果你需要用test-only去验证,那么batch一定要设置为1)')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size (h, w)')

    parser.add_argument('--format', '-f', default='onnx', choices=['onnx', 'engine'], type=str, help='export model platform type')

    args = parser.parse_args()

    print(ORANGE + "========== Parsed Arguments ==========")
    print(json.dumps(vars(args), indent=4, sort_keys=False))
    print("======================================" + RESET)

    main(args)