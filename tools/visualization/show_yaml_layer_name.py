import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
import torch, argparse
from engine.core import YAMLConfig

RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default= "configs/dfine/dfine_hgnetv2_l_coco.yml", type=str)
    args = parser.parse_args()

    model = YAMLConfig(args.config, resume=None).model

    for name, module in model.named_modules():
            print(BLUE + f"Layer Name: " + ORANGE + name + BLUE + ", Layer Type: ", ORANGE, module.__class__.__name__, RESET)