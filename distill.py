"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""
import warnings
warnings.filterwarnings('ignore')

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Windows使用这个来指定显卡 多卡的话假设我用的是第一第三第四第五张卡就是"0,2,3,4""
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import argparse

from engine.extre_module.torch_utils import check_cuda
from engine.misc import dist_utils
from engine.core import YAMLConfig
from engine.solver import TASKS

RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"

debug=False

if debug:
    import torch
    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr

def main(args, ) -> None:
    """main
    """
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)
    check_cuda()

    student_cfg = YAMLConfig(args.student_config, **{'resume':args.resume})
    teacher_cfg = YAMLConfig(args.teacher_config, **{'tuning':args.teacher_weights})

    student_cfg_str = json.dumps(student_cfg.__dict__, indent=4, ensure_ascii=False)
    teacher_cfg_str = json.dumps(teacher_cfg.__dict__, indent=4, ensure_ascii=False)
    print(RED + f'---------- student_cfg ----------\n' + GREEN + student_cfg_str + RESET)
    print(RED + f'---------- teacher_cfg ----------\n' + GREEN + teacher_cfg_str + RESET)

    solver = TASKS[student_cfg.yaml_cfg['task']](student_cfg)
    solver.distill(student_cfg_str, teacher_cfg_str, teacher_cfg)

    dist_utils.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # priority 0
    parser.add_argument('-sc', '--student-config', type=str, required=True)
    parser.add_argument('-tc', '--teacher-config', type=str, required=True)
    parser.add_argument('-tw', '--teacher-weights', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, help='resume from checkpoint')
    parser.add_argument('-d', '--device', type=str, help='device',)
    parser.add_argument('--seed', type=int, help='exp reproducibility')
    parser.add_argument('--output-dir', type=str, help='output directoy')
    parser.add_argument('--summary-dir', type=str, help='tensorboard summry')

    # env
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')

    parser.add_argument('--local-rank', type=int, help='local rank id')
    args = parser.parse_args()

    main(args)
