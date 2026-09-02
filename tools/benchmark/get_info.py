"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import argparse, thop, yaml
from calflops import calculate_flops
from engine.core import YAMLConfig
from engine.extre_module.tasks import DEIM_MG, OVDEIM_MG
from engine.logger_module import get_logger

import torch
import torch.nn as nn

RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
logger = get_logger(__name__)

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr


class _TensorInputDict(dict):
    """Dict input wrapper compatible with profilers that call `.to(device)` on each arg."""

    @staticmethod
    def _move(value, device):
        if hasattr(value, "to"):
            return value.to(device)
        if isinstance(value, dict):
            return {k: _TensorInputDict._move(v, device) for k, v in value.items()}
        if isinstance(value, list):
            return [_TensorInputDict._move(v, device) for v in value]
        if isinstance(value, tuple):
            return tuple(_TensorInputDict._move(v, device) for v in value)
        return value

    def to(self, device):
        return _TensorInputDict({k: self._move(v, device) for k, v in self.items()})


class Model_for_flops(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model.deploy()

    def forward(self, images):
        outputs = self.model(images)
        return outputs


def _load_yaml_file(config_path):
    with open(config_path, errors="ignore", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _is_direct_ultralytics_yaml(config):
    return isinstance(config, dict) and {"backbone", "encoder", "decoder"}.issubset(config)


def _is_direct_ov_ultralytics_yaml(config):
    if not _is_direct_ultralytics_yaml(config):
        return False

    for section in ("backbone", "encoder", "decoder"):
        for layer in config.get(section, []):
            if len(layer) >= 2 and str(layer[1]).startswith("OV"):
                return True
    return False


def _infer_ov_img_dim(config):
    guide_dims = []
    for layer in config.get("encoder", []):
        if len(layer) < 3:
            continue
        _, module_name, module_args = layer
        if module_name != "MaxSigmoidAttnBlock":
            continue
        guide_dims.append(module_args[2] if len(module_args) > 2 else 256)

    if not guide_dims:
        return 256
    unique_dims = set(guide_dims)
    if len(unique_dims) != 1:
        raise ValueError(f"MaxSigmoidAttnBlock guide dims must match, got {sorted(unique_dims)}")
    return guide_dims[0]


def build_profile_model(
    config_path,
    num_classes=1,
    eval_spatial_size=(640, 640),
    text_dim=768,
    img_dim=None,
    text_adapter_layers=1,
):
    config = _load_yaml_file(config_path)

    if _is_direct_ov_ultralytics_yaml(config):
        resolved_img_dim = img_dim if img_dim is not None else _infer_ov_img_dim(config)
        model = OVDEIM_MG(
            yaml_path=config_path,
            num_classes=num_classes,
            eval_spatial_size=tuple(eval_spatial_size),
            img_dim=resolved_img_dim,
            text_dim=text_dim,
            text_adapter_layers=text_adapter_layers,
        )
        model.text_feats = torch.randn(num_classes, text_dim)
        return Model_for_flops(model).eval(), (1, 3, *tuple(eval_spatial_size))

    if _is_direct_ultralytics_yaml(config):
        model = DEIM_MG(
            yaml_path=config_path,
            num_classes=num_classes,
            eval_spatial_size=tuple(eval_spatial_size),
        )
        return Model_for_flops(model).eval(), (1, 3, *tuple(eval_spatial_size))

    cfg = YAMLConfig(config_path, resume=None)
    return Model_for_flops(cfg.model).eval(), (1, 3, *cfg.yaml_cfg["eval_spatial_size"])


def _unwrap_profile_target(model: nn.Module) -> nn.Module:
    core_model = model.module if hasattr(model, "module") else model
    wrapped_model = getattr(core_model, "model", None)
    if isinstance(wrapped_model, nn.Module):
        return wrapped_model
    return core_model


def _is_multimodal_profile_model(model: nn.Module) -> bool:
    core_model = _unwrap_profile_target(model)
    backbone = getattr(core_model, "backbone", None)
    if backbone is None or not hasattr(backbone, "children"):
        return False

    for module in list(backbone.children())[:4]:
        if module.__class__.__name__ in {"GetRGBModalityTensor", "GetNPYModalityTensor"}:
            return True
    return False


def _build_profile_inputs(input_shape, device, multimodal):
    b, _, h, w = input_shape
    if multimodal:
        return (_TensorInputDict({
            "rgb": torch.randn(size=(b, 3, h, w), device=device),
            "npy": torch.randn(size=(b, 1, h, w), device=device),
        }),)
    return (torch.randn(size=input_shape, device=device),)


def profile_model_complexity(model: nn.Module, input_shape, device, print_detailed=True):
    is_multimodal_model = _is_multimodal_profile_model(model)
    profile_inputs = _build_profile_inputs(input_shape, device, is_multimodal_model)
    flops, macs = None, None

    try:
        if is_multimodal_model:
            flops, macs, _ = calculate_flops(
                model=model,
                input_shape=None,
                args=list(profile_inputs),
                kwargs={},
                output_as_string=True,
                output_precision=4,
                print_detailed=print_detailed,
            )
        else:
            flops, macs, _ = calculate_flops(
                model=model,
                input_shape=input_shape,
                output_as_string=True,
                output_precision=4,
                print_detailed=print_detailed,
            )
    except Exception:
        logger.warning(RED + "calculate_flops failed.. using thop instead.." + RESET)

    if flops is None or macs is None:
        macs_val = thop.profile(model, inputs=profile_inputs, verbose=False)[0]
        macs, flops = thop.clever_format([macs_val, macs_val * 2], format="%.3f")

    params = sum(p.numel() for p in model.parameters())
    return flops, macs, params


def render_profile_report(raw_output, config_path, input_shape, device, flops, macs, params):
    detail_marker = "-------------------------------- Detailed Calculated FLOPs Results --------------------------------"
    detail_start = raw_output.find(detail_marker)
    details = raw_output[detail_start:] if detail_start >= 0 else raw_output

    summary_rows = [
        ("Config", config_path),
        ("Input shape", str(input_shape)),
        ("Device", str(device)),
        ("FLOPs", str(flops)),
        ("MACs", str(macs)),
        ("Params", f"{params:,}"),
    ]
    table_width = 86
    lines = [
        "+" + "-" * table_width + "+",
        "| Model Profile Summary".ljust(table_width + 1) + "|",
        "+" + "-" * table_width + "+",
    ]
    for key, value in summary_rows:
        lines.append(f"| {key:<12}: {value}".ljust(table_width + 1) + "|")
    lines.append("+" + "-" * table_width + "+")

    return "\n".join(lines) + "\n\n" + details.strip()


def main(args, ):
    """main
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, input_shape = build_profile_model(
        args.config,
        num_classes=args.num_classes,
        eval_spatial_size=tuple(args.eval_spatial_size),
        text_dim=args.text_dim,
        img_dim=args.img_dim,
        text_adapter_layers=args.text_adapter_layers,
    )
    model = model.to(device)

    flops, macs, params = profile_model_complexity(
        model=model,
        input_shape=input_shape,
        device=device,
        print_detailed=True,
    )
    print("Model FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default= "configs/dfine/dfine_hgnetv2_l_coco.yml", type=str)
    parser.add_argument('--num-classes', default=1, type=int, help='num_classes for direct ultralytics yaml configs')
    parser.add_argument('--text-dim', default=768, type=int, help='text feature dim for direct OV ultralytics yaml configs')
    parser.add_argument('--img-dim', default=None, type=int, help='image/text adapter dim for direct OV ultralytics yaml configs')
    parser.add_argument('--text-adapter-layers', default=1, type=int, help='text adapter layers for direct OV ultralytics yaml configs')
    parser.add_argument(
        '--eval-spatial-size',
        default=(640, 640),
        nargs=2,
        type=int,
        metavar=('H', 'W'),
        help='eval input size for direct ultralytics yaml configs',
    )
    args = parser.parse_args()

    main(args)
