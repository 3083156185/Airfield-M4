"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import tqdm
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from engine.core import YAMLConfig
from engine.extre_module.utils import increment_path
from engine.logger_module import get_logger
from engine.misc.modality_utils import normalize_tensor_minmax_per_sample
from tools.inference.utils import draw

logger = get_logger(__name__)

RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
CLASS_NAME = None

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MODALITY_EXTENSIONS = {".npy", ".npz"}
INFERENCE_SIZE = (640, 640)


def is_image_file(path):
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def is_modality_file(path):
    return Path(path).suffix.lower() in MODALITY_EXTENSIONS


def resolve_modality_path(rgb_path, npy_dir):
    rgb_path = Path(rgb_path)
    npy_dir = Path(npy_dir)
    candidates = [npy_dir / f"{rgb_path.stem}.npy", npy_dir / f"{rgb_path.stem}.npz"]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = " and ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Missing modality file for {rgb_path}. Tried: {tried}")


def iter_multimodal_inputs(rgb_input, npy_input):
    rgb_path = Path(rgb_input)
    npy_path = Path(npy_input)

    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB input does not exist: {rgb_path}")
    if not npy_path.exists():
        raise FileNotFoundError(f"NPY input does not exist: {npy_path}")

    if rgb_path.is_dir():
        if npy_path.is_file():
            raise ValueError("When RGB input is a directory, NPY input must also be a directory.")
        rgb_files = sorted(path for path in rgb_path.iterdir() if path.is_file() and is_image_file(path))
        if not rgb_files:
            raise FileNotFoundError(f"No image files found in RGB directory: {rgb_path}")
        return [(path, resolve_modality_path(path, npy_path)) for path in rgb_files]

    if not is_image_file(rgb_path):
        raise ValueError(f"RGB input must be an image file or directory, got: {rgb_path}")

    if npy_path.is_dir():
        return [(rgb_path, resolve_modality_path(rgb_path, npy_path))]

    if not is_modality_file(npy_path):
        raise ValueError(f"NPY input must be a .npy/.npz file or directory, got: {npy_path}")

    return [(rgb_path, npy_path)]


def load_npy_tensor(npy_path):
    raw = np.load(npy_path, allow_pickle=True)
    if type(raw) is np.lib.npyio.NpzFile:
        try:
            npy = raw["arr_0"]
        except KeyError as e:
            raise ValueError(f"NPZ file must contain key 'arr_0', got keys={list(raw.keys())} for {npy_path}") from e
        finally:
            raw.close()
    else:
        npy = raw

    if npy.ndim != 2:
        raise ValueError(f"NPY modality must be 2D (H, W), got shape {tuple(npy.shape)} for {npy_path}")

    npy = npy.astype(np.float32, copy=False)
    return torch.from_numpy(npy).unsqueeze(0)


def build_multimodal_sample(rgb_path, npy_path, device):
    im_pil = Image.open(rgb_path).convert("RGB")
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]], device=device)

    npy_tensor = load_npy_tensor(npy_path)
    npy_h, npy_w = npy_tensor.shape[-2:]
    if (npy_h, npy_w) != (h, w):
        raise ValueError(f"NPY/RGB size mismatch for {rgb_path}: npy=({npy_h}, {npy_w}), rgb=({h}, {w})")

    rgb_transforms = T.Compose([
        T.Resize(INFERENCE_SIZE),
        T.ToTensor(),
    ])
    rgb_data = rgb_transforms(im_pil).unsqueeze(0).to(device)

    npy_data = F.interpolate(
        npy_tensor.unsqueeze(0),
        size=INFERENCE_SIZE,
        mode="bilinear",
        align_corners=False,
    )
    npy_data = normalize_tensor_minmax_per_sample(npy_data).to(device)

    return im_pil, {"rgb": rgb_data, "npy": npy_data}, orig_size


def process_image(model, device, rgb_path, npy_path, output_path, thrh):
    im_pil, sample, orig_size = build_multimodal_sample(rgb_path, npy_path, device)

    with torch.no_grad():
        output = model(sample, orig_size)
    labels, boxes, scores = output

    im_pil = draw([im_pil], labels, boxes, scores, thrh=thrh, class_name=CLASS_NAME)
    im_pil.save(output_path / Path(rgb_path).name)


def get_device(device_arg):
    if device_arg == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")

    cuda_index = str(device_arg).split(",")[0]
    return torch.device(f"cuda:{cuda_index}")


def main(args):
    """Main function"""
    global CLASS_NAME
    cfg = YAMLConfig(args.config, resume=args.resume)

    output_path = increment_path(args.output)
    logger.info(RED + f"output_dir:{str(output_path)}" + RESET)
    output_path.mkdir(parents=True, exist_ok=True)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if checkpoint.get("name", None) != None:
            CLASS_NAME = checkpoint["name"]
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
    else:
        raise AttributeError("Only support resume to load model.state_dict by now.")

    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, sample, orig_target_sizes):
            outputs = self.model(sample)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    device = get_device(args.device)
    model = Model().to(device).eval()

    input_pairs = iter_multimodal_inputs(args.input_rgb, args.input_npy)
    for rgb_path, npy_path in tqdm.tqdm(input_pairs, desc="Processing multimodal images"):
        process_image(model, device, rgb_path, npy_path, output_path, args.thrh)

    logger.info("Multimodal image processing complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, required=True)
    parser.add_argument("-i-rgb", "--input-rgb", type=str, required=True)
    parser.add_argument("-i-npy", "--input-npy", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, default="inference_results/exp")
    parser.add_argument("-t", "--thrh", type=float, default=0.2)
    parser.add_argument("-d", "--device", type=str, default="0")
    args = parser.parse_args()
    main(args)
