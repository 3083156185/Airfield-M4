#!/usr/bin/env python3
"""Extract reproducible P3/P4/P5 feature responses for RGB, IR, and PAPF."""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from engine.extre_module.tasks import DEIM_MG


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/feature_response_papf"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

RGB_YAML = ROOT / "configs/dfine/dfine_s.yml"
IR_YAML = ROOT / "configs/multimodal/yaml/dfine_s_aux_only.yml"
MM_YAML = ROOT / "configs/multimodal/yaml/dfine_s_papf.yml"

RGB_WEIGHT = ROOT / "weights/airfield_m4/rgb.pth"
IR_WEIGHT = ROOT / "weights/airfield_m4/ir.pth"
MM_WEIGHT = ROOT / "weights/airfield_m4/rgb_ir_papf.pth"

RGB_DIR = ROOT / "data/Airfield-M4/rgb/val/images"
IR_DIR = ROOT / "data/Airfield-M4/ir/val/images"
NPY_DIR = ROOT / "data/Airfield-M4/ir/val/npy"

SCENES = {
    "000575": {
        "label": "(a) Nighttime",
        "rois": [(755, 385, 1195, 615)],
    },
    "001826": {
        "label": "(b) Daytime",
        "rois": [(350, 380, 490, 570), (830, 375, 955, 580)],
    },
    "002622": {
        "label": "(c) Strong shadows",
        "rois": [(185, 390, 400, 585)],
    },
}

SCALE_NAMES = ("P3", "P4", "P5")
MODEL_KEYS = ("rgb", "ir", "papf_aux", "papf_fused")


def load_checkpoint_model(yaml_path, weight_path):
    model = DEIM_MG(
        yaml_path=str(yaml_path),
        pretrained=None,
        num_classes=7,
        eval_spatial_size=(640, 640),
    )
    checkpoint = torch.load(weight_path, map_location="cpu")
    state = (
        checkpoint["ema"]["module"]
        if "ema" in checkpoint
        else checkpoint["model"]
    )
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    metadata = {
        "path": str(weight_path),
        "epoch": checkpoint.get("epoch"),
        "date": checkpoint.get("date"),
    }
    return model, metadata


class ActivationCollector:
    def __init__(self, modules):
        self.outputs = {}
        self.handles = []
        for name, module in modules.items():
            self.handles.append(
                module.register_forward_hook(self._make_hook(name))
            )

    def _make_hook(self, name):
        def hook(_module, _inputs, output):
            if not torch.is_tensor(output):
                raise TypeError(
                    f"Expected tensor output from {name}, got {type(output)}"
                )
            self.outputs[name] = output.detach().float().cpu()

        return hook

    def clear(self):
        self.outputs.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()


def rgb_tensor(path):
    image = Image.open(path).convert("RGB").resize(
        (640, 640), Image.Resampling.BILINEAR
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def npy_tensor(path):
    array = np.load(path).astype(np.float32)
    if array.ndim == 2:
        array = array[None, :, :]
    elif array.ndim == 3 and array.shape[-1] <= 4:
        array = np.moveaxis(array, -1, 0)
    if array.ndim != 3:
        raise ValueError(f"Unsupported NPY shape {array.shape}: {path}")
    tensor = torch.from_numpy(array).unsqueeze(0)
    minimum = tensor.amin(dim=(1, 2, 3), keepdim=True)
    maximum = tensor.amax(dim=(1, 2, 3), keepdim=True)
    tensor = (tensor - minimum) / (maximum - minimum).clamp_min(1e-6)
    tensor = F.interpolate(
        tensor,
        size=(640, 640),
        mode="bilinear",
        align_corners=False,
    )
    return tensor


def channel_normalized_rms(feature):
    """Spatial response with channel-scale effects removed.

    Each channel is normalized by its own spatial RMS before channel
    aggregation. This makes response contrast comparable across separately
    trained networks and across feature tensors with different channel counts.
    """

    feature = feature[0].numpy().astype(np.float32)
    channel_scale = np.sqrt(np.mean(feature * feature, axis=(1, 2), keepdims=True))
    normalized = feature / np.maximum(channel_scale, 1e-6)
    response = np.sqrt(np.mean(normalized * normalized, axis=0))
    return np.log1p(response)


def dashed_rectangle(draw, box, color=(235, 33, 33), width=5, dash=18, gap=10):
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    for start in range(x1, x2, dash + gap):
        draw.line((start, y1, min(start + dash, x2), y1), fill=color, width=width)
        draw.line((start, y2, min(start + dash, x2), y2), fill=color, width=width)
    for start in range(y1, y2, dash + gap):
        draw.line((x1, start, x1, min(start + dash, y2)), fill=color, width=width)
        draw.line((x2, start, x2, min(start + dash, y2)), fill=color, width=width)


def draw_rois(image, rois):
    image = image.copy().convert("RGB")
    draw = ImageDraw.Draw(image)
    for roi in rois:
        dashed_rectangle(draw, roi)
    return image


def response_overlay(rgb_image, response, lower, upper, rois):
    width, height = rgb_image.size
    response = cv2.resize(response, (width, height), interpolation=cv2.INTER_CUBIC)
    normalized = np.clip((response - lower) / max(upper - lower, 1e-6), 0, 1)
    heat = cv2.applyColorMap(
        np.round(normalized * 255).astype(np.uint8),
        cv2.COLORMAP_VIRIDIS,
    )
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    background = np.asarray(rgb_image.convert("L").convert("RGB"), dtype=np.uint8)
    overlay = np.round(0.34 * background + 0.66 * heat).clip(0, 255).astype(np.uint8)
    return draw_rois(Image.fromarray(overlay), rois)


def roi_background_contrast(response, rois, image_size):
    width, height = image_size
    map_height, map_width = response.shape
    roi_mask = np.zeros_like(response, dtype=bool)
    for x1, y1, x2, y2 in rois:
        mx1 = max(0, min(map_width - 1, int(math.floor(x1 * map_width / width))))
        my1 = max(0, min(map_height - 1, int(math.floor(y1 * map_height / height))))
        mx2 = max(mx1 + 1, min(map_width, int(math.ceil(x2 * map_width / width))))
        my2 = max(my1 + 1, min(map_height, int(math.ceil(y2 * map_height / height))))
        roi_mask[my1:my2, mx1:mx2] = True
    roi_mean = float(response[roi_mask].mean())
    background = response[~roi_mask]
    background_mean = float(background.mean())
    return {
        "roi_mean": roi_mean,
        "background_mean": background_mean,
        "roi_to_background": roi_mean / max(background_mean, 1e-8),
    }


def run_model(model, collector, inputs):
    collector.clear()
    with torch.inference_mode():
        model(inputs)
    return {key: value.clone() for key, value in collector.outputs.items()}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "panels").mkdir(exist_ok=True)
    (OUTPUT / "responses").mkdir(exist_ok=True)

    rgb_model, rgb_metadata = load_checkpoint_model(RGB_YAML, RGB_WEIGHT)
    ir_model, ir_metadata = load_checkpoint_model(IR_YAML, IR_WEIGHT)
    mm_model, mm_metadata = load_checkpoint_model(MM_YAML, MM_WEIGHT)

    rgb_collector = ActivationCollector(
        {
            "P3": rgb_model.encoder[0],
            "P4": rgb_model.encoder[1],
            "P5": rgb_model.encoder[2],
        }
    )
    ir_collector = ActivationCollector(
        {
            "P3": ir_model.encoder[0],
            "P4": ir_model.encoder[1],
            "P5": ir_model.encoder[2],
        }
    )
    mm_collector = ActivationCollector(
        {
            "aux_P3": mm_model.backbone[8],
            "aux_P4": mm_model.backbone[11],
            "aux_P5": mm_model.backbone[14],
            "fused_P3": mm_model.encoder[0],
            "fused_P4": mm_model.encoder[1],
            "fused_P5": mm_model.encoder[2],
        }
    )

    report = {
        "device": str(DEVICE),
        "models": {
            "rgb": rgb_metadata,
            "ir": ir_metadata,
            "papf": mm_metadata,
        },
        "response_definition": (
            "log1p(channel RMS after per-channel spatial RMS normalization)"
        ),
        "normalization": (
            "shared 2nd--99.5th percentile within each scene and scale"
        ),
        "scenes": {},
    }

    for scene_name, scene in SCENES.items():
        rgb_path = RGB_DIR / f"{scene_name}.jpg"
        ir_path = IR_DIR / f"{scene_name}.jpg"
        npy_path = NPY_DIR / f"{scene_name}.npy"

        rgb_image = Image.open(rgb_path).convert("RGB")
        ir_image = Image.open(ir_path).convert("RGB")
        image_size = rgb_image.size

        rgb_input = rgb_tensor(rgb_path).to(DEVICE)
        ir_input = npy_tensor(npy_path).to(DEVICE)
        multimodal_input = {"rgb": rgb_input, "npy": ir_input}

        rgb_features = run_model(rgb_model, rgb_collector, rgb_input)
        ir_features = run_model(ir_model, ir_collector, multimodal_input)
        mm_features = run_model(mm_model, mm_collector, multimodal_input)

        scene_dir = OUTPUT / "panels" / scene_name
        response_dir = OUTPUT / "responses" / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        response_dir.mkdir(parents=True, exist_ok=True)

        draw_rois(rgb_image, scene["rois"]).save(scene_dir / "rgb_image.png")
        draw_rois(ir_image, scene["rois"]).save(scene_dir / "ir_image.png")

        scene_report = {
            "label": scene["label"],
            "rois": scene["rois"],
            "image_size": image_size,
            "scales": {},
        }
        for scale in SCALE_NAMES:
            responses = {
                "rgb": channel_normalized_rms(rgb_features[scale]),
                "ir": channel_normalized_rms(ir_features[scale]),
                "papf_aux": channel_normalized_rms(
                    mm_features[f"aux_{scale}"]
                ),
                "papf_fused": channel_normalized_rms(
                    mm_features[f"fused_{scale}"]
                ),
            }
            pooled = np.concatenate([value.ravel() for value in responses.values()])
            lower, upper = np.percentile(pooled, [2.0, 99.5]).tolist()
            scale_report = {
                "shared_lower": lower,
                "shared_upper": upper,
                "features": {},
            }
            for key, response in responses.items():
                np.save(response_dir / f"{scale}_{key}.npy", response)
                response_overlay(
                    rgb_image,
                    response,
                    lower,
                    upper,
                    scene["rois"],
                ).save(scene_dir / f"{scale}_{key}.png")
                scale_report["features"][key] = {
                    "shape": list(response.shape),
                    **roi_background_contrast(
                        response,
                        scene["rois"],
                        image_size,
                    ),
                }
            scene_report["scales"][scale] = scale_report
        report["scenes"][scene_name] = scene_report

    with (OUTPUT / "metrics.json").open("w") as stream:
        json.dump(report, stream, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
