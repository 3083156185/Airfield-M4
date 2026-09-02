#!/usr/bin/env python3
"""Run a D-FINE validation pass with deterministic online sensor corruption."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import runpy
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter


RGB_GAMMA = (1.0, 1.4, 1.8, 2.2, 2.6)
HAZE_ALPHA = (0.0, 0.08, 0.16, 0.24, 0.32)
BLUR_SIGMA = (0.0, 0.6, 1.2, 1.8, 2.4)
IR_CONTRAST = (1.0, 0.75, 0.50, 0.30, 0.15)
IR_NOISE = (0.0, 0.02, 0.05, 0.09, 0.14)
SHIFT_AT_640 = (0, 2, 4, 8, 16)
IR_MISSING_RATIO = (0.0, 0.25, 0.50, 0.75, 1.0)


def _stable_fraction(image_id: int, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{image_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _degrade_rgb(image: Image.Image, corruption: str, severity: int) -> Image.Image:
    if severity == 0:
        return image

    if corruption == "rgb_lowlight":
        gamma = RGB_GAMMA[severity]
        lut = [
            int(round(255.0 * ((value / 255.0) ** gamma)))
            for value in range(256)
        ]
        if image.mode == "RGB":
            lut = lut * 3
        return image.point(lut)

    if corruption == "rgb_visibility":
        sigma = BLUR_SIGMA[severity]
        alpha = HAZE_ALPHA[severity]
        blurred = image.filter(ImageFilter.GaussianBlur(radius=sigma))
        arr = np.asarray(blurred, dtype=np.float32)
        arr = arr * (1.0 - alpha) + 255.0 * alpha
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode=image.mode)

    return image


def _shift_with_fill(x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    if dx == 0 and dy == 0:
        return x

    channels, height, width = x.shape
    fill = x.reshape(channels, -1).median(dim=1).values[:, None, None]
    out = fill.expand_as(x).clone()

    src_x0 = max(0, -dx)
    src_x1 = min(width, width - dx)
    src_y0 = max(0, -dy)
    src_y1 = min(height, height - dy)
    dst_x0 = max(0, dx)
    dst_x1 = min(width, width + dx)
    dst_y0 = max(0, dy)
    dst_y1 = min(height, height + dy)

    if src_x1 > src_x0 and src_y1 > src_y0:
        out[:, dst_y0:dst_y1, dst_x0:dst_x1] = x[
            :, src_y0:src_y1, src_x0:src_x1
        ]
    return out


def _degrade_ir(
    x: torch.Tensor,
    image_id: int,
    corruption: str,
    severity: int,
) -> torch.Tensor:
    if severity == 0:
        return x

    if corruption == "ir_quality":
        x = x.float()
        x_min = x.amin()
        x_max = x.amax()
        scale = (x_max - x_min).clamp_min(1e-6)
        normalized = (x - x_min) / scale
        mean = normalized.mean()

        generator = torch.Generator(device="cpu")
        generator.manual_seed(20260729 + 1009 * severity + int(image_id))
        noise = torch.randn(
            normalized.shape,
            generator=generator,
            dtype=normalized.dtype,
            device=normalized.device,
        )
        degraded = (
            mean
            + IR_CONTRAST[severity] * (normalized - mean)
            + IR_NOISE[severity] * noise
        )
        # Preserve the original NPY intensity range expected by the trained model.
        return degraded.clamp_(0.0, 1.0) * scale + x_min

    if corruption == "misalignment":
        target_shift = SHIFT_AT_640[severity]
        _, height, width = x.shape
        dx = int(round(target_shift * width / 640.0))
        dy = int(round(target_shift * height / 640.0))
        return _shift_with_fill(x, dx=dx, dy=dy)

    if corruption == "ir_missing":
        ratio = IR_MISSING_RATIO[severity]
        if _stable_fraction(image_id, "ir_missing_2026") < ratio:
            return torch.zeros_like(x)

    return x


def install_corruption_patch(corruption: str, severity: int) -> None:
    from engine.data.dataset.multimodal_coco_dataset import (
        MultimodalCocoDetection,
    )

    original_load_item = MultimodalCocoDetection.load_item
    original_load_npy = MultimodalCocoDetection._load_npy

    def patched_load_item(self, idx):
        sample, target = original_load_item(self, idx)
        image_id = int(self.ids[idx])
        sample["rgb"] = _degrade_rgb(sample["rgb"], corruption, severity)
        return sample, target

    def patched_load_npy(self, image_id, rgb_image):
        npy_tensor = original_load_npy(self, image_id, rgb_image)
        return _degrade_ir(
            npy_tensor,
            int(image_id),
            corruption,
            severity,
        )

    MultimodalCocoDetection.load_item = patched_load_item
    MultimodalCocoDetection._load_npy = patched_load_npy


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--corruption",
        choices=(
            "clean",
            "rgb_lowlight",
            "rgb_visibility",
            "ir_quality",
            "misalignment",
            "ir_missing",
        ),
        required=True,
    )
    parser.add_argument("--severity", type=int, choices=range(5), required=True)
    args, remaining = parser.parse_known_args()

    install_corruption_patch(args.corruption, args.severity)
    os.environ["ROBUSTNESS_CORRUPTION"] = args.corruption
    os.environ["ROBUSTNESS_SEVERITY"] = str(args.severity)

    project_root = Path(__file__).resolve().parent
    train_script = project_root / "train.py"
    if not train_script.exists():
        raise FileNotFoundError(
            "Place eval_corrupted.py in the D-FINE project root next to train.py"
        )

    sys.argv = [str(train_script), *remaining]
    runpy.run_path(str(train_script), run_name="__main__")


if __name__ == "__main__":
    main()
