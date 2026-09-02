#!/usr/bin/env python3
"""Select and render representative multimodal detection comparisons.

The script consumes COCO ground truth plus prediction JSON files from five
modality settings. It ranks validation images by whether RGB+IR improves
small-object detection over the single-modality and tri-modality settings,
then exports auditable candidate sheets and final publication panels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


MODEL_ORDER = ["RGB-only", "IR-only", "Depth-only", "RGB+IR", "RGB+IR+Depth"]
MODEL_SLUG = {
    "RGB-only": "rgb",
    "IR-only": "ir",
    "Depth-only": "depth",
    "RGB+IR": "rgb_ir",
    "RGB+IR+Depth": "rgb_ir_depth",
}
CLASS_COLORS = [
    "#E64B35",
    "#4DBBD5",
    "#00A087",
    "#F39B7F",
    "#3C5488",
    "#8491B4",
    "#B09C85",
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def xywh_to_xyxy(box: Sequence[float]) -> np.ndarray:
    x, y, w, h = map(float, box)
    return np.asarray([x, y, x + w, y + h], dtype=np.float32)


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(
        0.0, float(box_a[3] - box_a[1])
    )
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(
        0.0, float(box_b[3] - box_b[1])
    )
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_image(
    gt_items: Sequence[Mapping],
    pred_items: Sequence[Mapping],
    iou_threshold: float,
) -> Dict[str, float]:
    ordered_predictions = sorted(
        enumerate(pred_items), key=lambda item: float(item[1]["score"]), reverse=True
    )
    used = set()
    matched_ious: List[float] = []
    matched_scores: List[float] = []
    matched_small = 0
    matched_staff = 0

    for gt in gt_items:
        gt_box = xywh_to_xyxy(gt["bbox"])
        best = None
        best_iou = 0.0
        for original_index, pred in ordered_predictions:
            if original_index in used or int(pred["category_id"]) != int(gt["category_id"]):
                continue
            overlap = iou(gt_box, xywh_to_xyxy(pred["bbox"]))
            if overlap > best_iou:
                best_iou = overlap
                best = (original_index, pred)
        if best is not None and best_iou >= iou_threshold:
            used.add(best[0])
            matched_ious.append(best_iou)
            matched_scores.append(float(best[1]["score"]))
            if float(gt.get("area", gt["bbox"][2] * gt["bbox"][3])) < 32.0**2:
                matched_small += 1
            if gt.get("_is_staff", False):
                matched_staff += 1

    tp = len(used)
    return {
        "tp": tp,
        "fp": max(0, len(pred_items) - tp),
        "fn": max(0, len(gt_items) - tp),
        "small_tp": matched_small,
        "staff_tp": matched_staff,
        "mean_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "mean_score": float(np.mean(matched_scores)) if matched_scores else 0.0,
    }


def find_image(image_dir: Path, file_name: str) -> Path:
    direct = image_dir / file_name
    if direct.exists():
        return direct
    base = Path(file_name).name
    direct = image_dir / base
    if direct.exists():
        return direct
    stem = Path(base).stem
    matches = sorted(image_dir.glob(f"{stem}.*"))
    if not matches:
        raise FileNotFoundError(f"Image not found for {file_name} in {image_dir}")
    return matches[0]


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def draw_predictions(
    source_image: Image.Image,
    predictions: Sequence[Mapping],
    category_names: Mapping[int, str],
    category_colors: Mapping[int, str],
    threshold: float,
    title: str,
) -> Image.Image:
    canvas = source_image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    line_width = max(2, round(min(canvas.size) / 320))
    label_font = font(max(14, round(min(canvas.size) / 42)), bold=True)

    kept = [item for item in predictions if float(item["score"]) >= threshold]
    kept.sort(key=lambda item: float(item["score"]), reverse=True)
    for item in kept:
        category_id = int(item["category_id"])
        color = category_colors.get(category_id, "#FFFFFF")
        x1, y1, x2, y2 = xywh_to_xyxy(item["bbox"])
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        label = f"{category_names.get(category_id, str(category_id))} {float(item['score']):.2f}"
        text_box = draw.textbbox((0, 0), label, font=label_font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        label_x = min(
            max(0, int(x1)),
            max(0, canvas.width - text_w - 8),
        )
        label_y = max(0, int(y1) - text_h - 5)
        draw.rectangle(
            (label_x, label_y, label_x + text_w + 8, label_y + text_h + 5),
            fill=color,
        )
        draw.text((label_x + 4, label_y + 1), label, fill="white", font=label_font)

    header_height = max(58, round(canvas.height * 0.09))
    panel = Image.new("RGB", (canvas.width, canvas.height + header_height), "white")
    panel.paste(canvas, (0, header_height))
    header_draw = ImageDraw.Draw(panel)
    title_font = font(max(18, round(canvas.width / 36)), bold=True)
    title_box = header_draw.textbbox((0, 0), title, font=title_font)
    title_w = title_box[2] - title_box[0]
    header_draw.text(
        ((canvas.width - title_w) / 2, (header_height - (title_box[3] - title_box[1])) / 2 - 1),
        title,
        fill="#1F2937",
        font=title_font,
    )
    return panel


def draw_gt(
    source_image: Image.Image,
    gt_items: Sequence[Mapping],
    category_names: Mapping[int, str],
    category_colors: Mapping[int, str],
) -> Image.Image:
    del category_colors
    canvas = source_image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    line_width = max(2, round(min(canvas.size) / 320))
    dash_length = max(8, line_width * 4)
    gap_length = max(5, line_width * 2)
    label_font = font(max(14, round(min(canvas.size) / 42)), bold=True)
    gt_color = "#F4D03F"

    def dashed_line(start, end):
        x1, y1 = start
        x2, y2 = end
        length = max(abs(x2 - x1), abs(y2 - y1))
        if length <= 0:
            return
        for offset in range(0, int(length) + 1, dash_length + gap_length):
            stop = min(offset + dash_length, length)
            t0, t1 = offset / length, stop / length
            draw.line(
                (
                    x1 + (x2 - x1) * t0,
                    y1 + (y2 - y1) * t0,
                    x1 + (x2 - x1) * t1,
                    y1 + (y2 - y1) * t1,
                ),
                fill=gt_color,
                width=line_width,
            )

    for item in gt_items:
        category_id = int(item["category_id"])
        x1, y1, x2, y2 = xywh_to_xyxy(item["bbox"])
        dashed_line((x1, y1), (x2, y1))
        dashed_line((x2, y1), (x2, y2))
        dashed_line((x2, y2), (x1, y2))
        dashed_line((x1, y2), (x1, y1))
        label = f"GT: {category_names.get(category_id, str(category_id))}"
        text_box = draw.textbbox((0, 0), label, font=label_font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        label_x = min(max(0, int(x1)), max(0, canvas.width - text_w - 8))
        label_y = max(0, int(y1) - text_h - 5)
        draw.rectangle(
            (label_x, label_y, label_x + text_w + 8, label_y + text_h + 5),
            fill=gt_color,
        )
        draw.text((label_x + 4, label_y + 1), label, fill="#111111", font=label_font)

    header_height = max(58, round(canvas.height * 0.09))
    panel = Image.new("RGB", (canvas.width, canvas.height + header_height), "white")
    panel.paste(canvas, (0, header_height))
    header_draw = ImageDraw.Draw(panel)
    title = "RGB image\nGT"
    title_font = font(max(18, round(canvas.width / 36)), bold=True)
    title_box = header_draw.multiline_textbbox(
        (0, 0), title, font=title_font, align="center", spacing=0
    )
    title_w = title_box[2] - title_box[0]
    title_h = title_box[3] - title_box[1]
    header_draw.multiline_text(
        ((canvas.width - title_w) / 2, (header_height - title_h) / 2 - 1),
        title,
        fill="#1F2937",
        font=title_font,
        align="center",
        spacing=0,
    )
    return panel


def concatenate_horizontally(panels: Sequence[Image.Image], gap: int = 12) -> Image.Image:
    height = max(panel.height for panel in panels)
    width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    composite = Image.new("RGB", (width, height), "white")
    x = 0
    for panel in panels:
        composite.paste(panel, (x, 0))
        x += panel.width + gap
    return composite


def concatenate_vertically(rows: Sequence[Image.Image], gap: int = 14) -> Image.Image:
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows) + gap * (len(rows) - 1)
    composite = Image.new("RGB", (width, height), "white")
    y = 0
    for row in rows:
        composite.paste(row, (0, y))
        y += row.height + gap
    return composite


def resize_to_width(image: Image.Image, width: int) -> Image.Image:
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def luminance(image: Image.Image) -> float:
    small = image.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
    return float(np.asarray(small, dtype=np.float32).mean() / 255.0)


def scene_bin(value: float) -> str:
    if value < 0.22:
        return "low-light"
    if value < 0.45:
        return "dim"
    if value < 0.68:
        return "normal"
    return "bright"


def build_index(items: Iterable[Mapping], score_threshold: float | None = None):
    output = defaultdict(list)
    for item in items:
        if score_threshold is not None and float(item.get("score", 1.0)) < score_threshold:
            continue
        output[int(item["image_id"])].append(dict(item))
    return output


def score_candidate(
    gt_items: Sequence[Mapping],
    metrics: Mapping[str, Mapping[str, float]],
) -> float:
    rgb = metrics["RGB-only"]
    ir = metrics["IR-only"]
    fused = metrics["RGB+IR"]
    auxiliary_models = [
        model
        for model in metrics
        if model not in {"RGB-only", "IR-only", "RGB+IR"}
    ]
    small_count = sum(
        float(item.get("area", item["bbox"][2] * item["bbox"][3])) < 32.0**2
        for item in gt_items
    )
    staff_count = sum(bool(item.get("_is_staff", False)) for item in gt_items)

    score = 0.0
    score += 4.0 * max(0.0, fused["tp"] - rgb["tp"])
    score += 2.0 * max(0.0, fused["tp"] - ir["tp"])
    score += 2.0 * max(0.0, fused["small_tp"] - rgb["small_tp"])
    score += 1.2 * max(0.0, rgb["fp"] - fused["fp"])
    score += 3.0 * max(0.0, fused["mean_iou"] - rgb["mean_iou"])
    for model in auxiliary_models:
        alternative = metrics[model]
        score += 2.5 * max(0.0, fused["tp"] - alternative["tp"])
        score += 2.0 * max(0.0, fused["small_tp"] - alternative["small_tp"])
        score += 1.0 * max(0.0, alternative["fp"] - fused["fp"])
    score += min(small_count, 5) * 0.8
    score += min(staff_count, 4) * 1.5
    if fused["tp"] == 0:
        score -= 10.0
    if fused["tp"] < rgb["tp"]:
        score -= 5.0
    return score


def export_contact_sheets(
    candidate_ids: Sequence[int],
    image_info: Mapping[int, Mapping],
    gt_index: Mapping[int, Sequence[Mapping]],
    pred_index: Mapping[str, Mapping[int, Sequence[Mapping]]],
    image_dir: Path,
    output_dir: Path,
    category_names: Mapping[int, str],
    category_colors: Mapping[int, str],
    threshold: float,
    rows_per_sheet: int,
) -> None:
    preview_dir = output_dir / "candidate_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Image.Image] = []
    for image_id in candidate_ids:
        info = image_info[image_id]
        source = Image.open(find_image(image_dir, info["file_name"])).convert("RGB")
        panels = [
            draw_predictions(
                source,
                pred_index[model].get(image_id, []),
                category_names,
                category_colors,
                threshold,
                model,
            )
            for model in MODEL_ORDER
        ]
        row = concatenate_horizontally([resize_to_width(panel, 430) for panel in panels], gap=8)
        draw = ImageDraw.Draw(row)
        draw.rectangle((0, 0, 150, 34), fill="white")
        draw.text((6, 6), f"ID {image_id}", fill="#111827", font=font(20, bold=True))
        row.save(preview_dir / f"{image_id:06d}_comparison.jpg", quality=94)
        all_rows.append(row)

    for start in range(0, len(all_rows), rows_per_sheet):
        sheet = concatenate_vertically(all_rows[start : start + rows_per_sheet], gap=10)
        sheet.save(
            output_dir / f"candidate_sheet_{start // rows_per_sheet + 1:02d}.jpg",
            quality=94,
        )


def export_final(
    selected_ids: Sequence[int],
    image_info: Mapping[int, Mapping],
    gt_index: Mapping[int, Sequence[Mapping]],
    pred_index: Mapping[str, Mapping[int, Sequence[Mapping]]],
    image_dir: Path,
    output_dir: Path,
    category_names: Mapping[int, str],
    category_colors: Mapping[int, str],
    threshold: float,
) -> None:
    single_dir = output_dir / "single_results"
    comparison_dir = output_dir / "scene_comparisons"
    source_dir = output_dir / "source_rgb"
    gt_dir = output_dir / "ground_truth"
    for directory in (single_dir, comparison_dir, source_dir, gt_dir):
        directory.mkdir(parents=True, exist_ok=True)

    paper_rows: List[Image.Image] = []
    for row_number, image_id in enumerate(selected_ids, start=1):
        info = image_info[image_id]
        source = Image.open(find_image(image_dir, info["file_name"])).convert("RGB")
        stem = f"scene_{row_number:02d}_id_{image_id}"
        source.save(source_dir / f"{stem}.png")
        gt_panel = draw_gt(source, gt_index.get(image_id, []), category_names, category_colors)
        gt_panel.save(gt_dir / f"{stem}_gt.png")

        panels = [gt_panel]
        for model in MODEL_ORDER:
            panel = draw_predictions(
                source,
                pred_index[model].get(image_id, []),
                category_names,
                category_colors,
                threshold,
                model,
            )
            panel.save(single_dir / f"{stem}_{MODEL_SLUG[model]}.png")
            panels.append(panel)
        row = concatenate_horizontally([resize_to_width(panel, 700) for panel in panels], gap=12)
        row.save(comparison_dir / f"{stem}_six_panels.png")
        paper_rows.append(resize_to_width(row, 3500))

    paper_panel = concatenate_vertically(paper_rows, gap=18)
    figure_stem = f"fig_qualitative_modality_{len(selected_ids)}x6"
    paper_panel.save(output_dir / f"{figure_stem}.png")
    paper_panel.convert("RGB").save(
        output_dir / f"{figure_stem}.tiff",
        compression="tiff_lzw",
        dpi=(600, 600),
    )
    paper_panel.save(output_dir / f"{figure_stem}.pdf", resolution=300.0)


def main() -> None:
    global MODEL_ORDER, MODEL_SLUG

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-profile",
        choices=("airfield", "m3fd"),
        default="airfield",
    )
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--candidate-count", type=int, default=30)
    parser.add_argument("--rows-per-sheet", type=int, default=5)
    parser.add_argument("--selected-count", type=int, default=5)
    parser.add_argument("--selected-ids", nargs="*", type=int, default=[])
    args = parser.parse_args()

    root = args.project_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_profile == "m3fd":
        MODEL_ORDER = [
            "RGB-only",
            "IR-only",
            "RGB+Text",
            "RGB+IR",
            "RGB+IR+Text",
        ]
        MODEL_SLUG = {
            "RGB-only": "rgb",
            "IR-only": "ir",
            "RGB+Text": "rgb_text",
            "RGB+IR": "rgb_ir",
            "RGB+IR+Text": "rgb_ir_text",
        }
        annotation_path = root / "data/M3FD/annotations/instances_val.json"
        image_dir = root / "data/M3FD/vi"
        prediction_paths = {
            "RGB-only": root / "outputs/m3fd/rgb/pred_bbox.json",
            "IR-only": root / "outputs/m3fd/ir/pred_bbox.json",
            "RGB+Text": root / "outputs/m3fd/rgb_text/pred_bbox.json",
            "RGB+IR": root / "outputs/m3fd/rgb_ir_papf/pred_bbox.json",
            "RGB+IR+Text": (
                root / "outputs/m3fd/rgb_ir_text/pred_bbox.json"
            ),
        }
    else:
        annotation_path = root / "data/Airfield-M4/annotations/instances_val.json"
        image_dir = root / "data/Airfield-M4/rgb/val/images"
        prediction_paths = {
            "RGB-only": root / "outputs/airfield_m4/rgb/pred_bbox.json",
            "IR-only": root / "outputs/airfield_m4/ir/pred_bbox.json",
            "Depth-only": root / "outputs/airfield_m4/depth/pred_bbox.json",
            "RGB+IR": root / "outputs/airfield_m4/rgb_ir_papf/pred_bbox.json",
            "RGB+IR+Depth": (
                root / "outputs/airfield_m4/rgb_ir_depth_papf/pred_bbox.json"
            ),
        }

    coco = load_json(annotation_path)
    image_info = {int(item["id"]): item for item in coco["images"]}
    categories = sorted(coco["categories"], key=lambda item: int(item["id"]))
    category_names = {int(item["id"]): str(item["name"]) for item in categories}
    category_colors = {
        int(item["id"]): CLASS_COLORS[index % len(CLASS_COLORS)]
        for index, item in enumerate(categories)
    }
    staff_ids = {
        category_id
        for category_id, name in category_names.items()
        if (
            "staff" in name.lower()
            or "person" in name.lower()
            or "people" in name.lower()
        )
    }

    gt_items = []
    for item in coco["annotations"]:
        copied = dict(item)
        copied["_is_staff"] = int(item["category_id"]) in staff_ids
        gt_items.append(copied)
    gt_index = build_index(gt_items)
    pred_index = {
        model: build_index(load_json(path), score_threshold=args.confidence)
        for model, path in prediction_paths.items()
    }

    rows = []
    for image_id, info in image_info.items():
        image_path = find_image(image_dir, info["file_name"])
        source = Image.open(image_path).convert("RGB")
        current_gt = gt_index.get(image_id, [])
        metrics = {
            model: match_image(
                current_gt,
                pred_index[model].get(image_id, []),
                args.iou,
            )
            for model in MODEL_ORDER
        }
        row = {
            "image_id": image_id,
            "file_name": info["file_name"],
            "brightness": luminance(source),
            "scene_bin": scene_bin(luminance(source)),
            "gt_count": len(current_gt),
            "small_gt": sum(
                float(item.get("area", item["bbox"][2] * item["bbox"][3])) < 32.0**2
                for item in current_gt
            ),
            "staff_gt": sum(bool(item.get("_is_staff", False)) for item in current_gt),
            "class_signature": "|".join(
                f"{category_names[category_id]}:{count}"
                for category_id, count in sorted(
                    Counter(int(item["category_id"]) for item in current_gt).items()
                )
            ),
            "selection_score": score_candidate(current_gt, metrics),
        }
        for model in MODEL_ORDER:
            slug = MODEL_SLUG[model]
            for metric_name, metric_value in metrics[model].items():
                row[f"{slug}_{metric_name}"] = metric_value
        rows.append(row)

    rows.sort(key=lambda row: float(row["selection_score"]), reverse=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with (output_dir / "candidate_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    candidate_ids = [
        int(row["image_id"]) for row in rows[: max(1, args.candidate_count)]
    ]
    export_contact_sheets(
        candidate_ids,
        image_info,
        gt_index,
        pred_index,
        image_dir,
        output_dir,
        category_names,
        category_colors,
        args.confidence,
        args.rows_per_sheet,
    )

    selected_ids = args.selected_ids or candidate_ids[: max(1, args.selected_count)]
    export_final(
        selected_ids,
        image_info,
        gt_index,
        pred_index,
        image_dir,
        output_dir,
        category_names,
        category_colors,
        args.confidence,
    )
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "confidence_threshold": args.confidence,
                "iou_threshold": args.iou,
                "selected_ids": selected_ids,
                "prediction_paths": {
                    model: str(path) for model, path in prediction_paths.items()
                },
                "annotation_path": str(annotation_path),
                "class_colors": {
                    category_names[category_id]: color
                    for category_id, color in category_colors.items()
                },
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
