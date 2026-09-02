#!/usr/bin/env python3
"""Evaluate four D-FINE fusion stages under five sensor corruptions."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = Path("/root/miniconda3/envs/pytorch_2_8_0_py311/bin/python")
OUTPUT_ROOT = ROOT / "outputs" / "sensor_robustness_dfine"
CSV_PATH = OUTPUT_ROOT / "robustness_metrics.csv"
METADATA_PATH = OUTPUT_ROOT / "corruption_protocol.json"

METHODS = {
    "Early": {
        "config": "configs_easy/dfine_visible_mm_early_self.yml",
        "weight": "outputs/dfine_visible_mm_early_self_seed0_1322/best_stg2.pth",
    },
    "Plain-middle": {
        "config": "configs_easy/dfine_visible_mm_plain_mid.yml",
        "weight": "outputs/dfine_visible_mm_plain_mid/best_stg2.pth",
    },
    "PAPF": {
        "config": "configs_easy/dfine_visible_mm.yml",
        "weight": "outputs/dfine_visible_mm_self/best_stg2.pth",
    },
    "Late-neck": {
        "config": "configs_easy/dfine_visible_mm_late_neck_self.yml",
        "weight": "outputs/dfine_visible_mm_late_neck_self_seed0_132/best_stg2.pth",
    },
}

CORRUPTIONS = {
    "rgb_lowlight": {
        "label": "RGB low illumination",
        "levels": ["gamma=1.0", "gamma=1.4", "gamma=1.8", "gamma=2.2", "gamma=2.6"],
    },
    "rgb_visibility": {
        "label": "RGB haze + blur",
        "levels": [
            "alpha=0.00, sigma=0.0",
            "alpha=0.08, sigma=0.6",
            "alpha=0.16, sigma=1.2",
            "alpha=0.24, sigma=1.8",
            "alpha=0.32, sigma=2.4",
        ],
    },
    "ir_quality": {
        "label": "IR noise + contrast loss",
        "levels": [
            "contrast=1.00, noise=0.00",
            "contrast=0.75, noise=0.02",
            "contrast=0.50, noise=0.05",
            "contrast=0.30, noise=0.09",
            "contrast=0.15, noise=0.14",
        ],
    },
    "misalignment": {
        "label": "Cross-modal misalignment",
        "levels": ["0 px", "2 px", "4 px", "8 px", "16 px"],
    },
    "ir_missing": {
        "label": "IR missing samples",
        "levels": ["0%", "25%", "50%", "75%", "100%"],
    },
}

FIELDS = [
    "method",
    "corruption",
    "corruption_label",
    "severity",
    "level_value",
    "AP",
    "AP50",
    "AP75",
    "APs",
    "APm",
    "APl",
    "AR",
    "elapsed_seconds",
    "log_path",
]


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def parse_metrics(log_path: Path) -> dict[str, float]:
    text = strip_ansi(log_path.read_text(errors="replace"))
    ap_patterns = {
        "AP": r"Average Precision\s+\(AP\) @\[ IoU=0\.50:0\.95 \| area=\s*all \| maxDets=300 \] = ([0-9.]+)",
        "AP50": r"Average Precision\s+\(AP\) @\[ IoU=0\.50\s+\| area=\s*all \| maxDets=300 \] = ([0-9.]+)",
        "AP75": r"Average Precision\s+\(AP\) @\[ IoU=0\.75\s+\| area=\s*all \| maxDets=300 \] = ([0-9.]+)",
        "APs": r"Average Precision\s+\(AP\) @\[ IoU=0\.50:0\.95 \| area=\s*small \| maxDets=300 \] = ([0-9.]+)",
        "APm": r"Average Precision\s+\(AP\) @\[ IoU=0\.50:0\.95 \| area=medium \| maxDets=300 \] = ([0-9.]+)",
        "APl": r"Average Precision\s+\(AP\) @\[ IoU=0\.50:0\.95 \| area=\s*large \| maxDets=300 \] = ([0-9.]+)",
        "AR": r"Average Recall\s+\(AR\) @\[ IoU=0\.50:0\.95 \| area=\s*all \| maxDets=300 \] = ([0-9.]+)",
    }
    parsed = {}
    for key, pattern in ap_patterns.items():
        matches = re.findall(pattern, text)
        if not matches:
            raise RuntimeError(f"Could not parse {key} from {log_path}")
        parsed[key] = float(matches[-1])
    return parsed


def load_rows() -> list[dict[str, str]]:
    if not CSV_PATH.exists():
        return []
    with CSV_PATH.open(newline="") as handle:
        return list(csv.DictReader(handle))


def save_rows(rows: list[dict]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    temp_path = CSV_PATH.with_suffix(".csv.tmp")
    with temp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(CSV_PATH)


def key_of(row: dict) -> tuple[str, str, int]:
    return row["method"], row["corruption"], int(row["severity"])


def run_one(method: str, corruption: str, severity: int) -> dict:
    method_spec = METHODS[method]
    corruption_spec = CORRUPTIONS[corruption]
    slug = f"{method.lower().replace('-', '_')}_{corruption}_s{severity}"
    run_dir = OUTPUT_ROOT / "runs" / slug
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "eval.log"

    effective_corruption = "clean" if severity == 0 else corruption
    command = [
        str(PYTHON),
        str(ROOT / "eval_corrupted.py"),
        "--corruption",
        effective_corruption,
        "--severity",
        str(severity),
        "-c",
        method_spec["config"],
        "-r",
        method_spec["weight"],
        "--test-only",
        "--output-dir",
        str(run_dir / "result"),
        "-u",
        "yolo_metrice=false",
    ]
    started = time.time()
    with log_path.open("w") as log_handle:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.time() - started
    if completed.returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
        raise RuntimeError(
            f"Evaluation failed for {method}/{corruption}/s{severity}\n{tail}"
        )

    metrics = parse_metrics(log_path)
    return {
        "method": method,
        "corruption": corruption,
        "corruption_label": corruption_spec["label"],
        "severity": severity,
        "level_value": corruption_spec["levels"][severity],
        **metrics,
        "elapsed_seconds": round(elapsed, 3),
        "log_path": str(log_path.relative_to(ROOT)),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(
        json.dumps(
            {
                "dataset": "Airfield-M4 validation split",
                "model": "D-FINE-S",
                "image_count": 444,
                "metric": "COCO AP, maxDets=300",
                "methods": METHODS,
                "corruptions": CORRUPTIONS,
                "deterministic_seed": 20260729,
                "notes": [
                    "No model is retrained or fine-tuned.",
                    "Severity zero is the unmodified validation set.",
                    "IR missing subsets are deterministic and nested.",
                    "Misalignment is a diagonal IR translation measured at 640x640 model input.",
                ],
            },
            indent=2,
        )
    )

    rows = load_rows()
    completed_keys = {key_of(row) for row in rows}
    total = len(METHODS) * (1 + len(CORRUPTIONS) * 4)
    done = len(completed_keys)

    for method in METHODS:
        clean_rows = [
            row
            for row in rows
            if row["method"] == method and int(row["severity"]) == 0
        ]
        if clean_rows:
            clean_template = clean_rows[0]
        else:
            first_corruption = next(iter(CORRUPTIONS))
            done += 1
            print(f"[{done:03d}/{total:03d}] {method} | clean", flush=True)
            clean_template = run_one(method, first_corruption, 0)

        for corruption, corruption_spec in CORRUPTIONS.items():
            key = (method, corruption, 0)
            if key in completed_keys:
                continue
            clean_row = dict(clean_template)
            clean_row.update(
                {
                    "corruption": corruption,
                    "corruption_label": corruption_spec["label"],
                    "severity": 0,
                    "level_value": corruption_spec["levels"][0],
                }
            )
            rows.append(clean_row)
            completed_keys.add(key)
        save_rows(rows)

        for corruption in CORRUPTIONS:
            for severity in range(1, 5):
                key = (method, corruption, severity)
                if key in completed_keys:
                    continue
                done += 1
                print(
                    f"[{done:03d}/{total:03d}] {method} | {corruption} | severity={severity}",
                    flush=True,
                )
                row = run_one(method, corruption, severity)
                rows.append(row)
                completed_keys.add(key)
                save_rows(rows)
                print(
                    f"  AP={row['AP']:.3f} AP50={row['AP50']:.3f} "
                    f"APs={row['APs']:.3f} time={row['elapsed_seconds']:.1f}s",
                    flush=True,
                )

    print(f"Completed {len(rows)} evaluations. Metrics: {CSV_PATH}", flush=True)


if __name__ == "__main__":
    main()
