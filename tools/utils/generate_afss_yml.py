import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import argparse, copy, yaml
from pathlib import Path
import yaml
from engine.core.yaml_utils import load_config


AFSS_DEFAULTS = {
    "enabled": True,
    "dump_refresh_json": False,
    "easy_threshold": 0.85,
    "moderate_threshold": 0.55,
    "easy_ratio": 0.02,
    "moderate_ratio": 0.4,
    "easy_review_gap": 10,
    "moderate_gap": 3,
    "update_interval": 5,
    "warmup_epochs": 5,
    "review_force_cap_ratio": 0.5,
    "iou_threshold": 0.5,
    "conf_threshold": 0.25,
    "random_seed": 0,
}


MULTIMODAL_DATASET_TYPES = {"MultimodalCocoDetection"}


def _unwrap_afss_dataset(dataset_cfg):
    cfg = copy.deepcopy(dataset_cfg)
    while isinstance(cfg, dict) and cfg.get("type") == "AFSSDataset" and isinstance(cfg.get("dataset"), dict):
        cfg = copy.deepcopy(cfg["dataset"])
    return cfg


def _is_segmentation_config(cfg, train_dataset_cfg):
    if cfg.get("postprocessor") == "SegPostProcessor":
        return True
    if bool(train_dataset_cfg.get("return_masks", False)):
        return True
    iou_types = cfg.get("evaluator", {}).get("iou_types", [])
    return "segm" in iou_types


def _is_multimodal_dataset(train_dataset_cfg):
    return train_dataset_cfg.get("type") in MULTIMODAL_DATASET_TYPES


def _build_lightweight_transforms(cfg, is_multimodal=False):
    resize_size = cfg.get("eval_spatial_size", [640, 640])
    ops = [
        {"type": "Resize", "size": list(resize_size)},
        {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
    ]
    if is_multimodal:
        ops.append({"type": "NormalizeNPYMinMax"})

    return {
        "type": "MultimodalCompose" if is_multimodal else "Compose",
        "ops": ops,
    }


def _build_afss_update_dataset(train_dataset_cfg, cfg, is_segmentation, is_multimodal=False):
    dataset_cfg = copy.deepcopy(train_dataset_cfg)
    dataset_cfg["return_masks"] = bool(is_segmentation)
    dataset_cfg["transforms"] = _build_lightweight_transforms(cfg, is_multimodal=is_multimodal)
    return dataset_cfg


def _build_train_wrapper(train_dataset_cfg, is_segmentation):
    dataset_cfg = copy.deepcopy(train_dataset_cfg)
    dataset_cfg["return_masks"] = bool(is_segmentation)
    return {
        "type": "AFSSDataset",
        "dataset": dataset_cfg,
    }


def _build_afss_update_loader(train_loader_cfg, train_dataset_cfg, cfg, is_segmentation, is_multimodal=False):
    update_loader = {
        "type": "DataLoader",
        "dataset": _build_afss_update_dataset(
            train_dataset_cfg,
            cfg,
            is_segmentation,
            is_multimodal=is_multimodal,
        ),
        "shuffle": False,
        "num_workers": int(train_loader_cfg.get("num_workers", 4)),
        "drop_last": False,
        "pin_memory": bool(train_loader_cfg.get("pin_memory", True)),
        "collate_fn": {
            "type": "BatchMultimodalCollateFunction" if is_multimodal else "BatchImageCollateFunction"
        },
    }

    if "total_batch_size" in train_loader_cfg:
        update_loader["total_batch_size"] = train_loader_cfg["total_batch_size"]
    elif "batch_size" in train_loader_cfg:
        update_loader["batch_size"] = train_loader_cfg["batch_size"]
    else:
        update_loader["total_batch_size"] = 8

    return update_loader


def _build_output_config(config_path, output_dir):
    cfg = load_config(str(config_path))
    train_loader_cfg = copy.deepcopy(cfg["train_dataloader"])
    train_dataset_cfg = _unwrap_afss_dataset(train_loader_cfg["dataset"])

    if train_dataset_cfg.get("type") is None:
        raise ValueError(f"Unsupported train dataset config in {config_path}: missing dataset.type")

    is_segmentation = _is_segmentation_config(cfg, train_dataset_cfg)
    is_multimodal = _is_multimodal_dataset(train_dataset_cfg)

    include_path = Path(os.path.relpath(config_path, output_dir))

    output_cfg = {
        "__include__": [include_path.as_posix()],
        "output_dir": f"./outputs/afss_training/{config_path.stem}_afss",
        "train_dataloader": {
            "dataset": _build_train_wrapper(train_dataset_cfg, is_segmentation),
        },
        "afss_update_dataloader": _build_afss_update_loader(
            train_loader_cfg,
            train_dataset_cfg,
            cfg,
            is_segmentation,
            is_multimodal=is_multimodal,
        ),
        "AFSS": copy.deepcopy(AFSS_DEFAULTS),
    }
    return output_cfg


def generate_afss_yaml(config_path, output_dir="configs/afss_training"):
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{config_path.stem}_afss.yml"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")

    output_cfg = _build_output_config(config_path, output_dir)
    output_path.write_text(yaml.safe_dump(output_cfg, sort_keys=False, allow_unicode=True))
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate an AFSS wrapper yml from an existing training yml.")
    parser.add_argument("--config", "-c", required=True, type=str, help="Source yml path")
    parser.add_argument(
        "--output-dir",
        default="configs/afss_training",
        type=str,
        help="Directory to place generated *_afss.yml",
    )
    args = parser.parse_args()

    output_path = generate_afss_yaml(args.config, args.output_dir)
    print(output_path)


if __name__ == "__main__":
    main()
