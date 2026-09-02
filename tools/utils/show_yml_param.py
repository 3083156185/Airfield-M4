import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from engine.core import YAMLConfig


DEFAULT_CONFIG = "configs/dfine/dfine_hgnetv2_n_custom.yml"
SECTION_WIDTH = 76
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_CYAN = "\033[36m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_MAGENTA = "\033[35m"
ANSI_BLUE = "\033[34m"

MODEL_DETAIL_FIELDS = {
    "architecture_cfg": [
        "backbone",
        "encoder",
        "decoder",
    ],
    "backbone_cfg": [
        "name",
        "pretrained",
        "return_idx",
        "freeze_at",
        "freeze_norm",
        "freeze_stem_only",
        "use_lab",
    ],
    "encoder_cfg": [
        "in_channels",
        "feat_strides",
        "hidden_dim",
        "use_encoder_idx",
        "num_encoder_layers",
        "nhead",
        "dim_feedforward",
        "dropout",
        "enc_act",
        "expansion",
        "depth_mult",
        "act",
    ],
    "decoder_cfg": [
        "feat_channels",
        "feat_strides",
        "hidden_dim",
        "num_levels",
        "num_layers",
        "eval_idx",
        "num_queries",
        "num_denoising",
        "label_noise_ratio",
        "box_noise_scale",
        "reg_max",
        "reg_scale",
        "layer_scale",
        "num_points",
        "cross_attn_method",
        "query_select_method",
        "dim_feedforward",
    ],
    "criterion_cfg": [
        "losses",
        "alpha",
        "gamma",
        "reg_max",
    ],
    "postprocessor_cfg": [
        "num_top_queries",
    ],
    "optimizer_cfg": [
        "type",
        "lr",
        "betas",
        "weight_decay",
        "params",
    ],
    "lr_scheduler_cfg": [
        "type",
        "milestones",
        "gamma",
    ],
    "lr_warmup_scheduler_cfg": [
        "type",
        "warmup_duration",
    ],
    "dataset_cfg": [
        "type",
        "img_folder",
        "ann_file",
        "return_masks",
    ],
    "transforms_cfg": [
        "type",
        "ops",
        "policy",
    ],
    "collate_fn_cfg": [
        "type",
        "base_size",
        "base_size_repeat",
        "stop_epoch",
        "ema_restart_decay",
    ],
    "scaler_cfg": [
        "type",
        "enabled",
    ],
    "ema_cfg": [
        "type",
        "decay",
        "warmups",
        "start",
    ],
    "evaluator_cfg": [
        "type",
        "iou_types",
    ],
}

SECTION_HIGHLIGHTS = {
    "Meta": {"task", "num_classes", "epoches"},
    "Model": {"model", "backbone", "encoder", "decoder"},
    "Optimize": {"optimizer", "lr", "scheduler", "warmup"},
    "Train": {"batch_size", "dataset"},
    "Val": {"batch_size", "dataset"},
    "Runtime": {"use_amp", "use_ema", "sync_bn"},
}


def _should_keep(value):
    return value is not None and value != {} and value != []


def _compact_dict(pairs):
    return {key: value for key, value in pairs if _should_keep(value)}


def _render_inline(value):
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _resolve_color_mode(color_mode):
    if color_mode == "always":
        return True
    if color_mode == "never":
        return False
    if os.getenv("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def _style(text, *codes, use_color=False):
    if not use_color or not codes:
        return text
    return f"{''.join(codes)}{text}{ANSI_RESET}"


def _format_op(op):
    if not isinstance(op, dict):
        return str(op)

    op_type = op.get("type", "Unknown")
    extras = []
    for key, value in op.items():
        if key in {"type", "ops", "policy"}:
            continue
        extras.append(f"{key}={_render_inline(value)}")

    if not extras:
        return op_type

    return f"{op_type}({', '.join(extras)})"


def _order_dict(cfg_section, preferred_fields):
    if not isinstance(cfg_section, dict):
        return {}

    ordered = {}
    for field in preferred_fields:
        if field in cfg_section:
            ordered[field] = cfg_section[field]

    for field, value in cfg_section.items():
        if field not in ordered:
            ordered[field] = value

    return ordered


def _extract_transforms(loader_cfg):
    ops = (
        loader_cfg.get("dataset", {})
        .get("transforms", {})
        .get("ops", [])
    )
    return [_format_op(op) for op in ops]


def _extract_dataset_label(loader_cfg):
    dataset_cfg = loader_cfg.get("dataset", {})
    return dataset_cfg.get("ann_file") or dataset_cfg.get("img_folder")


def _build_loader_section(loader_cfg):
    if not loader_cfg:
        return {}

    return _compact_dict(
        [
            ("batch_size", loader_cfg.get("total_batch_size")),
            ("num_workers", loader_cfg.get("num_workers")),
            ("shuffle", loader_cfg.get("shuffle")),
            ("dataset", _extract_dataset_label(loader_cfg)),
            ("transforms", _extract_transforms(loader_cfg)),
        ]
    )


def _pick_fields(cfg_section, field_names):
    if not isinstance(cfg_section, dict):
        return {}

    return _compact_dict((field, cfg_section.get(field)) for field in field_names)


def _extract_model_section(cfg):
    model_name = cfg.get("model")
    model_cfg = cfg.get(model_name, {}) if isinstance(model_name, str) else {}
    backbone_name = model_cfg.get("backbone")
    encoder_name = model_cfg.get("encoder")
    decoder_name = model_cfg.get("decoder")
    criterion_name = cfg.get("criterion")
    postprocessor_name = cfg.get("postprocessor")

    return _compact_dict(
        [
            ("model", model_name),
            ("architecture_cfg", _order_dict(model_cfg, MODEL_DETAIL_FIELDS["architecture_cfg"])),
            ("backbone", backbone_name),
            ("backbone_cfg", _order_dict(cfg.get(backbone_name, {}), MODEL_DETAIL_FIELDS["backbone_cfg"])),
            ("encoder", encoder_name),
            ("encoder_cfg", _order_dict(cfg.get(encoder_name, {}), MODEL_DETAIL_FIELDS["encoder_cfg"])),
            ("decoder", decoder_name),
            ("decoder_cfg", _order_dict(cfg.get(decoder_name, {}), MODEL_DETAIL_FIELDS["decoder_cfg"])),
            ("criterion", criterion_name),
            ("criterion_cfg", _order_dict(cfg.get(criterion_name, {}), MODEL_DETAIL_FIELDS["criterion_cfg"])),
            ("postprocessor", postprocessor_name),
            ("postprocessor_cfg", _order_dict(cfg.get(postprocessor_name, {}), MODEL_DETAIL_FIELDS["postprocessor_cfg"])),
        ]
    )


def _extract_loader_summary(loader_cfg):
    if not loader_cfg:
        return {}

    dataset_cfg = loader_cfg.get("dataset", {})
    transforms_cfg = dataset_cfg.get("transforms", {})
    dataset_without_transforms = {
        key: value for key, value in dataset_cfg.items() if key != "transforms"
    }

    return _compact_dict(
        [
            ("type", loader_cfg.get("type")),
            ("batch_size", loader_cfg.get("total_batch_size")),
            ("total_batch_size", loader_cfg.get("total_batch_size")),
            ("num_workers", loader_cfg.get("num_workers")),
            ("shuffle", loader_cfg.get("shuffle")),
            ("drop_last", loader_cfg.get("drop_last")),
            ("pin_memory", loader_cfg.get("pin_memory")),
            ("dataset", _extract_dataset_label(loader_cfg)),
            ("dataset_cfg", _order_dict(dataset_without_transforms, MODEL_DETAIL_FIELDS["dataset_cfg"])),
            ("transforms", _extract_transforms(loader_cfg)),
            ("transforms_cfg", _order_dict(transforms_cfg, MODEL_DETAIL_FIELDS["transforms_cfg"])),
            ("collate_fn_cfg", _order_dict(loader_cfg.get("collate_fn", {}), MODEL_DETAIL_FIELDS["collate_fn_cfg"])),
        ]
    )


def build_summary(cfg):
    optimizer_cfg = cfg.get("optimizer", {})
    lr_scheduler_cfg = cfg.get("lr_scheduler", {})
    warmup_cfg = cfg.get("lr_warmup_scheduler", {})
    consumed_keys = set()

    summary = {}

    summary["Meta"] = _compact_dict(
        [
            ("task", cfg.get("task")),
            ("num_classes", cfg.get("num_classes")),
            ("epoches", cfg.get("epoches")),
            ("eval_spatial_size", cfg.get("eval_spatial_size")),
            ("output_dir", cfg.get("output_dir")),
        ]
    )
    consumed_keys.update(summary["Meta"].keys())

    summary["Model"] = _extract_model_section(cfg)
    consumed_keys.update(
        {
            "model",
            "criterion",
            "postprocessor",
            cfg.get("model"),
            cfg.get("criterion"),
            cfg.get("postprocessor"),
        }
    )
    model_cfg = cfg.get(cfg.get("model"), {})
    if isinstance(model_cfg, dict):
        consumed_keys.add(cfg.get("model"))
        for part in ("backbone", "encoder", "decoder"):
            part_name = model_cfg.get(part)
            if part_name:
                consumed_keys.add(part_name)

    summary["Optimize"] = _compact_dict(
        [
            ("optimizer", optimizer_cfg.get("type")),
            ("optimizer_cfg", _order_dict(optimizer_cfg, MODEL_DETAIL_FIELDS["optimizer_cfg"])),
            ("lr", optimizer_cfg.get("lr")),
            ("weight_decay", optimizer_cfg.get("weight_decay")),
            ("scheduler", lr_scheduler_cfg.get("type")),
            ("lr_scheduler_cfg", _order_dict(lr_scheduler_cfg, MODEL_DETAIL_FIELDS["lr_scheduler_cfg"])),
            ("milestones", lr_scheduler_cfg.get("milestones")),
            ("gamma", lr_scheduler_cfg.get("gamma")),
            ("warmup", warmup_cfg.get("warmup_duration") or cfg.get("warmup_iter")),
            ("lr_warmup_scheduler_cfg", _order_dict(warmup_cfg, MODEL_DETAIL_FIELDS["lr_warmup_scheduler_cfg"])),
            ("clip_max_norm", cfg.get("clip_max_norm")),
            ("lrsheduler", cfg.get("lrsheduler")),
            ("lr_gamma", cfg.get("lr_gamma")),
            ("warmup_iter", cfg.get("warmup_iter")),
            ("flat_epoch", cfg.get("flat_epoch")),
            ("no_aug_epoch", cfg.get("no_aug_epoch")),
        ]
    )
    consumed_keys.update(
        {
            "optimizer",
            "lr_scheduler",
            "lr_warmup_scheduler",
            "clip_max_norm",
            "lrsheduler",
            "lr_gamma",
            "warmup_iter",
            "flat_epoch",
            "no_aug_epoch",
        }
    )

    summary["Train"] = _extract_loader_summary(cfg.get("train_dataloader", {}))
    summary["Val"] = _extract_loader_summary(cfg.get("val_dataloader", {}))
    consumed_keys.update({"train_dataloader", "val_dataloader"})

    summary["Runtime"] = _compact_dict(
        [
            ("use_amp", cfg.get("use_amp")),
            ("use_ema", cfg.get("use_ema")),
            ("sync_bn", cfg.get("sync_bn")),
            ("find_unused_parameters", cfg.get("find_unused_parameters")),
            ("cache_imgsz", cfg.get("cache_imgsz")),
            ("print_freq", cfg.get("print_freq")),
            ("checkpoint_freq", cfg.get("checkpoint_freq")),
            ("plot_train_batch_freq", cfg.get("plot_train_batch_freq")),
            ("verbose_type", cfg.get("verbose_type")),
            ("ram_cache", cfg.get("ram_cache")),
            ("yolo_metrice", cfg.get("yolo_metrice")),
            ("remap_mscoco_category", cfg.get("remap_mscoco_category")),
            ("use_focal_loss", cfg.get("use_focal_loss")),
        ]
    )
    if "resume" in cfg:
        summary["Runtime"]["resume"] = cfg.get("resume")
    consumed_keys.update(summary["Runtime"].keys())

    summary["Scaler"] = _order_dict(cfg.get("scaler", {}), MODEL_DETAIL_FIELDS["scaler_cfg"])
    summary["EMA"] = _order_dict(cfg.get("ema", {}), MODEL_DETAIL_FIELDS["ema_cfg"])
    summary["Evaluator"] = _order_dict(cfg.get("evaluator", {}), MODEL_DETAIL_FIELDS["evaluator_cfg"])
    summary["Includes"] = _compact_dict(
        [
            ("files", cfg.get("__include__")),
        ]
    )
    consumed_keys.update({"scaler", "ema", "evaluator", "__include__"})

    extra = {
        key: value
        for key, value in cfg.items()
        if key not in consumed_keys and _should_keep(value)
    }
    if extra:
        summary["Extra"] = extra

    return {section: values for section, values in summary.items() if values}


def _is_scalar(value):
    return isinstance(value, (str, int, float, bool)) or value is None


def _should_render_list_inline(value):
    return bool(value) and all(_is_scalar(item) for item in value)


def _append_complex_value(lines, value, indent_level, use_color):
    indent = "   " * indent_level

    if isinstance(value, dict):
        if not value:
            lines.append(f"{indent}{_style('-', ANSI_BLUE, use_color=use_color)} {_style('{}', ANSI_GREEN, use_color=use_color)}")
            return

        nested_key_width = max(len(str(nested_key)) for nested_key in value)
        for nested_key, nested_value in value.items():
            nested_bullet = _style("-", ANSI_BLUE, use_color=use_color)
            nested_label = _style(f"{nested_key:<{nested_key_width}}", ANSI_CYAN, use_color=use_color)
            if isinstance(nested_value, dict):
                lines.append(f"{indent}{nested_bullet} {nested_label} :")
                _append_complex_value(lines, nested_value, indent_level + 1, use_color)
                continue

            if isinstance(nested_value, list) and not _should_render_list_inline(nested_value):
                lines.append(f"{indent}{nested_bullet} {nested_label} :")
                _append_complex_value(lines, nested_value, indent_level + 1, use_color)
                continue

            rendered_value = _style(_render_inline(nested_value), ANSI_GREEN, use_color=use_color)
            lines.append(f"{indent}{nested_bullet} {nested_label} : {rendered_value}")
        return

    if isinstance(value, list):
        for item in value:
            bullet = _style("*", ANSI_YELLOW, use_color=use_color)
            if isinstance(item, dict):
                lines.append(f"{indent}{bullet}")
                _append_complex_value(lines, item, indent_level + 1, use_color)
                continue
            rendered_item = _style(_render_inline(item), ANSI_GREEN, use_color=use_color)
            lines.append(f"{indent}{bullet} {rendered_item}")
        return


def render_summary(summary, use_color=False):
    lines = []
    for section, values in summary.items():
        if not values:
            continue

        key_width = max(len(key) for key in values)
        highlight_keys = SECTION_HIGHLIGHTS.get(section, set())
        border = _style("+" + ("-" * SECTION_WIDTH) + "+", ANSI_BLUE, use_color=use_color)
        title_text = _style(f" {section}".ljust(SECTION_WIDTH), ANSI_BOLD, ANSI_MAGENTA, use_color=use_color)
        title_line = f"|{title_text}|"

        lines.append(border)
        lines.append(title_line)
        lines.append(border)

        for key, value in values.items():
            is_highlight = key in highlight_keys
            prefix = _style(">>" if is_highlight else "  ", ANSI_YELLOW if is_highlight else ANSI_BLUE, use_color=use_color)
            label = _style(f"{key:<{key_width}}", ANSI_BOLD, ANSI_CYAN, use_color=use_color)
            expand_list = isinstance(value, list) and (key == "transforms" or not _should_render_list_inline(value))

            if isinstance(value, dict) or expand_list:
                lines.append(f"{prefix} {label} :")
                _append_complex_value(lines, value, 1, use_color)
                continue

            rendered_value = _style(_render_inline(value), ANSI_GREEN, use_color=use_color)
            lines.append(f"{prefix} {label} : {rendered_value}")

        lines.append("")

    return "\n".join(lines).strip()


def format_output(payload, output_format):
    if output_format == "json":
        return json.dumps(payload, indent=4, ensure_ascii=False)

    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Show a readable summary or full export for a YAML config.")
    parser.add_argument(
        "--config",
        "-c",
        default=DEFAULT_CONFIG,
        type=str,
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print the full merged YAML config instead of the compact summary.",
    )
    parser.add_argument(
        "--format",
        choices=("yaml", "json"),
        default="yaml",
        help="Output format. Summary mode uses section layout for yaml and JSON for json.",
    )
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="Color mode for summary output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"Config not found: {args.config}")

    cfg = YAMLConfig(str(config_path), resume=None)

    if args.full:
        print(format_output(cfg.yaml_cfg, args.format))
        return

    summary = build_summary(cfg.yaml_cfg)
    if args.format == "json":
        print(format_output(summary, "json"))
    else:
        print(render_summary(summary, use_color=_resolve_color_mode(args.color)))


if __name__ == "__main__":
    main()
