import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path

import yaml


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.core.yaml_utils import INCLUDE_KEY, load_config


class FlowStyleDict(dict):
    pass


class FlatConfigDumper(yaml.SafeDumper):
    pass


def _represent_flow_style_dict(dumper, data):
    return dumper.represent_mapping("tag:yaml.org,2002:map", data, flow_style=True)


def _is_scalar(value):
    return value is None or isinstance(value, (str, int, float, bool))


def _represent_list(dumper, data):
    flow_style = all(_is_scalar(item) for item in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow_style)


FlatConfigDumper.add_representer(FlowStyleDict, _represent_flow_style_dict)
FlatConfigDumper.add_representer(list, _represent_list)


def default_output_path(config_path):
    config_path = Path(config_path)
    return config_path.with_name(f"{config_path.stem}_flat{config_path.suffix}")


def _format_dataloader_transform_ops(cfg):
    cfg = copy.deepcopy(cfg)
    for dataloader_key in ("train_dataloader", "val_dataloader"):
        ops = (
            cfg.get(dataloader_key, {})
            .get("dataset", {})
            .get("transforms", {})
            .get("ops")
        )
        if not isinstance(ops, list):
            continue

        for index, op in enumerate(ops):
            if isinstance(op, dict) and "type" in op:
                ops[index] = FlowStyleDict(op)

    return cfg


def dump_flat_config(cfg):
    return yaml.dump(
        _format_dataloader_transform_ops(cfg),
        Dumper=FlatConfigDumper,
        sort_keys=False,
        allow_unicode=True,
    )


def flatten_config(config_path):
    cfg = copy.deepcopy(load_config(str(config_path)))
    cfg.pop(INCLUDE_KEY, None)
    return cfg


def export_flat_config(config_path, output_path=None):
    config_path = Path(config_path)
    if output_path is None:
        output_path = default_output_path(config_path)
    output_path = Path(output_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = flatten_config(config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        dump_flat_config(cfg),
        encoding="utf-8",
    )
    return cfg


def run_compare(original_path, flat_path):
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "utils" / "compare_yml.py"),
            str(original_path),
            str(flat_path),
            "--color",
            "never",
        ],
        cwd=str(ROOT),
        check=False,
        text=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a YAML config with __include__ dependencies expanded into one file."
    )
    parser.add_argument("config", type=str, help="Input YAML config path.")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output YAML path. Defaults to '<input_stem>_flat.yml'.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run tools/utils/compare_yml.py against the original and exported configs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.config)
    output_path = Path(args.output) if args.output else default_output_path(input_path)

    export_flat_config(input_path, output_path)
    print(f"Exported flat YAML: {output_path}", flush=True)

    if args.check:
        result = run_compare(input_path, output_path)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
