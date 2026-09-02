import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from engine.core import YAMLConfig

RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"


def _resolve_color_mode(color_mode):
    if color_mode == "always":
        return True
    if color_mode == "never":
        return False
    if os.getenv("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def _style(text, color, use_color):
    if not use_color:
        return text
    return f"{color}{text}{RESET}"


def _render_inline(value):
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _join_path(base, part):
    if not base:
        return part
    if part.startswith("["):
        return f"{base}{part}"
    return f"{base}.{part}"


def _append_added(entries, path, value):
    entries.append((path, value))


def _append_removed(entries, path, value):
    entries.append((path, value))


def _diff_values(left, right, path, changed, added, removed):
    if isinstance(left, dict) and isinstance(right, dict):
        left_keys = set(left)
        right_keys = set(right)

        for key in sorted(left_keys & right_keys):
            _diff_values(
                left[key],
                right[key],
                _join_path(path, str(key)),
                changed,
                added,
                removed,
            )

        for key in sorted(right_keys - left_keys):
            _append_added(added, _join_path(path, str(key)), right[key])

        for key in sorted(left_keys - right_keys):
            _append_removed(removed, _join_path(path, str(key)), left[key])
        return

    if isinstance(left, list) and isinstance(right, list):
        shared = min(len(left), len(right))
        for index in range(shared):
            _diff_values(
                left[index],
                right[index],
                _join_path(path, f"[{index}]"),
                changed,
                added,
                removed,
            )

        for index in range(shared, len(right)):
            _append_added(added, _join_path(path, f"[{index}]"), right[index])

        for index in range(shared, len(left)):
            _append_removed(removed, _join_path(path, f"[{index}]"), left[index])
        return

    if left != right:
        changed.append((path, left, right))


def diff_configs(left, right):
    changed = []
    added = []
    removed = []
    _diff_values(left, right, "", changed, added, removed)
    return {
        "changed": changed,
        "added": added,
        "removed": removed,
    }


def render_diff(left_label, right_label, diff, use_color=False):
    lines = [
        _style("Comparing:", BLUE, use_color),
        f"  {_style('A', BLUE, use_color)}: {_style(left_label, YELLOW, use_color)}",
        f"  {_style('B', BLUE, use_color)}: {_style(right_label, YELLOW, use_color)}",
        "",
    ]

    has_diff = False
    if diff["changed"]:
        has_diff = True
        lines.append(_style("Changed:", ORANGE, use_color))
        for path, left_value, right_value in diff["changed"]:
            lines.append(
                f"  {_style(path, BLUE, use_color)}: "
                f"{_style(_render_inline(left_value), RED, use_color)} "
                f"{_style('->', YELLOW, use_color)} "
                f"{_style(_render_inline(right_value), GREEN, use_color)}"
            )
        lines.append("")

    if diff["added"]:
        has_diff = True
        lines.append(_style("Added in B:", GREEN, use_color))
        for path, value in diff["added"]:
            lines.append(
                f"  {_style(path, BLUE, use_color)}: {_style(_render_inline(value), GREEN, use_color)}"
            )
        lines.append("")

    if diff["removed"]:
        has_diff = True
        lines.append(_style("Removed from B:", RED, use_color))
        for path, value in diff["removed"]:
            lines.append(
                f"  {_style(path, BLUE, use_color)}: {_style(_render_inline(value), RED, use_color)}"
            )
        lines.append("")

    if not has_diff:
        lines.append(_style("No differences found.", GREEN, use_color))

    return "\n".join(lines).rstrip()


def load_final_config(config_path):
    cfg = YAMLConfig(str(config_path), resume=None)
    final_cfg = dict(cfg.yaml_cfg)
    final_cfg.pop("__include__", None)
    return final_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two YAML configs after include/merge expansion."
    )
    parser.add_argument("left", type=str, help="Path to the first YAML config.")
    parser.add_argument("right", type=str, help="Path to the second YAML config.")
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="always",
        help="Color mode for diff output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    left_path = Path(args.left)
    right_path = Path(args.right)

    if not left_path.exists():
        raise SystemExit(f"Config not found: {args.left}")
    if not right_path.exists():
        raise SystemExit(f"Config not found: {args.right}")

    left_cfg = load_final_config(left_path)
    right_cfg = load_final_config(right_path)
    diff = diff_configs(left_cfg, right_cfg)
    print(render_diff(args.left, args.right, diff, use_color=_resolve_color_mode(args.color)))


if __name__ == "__main__":
    main()
