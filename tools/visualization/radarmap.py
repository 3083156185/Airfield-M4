import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


# DEFAULT_DATA = {
#     "names": ["AP", "AP50", "AP75", "APs", "APm", "APl", "APtiny"],
#     "yolov7-tiny":  [0.531, 0.847, 0.748, 0.601, 0.548, 0.721, 0.278],
#     "yolov8n":      [0.560, 0.870, 0.780, 0.670, 0.578, 0.760, 0.312],
#     "yolov9-tiny":  [0.553, 0.863, 0.762, 0.641, 0.563, 0.738, 0.295],
#     "yolov10n":     [0.546, 0.858, 0.755, 0.628, 0.556, 0.731, 0.287],
#     "yolo11n":      [0.580, 0.875, 0.753, 0.687, 0.582, 0.750, 0.328],
#     "rtdetr-r18":   [0.650, 0.912, 0.812, 0.662, 0.612, 0.791, 0.381],
#     "dfine-n":      [0.623, 0.895, 0.798, 0.645, 0.598, 0.772, 0.356],
#     "deim-n":       [0.638, 0.903, 0.805, 0.658, 0.607, 0.783, 0.368],
#     "faster-rcnn":  [0.542, 0.856, 0.761, 0.623, 0.561, 0.742, 0.289],
#     "cascade-rcnn": [0.568, 0.874, 0.778, 0.648, 0.572, 0.758, 0.308],
#     "ATSS":         [0.612, 0.888, 0.792, 0.638, 0.589, 0.768, 0.341],
#     "TOOD":         [0.613, 0.872, 0.776, 0.632, 0.575, 0.751, 0.335],
#     "RTMDet-N":     [0.572, 0.872, 0.763, 0.678, 0.580, 0.755, 0.319],
# }

DEFAULT_DATA = {
    "names": ["AP", "AP50", "APs", "APm", "APl"],
    "Faster-RCNN-R50-FPN-CIOU": [0.194, 0.329, 0.095, 0.309, 0.429],
    "Cascade-RCNN-R50-FPN": [0.197, 0.326, 0.099, 0.309, 0.406],
    "ATSS-R50-FPN-DyHead": [0.204, 0.338, 0.100, 0.317, 0.485],
    "TOOD-R50": [0.204, 0.339, 0.102, 0.317, 0.403],
    "DINO": [0.253, 0.445, 0.150, 0.371, 0.503],
    "DDQ": [0.268, 0.463, 0.159, 0.390, 0.526],
    "GFL": [0.193, 0.321, 0.094, 0.300, 0.409],
    "RetinaNet-R50-FPN": [0.164, 0.276, 0.060, 0.274, 0.427],
    "D-Fine-Dinov3(ConvNext-Tiny)-L-4scale(P2345)": [0.284, 0.480, 0.178, 0.398, 0.526],
    "RTDETR-R18": [0.208, 0.363, 0.113, 0.305, 0.413],
    "RTDETRV2-R18": [0.222, 0.391, 0.127, 0.321, 0.456],
    "YOLO8m": [0.190, 0.332, 0.090, 0.294, 0.417],
    "YOLO10m": [0.195, 0.345, 0.097, 0.300, 0.414],
    "YOLO11m": [0.203, 0.350, 0.098, 0.312, 0.413],
    "YOLO12m": [0.192, 0.336, 0.094, 0.298, 0.386],
    "YOLO26m": [0.186, 0.332, 0.096, 0.281, 0.361]
}

COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#E64B35",
    "#4DBBD5",
    "#00A087",
    "#3C5488",
    "#F39B7F",
    "#8491B4",
    "#91D1C2",
    "#7E6148",
    "#B09C85",
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p", "8", "H", "d"]


def load_data():
    """Return a copy of the built-in chart data."""
    return {key: list(value) for key, value in DEFAULT_DATA.items()}


def validate_chart_data(data):
    """Validate the radar chart data and return metric names plus model values."""
    if not isinstance(data, dict):
        raise ValueError("Chart data must be a JSON object.")

    names = data.get("names")
    if not isinstance(names, list) or not names:
        raise ValueError("Chart data must contain a non-empty 'names' list.")

    if not all(isinstance(name, str) and name for name in names):
        raise ValueError("Every metric name in 'names' must be a non-empty string.")

    expected_len = len(names)
    models = {key: value for key, value in data.items() if key != "names"}
    if not models:
        raise ValueError("Chart data must contain at least one model.")

    errors = []
    for model_name, values in models.items():
        if not isinstance(values, list):
            errors.append(f"Model '{model_name}' values must be a list.")
            continue

        if len(values) != expected_len:
            errors.append(
                f"Model '{model_name}' has {len(values)} values, "
                f"but expected {expected_len} values to match 'names'."
            )
            continue

        non_numeric = [
            value for value in values if not isinstance(value, (int, float))
        ]
        if non_numeric:
            errors.append(f"Model '{model_name}' contains non-numeric values.")

    if errors:
        raise ValueError("Data validation failed:\n" + "\n".join(errors))

    return list(names), {key: list(value) for key, value in models.items()}


def setup_matplotlib_style():
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    serif_font = "Times New Roman" if "Times New Roman" in available_fonts else "DejaVu Serif"

    plt.rcParams.update(
        {
            "font.family": serif_font,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 9,
            "legend.fontsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def build_angles(metric_count):
    angles = np.linspace(0, 2 * np.pi, metric_count, endpoint=False).tolist()
    return angles + angles[:1]


def build_metric_ranges(models, step=0.1):
    """Build an independent rounded axis range for each metric."""
    metric_ranges = []
    values_by_metric = zip(*models.values())

    for values in values_by_metric:
        metric_min = min(values)
        metric_max = max(values)
        lower = math.floor(metric_min / step) * step
        upper = math.ceil(metric_max / step) * step

        if math.isclose(lower, upper):
            lower = max(0.0, lower - step)
            upper = upper + step

        tick_count = int(round((upper - lower) / step)) + 1
        ticks = [round(lower + index * step, 10) for index in range(tick_count)]

        metric_ranges.append(
            {
                "min": round(lower, 10),
                "max": round(upper, 10),
                "ticks": ticks,
            }
        )

    return metric_ranges


def normalize_models(models, metric_ranges):
    normalized = {}
    for model_name, values in models.items():
        normalized[model_name] = [
            (value - metric_range["min"]) / (metric_range["max"] - metric_range["min"])
            for value, metric_range in zip(values, metric_ranges)
        ]
    return normalized


def find_best_metric_points(models):
    """Return the highest scoring model/value for each metric."""
    best_points = []
    for metric_index, values in enumerate(zip(*models.values())):
        model_names = list(models)
        best_model_index, best_value = max(
            enumerate(values),
            key=lambda item: item[1],
        )
        best_points.append(
            {
                "model": model_names[best_model_index],
                "value": best_value,
                "metric_index": metric_index,
            }
        )
    return best_points


def add_metric_tick_labels(ax, angles, metric_ranges):
    for angle, metric_range in zip(angles[:-1], metric_ranges):
        lower = metric_range["min"]
        upper = metric_range["max"]
        span = upper - lower

        for tick in metric_range["ticks"]:
            radius = (tick - lower) / span
            if math.isclose(radius, 0.0):
                continue

            ax.text(
                angle,
                radius,
                f"{tick:.1f}",
                fontsize=5.5,
                color="grey",
                ha="center",
                va="center",
                alpha=0.75,
                zorder=1,
            )


def get_best_label_radius(independent_metric_scale=True, ylim=(0.0, 1.0)):
    if independent_metric_scale:
        return 1.12

    y_min, y_max = ylim
    return y_max + (y_max - y_min) * 0.12


def get_polar_label_alignment(angle):
    x_direction = np.cos(angle)
    if x_direction > 0.2:
        return "left"
    if x_direction < -0.2:
        return "right"
    return "center"


def get_best_value_label_offset(angle, distance=9):
    x_direction = math.cos(angle)
    y_direction = math.sin(angle)

    if abs(x_direction) > 0.2:
        return (distance if x_direction > 0 else -distance, 0)

    return (0, distance if y_direction >= 0 else -distance)


def get_legend_layout(model_count):
    if model_count <= 8:
        return {
            "loc": "upper right",
            "bbox_to_anchor": (1.32, 1.15),
            "ncol": 1,
        }

    return {
        "loc": "upper center",
        "bbox_to_anchor": (0.5, -0.08),
        "ncol": min(5, math.ceil(model_count / 3)),
    }


def plot_radar_chart(
    data,
    output_prefix="radar_chart",
    title="Performance Comparison of Object Detection Models",
    show=False,
    annotate_values=True,
    ylim=(0.0, 1.0),
    independent_metric_scale=True,
    metric_step=0.1,
    highlight_best=True,
):
    names, models = validate_chart_data(data)
    setup_matplotlib_style()

    angles = build_angles(len(names))
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    label_radius = get_best_label_radius(
        independent_metric_scale=independent_metric_scale,
        ylim=ylim,
    )

    if independent_metric_scale:
        metric_ranges = build_metric_ranges(models, step=metric_step)
        plot_models = normalize_models(models, metric_ranges)
        ax.set_ylim(0.0, label_radius if highlight_best else 1.0)
        ax.set_yticks(np.linspace(0.2, 1.0, 5))
        ax.set_yticklabels([])
        add_metric_tick_labels(ax, angles, metric_ranges)
    else:
        plot_models = models
        y_min, y_max = ylim
        ax.set_ylim(y_min, label_radius if highlight_best else y_max)
        r_ticks = np.linspace(y_min, y_max, 6)[1:]
        ax.set_yticks(r_ticks)
        ax.set_yticklabels([f"{tick:.1f}" for tick in r_ticks], color="grey", size=8)

    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="grey", alpha=0.6)
    ax.xaxis.grid(True, linestyle="-", linewidth=0.6, color="grey", alpha=0.4)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(names, size=11, fontweight="bold")
    ax.tick_params(axis="x", pad=12)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.spines["polar"].set_visible(False)

    best_points = find_best_metric_points(models) if highlight_best else []
    best_lookup = {
        (point["model"], point["metric_index"]) for point in best_points
    }

    for index, (model_name, values) in enumerate(plot_models.items()):
        color = COLORS[index % len(COLORS)]
        marker = MARKERS[index % len(MARKERS)]
        closed_values = values + values[:1]
        raw_values = models[model_name]

        ax.fill(angles, closed_values, color=color, alpha=0.08)
        ax.plot(
            angles,
            closed_values,
            color=color,
            linewidth=1.8,
            linestyle="-",
            marker=marker,
            markersize=5,
            markerfacecolor=color,
            markeredgewidth=0.8,
            markeredgecolor="white",
            label=model_name,
            zorder=3,
        )

        if annotate_values:
            for metric_index, (angle, value, raw_value) in enumerate(
                zip(angles[:-1], values, raw_values)
            ):
                if (model_name, metric_index) in best_lookup:
                    continue

                x_offset = 0.045 * np.cos(angle)
                y_offset = 0.045 * np.sin(angle)
                ax.annotate(
                    f"{raw_value:.3f}",
                    xy=(angle, value),
                    xytext=(angle + x_offset * 0.5, value + y_offset),
                    fontsize=6.5,
                    color=color,
                    ha="center",
                    va="center",
                    fontweight="normal",
                    zorder=4,
                )

    if highlight_best:
        model_to_color = {
            model_name: COLORS[index % len(COLORS)]
            for index, model_name in enumerate(models)
        }

        for point in best_points:
            model_name = point["model"]
            metric_index = point["metric_index"]
            angle = angles[metric_index]
            value = plot_models[model_name][metric_index]
            raw_value = point["value"]
            color = model_to_color[model_name]

            ax.scatter(
                [angle],
                [value],
                s=88,
                marker="*",
                color=color,
                edgecolors="black",
                linewidths=0.8,
                zorder=6,
                label="_nolegend_",
            )
            ax.scatter(
                [angle],
                [value],
                s=150,
                marker="o",
                facecolors="none",
                edgecolors="black",
                linewidths=1.0,
                zorder=5,
                label="_nolegend_",
            )
            ax.annotate(
                f"{raw_value:.3f}",
                xy=(angle, value),
                xytext=get_best_value_label_offset(angle),
                textcoords="offset points",
                fontsize=7.5,
                color=color,
                ha=get_polar_label_alignment(angle),
                va="center",
                fontweight="bold",
                annotation_clip=False,
                zorder=7,
            )

    legend_layout = get_legend_layout(len(models))
    legend = ax.legend(
        loc=legend_layout["loc"],
        bbox_to_anchor=legend_layout["bbox_to_anchor"],
        frameon=True,
        framealpha=0.9,
        edgecolor="lightgrey",
        fancybox=True,
        title="Models",
        title_fontsize=11,
        fontsize=9.5,
        ncol=legend_layout["ncol"],
    )
    legend.get_title().set_fontweight("bold")

    ax.set_title(title, size=13, fontweight="bold", pad=25, va="bottom")

    prefix = Path(output_prefix)
    pdf_path = prefix.with_suffix(".pdf")
    png_path = prefix.with_suffix(".png")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return pdf_path, png_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a radar chart from the built-in data at the top of this file."
    )
    parser.add_argument(
        "--output",
        default="radar_chart",
        help="Output file prefix. The script writes both PDF and PNG.",
    )
    parser.add_argument(
        "--title",
        default="Performance Comparison of Object Detection Models",
        help="Chart title.",
    )
    parser.add_argument("--show", action="store_true", help="Display the chart window.")
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Hide numeric labels on radar points.",
    )
    parser.add_argument(
        "--global-scale",
        action="store_true",
        help="Use one shared radial scale instead of per-metric rounded scales.",
    )
    parser.add_argument(
        "--no-highlight-best",
        action="store_true",
        help="Disable star markers and value labels for each metric's best score.",
    )
    parser.add_argument(
        "--metric-step",
        type=float,
        default=0.1,
        help="Step size for per-metric rounded scales.",
    )
    parser.add_argument("--ylim-min", type=float, default=0.0, help="Minimum radial axis.")
    parser.add_argument("--ylim-max", type=float, default=1.0, help="Maximum radial axis.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.ylim_min >= args.ylim_max:
        raise ValueError("--ylim-min must be smaller than --ylim-max.")
    if args.metric_step <= 0:
        raise ValueError("--metric-step must be greater than 0.")

    data = load_data()
    names, models = validate_chart_data(data)
    print(f"[OK] Validation passed: {len(models)} models x {len(names)} metrics.")

    pdf_path, png_path = plot_radar_chart(
        data,
        output_prefix=args.output,
        title=args.title,
        show=args.show,
        annotate_values=not args.no_annotate,
        ylim=(args.ylim_min, args.ylim_max),
        independent_metric_scale=not args.global_scale,
        metric_step=args.metric_step,
        highlight_best=not args.no_highlight_best,
    )
    print(f"[Saved] {pdf_path} & {png_path}")


if __name__ == "__main__":
    main()
