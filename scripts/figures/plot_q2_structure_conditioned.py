#!/usr/bin/env python3
"""Plot Q2 structure-conditioned pairwise delta-MSE trends.

This script uses the final cross-scene complexity-stratified result CSVs only.
It does not recompute patch statistics; it visualizes the recorded binned
median pairwise deltas and verifies that plotted trend directions match the
recorded trend tests before writing figure files.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


SCENES = ("garden", "bicycle", "room")
SCENE_TITLES = {
    "garden": "Garden",
    "bicycle": "Bicycle",
    "room": "Room",
}

DESCRIPTORS = (
    "mean_gradient_magnitude",
    "high_frequency_energy",
    "edge_strength",
)
DESCRIPTOR_TITLES = {
    "mean_gradient_magnitude": "Mean Gradient Magnitude",
    "high_frequency_energy": "High-Frequency Energy",
    "edge_strength": "Edge Strength",
}
ROW_PANEL_LABELS = {
    "mean_gradient_magnitude": "(a) Mean Gradient Magnitude",
    "high_frequency_energy": "(b) High-Frequency Energy",
    "edge_strength": "(c) Edge Strength",
}

COMPARISONS = ("ges_vs_3dgs", "drk_vs_3dgs", "ges_vs_drk")
COMPARISON_LABELS = {
    "ges_vs_3dgs": "GES - 3DGS",
    "drk_vs_3dgs": "DRK - 3DGS",
    "ges_vs_drk": "GES - DRK",
}
COMPARISON_STYLES = {
    "ges_vs_3dgs": {"color": "#2C7A7B", "linestyle": "-", "marker": "o"},
    "drk_vs_3dgs": {"color": "#C05621", "linestyle": "--", "marker": "s"},
    "ges_vs_drk": {"color": "#4C51BF", "linestyle": "-.", "marker": "^"},
}

REQUIRED_PAIRED_COLUMNS = {
    "scene",
    "descriptor",
    "comparison",
    "bin_index",
    "median_delta_mse",
    "median_delta_mse_ci_low",
    "median_delta_mse_ci_high",
}
REQUIRED_TREND_COLUMNS = {
    "scene",
    "descriptor",
    "comparison",
    "binned_median_delta_slope",
}


@dataclass(frozen=True)
class TrendCheck:
    scene: str
    descriptor: str
    comparison: str
    plotted_slope: float
    plotted_direction: str
    recorded_slope: float
    recorded_direction: str
    consistency: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    default_results = root / "results" / "cross_scene" / "complexity_stratified_errors_p32"
    parser = argparse.ArgumentParser(
        description="Create the Q2 structure-conditioned specialization figure."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_results,
        help="Directory containing paired_delta_summary.csv, trend_tests.csv, and cross_scene_trend_consistency.csv.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=root / "figures" / "q2_structure_conditioned.pdf",
        help="Vector PDF output path.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=root / "figures" / "q2_structure_conditioned.png",
        help="High-resolution PNG output path.",
    )
    parser.add_argument("--dpi", type=int, default=400, help="PNG resolution.")
    return parser


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def require_files(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Required final result file(s) are missing. Run this script in the "
            "checkout that contains the completed Q2 result CSVs:\n" + formatted
        )


def require_columns(rows: list[dict[str, str]], required: set[str], path: Path) -> None:
    if not rows:
        raise ValueError(f"{path} is empty.")
    missing = required.difference(rows[0].keys())
    if missing:
        raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")


def as_float(row: dict[str, str], column: str) -> float:
    value = float(row[column])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite value in column {column}: {row[column]}")
    return value


def sign(value: float, eps: float = 1e-12) -> str:
    if abs(value) <= eps:
        return "zero"
    return "positive" if value > 0 else "negative"


def linear_slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("Need at least two matched x/y points to compute a slope.")
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        raise ValueError("Cannot compute slope with identical x positions.")
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom


def x_value(row: dict[str, str]) -> float:
    if row.get("normalized_bin_position", "") != "":
        return as_float(row, "normalized_bin_position")
    return as_float(row, "bin_index")


def rows_for(
    rows: list[dict[str, str]], scene: str, descriptor: str, comparison: str
) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row["scene"] == scene
        and row["descriptor"] == descriptor
        and row["comparison"] == comparison
    ]
    selected.sort(key=lambda row: (x_value(row), int(float(row["bin_index"]))))
    return selected


def trend_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    lookup: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["scene"], row["descriptor"], row["comparison"])
        if key in lookup:
            raise ValueError(f"Duplicate trend_tests row for {key}")
        lookup[key] = row
    return lookup


def consistency_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    for row in rows:
        if "descriptor" not in row or "comparison" not in row:
            continue
        status = (
            row.get("direction_consistency")
            or row.get("consistency")
            or row.get("direction_consistent_across_scenes")
            or row.get("classification")
            or ""
        )
        lookup[(row["descriptor"], row["comparison"])] = status
    return lookup


def validate_direction_checks(
    paired_rows: list[dict[str, str]],
    trends: dict[tuple[str, str, str], dict[str, str]],
    consistency: dict[tuple[str, str], str],
) -> list[TrendCheck]:
    checks: list[TrendCheck] = []
    mismatches: list[str] = []
    for scene in SCENES:
        for descriptor in DESCRIPTORS:
            for comparison in COMPARISONS:
                selected = rows_for(paired_rows, scene, descriptor, comparison)
                if len(selected) != 10:
                    raise ValueError(
                        f"Expected 10 binned rows for {scene}/{descriptor}/{comparison}, "
                        f"found {len(selected)}."
                    )
                xs = [x_value(row) for row in selected]
                ys = [as_float(row, "median_delta_mse") for row in selected]
                for row in selected:
                    as_float(row, "median_delta_mse_ci_low")
                    as_float(row, "median_delta_mse_ci_high")
                plotted_slope = linear_slope(xs, ys)

                key = (scene, descriptor, comparison)
                if key not in trends:
                    raise ValueError(f"Missing trend_tests row for {key}.")
                recorded_slope = as_float(trends[key], "binned_median_delta_slope")
                plotted_direction = sign(plotted_slope)
                recorded_direction = sign(recorded_slope)
                if plotted_direction != recorded_direction:
                    mismatches.append(
                        f"{scene}/{descriptor}/{comparison}: plotted slope "
                        f"{plotted_slope:.12g} ({plotted_direction}) != recorded "
                        f"{recorded_slope:.12g} ({recorded_direction})"
                    )
                checks.append(
                    TrendCheck(
                        scene=scene,
                        descriptor=descriptor,
                        comparison=comparison,
                        plotted_slope=plotted_slope,
                        plotted_direction=plotted_direction,
                        recorded_slope=recorded_slope,
                        recorded_direction=recorded_direction,
                        consistency=consistency.get((descriptor, comparison), ""),
                    )
                )
    if mismatches:
        raise ValueError(
            "Plotted trend direction(s) disagree with trend_tests.csv:\n"
            + "\n".join(f"  - {item}" for item in mismatches)
        )
    return checks


def plot_figure(
    paired_rows: list[dict[str, str]],
    output_pdf: Path,
    output_png: Path,
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.2,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(7.0, 5.9),
        sharex=True,
        constrained_layout=False,
    )

    legend_handles = []
    legend_labels = []
    for row_idx, descriptor in enumerate(DESCRIPTORS):
        for col_idx, scene in enumerate(SCENES):
            ax = axes[row_idx][col_idx]
            ax.axhline(0.0, color="0.25", linewidth=0.7, alpha=0.75, zorder=0)
            for comparison in COMPARISONS:
                selected = rows_for(paired_rows, scene, descriptor, comparison)
                xs = [x_value(row) for row in selected]
                ys = [as_float(row, "median_delta_mse") for row in selected]
                lows = [as_float(row, "median_delta_mse_ci_low") for row in selected]
                highs = [as_float(row, "median_delta_mse_ci_high") for row in selected]
                style = COMPARISON_STYLES[comparison]
                (line,) = ax.plot(
                    xs,
                    ys,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markersize=3.0,
                    linewidth=1.35,
                    label=COMPARISON_LABELS[comparison],
                    zorder=3,
                )
                ax.fill_between(
                    xs,
                    lows,
                    highs,
                    color=style["color"],
                    alpha=0.15,
                    linewidth=0,
                    zorder=2,
                )
                if row_idx == 0 and col_idx == 0:
                    legend_handles.append(line)
                    legend_labels.append(COMPARISON_LABELS[comparison])

            if row_idx == 0:
                ax.set_title(SCENE_TITLES[scene], pad=6, fontsize=9.2)
            if col_idx == 0:
                ax.set_ylabel(
                    ROW_PANEL_LABELS[descriptor],
                    labelpad=8,
                    fontsize=8.3,
                    fontweight="regular",
                )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", length=2.5, width=0.7)
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-2, 2))
            ax.yaxis.set_major_formatter(formatter)
            ax.grid(False)

    fig.supxlabel("Descriptor Quantile (Low \u2192 High)", y=0.060, fontsize=9.5)
    fig.supylabel("Median \u0394MSE", x=0.025, fontsize=9.5)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.54, 0.008),
        handlelength=2.4,
        columnspacing=1.8,
    )
    fig.subplots_adjust(left=0.13, right=0.985, top=0.91, bottom=0.155, wspace=0.28, hspace=0.22)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def print_report(checks: list[TrendCheck], paired_rows: list[dict[str, str]], args: argparse.Namespace) -> None:
    used_count = 0
    for scene in SCENES:
        for descriptor in DESCRIPTORS:
            for comparison in COMPARISONS:
                used_count += len(rows_for(paired_rows, scene, descriptor, comparison))
    print("Q2 structure-conditioned figure generated.")
    print(
        "Rows/data used: "
        f"{used_count} paired rows = 3 scenes x 3 descriptors x 3 comparisons x 10 bins."
    )
    print(
        "Fields used: normalized_bin_position or bin_index, median_delta_mse, "
        "median_delta_mse_ci_low, median_delta_mse_ci_high."
    )
    print("Trend directions verified against trend_tests.csv:")
    for check in checks:
        print(
            f"  {check.scene},{check.descriptor},{check.comparison}: "
            f"{check.recorded_direction} "
            f"(slope={check.recorded_slope:.12g}; consistency={check.consistency or 'not_recorded'})"
        )
    print(f"PDF: {args.output_pdf}")
    print(f"PNG: {args.output_png}")


def main() -> int:
    args = build_parser().parse_args()
    results_dir = args.results_dir
    paired_path = results_dir / "paired_delta_summary.csv"
    trend_path = results_dir / "trend_tests.csv"
    consistency_path = results_dir / "cross_scene_trend_consistency.csv"
    require_files((paired_path, trend_path, consistency_path))

    paired_rows = read_csv(paired_path)
    trend_rows = read_csv(trend_path)
    consistency_rows = read_csv(consistency_path)
    require_columns(paired_rows, REQUIRED_PAIRED_COLUMNS, paired_path)
    require_columns(trend_rows, REQUIRED_TREND_COLUMNS, trend_path)

    trends = trend_lookup(trend_rows)
    consistency = consistency_lookup(consistency_rows)
    checks = validate_direction_checks(paired_rows, trends, consistency)
    plot_figure(paired_rows, args.output_pdf, args.output_png, args.dpi)
    print_report(checks, paired_rows, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
