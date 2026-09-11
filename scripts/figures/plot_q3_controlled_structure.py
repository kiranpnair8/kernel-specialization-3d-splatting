#!/usr/bin/env python3
"""Plot Figure 3 for the controlled Phase-3 synthetic experiment.

This script reads the existing seed-level paired-delta table and does not
recompute experiment outputs. It preserves the Phase-3 matched-seed delta
convention used by scripts/synthetic/analyze_phase3_seed_interactions.py.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable


FACTORS = ("edge_sharpness", "spatial_frequency", "curvature")
FACTOR_TITLES = {
    "edge_sharpness": "Edge Sharpness",
    "spatial_frequency": "Spatial Frequency",
    "curvature": "Surface Curvature",
}
PANEL_LABELS = {
    "edge_sharpness": "(a)",
    "spatial_frequency": "(b)",
    "curvature": "(c)",
}

LEVELS = ("low", "medium", "high")
LEVEL_LABELS = ("Low", "Medium", "High")
EXPECTED_SEEDS = (0, 1, 2, 3, 4)

COMPARISONS = ("drk_minus_3dgs", "drk_minus_ges", "ges_minus_3dgs")
COMPARISON_LABELS = {
    "drk_minus_3dgs": "DRK - 3DGS",
    "drk_minus_ges": "DRK - GES",
    "ges_minus_3dgs": "GES - 3DGS",
}
COMPARISON_STYLES = {
    "drk_minus_3dgs": {"color": "#B24E3A", "linestyle": "-", "marker": "o"},
    "drk_minus_ges": {"color": "#4C51BF", "linestyle": "--", "marker": "s"},
    "ges_minus_3dgs": {"color": "#22845E", "linestyle": "-.", "marker": "^"},
}

EXPECTED_MEANS = {
    ("edge_sharpness", "drk_minus_3dgs"): (13.633836, 14.628632, 15.034534),
    ("edge_sharpness", "drk_minus_ges"): (18.046135, 18.367522, 19.142517),
    ("edge_sharpness", "ges_minus_3dgs"): (-4.412299, -3.738890, -4.107983),
    ("spatial_frequency", "drk_minus_3dgs"): (11.619193, 6.926912, 2.238960),
    ("spatial_frequency", "drk_minus_ges"): (16.889740, 13.759143, 10.482526),
    ("spatial_frequency", "ges_minus_3dgs"): (-5.270547, -6.832231, -8.243566),
    ("curvature", "drk_minus_3dgs"): (13.249753, 16.612832, 12.050694),
    ("curvature", "drk_minus_ges"): (15.578590, 17.040367, 12.376600),
    ("curvature", "ges_minus_3dgs"): (-2.328836, -0.427536, -0.325906),
}

REQUIRED_COLUMNS = {"sweep_family", "level", "seed", "comparison", "delta_psnr"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=root
        / "results"
        / "synthetic"
        / "phase3_controlled_pilot"
        / "evaluation"
        / "phase3_seed_paired_deltas.csv",
        help="Existing Phase-3 seed-level paired delta CSV.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=root / "figures" / "q3_controlled_structure.pdf",
        help="Vector PDF output path.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=root / "figures" / "q3_controlled_structure.png",
        help="High-resolution PNG output path.",
    )
    parser.add_argument("--dpi", type=int, default=400, help="PNG resolution.")
    parser.add_argument(
        "--mean-tolerance",
        type=float,
        default=5e-6,
        help="Absolute tolerance for validating means against checkpoint values.",
    )
    return parser


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required seed-level paired-delta CSV does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} is empty.")
    missing = REQUIRED_COLUMNS.difference(rows[0].keys())
    if missing:
        raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")
    return rows


def seed_value(row: dict[str, str]) -> int:
    return int(float(row["seed"]))


def psnr_value(row: dict[str, str]) -> float:
    value = float(row["delta_psnr"])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite delta_psnr value: {row['delta_psnr']}")
    return value


def collect_values(rows: Iterable[dict[str, str]]) -> dict[tuple[str, str, str], dict[int, float]]:
    values: dict[tuple[str, str, str], dict[int, float]] = {
        (factor, level, comparison): {}
        for factor in FACTORS
        for level in LEVELS
        for comparison in COMPARISONS
    }
    unexpected: list[tuple[str, str, str]] = []
    for row in rows:
        factor = row["sweep_family"]
        level = row["level"]
        comparison = row["comparison"]
        key = (factor, level, comparison)
        if key not in values:
            if factor in FACTORS or comparison in COMPARISONS:
                unexpected.append(key)
            continue
        seed = seed_value(row)
        if seed in values[key]:
            raise ValueError(f"Duplicate row for {key} seed={seed}")
        values[key][seed] = psnr_value(row)
    if unexpected:
        preview = ", ".join(str(item) for item in unexpected[:5])
        raise ValueError(f"Unexpected Phase-3 paired-delta rows encountered: {preview}")
    for key, seed_values in values.items():
        seeds = tuple(sorted(seed_values))
        if seeds != EXPECTED_SEEDS:
            raise ValueError(f"{key} has seeds {seeds}; expected {EXPECTED_SEEDS}")
    return values


def mean(items: Iterable[float]) -> float:
    values = list(items)
    if not values:
        raise ValueError("Cannot compute mean of an empty sequence.")
    return sum(values) / len(values)


def computed_means(values: dict[tuple[str, str, str], dict[int, float]]) -> dict[tuple[str, str], tuple[float, float, float]]:
    means: dict[tuple[str, str], tuple[float, float, float]] = {}
    for factor in FACTORS:
        for comparison in COMPARISONS:
            means[(factor, comparison)] = tuple(
                mean(values[(factor, level, comparison)][seed] for seed in EXPECTED_SEEDS)
                for level in LEVELS
            )
    return means


def validate_means(means: dict[tuple[str, str], tuple[float, float, float]], tolerance: float) -> None:
    mismatches: list[str] = []
    for key, expected_values in EXPECTED_MEANS.items():
        observed_values = means[key]
        for level, observed, expected in zip(LEVELS, observed_values, expected_values):
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=tolerance):
                mismatches.append(
                    f"{key[0]}/{level}/{key[1]} observed={observed:.9f} expected={expected:.9f}"
                )
    if mismatches:
        raise ValueError(
            "Computed seed means do not match the expected Phase-3 checkpoint values:\n"
            + "\n".join(f"  - {item}" for item in mismatches)
        )


def y_limits(values: dict[tuple[str, str, str], dict[int, float]], means: dict[tuple[str, str], tuple[float, float, float]]) -> tuple[float, float]:
    all_values: list[float] = []
    for seed_values in values.values():
        all_values.extend(seed_values.values())
    for mean_values in means.values():
        all_values.extend(mean_values)
    lower = min(all_values)
    upper = max(all_values)
    span = upper - lower
    pad = max(0.75, span * 0.08)
    return lower - pad, upper + pad


def plot_figure(
    values: dict[tuple[str, str, str], dict[int, float]],
    means: dict[tuple[str, str], tuple[float, float, float]],
    output_pdf: Path,
    output_png: Path,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.6,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.45), sharey=True)
    x = list(range(len(LEVELS)))
    ymin, ymax = y_limits(values, means)
    legend_handles = []
    legend_labels = []

    for panel_idx, (ax, factor) in enumerate(zip(axes, FACTORS)):
        ax.axhline(0.0, color="0.28", linewidth=0.75, alpha=0.8, zorder=0)
        for comparison in COMPARISONS:
            style = COMPARISON_STYLES[comparison]
            for seed in EXPECTED_SEEDS:
                y = [values[(factor, level, comparison)][seed] for level in LEVELS]
                ax.plot(
                    x,
                    y,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markersize=2.3,
                    linewidth=0.75,
                    alpha=0.22,
                    zorder=1,
                )
            mean_y = means[(factor, comparison)]
            (line,) = ax.plot(
                x,
                mean_y,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=4.0,
                linewidth=2.0,
                zorder=3,
                label=COMPARISON_LABELS[comparison],
            )
            if panel_idx == 0:
                legend_handles.append(line)
                legend_labels.append(COMPARISON_LABELS[comparison])

        ax.set_title(FACTOR_TITLES[factor], pad=6)
        ax.text(
            0.025,
            0.965,
            PANEL_LABELS[factor],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.7,
            fontweight="bold",
            color="0.08",
        )
        ax.set_xticks(x, LEVEL_LABELS)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, axis="y", color="0.86", linewidth=0.55, alpha=0.7)
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", length=2.5, width=0.7)
        if panel_idx == 0:
            ax.set_ylabel("Delta PSNR (dB)")

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.52, -0.015),
        handlelength=2.4,
        columnspacing=1.6,
    )
    fig.subplots_adjust(left=0.085, right=0.992, top=0.83, bottom=0.265, wspace=0.12)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def print_means(means: dict[tuple[str, str], tuple[float, float, float]]) -> None:
    print("Validated seed means for delta PSNR (dB):")
    for factor in FACTORS:
        print(FACTOR_TITLES[factor] + ":")
        for comparison in COMPARISONS:
            low, medium, high = means[(factor, comparison)]
            print(
                f"  {COMPARISON_LABELS[comparison]}: "
                f"low={low:.6f}, medium={medium:.6f}, high={high:.6f}"
            )


def main() -> int:
    args = build_parser().parse_args()
    rows = read_rows(args.input)
    values = collect_values(rows)
    means = computed_means(values)
    validate_means(means, args.mean_tolerance)
    print_means(means)
    plot_figure(values, means, args.output_pdf, args.output_png, args.dpi)
    print(f"Wrote {args.output_pdf}")
    print(f"Wrote {args.output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
