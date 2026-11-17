from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Mapping

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_cross_family import combination_error_indices, cross_family_combinations
from analyze_nine_model_oracle import compute_oracle_arrays, load_patch_coordinates
from common import atomic_json, load_config, load_error_bundle, resolve_project_path, write_csv


CATEGORY_NAMES = np.asarray(["low", "medium", "high"])
DEFAULT_EDGE_THRESHOLD = 0.05


def integral_image(values: np.ndarray) -> np.ndarray:
    return np.pad(values.astype(np.float64), ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)


def rectangle_sums(
    integral: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    width: np.ndarray,
    height: np.ndarray,
) -> np.ndarray:
    x1 = x + width
    y1 = y + height
    return integral[y1, x1] - integral[y, x1] - integral[y1, x] + integral[y, x]


def image_patch_descriptors(
    rgb: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    width: np.ndarray,
    height: np.ndarray,
    edge_threshold: float,
) -> dict[str, np.ndarray]:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image, found shape {rgb.shape}")
    if np.any(x < 0) or np.any(y < 0) or np.any(width <= 0) or np.any(height <= 0):
        raise ValueError("Patch coordinates contain negative positions or non-positive sizes")
    if np.any(x + width > rgb.shape[1]) or np.any(y + height > rgb.shape[0]):
        raise ValueError("Patch coordinates exceed ground-truth image dimensions")
    gray = (
        0.2126 * rgb[..., 0].astype(np.float64)
        + 0.7152 * rgb[..., 1].astype(np.float64)
        + 0.0722 * rgb[..., 2].astype(np.float64)
    )
    gradient_y, gradient_x = np.gradient(gray)
    gradient = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
    area = (width * height).astype(np.float64)
    gradient_sum = rectangle_sums(integral_image(gradient), x, y, width, height)
    edge_sum = rectangle_sums(integral_image(gradient > edge_threshold), x, y, width, height)
    gray_sum = rectangle_sums(integral_image(gray), x, y, width, height)
    gray_square_sum = rectangle_sums(integral_image(gray * gray), x, y, width, height)
    mean = gray_sum / area
    variance = np.maximum(gray_square_sum / area - mean * mean, 0.0)
    return {
        "mean_gradient_magnitude": gradient_sum / area,
        "edge_density": edge_sum / area,
        "local_texture_variance": variance,
    }


def extract_descriptors(
    config: Mapping[str, object],
    views: np.ndarray,
    coordinates: Mapping[str, np.ndarray],
    edge_threshold: float,
) -> dict[str, np.ndarray]:
    from analysis.local_error.image_io import load_rgb

    gt_root = resolve_project_path(config, str(config["reference_gt"]))
    if not gt_root.is_dir():
        raise FileNotFoundError(
            f"Canonical ground-truth directory is unavailable: {gt_root}. "
            "Structural descriptors must not be computed from reconstructed images."
        )
    descriptors = {
        "mean_gradient_magnitude": np.zeros(len(views), dtype=np.float64),
        "edge_density": np.zeros(len(views), dtype=np.float64),
        "local_texture_variance": np.zeros(len(views), dtype=np.float64),
    }
    for view_index, view in enumerate(sorted(set(views))):
        mask = views == view
        gt_path = gt_root / view
        if not gt_path.is_file():
            raise FileNotFoundError(
                f"Missing canonical GT for held-out view {view}: {gt_path}. "
                "No reconstructed-image substitute will be used."
            )
        values = image_patch_descriptors(
            load_rgb(gt_path),
            coordinates["x"][mask], coordinates["y"][mask],
            coordinates["width"][mask], coordinates["height"][mask],
            edge_threshold,
        )
        for name, descriptor_values in values.items():
            descriptors[name][mask] = descriptor_values
        if (view_index + 1) % 10 == 0 or view_index + 1 == len(set(views)):
            print(f"[structural] descriptors {view_index + 1}/{len(set(views))} views")
    if not all(np.isfinite(values).all() for values in descriptors.values()):
        raise ValueError("Non-finite structural descriptor detected")
    return descriptors


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks / max(len(values) - 1, 1)


def complexity_categories(
    descriptors: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    descriptor_ranks = {name: percentile_ranks(values) for name, values in descriptors.items()}
    score = np.mean(np.column_stack(list(descriptor_ranks.values())), axis=1)
    thresholds = np.quantile(score, [1.0 / 3.0, 2.0 / 3.0])
    category_index = np.digitize(score, thresholds, right=True).astype(np.int64)
    if set(category_index.tolist()) != {0, 1, 2}:
        raise ValueError("Complexity tertiles did not produce all three categories")
    return score, category_index, descriptor_ranks


def matched_size_oracles(
    errors: np.ndarray,
    run_ids: list[str],
    seeds: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int, int]]]:
    combinations = cross_family_combinations(seeds)
    if len(combinations) != 27:
        raise ValueError(f"Expected 27 matched-size cross-family combinations, found {len(combinations)}")
    combination_indices = combination_error_indices(run_ids, combinations)
    cross_oracles = np.empty((len(errors), len(combinations)), dtype=np.float64)
    for index, model_indices in enumerate(combination_indices):
        cross_oracles[:, index] = errors[:, model_indices].min(axis=1)
    ges_indices = [run_ids.index(f"ges_seed_{seed}") for seed in seeds]
    drk_indices = [run_ids.index(f"drk_seed_{seed}") for seed in seeds]
    return cross_oracles, errors[:, ges_indices].min(axis=1), errors[:, drk_indices].min(axis=1), combinations


def precompute_category_view_statistics(
    views: np.ndarray,
    category_index: np.ndarray,
    oracle_arrays: Mapping[str, np.ndarray],
    cross_oracles: np.ndarray,
    ges_three_oracle: np.ndarray,
    drk_three_oracle: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    unique_views = np.asarray(sorted(set(views.astype(str))), dtype=str)
    shape = (3, len(unique_views))
    statistics = {
        "patch_count": np.zeros(shape, dtype=np.int64),
        "baseline_sum": np.zeros(shape, dtype=np.float64),
        "full_sum": np.zeros(shape, dtype=np.float64),
        "ges_augmented_sum": np.zeros(shape, dtype=np.float64),
        "drk_augmented_sum": np.zeros(shape, dtype=np.float64),
        "ges_three_sum": np.zeros(shape, dtype=np.float64),
        "drk_three_sum": np.zeros(shape, dtype=np.float64),
        "improved_count": np.zeros(shape, dtype=np.int64),
        "matched_cross_sum": np.zeros((*shape, cross_oracles.shape[1]), dtype=np.float64),
    }
    for category in range(3):
        for view_index, view in enumerate(unique_views):
            mask = (category_index == category) & (views == view)
            count = int(mask.sum())
            statistics["patch_count"][category, view_index] = count
            if count == 0:
                continue
            statistics["baseline_sum"][category, view_index] = oracle_arrays["baseline_mse"][mask].sum(dtype=np.float64)
            statistics["full_sum"][category, view_index] = oracle_arrays["full_mse"][mask].sum(dtype=np.float64)
            statistics["ges_augmented_sum"][category, view_index] = oracle_arrays["ges_augmented_mse"][mask].sum(dtype=np.float64)
            statistics["drk_augmented_sum"][category, view_index] = oracle_arrays["drk_augmented_mse"][mask].sum(dtype=np.float64)
            statistics["ges_three_sum"][category, view_index] = ges_three_oracle[mask].sum(dtype=np.float64)
            statistics["drk_three_sum"][category, view_index] = drk_three_oracle[mask].sum(dtype=np.float64)
            statistics["improved_count"][category, view_index] = np.count_nonzero(oracle_arrays["improved"][mask])
            statistics["matched_cross_sum"][category, view_index] = cross_oracles[mask].sum(axis=0, dtype=np.float64)
    if np.any(statistics["patch_count"].sum(axis=1) == 0):
        raise ValueError("At least one structural category has zero patches")
    return unique_views, statistics


def category_metrics(
    statistics: Mapping[str, np.ndarray],
    category: int,
    view_indices: np.ndarray | None = None,
) -> dict[str, object]:
    selected = np.arange(statistics["patch_count"].shape[1]) if view_indices is None else view_indices
    patch_count = int(statistics["patch_count"][category, selected].sum(dtype=np.int64))
    if patch_count <= 0:
        raise ValueError(f"No patches for category {CATEGORY_NAMES[category]}")
    baseline_sum = float(statistics["baseline_sum"][category, selected].sum(dtype=np.float64))
    full_sum = float(statistics["full_sum"][category, selected].sum(dtype=np.float64))
    ges_augmented_sum = float(statistics["ges_augmented_sum"][category, selected].sum(dtype=np.float64))
    drk_augmented_sum = float(statistics["drk_augmented_sum"][category, selected].sum(dtype=np.float64))
    ges_three_sum = float(statistics["ges_three_sum"][category, selected].sum(dtype=np.float64))
    drk_three_sum = float(statistics["drk_three_sum"][category, selected].sum(dtype=np.float64))
    cross_sums = statistics["matched_cross_sum"][category, selected].sum(axis=0, dtype=np.float64)
    cross_gains = (baseline_sum - cross_sums) / baseline_sum
    return {
        "patch_count": patch_count,
        "baseline_3dgs_three_seed_oracle_mse": baseline_sum / patch_count,
        "nine_model_oracle_mse": full_sum / patch_count,
        "nine_model_incremental_gain_fraction": (baseline_sum - full_sum) / baseline_sum,
        "nine_model_incremental_mse_reduction": (baseline_sum - full_sum) / patch_count,
        "improved_patch_count": int(statistics["improved_count"][category, selected].sum(dtype=np.int64)),
        "improved_patch_fraction": float(statistics["improved_count"][category, selected].sum(dtype=np.int64) / patch_count),
        "ges_augmented_incremental_gain_fraction": (baseline_sum - ges_augmented_sum) / baseline_sum,
        "drk_augmented_incremental_gain_fraction": (baseline_sum - drk_augmented_sum) / baseline_sum,
        "ges_three_seed_oracle_gain_vs_3dgs_fraction": (baseline_sum - ges_three_sum) / baseline_sum,
        "drk_three_seed_oracle_gain_vs_3dgs_fraction": (baseline_sum - drk_three_sum) / baseline_sum,
        "matched_cross_family_gain_mean": float(cross_gains.mean()),
        "matched_cross_family_gain_std": float(cross_gains.std(ddof=1)),
        "matched_cross_family_gain_median": float(np.median(cross_gains)),
        "matched_cross_family_gain_min": float(cross_gains.min()),
        "matched_cross_family_gain_max": float(cross_gains.max()),
        "matched_cross_family_outperform_count": int(np.count_nonzero(cross_gains > 0.0)),
        "matched_cross_family_outperform_fraction": float(np.mean(cross_gains > 0.0)),
    }


def bootstrap_categories(
    statistics: Mapping[str, np.ndarray],
    replicates: int,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    view_count = statistics["patch_count"].shape[1]
    rng = np.random.default_rng(seed)
    sampled = rng.choice(view_count, size=(replicates, view_count), replace=True)
    output: dict[str, dict[str, np.ndarray]] = {}
    for category in range(3):
        patch_count = statistics["patch_count"][category, sampled].sum(axis=1, dtype=np.int64)
        valid = patch_count > 0
        if not np.any(valid):
            raise ValueError(f"Every bootstrap replicate contains zero {CATEGORY_NAMES[category]} patches")
        baseline_sum = statistics["baseline_sum"][category, sampled].sum(axis=1, dtype=np.float64)
        full_sum = statistics["full_sum"][category, sampled].sum(axis=1, dtype=np.float64)
        ges_sum = statistics["ges_augmented_sum"][category, sampled].sum(axis=1, dtype=np.float64)
        drk_sum = statistics["drk_augmented_sum"][category, sampled].sum(axis=1, dtype=np.float64)
        improved_count = statistics["improved_count"][category, sampled].sum(axis=1, dtype=np.int64)
        cross_sums = statistics["matched_cross_sum"][category, sampled].sum(axis=1, dtype=np.float64)
        cross_gains = np.full_like(cross_sums, np.nan, dtype=np.float64)
        cross_gains[valid] = (baseline_sum[valid, None] - cross_sums[valid]) / baseline_sum[valid, None]
        def valid_metric(values: np.ndarray) -> np.ndarray:
            result = np.full(replicates, np.nan, dtype=np.float64)
            result[valid] = values
            return result
        output[str(CATEGORY_NAMES[category])] = {
            "nine_model_incremental_gain_fraction": valid_metric((baseline_sum[valid] - full_sum[valid]) / baseline_sum[valid]),
            "improved_patch_fraction": valid_metric(improved_count[valid] / patch_count[valid]),
            "ges_augmented_incremental_gain_fraction": valid_metric((baseline_sum[valid] - ges_sum[valid]) / baseline_sum[valid]),
            "drk_augmented_incremental_gain_fraction": valid_metric((baseline_sum[valid] - drk_sum[valid]) / baseline_sum[valid]),
            "matched_cross_family_gain_mean": valid_metric(np.nanmean(cross_gains[valid], axis=1)),
        }
    return output


def percentile_intervals(bootstrap: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, dict[str, dict[str, float]]]:
    output = {}
    for category, metrics in bootstrap.items():
        output[category] = {}
        for metric, values in metrics.items():
            finite = values[np.isfinite(values)]
            if len(finite) == 0:
                raise ValueError(f"No finite bootstrap estimates for {category}/{metric}")
            output[category][metric] = {
                "low": float(np.quantile(finite, 0.025)),
                "high": float(np.quantile(finite, 0.975)),
                "effective_replicates": int(len(finite)),
            }
    return output


def build_summary_rows(
    descriptors: Mapping[str, np.ndarray],
    score: np.ndarray,
    category_index: np.ndarray,
    statistics: Mapping[str, np.ndarray],
    intervals: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> list[dict[str, object]]:
    rows = []
    for category in range(3):
        mask = category_index == category
        name = str(CATEGORY_NAMES[category])
        metrics = category_metrics(statistics, category)
        positive_views = 0
        eligible_views = 0
        for view_index in range(statistics["patch_count"].shape[1]):
            if statistics["patch_count"][category, view_index] == 0:
                continue
            eligible_views += 1
            view_metric = category_metrics(statistics, category, np.asarray([view_index]))
            positive_views += int(float(view_metric["nine_model_incremental_gain_fraction"]) > 0.0)
        row: dict[str, object] = {
            "complexity_category": name,
            "complexity_score_min": float(score[mask].min()),
            "complexity_score_median": float(np.median(score[mask])),
            "complexity_score_max": float(score[mask].max()),
            "view_count_with_category": eligible_views,
            "views_with_positive_nine_model_gain": positive_views,
            "positive_view_fraction": positive_views / eligible_views,
            **metrics,
        }
        for descriptor_name, values in descriptors.items():
            row[f"{descriptor_name}_median"] = float(np.median(values[mask]))
            row[f"{descriptor_name}_q25"] = float(np.quantile(values[mask], 0.25))
            row[f"{descriptor_name}_q75"] = float(np.quantile(values[mask], 0.75))
        for metric, interval in intervals[name].items():
            row[f"{metric}_ci_low"] = interval["low"]
            row[f"{metric}_ci_high"] = interval["high"]
        rows.append(row)
    return rows


def build_per_view_rows(
    unique_views: np.ndarray,
    statistics: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    rows = []
    for category in range(3):
        for view_index, view in enumerate(unique_views):
            if statistics["patch_count"][category, view_index] == 0:
                continue
            rows.append({
                "view": str(view),
                "complexity_category": str(CATEGORY_NAMES[category]),
                **category_metrics(statistics, category, np.asarray([view_index], dtype=np.int64)),
            })
    return rows


def representative_regions(
    views: np.ndarray,
    coordinates: Mapping[str, np.ndarray],
    category_index: np.ndarray,
    score: np.ndarray,
    oracle_arrays: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    rows = []
    for category in range(3):
        candidates = np.flatnonzero((category_index == category) & oracle_arrays["improved"])
        if len(candidates) == 0:
            candidates = np.flatnonzero(category_index == category)
        best = int(candidates[np.argmax(oracle_arrays["improvement_mse"][candidates])])
        rows.append({
            "complexity_category": str(CATEGORY_NAMES[category]),
            "selection_rule": "maximum nine-model incremental patch-MSE reduction within category",
            "view": str(views[best]),
            "x": int(coordinates["x"][best]),
            "y": int(coordinates["y"][best]),
            "width": int(coordinates["width"][best]),
            "height": int(coordinates["height"][best]),
            "complexity_score": float(score[best]),
            "winning_family": str(oracle_arrays["full_family"][best]),
            "winning_seed": int(oracle_arrays["full_seed"][best]),
            "baseline_3dgs_oracle_mse": float(oracle_arrays["baseline_mse"][best]),
            "nine_model_oracle_mse": float(oracle_arrays["full_mse"][best]),
            "improvement_mse": float(oracle_arrays["improvement_mse"][best]),
        })
    return rows


def make_figure(
    path_root: Path,
    config: Mapping[str, object],
    views: np.ndarray,
    coordinates: Mapping[str, np.ndarray],
    category_index: np.ndarray,
    oracle_arrays: Mapping[str, np.ndarray],
    representatives: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analysis.local_error.image_io import load_rgb
    from analysis.local_error.patches import Patch
    from analysis.winner_maps.visualize import rasterize_patch_values

    gt_root = resolve_project_path(config, str(config["reference_gt"]))
    figure = plt.figure(figsize=(7.1, 6.6))
    grid = figure.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.15], hspace=0.22, wspace=0.08)
    positive_improvements = oracle_arrays["improvement_mse"][oracle_arrays["improved"]]
    vmax = max(float(np.quantile(positive_improvements, 0.99)), 1e-12) if len(positive_improvements) else 1e-12
    image_artist = None
    context_size = 192
    for column, representative in enumerate(representatives):
        view = str(representative["view"])
        image = load_rgb(gt_root / view)
        mask = views == view
        patches = [
            Patch(int(x), int(y), int(width), int(height))
            for x, y, width, height in zip(
                coordinates["x"][mask], coordinates["y"][mask],
                coordinates["width"][mask], coordinates["height"][mask],
            )
        ]
        heatmap = rasterize_patch_values(image.shape[:2], patches, oracle_arrays["improvement_mse"][mask])
        center_x = int(representative["x"]) + int(representative["width"]) // 2
        center_y = int(representative["y"]) + int(representative["height"]) // 2
        x0 = max(0, min(image.shape[1] - context_size, center_x - context_size // 2))
        y0 = max(0, min(image.shape[0] - context_size, center_y - context_size // 2))
        x1 = min(image.shape[1], x0 + context_size)
        y1 = min(image.shape[0], y0 + context_size)
        original_axis = figure.add_subplot(grid[0, column])
        original_axis.imshow(image[y0:y1, x0:x1])
        original_axis.set_title(str(representative["complexity_category"]).title(), fontweight="bold")
        original_axis.axis("off")
        heatmap_axis = figure.add_subplot(grid[1, column])
        image_artist = heatmap_axis.imshow(heatmap[y0:y1, x0:x1], cmap="magma", vmin=0.0, vmax=vmax)
        heatmap_axis.axis("off")
    figure.text(0.015, 0.79, "GT region", rotation=90, va="center", fontsize=9)
    figure.text(0.015, 0.51, "Incremental MSE reduction", rotation=90, va="center", fontsize=9)
    colorbar_axis = figure.add_axes([0.92, 0.39, 0.012, 0.24])
    figure.colorbar(image_artist, cax=colorbar_axis, label="Patch MSE reduction")

    comparison_axis = figure.add_subplot(grid[2, :])
    x_positions = np.arange(3)
    width = 0.22
    series = [
        ("Nine-model", "nine_model_incremental_gain_fraction", "#6f4c9b"),
        ("+ GES seeds", "ges_augmented_incremental_gain_fraction", "#2a9d8f"),
        ("+ DRK seeds", "drk_augmented_incremental_gain_fraction", "#e76f51"),
    ]
    for series_index, (label, key, color) in enumerate(series):
        values = np.asarray([100.0 * float(row[key]) for row in summary_rows])
        low = np.asarray([100.0 * float(row[f"{key}_ci_low"]) for row in summary_rows])
        high = np.asarray([100.0 * float(row[f"{key}_ci_high"]) for row in summary_rows])
        lower_error = np.maximum(values - low, 0.0)
        upper_error = np.maximum(high - values, 0.0)
        comparison_axis.bar(
            x_positions + (series_index - 1) * width, values, width,
            yerr=np.vstack((lower_error, upper_error)), capsize=3, label=label, color=color,
        )
    matched = np.asarray([100.0 * float(row["matched_cross_family_gain_mean"]) for row in summary_rows])
    comparison_axis.plot(x_positions, matched, color="#333333", marker="D", linestyle="--", label="Matched-size cross-family mean")
    comparison_axis.axhline(0.0, color="#555555", linewidth=0.8)
    comparison_axis.set_xticks(x_positions, ["Low", "Medium", "High"])
    comparison_axis.set_ylabel("Gain vs. three-seed 3DGS oracle (%)")
    comparison_axis.set_xlabel("Ground-truth structural complexity")
    comparison_axis.grid(axis="y", alpha=0.18, linewidth=0.6)
    comparison_axis.legend(frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    figure.subplots_adjust(left=0.06, right=0.90, top=0.96, bottom=0.16)
    path_root.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path_root.with_suffix(".png"), dpi=300)
    figure.savefig(path_root.with_suffix(".pdf"))
    plt.close(figure)


def write_latex(path: Path, rows: list[dict[str, object]]) -> None:
    body = []
    for row in rows:
        body.append(
            f"{str(row['complexity_category']).title()} & {int(row['patch_count']):,} & "
            f"{100 * float(row['nine_model_incremental_gain_fraction']):.3f} "
            f"[{100 * float(row['nine_model_incremental_gain_fraction_ci_low']):.3f}, {100 * float(row['nine_model_incremental_gain_fraction_ci_high']):.3f}] & "
            f"{100 * float(row['improved_patch_fraction']):.2f} & "
            f"{100 * float(row['ges_augmented_incremental_gain_fraction']):.3f} & "
            f"{100 * float(row['drk_augmented_incremental_gain_fraction']):.3f} & "
            f"{100 * float(row['matched_cross_family_gain_mean']):.3f} \\\\"
        )
    latex = "\n".join([
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Room multi-seed structural analysis. Complexity is the mean scene-wide percentile rank of ground-truth gradient magnitude, edge density, and texture variance, divided at empirical tertiles. Nine-model gains use the best of three 3DGS seeds as the baseline; the matched-size column averages the 27 dependent one-seed-per-family controls. View-cluster bootstrap 95\% confidence intervals are shown in brackets. The nine-model pool is larger and cannot worsen by construction, so these results are descriptive rather than causal evidence of kernel specialization.}",
        r"\label{tab:rebuttal_structural_specialization}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Complexity & Patches & Nine-model gain (\%) & Improved (\%) & +GES (\%) & +DRK (\%) & Matched-size mean (\%) \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])
    path.write_text(latex, encoding="utf-8")


def run_analysis(config_path: Path, skip_figure: bool = False) -> dict[str, object]:
    started = time.perf_counter()
    config = load_config(config_path)
    result_root = resolve_project_path(config, str(config["result_root"]))
    figure_root = resolve_project_path(config, str(config["figure_root"]))
    bundle_path = result_root / "patch_errors.npz"
    errors, run_ids, views, metadata = load_error_bundle(bundle_path)
    coordinates = load_patch_coordinates(bundle_path, len(errors))
    seeds = [int(seed) for seed in config["seeds"]]
    edge_threshold = float(config.get("structural_edge_threshold", DEFAULT_EDGE_THRESHOLD))
    print(f"[structural] loaded {len(errors):,} patches across {len(set(views))} views")
    descriptors = extract_descriptors(config, views, coordinates, edge_threshold)
    score, category_index, descriptor_ranks = complexity_categories(descriptors)
    category_thresholds = np.quantile(score, [1.0 / 3.0, 2.0 / 3.0])
    oracle_arrays = compute_oracle_arrays(errors, run_ids, seeds)
    cross_oracles, ges_three_oracle, drk_three_oracle, combinations = matched_size_oracles(errors, run_ids, seeds)
    unique_views, statistics = precompute_category_view_statistics(
        views, category_index, oracle_arrays, cross_oracles, ges_three_oracle, drk_three_oracle
    )
    bootstrap = bootstrap_categories(
        statistics,
        int(config.get("bootstrap_replicates", 2000)),
        int(config.get("bootstrap_seed", 2027)),
    )
    intervals = percentile_intervals(bootstrap)
    summary_rows = build_summary_rows(descriptors, score, category_index, statistics, intervals)
    per_view = build_per_view_rows(unique_views, statistics)
    representatives = representative_regions(views, coordinates, category_index, score, oracle_arrays)

    write_csv(result_root / "structural_specialization_summary.csv", summary_rows)
    write_csv(result_root / "structural_specialization_per_view.csv", per_view)
    write_csv(result_root / "structural_specialization_representatives.csv", representatives)
    write_latex(result_root / "structural_specialization_table.tex", summary_rows)
    figure_paths = []
    if not skip_figure:
        figure_base = figure_root / "rebuttal_structural_specialization"
        make_figure(
            figure_base, config, views, coordinates, category_index,
            oracle_arrays, representatives, summary_rows,
        )
        figure_paths = [str(figure_base.with_suffix(".png")), str(figure_base.with_suffix(".pdf"))]
    document = {
        "config": {
            "scene": config["scene"],
            "patch_count": len(errors),
            "view_count": len(unique_views),
            "patch_size": metadata.get("patch_size", config.get("patch_size")),
            "stride": metadata.get("stride", config.get("stride")),
            "edge_density_gradient_threshold": edge_threshold,
            "bootstrap_replicates": int(config.get("bootstrap_replicates", 2000)),
            "bootstrap_seed": int(config.get("bootstrap_seed", 2027)),
            "matched_cross_family_combination_count": len(combinations),
        },
        "descriptor_definitions": {
            "input": "Canonical ground-truth RGB converted to float [0,1]; reconstructed images are never used.",
            "grayscale": "0.2126 R + 0.7152 G + 0.0722 B.",
            "mean_gradient_magnitude": "Mean sqrt(gx^2 + gy^2) within each patch, using numpy.gradient on the complete GT image before patch aggregation.",
            "edge_density": f"Fraction of patch pixels with GT gradient magnitude > {edge_threshold}.",
            "local_texture_variance": "Patch grayscale variance from first and second moments.",
            "complexity_score": "Mean of the scene-wide percentile ranks of the three descriptors.",
            "categories": "Low/medium/high are empirical complexity-score tertiles; thresholds are recorded below.",
        },
        "complexity_score_thresholds": {
            "low_to_medium": float(category_thresholds[0]),
            "medium_to_high": float(category_thresholds[1]),
        },
        "summary": summary_rows,
        "view_cluster_bootstrap_95_ci": intervals,
        "representative_regions": representatives,
        "figures": figure_paths,
        "notes": {
            "model_count": "The nine-model oracle has a larger candidate pool than the three-seed 3DGS oracle and cannot be worse by construction.",
            "matched_size_control": "Matched-size results summarize all 27 dependent groups containing one 3DGS, one GES, and one DRK seed; they are not 27 independent experiments.",
            "interpretation": "Descriptor-conditioned gains are descriptive local complementarity, not causal proof of kernel specialization.",
            "statistical_unit": "Confidence intervals resample complete held-out views with replacement and preserve patch-count weighting.",
            "representative_selection": "Each displayed region is the maximum incremental-MSE-reduction patch within its structural category and is explicitly labeled as such in the representatives CSV.",
        },
    }
    atomic_json(result_root / "structural_specialization_summary.json", document)
    print(f"[structural] wrote summaries to {result_root}; elapsed {time.perf_counter() - started:.2f}s")
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze structural conditioning of incremental nine-model oracle coverage.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-figure", action="store_true", help="Skip the publication figure; numerical outputs are unchanged.")
    args = parser.parse_args()
    run_analysis(args.config, skip_figure=args.skip_figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
