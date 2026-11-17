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

from common import atomic_json, load_config, load_error_bundle, resolve_project_path, write_csv


METHODS = ("3dgs", "ges", "drk")


def model_indices_and_labels(
    run_ids: list[str], seeds: list[int]
) -> tuple[dict[str, np.ndarray], list[tuple[str, int, str]]]:
    indices: dict[str, np.ndarray] = {}
    ordered_models: list[tuple[str, int, str]] = []
    for method in METHODS:
        method_indices = []
        for seed in seeds:
            run_id = f"{method}_seed_{seed}"
            if run_id not in run_ids:
                raise ValueError(f"Missing required run in patch-error bundle: {run_id}")
            method_indices.append(run_ids.index(run_id))
            ordered_models.append((method, seed, run_id))
        indices[method] = np.asarray(method_indices, dtype=np.int64)
    if len({item[2] for item in ordered_models}) != len(ordered_models):
        raise AssertionError("Nine-model oracle contains duplicate run identifiers")
    return indices, ordered_models


def compute_oracle_arrays(
    errors: np.ndarray,
    run_ids: list[str],
    seeds: list[int],
) -> dict[str, np.ndarray]:
    values = np.asarray(errors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(run_ids):
        raise ValueError("errors must have shape [patches, runs]")
    if not np.isfinite(values).all():
        raise ValueError("Patch-error bundle contains NaN or infinite values")

    indices, ordered_models = model_indices_and_labels(run_ids, seeds)
    ordered_indices = np.concatenate([indices[method] for method in METHODS])
    nine_errors = values[:, ordered_indices]
    three_errors = values[:, indices["3dgs"]]
    ges_errors = values[:, indices["ges"]]
    drk_errors = values[:, indices["drk"]]

    baseline_local_index = np.argmin(three_errors, axis=1)
    baseline_mse = three_errors[np.arange(len(values)), baseline_local_index]
    baseline_seed = np.asarray(seeds, dtype=np.int64)[baseline_local_index]

    full_local_index = np.argmin(nine_errors, axis=1)
    full_mse = nine_errors[np.arange(len(values)), full_local_index]
    full_family = np.asarray([item[0] for item in ordered_models], dtype=str)[full_local_index]
    full_seed = np.asarray([item[1] for item in ordered_models], dtype=np.int64)[full_local_index]
    full_run_id = np.asarray([item[2] for item in ordered_models], dtype=str)[full_local_index]
    full_tie_count = np.sum(nine_errors == full_mse[:, None], axis=1, dtype=np.int64)

    ges_best = ges_errors.min(axis=1)
    drk_best = drk_errors.min(axis=1)
    ges_augmented_mse = np.minimum(baseline_mse, ges_best)
    drk_augmented_mse = np.minimum(baseline_mse, drk_best)
    improved = full_mse < baseline_mse
    ges_improved = ges_best < baseline_mse
    drk_improved = drk_best < baseline_mse

    if np.any(full_mse > baseline_mse):
        raise AssertionError("Nine-model oracle exceeds the three-seed 3DGS oracle")
    if np.any(improved & (full_family == "3dgs")):
        raise AssertionError("A strict improvement over the 3DGS oracle cannot be assigned to 3DGS")

    return {
        "baseline_mse": baseline_mse,
        "baseline_seed": baseline_seed,
        "full_mse": full_mse,
        "full_family": full_family,
        "full_seed": full_seed,
        "full_run_id": full_run_id,
        "full_tie_count": full_tie_count,
        "improvement_mse": baseline_mse - full_mse,
        "improved": improved,
        "ges_augmented_mse": ges_augmented_mse,
        "drk_augmented_mse": drk_augmented_mse,
        "ges_improved": ges_improved,
        "drk_improved": drk_improved,
    }


def precompute_view_statistics(
    views: np.ndarray,
    arrays: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    unique_views = np.asarray(sorted(set(views.astype(str))), dtype=str)
    statistics = {
        "patch_count": np.zeros(len(unique_views), dtype=np.int64),
        "baseline_sum": np.zeros(len(unique_views), dtype=np.float64),
        "full_sum": np.zeros(len(unique_views), dtype=np.float64),
        "ges_augmented_sum": np.zeros(len(unique_views), dtype=np.float64),
        "drk_augmented_sum": np.zeros(len(unique_views), dtype=np.float64),
        "improved_count": np.zeros(len(unique_views), dtype=np.int64),
        "ges_improved_count": np.zeros(len(unique_views), dtype=np.int64),
        "drk_improved_count": np.zeros(len(unique_views), dtype=np.int64),
        "ges_winner_count": np.zeros(len(unique_views), dtype=np.int64),
        "drk_winner_count": np.zeros(len(unique_views), dtype=np.int64),
    }
    for index, view in enumerate(unique_views):
        mask = views == view
        count = int(mask.sum())
        if count <= 0:
            raise ValueError(f"View has no aligned patches: {view}")
        statistics["patch_count"][index] = count
        statistics["baseline_sum"][index] = arrays["baseline_mse"][mask].sum(dtype=np.float64)
        statistics["full_sum"][index] = arrays["full_mse"][mask].sum(dtype=np.float64)
        statistics["ges_augmented_sum"][index] = arrays["ges_augmented_mse"][mask].sum(dtype=np.float64)
        statistics["drk_augmented_sum"][index] = arrays["drk_augmented_mse"][mask].sum(dtype=np.float64)
        statistics["improved_count"][index] = np.count_nonzero(arrays["improved"][mask])
        statistics["ges_improved_count"][index] = np.count_nonzero(arrays["ges_improved"][mask])
        statistics["drk_improved_count"][index] = np.count_nonzero(arrays["drk_improved"][mask])
        statistics["ges_winner_count"][index] = np.count_nonzero(arrays["improved"][mask] & (arrays["full_family"][mask] == "ges"))
        statistics["drk_winner_count"][index] = np.count_nonzero(arrays["improved"][mask] & (arrays["full_family"][mask] == "drk"))
    return unique_views, statistics


def aggregate_statistics(statistics: Mapping[str, np.ndarray], indices: np.ndarray | None = None) -> dict[str, float | int]:
    selected = np.arange(len(statistics["patch_count"])) if indices is None else indices
    patch_count = int(statistics["patch_count"][selected].sum(dtype=np.int64))
    if patch_count <= 0:
        raise ValueError("Cannot aggregate zero patches")
    baseline_sum = float(statistics["baseline_sum"][selected].sum(dtype=np.float64))
    full_sum = float(statistics["full_sum"][selected].sum(dtype=np.float64))
    ges_sum = float(statistics["ges_augmented_sum"][selected].sum(dtype=np.float64))
    drk_sum = float(statistics["drk_augmented_sum"][selected].sum(dtype=np.float64))
    if baseline_sum <= 0.0:
        raise ValueError("Three-seed 3DGS oracle MSE must be positive")
    return {
        "patch_count": patch_count,
        "baseline_3dgs_oracle_mse": baseline_sum / patch_count,
        "nine_model_oracle_mse": full_sum / patch_count,
        "incremental_mse_reduction": (baseline_sum - full_sum) / patch_count,
        "incremental_gain_fraction": (baseline_sum - full_sum) / baseline_sum,
        "improved_patch_count": int(statistics["improved_count"][selected].sum(dtype=np.int64)),
        "improved_patch_fraction": float(statistics["improved_count"][selected].sum(dtype=np.int64) / patch_count),
        "ges_augmented_oracle_mse": ges_sum / patch_count,
        "ges_incremental_mse_reduction": (baseline_sum - ges_sum) / patch_count,
        "ges_incremental_gain_fraction": (baseline_sum - ges_sum) / baseline_sum,
        "ges_improved_patch_count": int(statistics["ges_improved_count"][selected].sum(dtype=np.int64)),
        "ges_improved_patch_fraction": float(statistics["ges_improved_count"][selected].sum(dtype=np.int64) / patch_count),
        "drk_augmented_oracle_mse": drk_sum / patch_count,
        "drk_incremental_mse_reduction": (baseline_sum - drk_sum) / patch_count,
        "drk_incremental_gain_fraction": (baseline_sum - drk_sum) / baseline_sum,
        "drk_improved_patch_count": int(statistics["drk_improved_count"][selected].sum(dtype=np.int64)),
        "drk_improved_patch_fraction": float(statistics["drk_improved_count"][selected].sum(dtype=np.int64) / patch_count),
        "ges_full_oracle_winner_count": int(statistics["ges_winner_count"][selected].sum(dtype=np.int64)),
        "drk_full_oracle_winner_count": int(statistics["drk_winner_count"][selected].sum(dtype=np.int64)),
    }


def bootstrap_view_clusters(
    statistics: Mapping[str, np.ndarray],
    replicates: int,
    seed: int,
) -> dict[str, np.ndarray]:
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    view_count = len(statistics["patch_count"])
    if view_count < 2:
        raise ValueError("At least two views are required for view-cluster bootstrap")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(view_count, size=(replicates, view_count), replace=True)
    patch_count = statistics["patch_count"][sampled].sum(axis=1, dtype=np.int64)
    baseline_sum = statistics["baseline_sum"][sampled].sum(axis=1, dtype=np.float64)
    full_sum = statistics["full_sum"][sampled].sum(axis=1, dtype=np.float64)
    ges_sum = statistics["ges_augmented_sum"][sampled].sum(axis=1, dtype=np.float64)
    drk_sum = statistics["drk_augmented_sum"][sampled].sum(axis=1, dtype=np.float64)
    improved_count = statistics["improved_count"][sampled].sum(axis=1, dtype=np.int64)
    ges_improved_count = statistics["ges_improved_count"][sampled].sum(axis=1, dtype=np.int64)
    drk_improved_count = statistics["drk_improved_count"][sampled].sum(axis=1, dtype=np.int64)
    return {
        "incremental_gain_fraction": (baseline_sum - full_sum) / baseline_sum,
        "incremental_mse_reduction": (baseline_sum - full_sum) / patch_count,
        "improved_patch_fraction": improved_count / patch_count,
        "ges_incremental_gain_fraction": (baseline_sum - ges_sum) / baseline_sum,
        "ges_incremental_mse_reduction": (baseline_sum - ges_sum) / patch_count,
        "ges_improved_patch_fraction": ges_improved_count / patch_count,
        "drk_incremental_gain_fraction": (baseline_sum - drk_sum) / baseline_sum,
        "drk_incremental_mse_reduction": (baseline_sum - drk_sum) / patch_count,
        "drk_improved_patch_fraction": drk_improved_count / patch_count,
    }


def confidence_intervals(bootstrap: Mapping[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "low": float(np.quantile(values, 0.025)),
            "high": float(np.quantile(values, 0.975)),
        }
        for name, values in bootstrap.items()
    }


def per_view_rows(unique_views: np.ndarray, statistics: Mapping[str, np.ndarray]) -> list[dict[str, object]]:
    rows = []
    for index, view in enumerate(unique_views):
        row = {"view": str(view), **aggregate_statistics(statistics, np.asarray([index], dtype=np.int64))}
        rows.append(row)
    return rows


def load_patch_coordinates(bundle_path: Path, patch_count: int) -> dict[str, np.ndarray]:
    required = ("x", "y", "width", "height")
    with np.load(bundle_path, allow_pickle=False) as bundle:
        missing = [name for name in required if name not in bundle]
        if missing:
            raise ValueError(f"Patch-error bundle lacks required spatial metadata: {missing}")
        coordinates = {name: np.asarray(bundle[name], dtype=np.int64) for name in required}
    for name, values in coordinates.items():
        if len(values) != patch_count:
            raise ValueError(f"Coordinate array {name} has {len(values)} rows; expected {patch_count}")
    return coordinates


def region_rows(
    views: np.ndarray,
    coordinates: Mapping[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    improved_indices = np.flatnonzero(arrays["improved"])
    order = improved_indices[np.argsort(-arrays["improvement_mse"][improved_indices], kind="stable")]
    view_patch_indices = np.zeros(len(views), dtype=np.int64)
    counters: dict[str, int] = {}
    for index, view in enumerate(views):
        key = str(view)
        view_patch_indices[index] = counters.get(key, 0)
        counters[key] = int(view_patch_indices[index] + 1)
    rows = []
    for rank, index in enumerate(order, start=1):
        baseline = float(arrays["baseline_mse"][index])
        rows.append({
            "rank": rank,
            "view": str(views[index]),
            "patch_index": int(view_patch_indices[index]),
            "x": int(coordinates["x"][index]),
            "y": int(coordinates["y"][index]),
            "width": int(coordinates["width"][index]),
            "height": int(coordinates["height"][index]),
            "winning_family": str(arrays["full_family"][index]),
            "winning_seed": int(arrays["full_seed"][index]),
            "winning_run_id": str(arrays["full_run_id"][index]),
            "winning_model_tie_count": int(arrays["full_tie_count"][index]),
            "baseline_3dgs_seed": int(arrays["baseline_seed"][index]),
            "baseline_3dgs_mse": baseline,
            "nine_model_oracle_mse": float(arrays["full_mse"][index]),
            "improvement_mse": float(arrays["improvement_mse"][index]),
            "relative_improvement_fraction": float(arrays["improvement_mse"][index] / baseline),
        })
    return rows


def save_region_heatmaps(
    config: Mapping[str, object],
    result_root: Path,
    views: np.ndarray,
    coordinates: Mapping[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
    view_rows: list[dict[str, object]],
) -> list[str]:
    from analysis.local_error.image_io import load_rgb
    from analysis.local_error.patches import Patch
    from analysis.winner_maps.visualize import rasterize_patch_values, save_float_map

    gt_root = resolve_project_path(config, str(config["reference_gt"]))
    output_root = result_root / "nine_model_oracle_regions"
    ranked_views = sorted(view_rows, key=lambda row: float(row["incremental_mse_reduction"]), reverse=True)[:5]
    outputs = []
    for row in ranked_views:
        view = str(row["view"])
        mask = views == view
        gt_path = gt_root / view
        if not gt_path.is_file():
            raise FileNotFoundError(f"Cannot determine heatmap dimensions; missing GT image: {gt_path}")
        image = load_rgb(gt_path)
        patches = [
            Patch(int(x), int(y), int(width), int(height))
            for x, y, width, height in zip(
                coordinates["x"][mask], coordinates["y"][mask],
                coordinates["width"][mask], coordinates["height"][mask],
            )
        ]
        heatmap = rasterize_patch_values(image.shape[:2], patches, arrays["improvement_mse"][mask])
        output = output_root / f"{Path(view).stem}_incremental_mse.png"
        save_float_map(output, heatmap)
        outputs.append(str(output))
    return outputs


def write_latex_table(path: Path, summary: Mapping[str, object], intervals: Mapping[str, Mapping[str, float]]) -> None:
    rows = [
        ("Three-seed 3DGS", float(summary["baseline_3dgs_oracle_mse"]), None, None, None),
        ("3DGS + GES seeds", float(summary["ges_augmented_oracle_mse"]), float(summary["ges_incremental_gain_fraction"]), float(summary["ges_improved_patch_fraction"]), intervals["ges_incremental_gain_fraction"]),
        ("3DGS + DRK seeds", float(summary["drk_augmented_oracle_mse"]), float(summary["drk_incremental_gain_fraction"]), float(summary["drk_improved_patch_fraction"]), intervals["drk_incremental_gain_fraction"]),
        ("All nine models", float(summary["nine_model_oracle_mse"]), float(summary["incremental_gain_fraction"]), float(summary["improved_patch_fraction"]), intervals["incremental_gain_fraction"]),
    ]
    body = []
    for label, mse, gain, improved, interval in rows:
        if gain is None:
            body.append(f"{label} & {mse:.7f} & -- & -- & -- \\\\")
        else:
            body.append(
                f"{label} & {mse:.7f} & {100 * gain:.3f} & {100 * improved:.3f} & "
                f"[{100 * interval['low']:.3f}, {100 * interval['high']:.3f}] \\\\")
    latex = "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Incremental Room oracle coverage beyond the best of three independently trained 3DGS models. Adding GES or DRK increases the candidate pool, so the oracle cannot worsen by construction. Results describe local complementarity and do not by themselves establish kernel specialization. Confidence intervals resample complete held-out views with patch-count weighting.}",
        r"\label{tab:nine_model_incremental_oracle}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Candidate pool & Oracle MSE & Gain (\%) & Improved (\%) & Gain 95\% CI \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    path.write_text(latex, encoding="utf-8")


def run_analysis(config_path: Path, skip_visualizations: bool = False) -> dict[str, object]:
    started = time.perf_counter()
    config = load_config(config_path)
    result_root = resolve_project_path(config, str(config["result_root"]))
    bundle_path = result_root / "patch_errors.npz"
    errors, run_ids, views, metadata = load_error_bundle(bundle_path)
    seeds = [int(seed) for seed in config["seeds"]]
    print(f"[nine-model] loaded {len(errors):,} aligned patches across {len(set(views))} views")

    arrays = compute_oracle_arrays(errors, run_ids, seeds)
    unique_views, view_statistics = precompute_view_statistics(views, arrays)
    summary = aggregate_statistics(view_statistics)
    bootstrap_started = time.perf_counter()
    bootstrap = bootstrap_view_clusters(
        view_statistics,
        int(config.get("bootstrap_replicates", 2000)),
        int(config.get("bootstrap_seed", 2027)),
    )
    intervals = confidence_intervals(bootstrap)
    print(f"[nine-model] {len(bootstrap['incremental_gain_fraction']):,} view-cluster bootstrap replicates: {time.perf_counter() - bootstrap_started:.2f}s")

    views_output = per_view_rows(unique_views, view_statistics)
    coordinates = load_patch_coordinates(bundle_path, len(errors))
    regions_output = region_rows(views, coordinates, arrays)
    if not regions_output:
        raise ValueError("Nine-model oracle produced no strict GES/DRK patch improvements")

    result_root.mkdir(parents=True, exist_ok=True)
    write_csv(result_root / "nine_model_oracle_per_view.csv", views_output)
    write_csv(result_root / "nine_model_oracle_regions.csv", regions_output)
    heatmaps = [] if skip_visualizations else save_region_heatmaps(
        config, result_root, views, coordinates, arrays, views_output
    )
    summary_document = {
        "config": {
            "scene": config["scene"],
            "seeds": seeds,
            "model_count": 9,
            "view_count": len(unique_views),
            "patch_size": metadata.get("patch_size", config.get("patch_size")),
            "stride": metadata.get("stride", config.get("stride")),
            "bootstrap_replicates": int(config.get("bootstrap_replicates", 2000)),
            "bootstrap_seed": int(config.get("bootstrap_seed", 2027)),
        },
        "summary": summary,
        "view_cluster_bootstrap_95_ci": intervals,
        "regional_visualizations": heatmaps,
        "notes": {
            "strict_improvement": "A patch is improved only when the best GES or DRK seed has MSE strictly below the best of the three 3DGS seeds; exact ties are excluded.",
            "family_contributions": "GES and DRK contributions are separate augmented-oracle analyses and are not additive.",
            "model_count_limitation": "The nine-model oracle has a larger candidate pool than the three-model 3DGS oracle and cannot be worse by construction. Results are descriptive local complementarity, not proof of kernel specialization.",
            "matched_size_context": "The existing three-model within-family and three-model cross-family comparisons remain the matched-size control and are not replaced by this analysis.",
            "winner_ties": "Exact ties among full-oracle models use stable method/seed order for the exported winner label; winning_model_tie_count records the multiplicity.",
            "uncertainty_interpretation": "A descriptive positive gain is not treated as statistically meaningful unless the view-cluster confidence interval excludes zero.",
            "statistical_unit": "Confidence intervals resample complete held-out views. Overlapping patches are not treated as independent replicates.",
        },
    }
    atomic_json(result_root / "nine_model_oracle_summary.json", summary_document)
    write_latex_table(result_root / "nine_model_oracle_table.tex", summary, intervals)
    print(f"[nine-model] strict improved patches: {summary['improved_patch_count']:,}/{summary['patch_count']:,} ({100 * float(summary['improved_patch_fraction']):.3f}%)")
    print(f"[nine-model] wrote outputs to {result_root}; elapsed {time.perf_counter() - started:.2f}s")
    return summary_document


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure nine-model oracle coverage beyond the three-seed 3DGS oracle.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-visualizations", action="store_true", help="Skip optional regional heatmaps; numerical outputs are unchanged.")
    args = parser.parse_args()
    run_analysis(args.config, skip_visualizations=args.skip_visualizations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
