from __future__ import annotations

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.local_error.image_io import MethodSpec, ensure_same_shape, load_rgb, pair_views
from analysis.local_error.metrics import abs_rgb_error, image_metrics, mse, psnr_from_mse, ssim_simple
from analysis.local_error.patches import Patch, crop, iter_patches
from analysis.oracle.oracle import oracle_summary
from analysis.statistics.predictor import evaluate_predictors
from analysis.structure_features.descriptors import patch_descriptors
from analysis.winner_maps.visualize import rasterize_patch_values, save_float_map, save_winner_map


def load_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return config


def resolve_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def inspect(config_path: Path, config: Dict[str, object]) -> List[object]:
    gt_dir = resolve_path(config_path, str(config["gt_dir"]))
    methods = [
        MethodSpec(
            name=str(item["name"]),
            render_dir=resolve_path(config_path, str(item["render_dir"])),
            render_name_template=item.get("render_name_template"),
        )
        for item in config["methods"]
    ]
    print(f"scene: {config['scene']}")
    print(f"gt_dir: {gt_dir}")
    for method in methods:
        print(f"{method.name}_render_dir: {method.render_dir}")
    views = pair_views(gt_dir, methods)
    print(f"paired_views: {len(views)}")
    print("first_views:", ", ".join(view.view for view in views[:5]))
    return views


def method_colors(methods: List[str]) -> Dict[str, tuple[int, int, int]]:
    palette = [(45, 114, 210), (218, 87, 67), (62, 150, 91), (180, 118, 35)]
    return {method: palette[index % len(palette)] for index, method in enumerate(methods)}


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def interpretation_note(config: Dict[str, object]) -> str:
    note = config.get("interpretation_note")
    if note:
        return str(note)
    return (
        "Pipeline validation only. Do not interpret 3DGS-vs-GES winners as evidence of "
        "cross-family specialization; Gaussian is nested inside GES."
    )


def analyze(config_path: Path, config: Dict[str, object], inspect_only: bool = False) -> Optional[Dict[str, object]]:
    views = inspect(config_path, config)
    if inspect_only:
        return None

    scene = str(config["scene"])
    patch_size = int(config.get("patch_size", 64))
    stride = int(config.get("stride", patch_size))
    tie_threshold = float(config.get("tie_threshold_mse", 1e-5))
    results_dir = resolve_path(config_path, str(config["results_dir"]))
    methods = [str(item["name"]) for item in config["methods"]]
    if len(methods) < 2:
        raise ValueError("At least two methods are required for winner/oracle analysis")

    all_rows: List[Dict[str, object]] = []
    image_rows: List[Dict[str, object]] = []
    colors = method_colors(methods)

    for view_paths in views:
        gt = load_rgb(view_paths.gt_path)
        renders = {name: load_rgb(path) for name, path in view_paths.render_paths.items()}
        ensure_same_shape([gt, *renders.values()], view_paths.view)

        image_row: Dict[str, object] = {"scene": scene, "view": view_paths.view}
        for name, render in renders.items():
            metrics = image_metrics(gt, render)
            for key, value in metrics.items():
                image_row[f"{name}_{key}"] = value

            error_map = abs_rgb_error(gt, render).mean(axis=-1)
            save_float_map(results_dir / "maps" / name / f"{Path(view_paths.view).stem}_error.png", error_map)
        image_rows.append(image_row)

        height, width = gt.shape[:2]
        patches = list(iter_patches(height, width, patch_size, stride))
        view_rows: List[Dict[str, object]] = []

        for patch_index, patch in enumerate(patches):
            gt_patch = crop(gt, patch)
            features = patch_descriptors(gt_patch)
            method_mse = {}
            method_psnr = {}
            method_ssim = {}
            for name, render in renders.items():
                render_patch = crop(render, patch)
                value = mse(gt_patch, render_patch)
                method_mse[name] = value
                method_psnr[name] = psnr_from_mse(value)
                method_ssim[name] = ssim_simple(gt_patch, render_patch)

            sorted_methods = sorted(method_mse, key=method_mse.get)
            best = sorted_methods[0]
            second = sorted_methods[1]
            margin = method_mse[second] - method_mse[best]
            winner = "tie" if margin < tie_threshold else best

            row: Dict[str, object] = {
                "scene": scene,
                "view": view_paths.view,
                "patch_index": patch_index,
                "x": patch.x,
                "y": patch.y,
                "width": patch.width,
                "height": patch.height,
                "winner": winner,
                "best_method": best,
                "second_method": second,
                "error_margin": margin,
            }
            row.update(features)
            for name in methods:
                row[f"{name}_mae"] = float(abs_rgb_error(gt_patch, crop(renders[name], patch)).mean())
                row[f"{name}_mse"] = method_mse[name]
                row[f"{name}_psnr"] = method_psnr[name]
                row[f"{name}_ssim_simple"] = method_ssim[name]
            view_rows.append(row)

        all_rows.extend(view_rows)

        for name in methods:
            values = [float(row[f"{name}_mse"]) for row in view_rows]
            patch_map = rasterize_patch_values((height, width), patches, values)
            save_float_map(results_dir / "maps" / name / f"{Path(view_paths.view).stem}_patch_mse.png", patch_map)

        for left, right in combinations(methods, 2):
            diff_values = [float(row[f"{right}_mse"]) - float(row[f"{left}_mse"]) for row in view_rows]
            diff_map = rasterize_patch_values((height, width), patches, diff_values)
            save_float_map(
                results_dir / "maps" / "difference" / f"{left}_vs_{right}" / f"{Path(view_paths.view).stem}_signed_mse.png",
                diff_map,
                signed=True,
            )

        margin_map = rasterize_patch_values((height, width), patches, [float(row["error_margin"]) for row in view_rows])
        save_float_map(results_dir / "maps" / "margin" / f"{Path(view_paths.view).stem}_margin.png", margin_map)
        save_winner_map(
            results_dir / "maps" / "winner" / f"{Path(view_paths.view).stem}_winner.png",
            (height, width),
            patches,
            [str(row["winner"]) for row in view_rows],
            colors,
        )

    feature_columns = [
        "mean_gradient_magnitude",
        "edge_strength",
        "laplacian_energy",
        "local_variance",
        "high_frequency_energy",
        "entropy",
    ]
    metric_columns = []
    for name in methods:
        metric_columns.extend([f"{name}_mae", f"{name}_mse", f"{name}_psnr", f"{name}_ssim_simple"])
    patch_fields = [
        "scene",
        "view",
        "patch_index",
        "x",
        "y",
        "width",
        "height",
        *feature_columns,
        *metric_columns,
        "winner",
        "best_method",
        "second_method",
        "error_margin",
    ]
    write_csv(results_dir / "patches.csv", all_rows, patch_fields)
    write_csv(results_dir / "per_image_metrics.csv", image_rows, list(image_rows[0].keys()))

    summary = {
        "config": {
            "scene": scene,
            "patch_size": patch_size,
            "stride": stride,
            "tie_threshold_mse": tie_threshold,
            "methods": methods,
        },
        "oracle": oracle_summary(all_rows, methods),
        "predictors": evaluate_predictors(all_rows, methods),
        "notes": {
            "lpips_patch_scale": (
                "Not computed. Naive patch LPIPS is not used because LPIPS is calibrated as an image-level "
                "perceptual metric and small patches require a separate resize/context protocol."
            ),
            "interpretation": interpretation_note(config),
        },
    }
    with (results_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"wrote: {results_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Local patch comparison for rendered splatting outputs.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        analyze(args.config.resolve(), config, inspect_only=args.inspect_only)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
