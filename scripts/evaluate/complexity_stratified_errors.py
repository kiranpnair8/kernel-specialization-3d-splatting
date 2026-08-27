#!/usr/bin/env python3
"""Complexity-stratified reconstruction-error analysis for Paper-1."""
from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENES = {
    "garden": "results/garden/3dgs_vs_ges_vs_drk_p32/patches.csv",
    "bicycle": "results/bicycle/3dgs_vs_ges_vs_drk_p32/patches.csv",
    "room": "results/room/3dgs_vs_ges_vs_drk_p32/patches.csv",
}
DEFAULT_METHODS = ["3dgs", "ges", "drk"]
DEFAULT_DESCRIPTORS = [
    "mean_gradient_magnitude",
    "edge_strength",
    "laplacian_energy",
    "local_variance",
    "high_frequency_energy",
    "entropy",
]
COMPARISONS = [("ges", "3dgs"), ("drk", "3dgs"), ("ges", "drk")]
METHOD_COLORS = {
    "3dgs": (48, 94, 160),
    "ges": (34, 132, 94),
    "drk": (178, 78, 58),
}
SCENE_COLORS = [
    (48, 94, 160),
    (220, 126, 52),
    (52, 145, 88),
    (146, 89, 166),
]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Input CSV is empty: {path}")
    return rows


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_scene(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Scene must use name=path syntax")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Scene name cannot be empty")
    path = resolve_path(path.strip())
    if path.is_dir():
        path = path / "patches.csv"
    return name, path


def as_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Field {field} contains a non-numeric value: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Field {field} contains NaN/inf: {value!r}")
    return result


def identify_view_column(columns: set[str]) -> str:
    for candidate in ("view", "image", "filename", "file"):
        if candidate in columns:
            return candidate
    raise ValueError("View identifiers are unavailable; expected a column like 'view'")


def identify_mse_columns(columns: set[str], methods: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for method in methods:
        candidates = [f"{method}_mse", f"{method}_patch_mse", f"mse_{method}"]
        for candidate in candidates:
            if candidate in columns:
                result[method] = candidate
                break
        if method not in result:
            raise ValueError(f"Could not identify patch MSE column for method {method}; tried {candidates}")
    return result


def validate_scene_rows(
    scene: str,
    rows: list[dict[str, str]],
    descriptors: list[str],
    methods: list[str],
) -> dict[str, Any]:
    columns = set(rows[0])
    missing_descriptors = [descriptor for descriptor in descriptors if descriptor not in columns]
    if missing_descriptors:
        raise ValueError(f"Scene {scene} is missing descriptor columns: {missing_descriptors}")
    view_column = identify_view_column(columns)
    mse_columns = identify_mse_columns(columns, methods)
    required = [view_column, *descriptors, *mse_columns.values()]
    views: set[str] = set()
    for index, row in enumerate(rows):
        view = str(row.get(view_column, "")).strip()
        if not view:
            raise ValueError(f"Scene {scene} row {index} has an empty view identifier")
        views.add(view)
        for field in required:
            if field != view_column:
                as_float(row[field], field)
    return {
        "scene": scene,
        "patch_count": len(rows),
        "view_count": len(views),
        "view_column": view_column,
        "mse_columns": mse_columns,
        "descriptors": descriptors,
    }


def quantile_bins(values: np.ndarray, bins: int) -> list[np.ndarray]:
    if bins < 1:
        raise ValueError("Number of bins must be positive")
    order = np.argsort(values, kind="mergesort")
    return [chunk.astype(np.int64) for chunk in np.array_split(order, min(bins, len(values))) if len(chunk)]


def mean(values: np.ndarray) -> float:
    return float(np.mean(values))


def median(values: np.ndarray) -> float:
    return float(np.median(values))


def bootstrap_cluster_ci(
    values: np.ndarray,
    views: np.ndarray,
    statistic,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float | None, float | None]:
    unique_views = np.unique(views)
    if len(unique_views) == 0:
        return None, None
    grouped = [values[views == view] for view in unique_views]
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.integers(0, len(grouped), size=len(grouped))
        estimates[index] = statistic(np.concatenate([grouped[item] for item in sampled]))
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and sorted_values[end] == sorted_values[index]:
            end += 1
        ranks[order[index:end]] = 0.5 * (index + 1 + end)
        index = end
    return ranks


def normal_two_sided_pvalue(z_value: float) -> float:
    return float(min(1.0, 1.0 - math.erf(abs(z_value) / math.sqrt(2.0))))


def spearman(values_x: np.ndarray, values_y: np.ndarray) -> tuple[float, float]:
    if len(values_x) != len(values_y) or len(values_x) < 3:
        return float("nan"), float("nan")
    try:
        from scipy.stats import spearmanr  # type: ignore

        stat = spearmanr(values_x, values_y)
        return float(stat.correlation), float(stat.pvalue)
    except Exception:
        rx = rankdata(values_x)
        ry = rankdata(values_y)
        corr = float(np.corrcoef(rx, ry)[0, 1])
        if not math.isfinite(corr):
            return corr, float("nan")
        z_value = corr * math.sqrt(max(1.0, len(values_x) - 1.0))
        return corr, normal_two_sided_pvalue(z_value)


def benjamini_hochberg(rows: list[dict[str, Any]], pvalue_key: str = "spearman_p_value") -> None:
    indexed = [
        (index, float(row[pvalue_key]))
        for index, row in enumerate(rows)
        if row.get(pvalue_key) is not None and math.isfinite(float(row[pvalue_key]))
    ]
    indexed.sort(key=lambda item: item[1])
    count = len(indexed)
    previous = 1.0
    for rank, (index, pvalue) in reversed(list(enumerate(indexed, start=1))):
        adjusted = min(previous, pvalue * count / rank)
        rows[index]["spearman_p_value_bh_fdr"] = float(min(1.0, adjusted))
        previous = adjusted
    for row in rows:
        row.setdefault("spearman_p_value_bh_fdr", None)


def sign(value: float | None, eps: float = 1e-12) -> str:
    if value is None or not math.isfinite(value):
        return "missing"
    if abs(value) <= eps:
        return "zero"
    return "positive" if value > 0 else "negative"


def consistency_status(signs: list[str]) -> str:
    if any(item == "missing" for item in signs):
        return "missing"
    nonzero = [item for item in signs if item != "zero"]
    if not nonzero:
        return "all_zero"
    if len(set(nonzero)) == 1 and len(nonzero) == len(signs):
        return "same_direction"
    if len(set(nonzero)) == 1:
        return "weak_or_zero_in_some_scenes"
    return "mixed_directions"


def analyze(
    scene_inputs: list[tuple[str, Path]],
    output_dir: Path,
    descriptors: list[str],
    methods: list[str],
    bin_count: int,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    scenes = []
    for scene_name, path in scene_inputs:
        rows = read_csv(path)
        meta = validate_scene_rows(scene_name, rows, descriptors, methods)
        scenes.append({"name": scene_name, "path": str(path), "rows": rows, "meta": meta})

    stratified_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    trend_rows: list[dict[str, Any]] = []

    for scene in scenes:
        rows = scene["rows"]
        meta = scene["meta"]
        view_column = meta["view_column"]
        mse_columns = meta["mse_columns"]
        views = np.asarray([str(row[view_column]) for row in rows])
        method_mse = {
            method: np.asarray([as_float(row[column], column) for row in rows], dtype=np.float64)
            for method, column in mse_columns.items()
        }

        for descriptor in descriptors:
            descriptor_values = np.asarray([as_float(row[descriptor], descriptor) for row in rows], dtype=np.float64)
            bins = quantile_bins(descriptor_values, bin_count)
            binned_delta_medians: dict[str, list[tuple[float, float]]] = defaultdict(list)

            for bin_index, indices in enumerate(bins):
                descriptor_bin = descriptor_values[indices]
                normalized_position = (bin_index + 0.5) / len(bins)
                stratified: dict[str, Any] = {
                    "scene": scene["name"],
                    "descriptor": descriptor,
                    "bin_index": bin_index,
                    "normalized_bin_position": normalized_position,
                    "patch_count": int(len(indices)),
                    "descriptor_min": float(np.min(descriptor_bin)),
                    "descriptor_max": float(np.max(descriptor_bin)),
                    "descriptor_median": median(descriptor_bin),
                }
                for method in methods:
                    values = method_mse[method][indices]
                    stratified[f"{method}_mean_mse"] = mean(values)
                    stratified[f"{method}_median_mse"] = median(values)
                stratified_rows.append(stratified)

                for left, right in COMPARISONS:
                    comparison = f"{left}_vs_{right}"
                    delta = method_mse[left][indices] - method_mse[right][indices]
                    mean_low, mean_high = bootstrap_cluster_ci(delta, views[indices], mean, bootstrap_replicates, rng)
                    median_low, median_high = bootstrap_cluster_ci(delta, views[indices], median, bootstrap_replicates, rng)
                    mean_delta = mean(delta)
                    median_delta = median(delta)
                    delta_rows.append(
                        {
                            "scene": scene["name"],
                            "descriptor": descriptor,
                            "comparison": comparison,
                            "left_method": left,
                            "right_method": right,
                            "bin_index": bin_index,
                            "normalized_bin_position": normalized_position,
                            "patch_count": int(len(indices)),
                            "descriptor_min": float(np.min(descriptor_bin)),
                            "descriptor_max": float(np.max(descriptor_bin)),
                            "descriptor_median": median(descriptor_bin),
                            "mean_delta_mse": mean_delta,
                            "mean_delta_mse_ci_low": mean_low,
                            "mean_delta_mse_ci_high": mean_high,
                            "median_delta_mse": median_delta,
                            "median_delta_mse_ci_low": median_low,
                            "median_delta_mse_ci_high": median_high,
                            "delta_sign_convention": "negative means left_method has lower MSE",
                        }
                    )
                    binned_delta_medians[comparison].append((normalized_position, median_delta))

            for left, right in COMPARISONS:
                comparison = f"{left}_vs_{right}"
                delta_all = method_mse[left] - method_mse[right]
                rho, pvalue = spearman(descriptor_values, delta_all)
                binned = binned_delta_medians[comparison]
                slope = None
                if len(binned) >= 2:
                    slope = float(np.polyfit([item[0] for item in binned], [item[1] for item in binned], deg=1)[0])
                trend_rows.append(
                    {
                        "scene": scene["name"],
                        "descriptor": descriptor,
                        "comparison": comparison,
                        "left_method": left,
                        "right_method": right,
                        "spearman_rho": rho,
                        "spearman_p_value": pvalue,
                        "binned_median_delta_slope": slope,
                        "direction": sign(slope),
                        "delta_sign_convention": "negative means left_method has lower MSE",
                    }
                )

    benjamini_hochberg(trend_rows)
    consistency_rows = build_cross_scene_consistency(trend_rows)
    summary = {
        "scenes": [
            {
                "scene": scene["name"],
                "path": scene["path"],
                "patch_count": scene["meta"]["patch_count"],
                "view_count": scene["meta"]["view_count"],
            }
            for scene in scenes
        ],
        "descriptors": descriptors,
        "methods": methods,
        "comparisons": [f"{left}_vs_{right}" for left, right in COMPARISONS],
        "quantile_bins": bin_count,
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "output_dir": str(output_dir),
        "interpretation": (
            "Observational complexity-stratified reconstruction-error analysis. "
            "Uncertainty uses view-level cluster bootstrap; scene is the replication unit for cross-scene interpretation."
        ),
    }

    write_csv(output_dir / "stratified_error_summary.csv", stratified_rows)
    write_json(output_dir / "stratified_error_summary.json", stratified_rows)
    write_csv(output_dir / "paired_delta_summary.csv", delta_rows)
    write_json(output_dir / "paired_delta_summary.json", delta_rows)
    write_csv(output_dir / "trend_tests.csv", trend_rows)
    write_json(output_dir / "trend_tests.json", trend_rows)
    write_csv(output_dir / "cross_scene_trend_consistency.csv", consistency_rows)
    write_json(output_dir / "cross_scene_trend_consistency.json", consistency_rows)
    write_json(output_dir / "analysis_summary.json", summary)
    make_plots(output_dir / "plots", stratified_rows, delta_rows, descriptors, methods)
    print_summary(summary)
    return {
        "summary": summary,
        "stratified_rows": stratified_rows,
        "delta_rows": delta_rows,
        "trend_rows": trend_rows,
        "consistency_rows": consistency_rows,
    }


def build_cross_scene_consistency(trend_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in trend_rows:
        grouped[(str(row["descriptor"]), str(row["comparison"]))][str(row["scene"])] = row
    scene_names = sorted({str(row["scene"]) for row in trend_rows})
    rows = []
    for (descriptor, comparison), by_scene in sorted(grouped.items()):
        out: dict[str, Any] = {"descriptor": descriptor, "comparison": comparison}
        directions = []
        for scene in scene_names:
            row = by_scene.get(scene)
            effect = None if row is None else row.get("binned_median_delta_slope")
            direction = sign(None if effect is None else float(effect))
            out[f"{scene}_effect_value"] = effect
            out[f"{scene}_direction"] = direction
            directions.append(direction)
        out["status"] = consistency_status(directions)
        out["direction_consistent_across_scenes"] = out["status"] == "same_direction"
        out["effect_value"] = "binned_median_delta_slope"
        out["delta_sign_convention"] = "negative means left method has lower MSE increasingly at higher descriptor quantiles"
        rows.append(out)
    return rows


def font(size: int = 13):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def grouped_rows(rows: list[Mapping[str, Any]], keys: list[str]) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return grouped


def draw_axes(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    y_zero: float | None,
    ymin: float,
    ymax: float,
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(210, 210, 210))
    draw.line((x0, y1, x1, y1), fill=(40, 40, 40), width=2)
    draw.line((x0, y0, x0, y1), fill=(40, 40, 40), width=2)
    draw.text((x0, y0 - 22), title, fill=(25, 25, 25), font=font(13))
    if y_zero is not None and ymin < y_zero < ymax:
        y = y1 - int((y_zero - ymin) / (ymax - ymin) * (y1 - y0))
        draw.line((x0, y, x1, y), fill=(120, 120, 120), width=1)


def make_plots(
    plot_dir: Path,
    stratified_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    descriptors: list[str],
    methods: list[str],
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    scene_names = sorted({str(row["scene"]) for row in stratified_rows})
    stratified_by = grouped_rows(stratified_rows, ["descriptor", "scene"])
    delta_by = grouped_rows(delta_rows, ["descriptor", "comparison", "scene"])
    for descriptor in descriptors:
        draw_mse_plot(plot_dir / f"{descriptor}_method_mse_by_quantile.png", descriptor, scene_names, methods, stratified_by)
        for left, right in COMPARISONS:
            comparison = f"{left}_vs_{right}"
            draw_delta_plot(plot_dir / f"{descriptor}_{comparison}_delta_by_quantile.png", descriptor, comparison, scene_names, delta_by)


def y_range(values: list[float], include_zero: bool = False) -> tuple[float, float]:
    if include_zero:
        values = [*values, 0.0]
    ymin = min(values)
    ymax = max(values)
    pad = max(abs(ymin) * 0.1, 1e-6) if ymin == ymax else (ymax - ymin) * 0.08
    return ymin - pad, ymax + pad


def draw_mse_plot(
    path: Path,
    descriptor: str,
    scenes: list[str],
    methods: list[str],
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]],
) -> None:
    width = max(1200, 420 * len(scenes))
    height = 520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((28, 18), f"Reconstruction MSE vs {descriptor} quantile", fill=(20, 20, 20), font=font(18))
    panel_width = (width - 110) // max(1, len(scenes))
    all_values = [
        float(row[f"{method}_mean_mse"])
        for scene in scenes
        for row in grouped.get((descriptor, scene), [])
        for method in methods
    ]
    ymin, ymax = y_range(all_values)
    for scene_index, scene in enumerate(scenes):
        rows = sorted(grouped.get((descriptor, scene), []), key=lambda row: int(row["bin_index"]))
        box = (60 + scene_index * panel_width, 92, 60 + scene_index * panel_width + panel_width - 45, height - 78)
        draw_axes(draw, box, scene, None, ymin, ymax)
        x0, y0, x1, y1 = box
        for method in methods:
            points = []
            for row in rows:
                px = x0 + int(float(row["normalized_bin_position"]) * (x1 - x0))
                py = y1 - int((float(row[f"{method}_mean_mse"]) - ymin) / (ymax - ymin) * (y1 - y0))
                points.append((px, py))
            if len(points) >= 2:
                draw.line(points, fill=METHOD_COLORS.get(method, (80, 80, 80)), width=3)
            for point in points:
                draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=METHOD_COLORS.get(method, (80, 80, 80)))
    draw_legend(draw, width - 150, 58, methods, METHOD_COLORS)
    image.save(path)


def draw_delta_plot(
    path: Path,
    descriptor: str,
    comparison: str,
    scenes: list[str],
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]],
) -> None:
    width = max(1200, 420 * len(scenes))
    height = 520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((28, 18), f"{comparison} MSE delta vs {descriptor} quantile", fill=(20, 20, 20), font=font(18))
    panel_width = (width - 110) // max(1, len(scenes))
    all_values = [float(row["median_delta_mse"]) for scene in scenes for row in grouped.get((descriptor, comparison, scene), [])]
    ymin, ymax = y_range(all_values, include_zero=True)
    for scene_index, scene in enumerate(scenes):
        rows = sorted(grouped.get((descriptor, comparison, scene), []), key=lambda row: int(row["bin_index"]))
        box = (60 + scene_index * panel_width, 92, 60 + scene_index * panel_width + panel_width - 45, height - 78)
        draw_axes(draw, box, scene, 0.0, ymin, ymax)
        x0, y0, x1, y1 = box
        color = SCENE_COLORS[scene_index % len(SCENE_COLORS)]
        points = []
        for row in rows:
            px = x0 + int(float(row["normalized_bin_position"]) * (x1 - x0))
            py = y1 - int((float(row["median_delta_mse"]) - ymin) / (ymax - ymin) * (y1 - y0))
            points.append((px, py))
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        for point in points:
            draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=color)
    image.save(path)


def draw_legend(draw: ImageDraw.ImageDraw, x: int, y: int, labels: list[str], colors: Mapping[str, tuple[int, int, int]]) -> None:
    for label in labels:
        color = colors.get(label, (80, 80, 80))
        draw.line((x, y, x + 28, y), fill=color, width=4)
        draw.text((x + 36, y - 8), label, fill=(35, 35, 35), font=font(12))
        y += 24


def print_summary(summary: Mapping[str, Any]) -> None:
    print("Complexity-stratified reconstruction-error analysis")
    print(f"Output directory: {summary['output_dir']}")
    print(f"Descriptors: {', '.join(summary['descriptors'])}")
    print(f"Bootstrap replicates: {summary['bootstrap_replicates']}")
    print(f"Seed: {summary['seed']}")
    for scene in summary["scenes"]:
        print(f"- {scene['scene']}: {scene['patch_count']} patches, {scene['view_count']} views")


def make_synthetic_rows() -> list[tuple[str, list[dict[str, str]]]]:
    scenes = []
    for scene_index, scene in enumerate(("synthetic_a", "synthetic_b")):
        rows = []
        for view_index in range(4):
            for patch_index in range(30):
                complexity = (patch_index + 30 * view_index) / 119.0
                base = 0.01 + 0.002 * scene_index + 0.01 * complexity
                row: dict[str, str] = {"scene": scene, "view": f"view_{view_index:03d}.png", "patch_index": str(patch_index)}
                for descriptor in DEFAULT_DESCRIPTORS:
                    row[descriptor] = f"{complexity:.8f}"
                row["3dgs_mse"] = f"{base:.8f}"
                row["ges_mse"] = f"{base - 0.001 - 0.004 * complexity:.8f}"
                row["drk_mse"] = f"{base - 0.002 + 0.006 * complexity:.8f}"
                rows.append(row)
        scenes.append((scene, rows))
    return scenes


def run_smoke_test(output_dir: Path, bootstrap_replicates: int, seed: int) -> int:
    with tempfile.TemporaryDirectory() as temp_name:
        temp_dir = Path(temp_name)
        scene_inputs = []
        for scene, rows in make_synthetic_rows():
            path = temp_dir / scene / "patches.csv"
            write_csv(path, rows)
            scene_inputs.append((scene, path))
        result = analyze(scene_inputs, output_dir, DEFAULT_DESCRIPTORS, DEFAULT_METHODS, 10, bootstrap_replicates, seed)
    assert len(result["stratified_rows"]) == 2 * len(DEFAULT_DESCRIPTORS) * 10
    first_delta = next(row for row in result["delta_rows"] if row["comparison"] == "ges_vs_3dgs")
    assert float(first_delta["median_delta_mse"]) < 0.0
    assert all("spearman_p_value_bh_fdr" in row for row in result["trend_rows"])
    assert result["consistency_rows"]
    for filename in (
        "stratified_error_summary.csv",
        "paired_delta_summary.csv",
        "trend_tests.csv",
        "cross_scene_trend_consistency.csv",
        "analysis_summary.json",
    ):
        assert (output_dir / filename).exists(), filename
    assert any((output_dir / "plots").glob("*.png"))
    print("Synthetic smoke test passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", action="append", type=parse_scene, help="Scene input as name=patches.csv or name=result_dir")
    parser.add_argument("--output-dir", default="results/cross_scene/complexity_stratified_errors_p32")
    parser.add_argument("--descriptors", default=",".join(DEFAULT_DESCRIPTORS))
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    output_dir = resolve_path(args.output_dir)
    descriptors = parse_csv_list(args.descriptors)
    methods = parse_csv_list(args.methods)
    if methods != DEFAULT_METHODS:
        raise SystemExit("This analysis currently expects methods: 3dgs,ges,drk")
    if args.bootstrap_replicates < 1:
        raise SystemExit("--bootstrap-replicates must be positive")
    if args.smoke_test:
        return run_smoke_test(output_dir, args.bootstrap_replicates, args.seed)

    scenes = args.scene or [(name, resolve_path(path)) for name, path in DEFAULT_SCENES.items()]
    analyze(scenes, output_dir, descriptors, methods, args.bins, args.bootstrap_replicates, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
