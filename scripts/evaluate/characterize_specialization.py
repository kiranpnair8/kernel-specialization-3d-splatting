from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import json
from math import erf, sqrt
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.local_error.image_io import MethodSpec, load_rgb, pair_views, save_rgb
from analysis.local_error.patches import Patch, crop
from scripts.evaluate.local_compare import load_config, resolve_path


DESCRIPTORS = [
    "mean_gradient_magnitude",
    "edge_strength",
    "laplacian_energy",
    "local_variance",
    "high_frequency_energy",
    "entropy",
]

DEFAULT_METHODS = ["3dgs", "ges", "drk"]
DEFAULT_PATCH_SIZE = 32
DEFAULT_STRIDE = 16
DEFAULT_TIE_THRESHOLD = 1e-5
COLORS = {
    "3dgs": (45, 114, 210),
    "ges": (218, 87, 67),
    "drk": (62, 150, 91),
    "tie": (130, 130, 130),
}


def _normal_sf_abs(z_value: float) -> float:
    return 0.5 * (1.0 - erf(abs(z_value) / sqrt(2.0)))


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and sorted_values[end] == sorted_values[index]:
            end += 1
        rank = 0.5 * (index + 1 + end)
        ranks[order[index:end]] = rank
        index = end
    return ranks


def mann_whitney_pvalue(left: np.ndarray, right: np.ndarray) -> Optional[float]:
    if len(left) == 0 or len(right) == 0:
        return None
    try:
        from scipy.stats import mannwhitneyu  # type: ignore

        return float(mannwhitneyu(left, right, alternative="two-sided").pvalue)
    except Exception:
        combined = np.concatenate([left, right])
        ranks = _rankdata(combined)
        n_left = float(len(left))
        n_right = float(len(right))
        rank_sum_left = float(ranks[: len(left)].sum())
        u_left = rank_sum_left - n_left * (n_left + 1.0) / 2.0
        mean_u = n_left * n_right / 2.0
        std_u = sqrt(n_left * n_right * (n_left + n_right + 1.0) / 12.0)
        if std_u == 0:
            return None
        z_value = (u_left - mean_u) / std_u
        return float(min(1.0, 2.0 * _normal_sf_abs(z_value)))


def benjamini_hochberg(rows: List[Dict[str, object]], pvalue_key: str = "p_value") -> None:
    indexed = [
        (index, float(row[pvalue_key]))
        for index, row in enumerate(rows)
        if row.get(pvalue_key) is not None and np.isfinite(float(row[pvalue_key]))
    ]
    indexed.sort(key=lambda item: item[1])
    count = len(indexed)
    previous = 1.0
    for rank, (index, pvalue) in reversed(list(enumerate(indexed, start=1))):
        adjusted = min(previous, pvalue * count / rank)
        rows[index]["p_value_bh_fdr"] = float(min(1.0, adjusted))
        previous = adjusted
    for row in rows:
        row.setdefault("p_value_bh_fdr", None)


def load_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Patch CSV does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Patch CSV is empty: {path}")
    return rows


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def descriptor_values(rows: Sequence[Mapping[str, object]], descriptor: str) -> np.ndarray:
    return np.asarray([float(row[descriptor]) for row in rows], dtype=np.float64)


def validate_patch_setting(rows: Sequence[Mapping[str, object]], patch_size: int, stride: int) -> Dict[str, object]:
    widths = sorted({int(float(row["width"])) for row in rows})
    heights = sorted({int(float(row["height"])) for row in rows})
    x_mods = sorted({int(float(row["x"])) % stride for row in rows}) if stride > 0 else []
    y_mods = sorted({int(float(row["y"])) % stride for row in rows}) if stride > 0 else []
    return {
        "expected_patch_size": patch_size,
        "expected_stride": stride,
        "observed_widths": widths,
        "observed_heights": heights,
        "x_mod_stride_values": x_mods,
        "y_mod_stride_values": y_mods,
        "patch_size_matches": widths == [patch_size] and heights == [patch_size],
        "stride_grid_matches": x_mods == [0] and y_mods == [0],
    }


def grouped_by_winner(rows: Sequence[Mapping[str, object]], labels: Sequence[str]) -> Dict[str, List[Mapping[str, object]]]:
    groups = {label: [] for label in labels}
    for row in rows:
        winner = str(row["winner"])
        if winner in groups:
            groups[winner].append(row)
    return groups


def summarize_descriptors(rows: Sequence[Mapping[str, object]], labels: Sequence[str]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    groups = grouped_by_winner(rows, list(labels) + ["tie"])
    summary_rows: List[Dict[str, object]] = []
    nested: Dict[str, object] = {}
    for descriptor in DESCRIPTORS:
        nested[descriptor] = {}
        for label in list(labels) + ["tie"]:
            values = descriptor_values(groups[label], descriptor) if groups[label] else np.asarray([], dtype=np.float64)
            if len(values):
                q25, median, q75 = np.percentile(values, [25, 50, 75])
                payload = {
                    "descriptor": descriptor,
                    "winner": label,
                    "count": int(len(values)),
                    "median": float(median),
                    "q25": float(q25),
                    "q75": float(q75),
                    "iqr": float(q75 - q25),
                }
            else:
                payload = {
                    "descriptor": descriptor,
                    "winner": label,
                    "count": 0,
                    "median": None,
                    "q25": None,
                    "q75": None,
                    "iqr": None,
                }
            summary_rows.append(payload)
            nested[descriptor][label] = payload
    return summary_rows, nested


def cohens_d(left: np.ndarray, right: np.ndarray) -> Optional[float]:
    if len(left) < 2 or len(right) < 2:
        return None
    var_left = float(np.var(left, ddof=1))
    var_right = float(np.var(right, ddof=1))
    pooled = sqrt(((len(left) - 1) * var_left + (len(right) - 1) * var_right) / (len(left) + len(right) - 2))
    if pooled == 0:
        return None
    return float((float(np.mean(left)) - float(np.mean(right))) / pooled)


def cliffs_delta(left: np.ndarray, right: np.ndarray) -> Optional[float]:
    if len(left) == 0 or len(right) == 0:
        return None
    sorted_right = np.sort(right)
    greater = np.searchsorted(sorted_right, left, side="left").sum()
    less_or_equal = np.searchsorted(sorted_right, left, side="right").sum()
    less = len(right) * len(left) - less_or_equal
    return float((greater - less) / (len(left) * len(right)))


def pairwise_effects(rows: Sequence[Mapping[str, object]], methods: Sequence[str]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    groups = grouped_by_winner(rows, methods)
    effect_rows: List[Dict[str, object]] = []
    for descriptor in DESCRIPTORS:
        for left_index, left_method in enumerate(methods):
            for right_method in methods[left_index + 1 :]:
                left_values = descriptor_values(groups[left_method], descriptor)
                right_values = descriptor_values(groups[right_method], descriptor)
                effect_rows.append(
                    {
                        "descriptor": descriptor,
                        "left_winner": left_method,
                        "right_winner": right_method,
                        "left_count": int(len(left_values)),
                        "right_count": int(len(right_values)),
                        "left_median": float(np.median(left_values)) if len(left_values) else None,
                        "right_median": float(np.median(right_values)) if len(right_values) else None,
                        "median_difference_left_minus_right": (
                            float(np.median(left_values) - np.median(right_values))
                            if len(left_values) and len(right_values)
                            else None
                        ),
                        "cohens_d_left_minus_right": cohens_d(left_values, right_values),
                        "cliffs_delta_left_minus_right": cliffs_delta(left_values, right_values),
                        "test": "mann_whitney_u_two_sided",
                        "p_value": mann_whitney_pvalue(left_values, right_values),
                    }
                )
    benjamini_hochberg(effect_rows)
    nested: Dict[str, object] = {}
    for row in effect_rows:
        nested.setdefault(str(row["descriptor"]), []).append(row)
    return effect_rows, nested


def quantile_probability_rows(
    rows: Sequence[Mapping[str, object]],
    methods: Sequence[str],
    bins: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    probability_rows: List[Dict[str, object]] = []
    nested: Dict[str, object] = {}
    for descriptor in DESCRIPTORS:
        values = descriptor_values(rows, descriptor)
        edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
        if len(edges) < 2:
            edges = np.asarray([float(values.min()), float(values.max()) + 1e-12], dtype=np.float64)
        nested[descriptor] = []
        for bin_index in range(len(edges) - 1):
            left = edges[bin_index]
            right = edges[bin_index + 1]
            if bin_index == len(edges) - 2:
                mask = (values >= left) & (values <= right)
            else:
                mask = (values >= left) & (values < right)
            selected = [rows[index] for index, keep in enumerate(mask) if keep]
            count = len(selected)
            payload: Dict[str, object] = {
                "descriptor": descriptor,
                "bin_index": bin_index,
                "bin_left": float(left),
                "bin_right": float(right),
                "bin_mid": float((left + right) / 2.0),
                "count": count,
            }
            for method in methods:
                wins = sum(1 for row in selected if str(row["winner"]) == method)
                payload[f"p_{method}_wins"] = float(wins / count) if count else None
            tie_count = sum(1 for row in selected if str(row["winner"]) == "tie")
            payload["p_tie"] = float(tie_count / count) if count else None
            probability_rows.append(payload)
            nested[descriptor].append(payload)
    return probability_rows, nested


def correlation_matrix(rows: Sequence[Mapping[str, object]]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    matrix = np.vstack([descriptor_values(rows, descriptor) for descriptor in DESCRIPTORS])
    corr = np.corrcoef(matrix)
    csv_rows = []
    nested: Dict[str, object] = {}
    for i, left in enumerate(DESCRIPTORS):
        nested[left] = {}
        for j, right in enumerate(DESCRIPTORS):
            value = float(corr[i, j])
            csv_rows.append({"descriptor": left, "other_descriptor": right, "pearson_r": value})
            nested[left][right] = value
    return csv_rows, nested


def draw_probability_plot(path: Path, descriptor: str, rows: Sequence[Mapping[str, object]], methods: Sequence[str]) -> None:
    width, height = 860, 520
    margin_left, margin_right, margin_top, margin_bottom = 80, 30, 40, 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw.rectangle(
        [margin_left, margin_top, margin_left + plot_width, margin_top + plot_height],
        outline=(40, 40, 40),
        width=1,
    )
    draw.text((margin_left, 12), f"Winner probability by {descriptor}", fill=(20, 20, 20))
    draw.text((margin_left, height - 42), descriptor, fill=(20, 20, 20))
    draw.text((12, margin_top + 4), "P(win)", fill=(20, 20, 20))

    x_values = np.asarray([float(row["bin_mid"]) for row in rows], dtype=np.float64)
    if len(x_values) == 0:
        return
    x_min = float(x_values.min())
    x_max = float(x_values.max())
    if x_min == x_max:
        x_max = x_min + 1e-12

    def to_xy(x_value: float, y_value: float) -> Tuple[int, int]:
        x_frac = (x_value - x_min) / (x_max - x_min)
        y_frac = max(0.0, min(1.0, y_value))
        return (
            int(margin_left + x_frac * plot_width),
            int(margin_top + (1.0 - y_frac) * plot_height),
        )

    for tick in np.linspace(0.0, 1.0, 6):
        _, y = to_xy(x_min, float(tick))
        draw.line([(margin_left, y), (margin_left + plot_width, y)], fill=(230, 230, 230))
        draw.text((margin_left - 45, y - 7), f"{tick:.1f}", fill=(60, 60, 60))

    for method in methods:
        points = []
        for row in rows:
            probability = row.get(f"p_{method}_wins")
            if probability is not None:
                points.append(to_xy(float(row["bin_mid"]), float(probability)))
        color = COLORS.get(method, (0, 0, 0))
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        for point in points:
            draw.ellipse([point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3], fill=color)

    legend_x = margin_left + plot_width - 150
    legend_y = margin_top + 16
    for offset, method in enumerate(methods):
        y = legend_y + offset * 24
        color = COLORS.get(method, (0, 0, 0))
        draw.line([(legend_x, y + 7), (legend_x + 28, y + 7)], fill=color, width=3)
        draw.text((legend_x + 36, y), method, fill=(20, 20, 20))

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def build_method_specs(config_path: Path, config: Mapping[str, object]) -> List[MethodSpec]:
    return [
        MethodSpec(
            name=str(item["name"]),
            render_dir=resolve_path(config_path, str(item["render_dir"])),
            render_name_template=item.get("render_name_template"),
        )
        for item in config["methods"]
    ]


def save_crop(path: Path, image: np.ndarray, patch: Patch) -> None:
    save_rgb(path, crop(image, patch))


def extract_exemplars(
    rows: Sequence[Mapping[str, object]],
    config_path: Path,
    config: Mapping[str, object],
    output_dir: Path,
    methods: Sequence[str],
    per_method: int,
) -> List[Dict[str, object]]:
    gt_dir = resolve_path(config_path, str(config["gt_dir"]))
    view_paths = {view.view: view for view in pair_views(gt_dir, build_method_specs(config_path, config))}
    exemplar_rows: List[Dict[str, object]] = []

    @lru_cache(maxsize=16)
    def load_view(view: str, method: str) -> np.ndarray:
        if method == "gt":
            return load_rgb(view_paths[view].gt_path)
        return load_rgb(view_paths[view].render_paths[method])

    for method in methods:
        method_rows = [
            row
            for row in rows
            if str(row["winner"]) == method and str(row.get("best_method", method)) == method
        ]
        method_rows.sort(key=lambda row: float(row.get("error_margin", 0.0)), reverse=True)
        for rank, row in enumerate(method_rows[:per_method], start=1):
            view = str(row["view"])
            patch = Patch(
                x=int(float(row["x"])),
                y=int(float(row["y"])),
                width=int(float(row["width"])),
                height=int(float(row["height"])),
            )
            stem = Path(view).stem
            exemplar_dir = output_dir / "exemplars" / method / f"{rank:03d}_{stem}_x{patch.x}_y{patch.y}"
            exemplar_dir.mkdir(parents=True, exist_ok=True)
            save_crop(exemplar_dir / "gt.png", load_view(view, "gt"), patch)
            for render_method in methods:
                save_crop(exemplar_dir / f"{render_method}.png", load_view(view, render_method), patch)

            metadata: Dict[str, object] = {
                "winner": method,
                "rank": rank,
                "view": view,
                "x": patch.x,
                "y": patch.y,
                "width": patch.width,
                "height": patch.height,
                "error_margin": float(row["error_margin"]),
                "local_errors": {render_method: float(row[f"{render_method}_mse"]) for render_method in methods},
                "descriptors": {descriptor: float(row[descriptor]) for descriptor in DESCRIPTORS},
            }
            write_json(exemplar_dir / "metadata.json", metadata)
            exemplar_rows.append(metadata)
    return exemplar_rows


def characterize(
    config_path: Path,
    patches_csv: Optional[Path],
    output_dir: Optional[Path],
    patch_size: int,
    stride: int,
    tie_threshold: float,
    quantile_bins: int,
    exemplars_per_method: int,
    skip_exemplars: bool,
) -> Dict[str, object]:
    config = load_config(config_path)
    results_dir = resolve_path(config_path, str(config["results_dir"]))
    resolved_patches_csv = patches_csv.resolve() if patches_csv else results_dir / "patches.csv"
    resolved_output_dir = output_dir.resolve() if output_dir else results_dir / "characterization"
    methods = [str(item["name"]) for item in config["methods"]]

    rows = load_rows(resolved_patches_csv)
    missing = [descriptor for descriptor in DESCRIPTORS if descriptor not in rows[0]]
    if missing:
        raise ValueError(f"Patch CSV is missing descriptor columns: {missing}")
    for method in methods:
        if f"{method}_mse" not in rows[0]:
            raise ValueError(f"Patch CSV is missing method error column: {method}_mse")

    validation = validate_patch_setting(rows, patch_size, stride)
    summary_rows, descriptor_summary = summarize_descriptors(rows, methods)
    effect_rows, effect_summary = pairwise_effects(rows, methods)
    probability_rows, probability_summary = quantile_probability_rows(rows, methods, quantile_bins)
    correlation_rows, correlation_summary = correlation_matrix(rows)

    write_csv(resolved_output_dir / "descriptor_summary.csv", summary_rows)
    write_json(resolved_output_dir / "descriptor_summary.json", descriptor_summary)
    write_csv(resolved_output_dir / "pairwise_effects.csv", effect_rows)
    write_json(resolved_output_dir / "pairwise_effects.json", effect_summary)
    write_csv(resolved_output_dir / "winner_probability_by_descriptor.csv", probability_rows)
    write_json(resolved_output_dir / "winner_probability_by_descriptor.json", probability_summary)
    write_csv(resolved_output_dir / "feature_correlation_matrix.csv", correlation_rows)
    write_json(resolved_output_dir / "feature_correlation_matrix.json", correlation_summary)

    for descriptor in DESCRIPTORS:
        draw_probability_plot(
            resolved_output_dir / "plots" / f"{descriptor}_winner_probability.png",
            descriptor,
            probability_summary[descriptor],
            methods,
        )

    exemplar_rows: List[Dict[str, object]] = []
    if not skip_exemplars:
        exemplar_rows = extract_exemplars(rows, config_path, config, resolved_output_dir, methods, exemplars_per_method)
        write_csv(resolved_output_dir / "exemplars.csv", exemplar_rows)
        write_json(resolved_output_dir / "exemplars.json", exemplar_rows)

    metadata = {
        "input_patches_csv": str(resolved_patches_csv),
        "output_dir": str(resolved_output_dir),
        "scene": config.get("scene"),
        "methods": methods,
        "patch_count": len(rows),
        "patch_size": patch_size,
        "stride": stride,
        "tie_threshold_mse": tie_threshold,
        "quantile_bins": quantile_bins,
        "exemplars_per_method": exemplars_per_method,
        "skip_exemplars": skip_exemplars,
        "validation": validation,
        "notes": {
            "interpretation": (
                "Strictly observational characterization of associations between local image-space "
                "descriptors and winning method labels. Do not interpret these associations as causal "
                "kernel-specialization evidence."
            ),
            "tests": (
                "Pairwise descriptor comparisons use two-sided Mann-Whitney U tests with "
                "Benjamini-Hochberg FDR correction across all descriptor/method-pair tests."
            ),
        },
    }
    write_json(resolved_output_dir / "characterization_summary.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Characterize local structures associated with method winners.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "garden_3dgs_ges_drk.json")
    parser.add_argument("--patches-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--tie-threshold-mse", type=float, default=DEFAULT_TIE_THRESHOLD)
    parser.add_argument("--quantile-bins", type=int, default=10)
    parser.add_argument("--exemplars-per-method", type=int, default=12)
    parser.add_argument("--skip-exemplars", action="store_true")
    args = parser.parse_args()

    try:
        summary = characterize(
            args.config.resolve(),
            args.patches_csv,
            args.output_dir,
            args.patch_size,
            args.stride,
            args.tie_threshold_mse,
            args.quantile_bins,
            args.exemplars_per_method,
            args.skip_exemplars,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    print(f"wrote: {summary['output_dir']}")
    validation = summary["validation"]
    if not validation["patch_size_matches"] or not validation["stride_grid_matches"]:
        print(f"warning: patch grid differs from requested setting: {validation}", file=sys.stderr)


if __name__ == "__main__":
    main()
