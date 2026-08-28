#!/usr/bin/env python3
"""Evaluate canonical Phase-III synthetic pilot outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import inventory_phase3_outputs as inventory

METHODS = list(inventory.METHODS)
METRICS = ["psnr", "ssim", "lpips"]
PAIRWISE = [("ges", "3dgs"), ("drk", "3dgs"), ("drk", "ges")]
LEVEL_ORDER = {"low": 0, "medium": 1, "high": 2}
METHOD_COLORS = {"3dgs": "#305ea0", "ges": "#22845e", "drk": "#b24e3a"}


@dataclass
class Phase3Result:
    scene_id: str
    sweep_family: str
    level: str
    parameter_value: str
    seed: int
    train_view_count: int
    test_view_count: int
    method: str
    psnr: float | None
    ssim: float | None
    lpips: float | None
    primitive_count: int | None
    final_iteration: int | None
    output_path: str
    metrics_source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/synthetic/phase3_controlled_pilot"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/synthetic/phase3_controlled_pilot"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/synthetic/phase3_controlled_pilot/evaluation"))
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--skip-image-bootstrap", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def rel(path: str | Path | None, root: Path) -> str:
    if path is None:
        return ""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(p)


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def as_int(row: Mapping[str, Any], key: str, default: int) -> int:
    return inventory.as_int(dict(row), key, default)


def canonical_status_for_scene(method_root: Path, method: str, scene_id: str, expected_test_views: int) -> inventory.CandidateStatus | None:
    expected_iteration = inventory.EXPECTED_ITERATIONS[method]
    candidates = inventory.find_candidates(method_root, scene_id)
    statuses = [inventory.candidate_status(path, method, expected_iteration, expected_test_views) for path in candidates]
    return max(statuses, key=inventory.candidate_sort_key) if statuses else None


def flatten_metric_values(obj: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            yield str(key), value
            yield from flatten_metric_values(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from flatten_metric_values(value)


def parse_gaussian_results(path: Path) -> tuple[dict[str, float | None], str]:
    metrics = {metric: None for metric in METRICS}
    results_path = path / "results.json"
    if not results_path.exists():
        return metrics, "missing results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))
    candidates: list[tuple[int, Mapping[str, Any]]] = []
    for key, value in flatten_metric_values(data):
        if isinstance(value, Mapping):
            match = re.search(r"ours_(\d+)", key)
            if match:
                candidates.append((int(match.group(1)), value))
    if candidates:
        source = max(candidates, key=lambda item: item[0])[1]
    elif isinstance(data, Mapping):
        source = data
    else:
        source = {}
    upper = {str(key).lower(): value for key, value in source.items()}
    for metric in METRICS:
        value = upper.get(metric)
        if value is not None:
            metrics[metric] = float(value)
    if any(value is not None for value in metrics.values()):
        return metrics, rel(results_path, REPO_ROOT)
    return metrics, "results.json parsed but metrics were not found"


def parse_drk_metric_text(path: Path) -> tuple[dict[str, float | None], str]:
    metrics = {metric: None for metric in METRICS}
    metric_files = sorted((path / "metric").glob("test_*.txt"))
    if not metric_files:
        return metrics, "missing metric/test_*.txt"
    metric_path = max(metric_files, key=lambda p: inventory.iteration_from_path(p) or -1)
    text = metric_path.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "psnr": r"PSNR\s*[:=]\s*([0-9.eE+-]+)",
        "ssim": r"SSIM\s*[:=]\s*([0-9.eE+-]+)",
        "lpips": r"LPIPS\s*[:=]\s*([0-9.eE+-]+)",
    }
    for metric, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            metrics[metric] = float(match.group(1))
    return metrics, rel(metric_path, REPO_ROOT)


def parse_official_metrics(path: Path, method: str) -> tuple[dict[str, float | None], str]:
    if method in {"3dgs", "ges"}:
        return parse_gaussian_results(path)
    return parse_drk_metric_text(path)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0


def psnr_from_mse(mse: float) -> float:
    if mse <= 1e-12:
        return float("inf")
    return float(-10.0 * math.log10(mse))


def ssim_value(gt: np.ndarray, render: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity

        return float(structural_similarity(gt, render, channel_axis=-1, data_range=1.0))
    except Exception:
        x = 0.2126 * gt[..., 0] + 0.7152 * gt[..., 1] + 0.0722 * gt[..., 2]
        y = 0.2126 * render[..., 0] + 0.7152 * render[..., 1] + 0.0722 * render[..., 2]
        c1 = 0.01**2
        c2 = 0.03**2
        mux = float(np.mean(x))
        muy = float(np.mean(y))
        varx = float(np.var(x))
        vary = float(np.var(y))
        cov = float(np.mean((x - mux) * (y - muy)))
        numerator = (2.0 * mux * muy + c1) * (2.0 * cov + c2)
        denominator = (mux * mux + muy * muy + c1) * (varx + vary + c2)
        return float(numerator / max(denominator, 1e-12))


def image_metrics(gt_path: Path, render_path: Path) -> dict[str, float]:
    gt = load_rgb(gt_path)
    render = load_rgb(render_path)
    if gt.shape != render.shape:
        raise ValueError(f"Shape mismatch: {gt_path} {gt.shape} vs {render_path} {render.shape}")
    mse = float(np.mean((gt - render) ** 2))
    return {"mse": mse, "psnr": psnr_from_mse(mse), "ssim": ssim_value(gt, render)}


def method_image_pairs(path: Path, method: str) -> dict[str, tuple[Path, Path]]:
    pairs: dict[str, tuple[Path, Path]] = {}
    if method in {"3dgs", "ges"}:
        test_dir = inventory.choose_highest_iteration_dir([p for p in (path / "test").glob("ours_*") if p.is_dir()])
        if test_dir is None:
            return pairs
        render_dir = test_dir / "renders"
        gt_dir = test_dir / "gt"
        for render in sorted(render_dir.glob("*.png")):
            gt = gt_dir / render.name
            if gt.exists():
                pairs[render.stem] = (gt, render)
        return pairs
    metric_dir = path / "metric" / "test"
    for render in sorted(metric_dir.glob("render_*.png")):
        stem = render.stem.removeprefix("render_")
        gt = metric_dir / f"gt_{stem}.png"
        if gt.exists():
            pairs[stem] = (gt, render)
    return pairs


def collect_per_view_metrics(selected: Mapping[tuple[str, str], Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (scene_id, method), output_path in selected.items():
        for view_id, (gt_path, render_path) in method_image_pairs(output_path, method).items():
            metrics = image_metrics(gt_path, render_path)
            rows.append(
                {
                    "scene_id": scene_id,
                    "method": method,
                    "view": view_id,
                    "mse": metrics["mse"],
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                    "gt_path": rel(gt_path, REPO_ROOT),
                    "render_path": rel(render_path, REPO_ROOT),
                }
            )
    return rows


def bootstrap_ci(values: np.ndarray, replicates: int, rng: np.random.Generator) -> tuple[float, float, float]:
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(values) == 1 or replicates <= 0:
        mean = float(np.mean(values))
        return mean, mean, mean
    samples = rng.choice(values, size=(replicates, len(values)), replace=True)
    means = np.mean(samples, axis=1)
    return float(np.mean(values)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap_rows(per_view_rows: list[dict[str, Any]], replicates: int, seed: int) -> list[dict[str, Any]]:
    by_scene_method: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in per_view_rows:
        by_scene_method[(str(row["scene_id"]), str(row["method"]))][str(row["view"])] = row

    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    scenes = sorted({key[0] for key in by_scene_method})
    for scene_id in scenes:
        for left, right in PAIRWISE:
            left_views = by_scene_method.get((scene_id, left), {})
            right_views = by_scene_method.get((scene_id, right), {})
            common_views = sorted(set(left_views) & set(right_views))
            if not common_views:
                continue
            for metric in ("mse", "psnr", "ssim"):
                deltas = np.asarray([float(left_views[v][metric]) - float(right_views[v][metric]) for v in common_views], dtype=np.float64)
                mean_delta, ci_low, ci_high = bootstrap_ci(deltas, replicates, rng)
                output.append(
                    {
                        "scene_id": scene_id,
                        "comparison": f"{left}_minus_{right}",
                        "metric": metric,
                        "view_count": len(common_views),
                        "mean_delta": mean_delta,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "bootstrap_replicates": replicates,
                    }
                )
    return output


def metric_key(row: Mapping[str, Any], metric: str) -> float:
    value = row.get(metric)
    if value is None or value == "":
        return float("nan")
    return float(value)


def pairwise_delta_rows(results: list[Phase3Result]) -> list[dict[str, Any]]:
    by_scene_method = {(row.scene_id, row.method): row for row in results}
    rows: list[dict[str, Any]] = []
    for scene_id in sorted({row.scene_id for row in results}):
        for left, right in PAIRWISE:
            left_row = by_scene_method.get((scene_id, left))
            right_row = by_scene_method.get((scene_id, right))
            if left_row is None or right_row is None:
                continue
            item: dict[str, Any] = {
                "scene_id": scene_id,
                "sweep_family": left_row.sweep_family,
                "level": left_row.level,
                "parameter_value": left_row.parameter_value,
                "comparison": f"{left}_minus_{right}",
            }
            for metric in METRICS:
                left_value = getattr(left_row, metric)
                right_value = getattr(right_row, metric)
                item[f"delta_{metric}"] = None if left_value is None or right_value is None else float(left_value - right_value)
            if left_row.primitive_count is not None and right_row.primitive_count is not None:
                item["delta_primitive_count"] = left_row.primitive_count - right_row.primitive_count
            rows.append(item)
    return rows


def summarize(values: list[float | None]) -> dict[str, float | int | None]:
    numeric = np.asarray([value for value in values if value is not None and math.isfinite(float(value))], dtype=np.float64)
    if len(numeric) == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(len(numeric)),
        "mean": float(np.mean(numeric)),
        "std": float(np.std(numeric, ddof=1)) if len(numeric) > 1 else 0.0,
        "min": float(np.min(numeric)),
        "max": float(np.max(numeric)),
    }


def factor_summary_rows(results: list[Phase3Result]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Phase3Result]] = defaultdict(list)
    for row in results:
        groups[(row.sweep_family, row.level, row.method)].append(row)
    rows: list[dict[str, Any]] = []
    for (family, level, method), items in sorted(groups.items(), key=lambda x: (x[0][0], LEVEL_ORDER.get(x[0][1], 99), x[0][2])):
        out: dict[str, Any] = {"sweep_family": family, "level": level, "method": method, "scene_count": len(items)}
        for metric in METRICS:
            stats = summarize([getattr(item, metric) for item in items])
            for key, value in stats.items():
                out[f"{metric}_{key}"] = value
        primitive_stats = summarize([item.primitive_count for item in items])
        for key, value in primitive_stats.items():
            out[f"primitive_count_{key}"] = value
        rows.append(out)
    return rows


def make_plots(results: list[Phase3Result], results_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_paths: list[str] = []
    for family in sorted({row.sweep_family for row in results}):
        for metric in METRICS:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for method in METHODS:
                items = [row for row in results if row.sweep_family == family and row.method == method]
                items.sort(key=lambda row: LEVEL_ORDER.get(row.level, 99))
                x = [row.level for row in items]
                y = [getattr(row, metric) for row in items]
                if not x or all(value is None for value in y):
                    continue
                ax.plot(x, y, marker="o", label=method.upper(), color=METHOD_COLORS.get(method))
            ax.set_title(f"{family.replace('_', ' ').title()} {metric.upper()}")
            ax.set_xlabel("Level")
            ax.set_ylabel(metric.upper())
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False)
            fig.tight_layout()
            path = results_dir / "plots" / f"{family}_{metric}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=160)
            plt.close(fig)
            plot_paths.append(rel(path, REPO_ROOT))
    return plot_paths


def best_by_metric(results: list[Phase3Result], scene_id: str, metric: str) -> str:
    candidates = [row for row in results if row.scene_id == scene_id and getattr(row, metric) is not None]
    if not candidates:
        return "unavailable"
    reverse = metric in {"psnr", "ssim"}
    row = sorted(candidates, key=lambda item: getattr(item, metric), reverse=reverse)[0]
    return f"{row.method.upper()} ({getattr(row, metric):.5f})"


def write_findings(path: Path, results: list[Phase3Result], pairwise_rows: list[dict[str, Any]], bootstrap_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase-III Synthetic Pilot Findings",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "This file reports measured outputs from the completed Phase-III controlled synthetic pilot. It does not make causal or speculative claims.",
        "",
        "## Coverage",
        "",
        f"- Method-scene records: {len(results)}",
        f"- Scenes: {len({row.scene_id for row in results})}",
        f"- Methods: {', '.join(method.upper() for method in METHODS)}",
        "",
        "## Per-scene Metric Leaders",
        "",
        "| Scene | PSNR leader | SSIM leader | LPIPS leader |",
        "| --- | --- | --- | --- |",
    ]
    for scene_id in sorted({row.scene_id for row in results}):
        lines.append(
            f"| {scene_id} | {best_by_metric(results, scene_id, 'psnr')} | "
            f"{best_by_metric(results, scene_id, 'ssim')} | {best_by_metric(results, scene_id, 'lpips')} |"
        )
    lines.extend(
        [
            "",
            "## Output Tables",
            "",
            "- `phase3_results.csv/json`: canonical aggregate metrics, primitive counts, final iterations, and output paths.",
            "- `phase3_pairwise_deltas.csv/json`: method-minus-method metric deltas for each synthetic scene.",
            "- `phase3_factor_summary.csv/json`: low/medium/high summaries by controlled factor and method.",
            "- `phase3_bootstrap_ci.csv/json`: paired view-level bootstrap intervals for image-derived metrics where render/GT views are available.",
            "- `plots/`: PSNR, SSIM, and LPIPS versus factor level for each sweep family.",
            "",
            "## Notes",
            "",
            "- Canonical runs are selected through `scripts/synthetic/inventory_phase3_outputs.py`.",
            "- Training was not rerun by this evaluation script.",
            "- Canonical outputs are read only.",
            f"- Pairwise delta rows: {len(pairwise_rows)}.",
            f"- Bootstrap CI rows: {len(bootstrap_rows)}.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    dataset_root = resolve_path(project_root, args.dataset_root)
    output_root = resolve_path(project_root, args.output_root)
    results_dir = resolve_path(project_root, args.results_dir)
    manifest = inventory.load_manifest(dataset_root)

    selected_paths: dict[tuple[str, str], Path] = {}
    results: list[Phase3Result] = []
    incomplete: list[str] = []

    for row in manifest:
        scene_id = str(row["scene_id"])
        expected_test_views = as_int(row, "test_view_count", 8)
        for method in METHODS:
            status = canonical_status_for_scene(output_root / method, method, scene_id, expected_test_views)
            if status is None:
                incomplete.append(f"{scene_id}/{method}: missing canonical output")
                continue
            if status.status != "complete":
                incomplete.append(f"{scene_id}/{method}: inventory status {status.status}")
            output_path = Path(status.path)
            selected_paths[(scene_id, method)] = output_path
            metrics, source = parse_official_metrics(output_path, method)
            results.append(
                Phase3Result(
                    scene_id=scene_id,
                    sweep_family=str(row.get("sweep_family", "")),
                    level=str(row.get("level", "")),
                    parameter_value=str(row.get("parameter_value", "")),
                    seed=as_int(row, "seed", 0),
                    train_view_count=as_int(row, "train_view_count", 24),
                    test_view_count=expected_test_views,
                    method=method,
                    psnr=metrics["psnr"],
                    ssim=metrics["ssim"],
                    lpips=metrics["lpips"],
                    primitive_count=status.primitive_count,
                    final_iteration=status.final_iteration,
                    output_path=rel(output_path, project_root),
                    metrics_source=source,
                )
            )

    if incomplete and not args.allow_incomplete:
        raise RuntimeError("Inventory did not report complete canonical outputs: " + "; ".join(incomplete))

    results_dir.mkdir(parents=True, exist_ok=True)
    result_rows = [asdict(row) for row in results]
    pairwise_rows = pairwise_delta_rows(results)
    factor_rows = factor_summary_rows(results)
    per_view_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    if not args.skip_image_bootstrap:
        per_view_rows = collect_per_view_metrics(selected_paths)
        bootstrap_rows = paired_bootstrap_rows(per_view_rows, args.bootstrap_replicates, args.bootstrap_seed)

    plot_paths = make_plots(results, results_dir)

    write_csv(results_dir / "phase3_results.csv", result_rows)
    write_json(results_dir / "phase3_results.json", result_rows)
    write_csv(results_dir / "phase3_pairwise_deltas.csv", pairwise_rows)
    write_json(results_dir / "phase3_pairwise_deltas.json", pairwise_rows)
    write_csv(results_dir / "phase3_factor_summary.csv", factor_rows)
    write_json(results_dir / "phase3_factor_summary.json", factor_rows)
    if per_view_rows:
        write_csv(results_dir / "phase3_per_view_metrics.csv", per_view_rows)
        write_json(results_dir / "phase3_per_view_metrics.json", per_view_rows)
    write_csv(results_dir / "phase3_bootstrap_ci.csv", bootstrap_rows)
    write_json(results_dir / "phase3_bootstrap_ci.json", bootstrap_rows)
    write_findings(results_dir / "phase3_findings.md", results, pairwise_rows, bootstrap_rows)

    manifest_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results_dir": rel(results_dir, project_root),
        "record_count": len(results),
        "pairwise_delta_count": len(pairwise_rows),
        "factor_summary_count": len(factor_rows),
        "per_view_metric_count": len(per_view_rows),
        "bootstrap_ci_count": len(bootstrap_rows),
        "plot_paths": plot_paths,
    }
    write_json(results_dir / "phase3_evaluation_manifest.json", manifest_payload)

    print(f"Phase-III evaluation complete: {len(results)} method-scene records")
    print(f"Wrote {rel(results_dir / 'phase3_results.csv', project_root)}")
    print(f"Wrote {rel(results_dir / 'phase3_findings.md', project_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
