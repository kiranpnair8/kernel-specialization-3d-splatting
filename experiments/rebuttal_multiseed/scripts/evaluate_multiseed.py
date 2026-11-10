from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis.local_error.image_io import load_rgb
from analysis.local_error.metrics import mse
from analysis.local_error.patches import crop, iter_patches
from common import atomic_json, load_config, resolve_project_path, write_csv


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ply_count(path: Path) -> int:
    with path.open("rb") as handle:
        for raw in handle:
            line = raw.decode("ascii", errors="strict").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise ValueError(f"No vertex count in PLY header: {path}")


def parse_metrics(method: str, model_path: Path, iteration: int) -> dict[str, float]:
    if method in {"3dgs", "ges"}:
        path = model_path / "results.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        key = f"ours_{iteration}"
        if key not in data:
            if len(data) != 1:
                raise KeyError(f"Missing {key} in {path}; keys={list(data)}")
            key = next(iter(data))
        record = data[key]
        return {"psnr": float(record["PSNR"]), "ssim": float(record["SSIM"]), "lpips": float(record["LPIPS"]), "metric_source": str(path)}
    path = model_path / "metric" / f"test_{iteration}.txt"
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            if key.strip().upper() in {"PSNR", "SSIM", "LPIPS"}:
                values[key.strip().lower()] = float(value.strip())
    if set(values) != {"psnr", "ssim", "lpips"}:
        raise ValueError(f"Incomplete DRK metrics in {path}: {values}")
    return {**values, "metric_source": str(path)}


def run_paths(method: str, model_path: Path, iteration: int) -> tuple[Path, str, Path, str]:
    if method in {"3dgs", "ges"}:
        base = model_path / "test" / f"ours_{iteration}"
        return base / "renders", "{stem}.png", base / "gt", "{stem}.png"
    base = model_path / "metric" / "test"
    return base, "render_{stem}.png", base, "gt_{stem}.png"


def final_ply(model_path: Path, iteration: int) -> Path:
    expected = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if expected.is_file():
        return expected
    candidates = list(model_path.glob(f"**/iteration_{iteration}/point_cloud.ply"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one final iteration-{iteration} PLY under {model_path}, found {candidates}")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit nine Room seed runs and compute the canonical p32 patch-MSE matrix once.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    result_root = resolve_project_path(config, str(config["result_root"]))
    output_root = resolve_project_path(config, str(config["output_root"]))
    reference_gt = resolve_project_path(config, str(config["reference_gt"]))
    reference_files = sorted(reference_gt.glob("*.png"))
    expected_views = int(config["expected_test_views"])
    if len(reference_files) != expected_views:
        raise ValueError(f"Expected {expected_views} reference GT files, found {len(reference_files)} in {reference_gt}")
    reference = {path.name: path for path in reference_files}
    reference_hashes = {name: sha256(path) for name, path in reference.items()}

    inventory: list[dict[str, object]] = []
    render_lookup: dict[str, dict[str, Path]] = {}
    for method, method_config in config["methods"].items():
        iteration = int(method_config["iterations"])
        for seed in config["seeds"]:
            run_id = f"{method}_seed_{seed}"
            status_path = output_root / method / f"seed_{seed}" / "run_status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("state") != "complete":
                raise RuntimeError(f"Run is not complete: {status_path} ({status.get('state')})")
            model_path = Path(status["model_path"])
            render_dir, render_template, gt_dir, gt_template = run_paths(method, model_path, iteration)
            run_renders: dict[str, Path] = {}
            shapes: set[tuple[int, int, int]] = set()
            for view, gt_path in reference.items():
                stem = Path(view).stem
                candidate_gt = gt_dir / gt_template.format(stem=stem)
                render_path = render_dir / render_template.format(stem=stem)
                if not candidate_gt.is_file() or not render_path.is_file():
                    raise FileNotFoundError(f"Missing aligned files for {run_id}/{view}: gt={candidate_gt}, render={render_path}")
                if sha256(candidate_gt) != reference_hashes[view]:
                    reference_array = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.int16)
                    candidate_array = np.asarray(Image.open(candidate_gt).convert("RGB"), dtype=np.int16)
                    if reference_array.shape != candidate_array.shape or not np.array_equal(reference_array, candidate_array):
                        maximum = "shape mismatch" if reference_array.shape != candidate_array.shape else int(np.max(np.abs(reference_array - candidate_array)))
                        raise ValueError(f"GT alignment failed for {run_id}/{view}; max pixel difference={maximum}")
                with Image.open(render_path) as image:
                    shapes.add((*image.size[::-1], 3))
                run_renders[view] = render_path
            if len(shapes) != 1:
                raise ValueError(f"Render dimensions vary within {run_id}: {shapes}")
            metrics = parse_metrics(method, model_path, iteration)
            ply = final_ply(model_path, iteration)
            inventory.append({
                "run_id": run_id, "method": method, "seed": seed, "iteration": iteration,
                "psnr": metrics["psnr"], "ssim": metrics["ssim"], "lpips": metrics["lpips"],
                "primitive_count": ply_count(ply), "model_path": str(model_path), "render_dir": str(render_dir),
                "metric_source": metrics["metric_source"], "final_ply": str(ply), "test_view_count": len(run_renders),
            })
            render_lookup[run_id] = run_renders

    run_ids = [str(row["run_id"]) for row in inventory]
    patch_errors: list[list[float]] = []
    patch_views: list[str] = []
    xs: list[int] = []
    ys: list[int] = []
    widths: list[int] = []
    heights: list[int] = []
    for view, gt_path in reference.items():
        gt = load_rgb(gt_path)
        renders = {run_id: load_rgb(render_lookup[run_id][view]) for run_id in run_ids}
        shapes = {gt.shape, *(image.shape for image in renders.values())}
        if len(shapes) != 1:
            raise ValueError(f"Shape mismatch for {view}: {shapes}")
        for patch in iter_patches(gt.shape[0], gt.shape[1], int(config["patch_size"]), int(config["stride"])):
            gt_patch = crop(gt, patch)
            patch_errors.append([mse(gt_patch, crop(renders[run_id], patch)) for run_id in run_ids])
            patch_views.append(view)
            xs.append(patch.x); ys.append(patch.y); widths.append(patch.width); heights.append(patch.height)
        print(f"evaluated {view}: {len(patch_errors)} cumulative patches")
    errors = np.asarray(patch_errors, dtype=np.float64)
    if not np.isfinite(errors).all():
        raise ValueError("Non-finite patch MSE detected")
    result_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        result_root / "patch_errors.npz", errors=errors, run_ids=np.asarray(run_ids), views=np.asarray(patch_views),
        x=np.asarray(xs), y=np.asarray(ys), width=np.asarray(widths), height=np.asarray(heights),
    )
    bundle_metadata = {
        "scene": config["scene"], "patch_size": config["patch_size"], "stride": config["stride"],
        "tie_threshold_mse": config["tie_threshold_mse"], "expected_test_views": expected_views,
        "reference_gt": str(reference_gt), "run_ids": run_ids, "patch_count": len(errors),
        "color_normalization": "PIL RGB conversion followed by float32 division by 255.0 (analysis.local_error.image_io.load_rgb)",
        "alignment": "Every run GT is pixel-identical to the canonical 3DGS Room GT for every filename.",
    }
    atomic_json(result_root / "patch_errors.json", bundle_metadata)
    write_csv(result_root / "run_inventory.csv", inventory)
    atomic_json(result_root / "run_inventory.json", inventory)
    print(f"Wrote {len(inventory)} runs and {len(errors)} patches to {result_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
