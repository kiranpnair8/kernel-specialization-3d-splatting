from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
for item in (SCRIPT_DIR, REPO_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from analysis.local_error.image_io import load_rgb
from analysis.local_error.metrics import mse
from analysis.local_error.patches import crop, iter_patches
from common import atomic_json, load_config, resolve, write_csv


def pixel_digest(path: Path) -> tuple[str, tuple[int, int, int]]:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest(), tuple(array.shape)


def ply_count(path: Path) -> int:
    with path.open("rb") as handle:
        for raw in handle:
            line = raw.decode("ascii", errors="strict").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise ValueError(f"No vertex count in {path}")


def final_ply(model: Path, iteration: int) -> Path:
    expected = model / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    candidates = [expected] if expected.is_file() else list(model.glob(f"**/iteration_{iteration}/point_cloud.ply"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one iteration-{iteration} PLY in {model}, found {candidates}")
    return candidates[0]


def parse_metrics(method: str, model: Path, iteration: int) -> tuple[dict[str, float], Path]:
    if method in {"3dgs", "ges"}:
        path = model / "results.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        key = f"ours_{iteration}"
        if key not in data:
            if len(data) != 1:
                raise KeyError(f"Missing {key} in {path}: {list(data)}")
            key = next(iter(data))
        record = data[key]
        return {name: float(record[source]) for name, source in (("psnr", "PSNR"), ("ssim", "SSIM"), ("lpips", "LPIPS"))}, path
    path = model / "metric" / f"test_{iteration}.txt"
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            name, value = line.split(":", 1)
            if name.strip().lower() in {"psnr", "ssim", "lpips"}:
                values[name.strip().lower()] = float(value.strip())
    if set(values) != {"psnr", "ssim", "lpips"}:
        raise ValueError(f"Incomplete metrics in {path}: {values}")
    return values, path


def render_sources(method: str, model: Path, iteration: int) -> tuple[Path, list[Path], dict[str, Path]]:
    if method in {"3dgs", "ges"}:
        root = model / "test" / f"ours_{iteration}"
        gt_files = sorted((root / "gt").glob("*.png"))
        renders = {path.name: root / "renders" / path.name for path in gt_files}
        return root / "gt", gt_files, renders
    root = model / "metric" / "test"
    gt_files = sorted(root.glob("gt_*.png"))
    renders = {path.name: root / path.name.replace("gt_", "render_", 1) for path in gt_files}
    return root, gt_files, renders


def align_by_gt_pixels(reference_files: list[Path], candidate_gt: list[Path], candidate_renders: dict[str, Path]) -> dict[str, dict[str, str]]:
    digest_to_reference: dict[str, tuple[str, tuple[int, int, int]]] = {}
    for path in reference_files:
        digest, shape = pixel_digest(path)
        if digest in digest_to_reference:
            raise ValueError(f"Ambiguous duplicate reference GT pixels: {path} and {digest_to_reference[digest][0]}")
        digest_to_reference[digest] = (path.name, shape)
    mapping: dict[str, dict[str, str]] = {}
    for candidate in candidate_gt:
        digest, shape = pixel_digest(candidate)
        if digest not in digest_to_reference:
            raise ValueError(f"Candidate GT has no pixel-identical reference: {candidate}")
        view, reference_shape = digest_to_reference[digest]
        if shape != reference_shape or view in mapping:
            raise ValueError(f"Ambiguous or dimension-mismatched alignment for {candidate}")
        render = candidate_renders[candidate.name]
        if not render.is_file():
            raise FileNotFoundError(render)
        if pixel_digest(render)[1] != shape:
            raise ValueError(f"Render/GT shape mismatch: {render}, {candidate}")
        mapping[view] = {"candidate_gt": str(candidate), "render": str(render)}
    expected = {path.name for path in reference_files}
    if set(mapping) != expected:
        raise ValueError(f"Alignment is incomplete: missing={sorted(expected - set(mapping))}, extra={sorted(set(mapping) - expected)}")
    return mapping


def evaluate_scene(config: dict, scene: str, result_root: Path) -> tuple[list[dict], dict]:
    output_root = resolve(config, config["output_root"])
    run_data: dict[str, dict] = {}
    inventory: list[dict] = []
    for method, method_config in config["methods"].items():
        status_path = output_root / scene / method / f"seed_{config['seed']}" / "run_status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") != "complete":
            raise RuntimeError(f"Incomplete run: {status_path}")
        model = Path(status["model_path"])
        iteration = int(method_config["iterations"])
        metrics, metric_path = parse_metrics(method, model, iteration)
        ply = final_ply(model, iteration)
        gt_dir, gt_files, renders = render_sources(method, model, iteration)
        run_data[method] = {"model": model, "gt_dir": gt_dir, "gt_files": gt_files, "renders": renders}
        inventory.append({
            "scene": scene, "method": method, "seed": config["seed"], "iteration": iteration,
            **metrics, "primitive_count": ply_count(ply), "model_path": str(model),
            "metric_source": str(metric_path), "final_ply": str(ply), "test_view_count": len(gt_files),
            "training_elapsed_seconds": next(record["elapsed_seconds"] for record in status["commands"] if record["stage"] == "train"),
        })
    reference_files = run_data["3dgs"]["gt_files"]
    if not reference_files:
        raise ValueError(f"No 3DGS test GT files for {scene}")
    alignments = {}
    for method in config["methods"]:
        source = run_data[method]
        alignments[method] = align_by_gt_pixels(reference_files, source["gt_files"], source["renders"])
    views = [path.name for path in reference_files]
    records = []
    errors: list[list[float]] = []
    patch_views: list[str] = []
    xs: list[int] = []; ys: list[int] = []; widths: list[int] = []; heights: list[int] = []
    for index, reference_path in enumerate(reference_files):
        view = reference_path.name
        gt = load_rgb(reference_path)
        rendered = {method: load_rgb(Path(alignments[method][view]["render"])) for method in config["methods"]}
        if len({gt.shape, *(value.shape for value in rendered.values())}) != 1:
            raise ValueError(f"Aligned image shapes differ for {scene}/{view}")
        for patch in iter_patches(gt.shape[0], gt.shape[1], int(config["patch_size"]), int(config["stride"])):
            target = crop(gt, patch)
            errors.append([mse(target, crop(rendered[method], patch)) for method in config["methods"]])
            patch_views.append(view); xs.append(patch.x); ys.append(patch.y); widths.append(patch.width); heights.append(patch.height)
        records.append({"view": view, "gt": str(reference_path), "renders": {method: alignments[method][view]["render"] for method in config["methods"]}})
        print(f"[{scene}] view {index + 1}/{len(reference_files)}: {len(errors):,} cumulative patches")
    values = np.asarray(errors, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite patch MSE in {scene}")
    scene_root = result_root / scene
    scene_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(scene_root / "patch_errors.npz", errors=values, methods=np.asarray(list(config["methods"])), views=np.asarray(patch_views), x=np.asarray(xs), y=np.asarray(ys), width=np.asarray(widths), height=np.asarray(heights))
    metadata = {
        "scene": scene, "methods": list(config["methods"]), "patch_count": len(values), "view_count": len(views),
        "patch_size": config["patch_size"], "stride": config["stride"], "tie_threshold_mse": config["tie_threshold_mse"],
        "alignment": "candidate GT mapped one-to-one to 3DGS GT by SHA-256 of RGB pixels and dimensions; renders inherit candidate camera identity",
        "local_mse": "mean squared error over all RGB values after PIL RGB conversion and division by 255",
        "views": records,
    }
    atomic_json(scene_root / "patch_errors.json", metadata)
    atomic_json(scene_root / "alignment_audit.json", {"scene": scene, "view_count": len(views), "status": "exact_pixel_alignment", "methods": alignments})
    return inventory, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Align official renders and compute reusable p32 patch errors for both independent scenes.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    result_root = resolve(config, config["result_root"])
    all_inventory = []
    for scene in config["scenes"]:
        inventory, _ = evaluate_scene(config, scene, result_root)
        all_inventory.extend(inventory)
    write_csv(result_root / "global_quality.csv", all_inventory)
    atomic_json(result_root / "global_quality.json", all_inventory)
    print(f"Wrote aligned evaluation for {len(config['scenes'])} scenes to {result_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
