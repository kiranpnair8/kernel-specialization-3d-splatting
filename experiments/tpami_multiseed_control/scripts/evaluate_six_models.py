from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
for item in (SCRIPT_DIR, REPO_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from analysis.local_error.image_io import load_rgb
from analysis.local_error.metrics import mse
from analysis.local_error.patches import crop, iter_patches
from common import atomic_json, expected_run_ids, load_config, resolve, write_csv


def load_experiment2_evaluator():
    path = REPO_ROOT / "experiments" / "rebuttal_independent_dataset" / "scripts" / "evaluate_independent.py"
    spec = importlib.util.spec_from_file_location("experiment2_evaluate_independent", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP2 = load_experiment2_evaluator()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def run_status_path(config: dict, method: str, seed: int) -> Path:
    root_key = "experiment2_output_root" if seed == 0 else "output_root"
    return resolve(config, config[root_key]) / config["scene"] / method / f"seed_{seed}" / "run_status.json"


def verify_seed_record(status: dict, method: str, seed: int) -> None:
    if int(status.get("seed", -1)) != seed or status.get("method") != method:
        raise ValueError(f"Wrong method/seed in run status: expected {method}/{seed}, found {status.get('method')}/{status.get('seed')}")
    training = next((record for record in status.get("commands", []) if record.get("stage") == "train"), None)
    if training is None:
        raise ValueError(f"Missing training command for {method}_s{seed}")
    command = [str(value) for value in training["command"]]
    if "--seed" not in command or command[command.index("--seed") + 1] != str(seed):
        raise ValueError(f"Training command did not explicitly propagate seed {seed}: {command}")


def source_state(root: Path) -> dict:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    diff = subprocess.check_output(["git", "diff", "--", "train.py", "train_ges.py", "utils/general_utils.py"], cwd=root, text=True)
    return {"commit": commit, "seed_patch_diff_sha256": sha256_text(diff), "seed_patch_diff_present": bool(diff.strip())}


def load_runs(config: dict) -> tuple[dict[str, dict], list[dict]]:
    runs = {}
    inventory = []
    for method, method_config in config["methods"].items():
        iteration = int(method_config["iterations"])
        for seed in config["seeds"]:
            run_id = f"{method}_s{seed}"
            status_path = run_status_path(config, method, int(seed))
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("state") != "complete":
                raise RuntimeError(f"Incomplete run: {status_path}")
            verify_seed_record(status, method, int(seed))
            model = Path(status["model_path"])
            metrics, metric_source = EXP2.parse_metrics(method, model, iteration)
            ply = EXP2.final_ply(model, iteration)
            gt_dir, gt_files, renders = EXP2.render_sources(method, model, iteration)
            runs[run_id] = {"method": method, "seed": int(seed), "model": model, "gt_files": gt_files, "renders": renders}
            inventory.append({
                "run_id": run_id, "method": method, "seed": seed, "iteration": iteration, **metrics,
                "primitive_count": EXP2.ply_count(ply), "model_path": str(model), "metric_source": str(metric_source),
                "final_ply": str(ply), "test_view_count": len(gt_files), "run_status": str(status_path),
                "training_elapsed_seconds": next(record.get("elapsed_seconds") for record in status["commands"] if record["stage"] == "train"),
                "test_split_sha256": status.get("test_split_sha256", "recorded in Experiment-2 manifest for seed 0"),
            })
    if list(runs) != expected_run_ids(config["seeds"]):
        raise ValueError(f"Unexpected run order: {list(runs)}")
    return runs, inventory


def verify_against_experiment2(config: dict, errors: np.ndarray, views: np.ndarray, coordinates: dict[str, np.ndarray]) -> dict:
    source = resolve(config, config["experiment2_result"]) / "patch_errors.npz"
    with np.load(source, allow_pickle=False) as bundle:
        original_errors = np.asarray(bundle["errors"], dtype=np.float64)
        original_methods = [str(value) for value in bundle["methods"]]
        if original_methods[:2] != ["3dgs", "ges"]:
            raise ValueError(f"Unexpected Experiment-2 methods: {original_methods}")
        if not np.array_equal(views, np.asarray(bundle["views"]).astype(str)):
            raise ValueError("Held-out view ordering differs from Experiment 2")
        for name in coordinates:
            if not np.array_equal(coordinates[name], np.asarray(bundle[name], dtype=np.int64)):
                raise ValueError(f"Patch coordinates differ from Experiment 2: {name}")
    if original_errors.shape[0] != errors.shape[0]:
        raise ValueError("Patch count differs from Experiment 2")
    differences = {
        "3dgs_s0_max_abs_mse_difference": float(np.max(np.abs(original_errors[:, 0] - errors[:, 0]))),
        "ges_s0_max_abs_mse_difference": float(np.max(np.abs(original_errors[:, 1] - errors[:, 3]))),
    }
    if max(differences.values()) > 1e-15:
        raise ValueError(f"Seed-0 patch errors do not reproduce Experiment 2: {differences}")
    return {"source": str(source), "status": "exact", **differences}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the aligned six-model Dr Johnson patch-error matrices.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    result_root = resolve(config, config["result_root"]); result_root.mkdir(parents=True, exist_ok=True)
    runs, inventory = load_runs(config)
    run_ids = list(runs)
    reference_files = runs["3dgs_s0"]["gt_files"]
    if len(reference_files) != 33:
        raise ValueError(f"Expected 33 held-out views, found {len(reference_files)}")
    alignments = {run_id: EXP2.align_by_gt_pixels(reference_files, run["gt_files"], run["renders"]) for run_id, run in runs.items()}
    settings = [(int(size), int(size) // 2) for size in config["sensitivity"]["patch_sizes"]]
    buffers = {size: {"errors": [], "views": [], "x": [], "y": [], "width": [], "height": []} for size, _ in settings}
    view_records = []
    for view_index, reference_path in enumerate(reference_files):
        view = reference_path.name
        gt = load_rgb(reference_path)
        rendered = {run_id: load_rgb(Path(alignments[run_id][view]["render"])) for run_id in run_ids}
        if len({gt.shape, *(image.shape for image in rendered.values())}) != 1:
            raise ValueError(f"Image dimensions differ for {view}")
        for size, stride in settings:
            target = buffers[size]
            for patch in iter_patches(gt.shape[0], gt.shape[1], size, stride):
                gt_patch = crop(gt, patch)
                target["errors"].append([mse(gt_patch, crop(rendered[run_id], patch)) for run_id in run_ids])
                target["views"].append(view); target["x"].append(patch.x); target["y"].append(patch.y)
                target["width"].append(patch.width); target["height"].append(patch.height)
        view_records.append({"view": view, "gt": str(reference_path), "shape": list(gt.shape), "renders": {run_id: alignments[run_id][view]["render"] for run_id in run_ids}})
        print(f"[six-model] view {view_index + 1}/33 {view}")
    validation = None
    sensitivity_root = result_root / "sensitivity"; sensitivity_root.mkdir(parents=True, exist_ok=True)
    for size, stride in settings:
        buffer = buffers[size]
        errors = np.asarray(buffer["errors"], dtype=np.float64)
        views = np.asarray(buffer["views"])
        coordinates = {name: np.asarray(buffer[name], dtype=np.int64) for name in ("x", "y", "width", "height")}
        if not np.isfinite(errors).all():
            raise ValueError(f"Non-finite p{size} errors")
        destination = result_root / "patch_errors.npz" if size == int(config["patch_size"]) else sensitivity_root / f"patch_errors_p{size}.npz"
        np.savez_compressed(destination, errors=errors, run_ids=np.asarray(run_ids), views=views, **coordinates)
        metadata = {
            "scene": config["scene"], "run_ids": run_ids, "patch_size": size, "stride": stride,
            "patch_count": len(errors), "view_count": len(reference_files), "views": view_records,
            "alignment": "all run GT images mapped one-to-one to 3DGS seed-0 GT by RGB-pixel hash and dimensions",
            "local_mse": "mean squared RGB error after conversion to [0,1]",
        }
        atomic_json(destination.with_suffix(".json"), metadata)
        if size == int(config["patch_size"]):
            validation = verify_against_experiment2(config, errors, views, coordinates)
    write_csv(result_root / "model_inventory.csv", inventory); atomic_json(result_root / "model_inventory.json", inventory)
    atomic_json(result_root / "alignment_audit.json", {"status": "exact", "view_count": 33, "run_ids": run_ids, "experiment2_reproduction": validation, "alignments": alignments})
    source_states = {method: source_state(resolve(config, method_config["root"])) for method, method_config in config["methods"].items()}
    manifest_candidates = [result_root / "dataset_manifest.json", resolve(config, config["experiment2_result"]).parent / "dataset_manifest.json"]
    experiment2_manifest_path = next((path for path in manifest_candidates if path.is_file()), None)
    if experiment2_manifest_path is None:
        raise FileNotFoundError(f"No strict dataset manifest found: {manifest_candidates}")
    experiment2_manifest = json.loads(experiment2_manifest_path.read_text(encoding="utf-8"))
    split_record = next(row for row in experiment2_manifest["scenes"] if row["scene"] == config["scene"])
    new_split_hashes = {str(row["test_split_sha256"]) for row in inventory if row["seed"] != 0}
    if new_split_hashes != {split_record["test_split_sha256"]}:
        raise ValueError(f"New-run split hashes differ from Experiment 2: {new_split_hashes}")
    project_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=resolve(config, "."), text=True).strip()
    manifest = {
        "experiment": config["experiment"], "scene": config["scene"], "dataset": str(resolve(config, config["dataset"])),
        "seeds": config["seeds"], "run_ids": run_ids, "pair_pool_size": 2,
        "split": {"train_views": 230, "test_views": 33, "train_split_sha256": split_record["train_split_sha256"], "test_split_sha256": split_record["test_split_sha256"], "source_manifest": str(experiment2_manifest_path)},
        "patch_settings": {"primary": [config["patch_size"], config["stride"]], "sensitivity": settings},
        "tie_thresholds_mse": config["sensitivity"]["tie_thresholds_mse"], "methods": config["methods"],
        "tracked_repository_commit": project_commit, "source_states": source_states, "experiment2_reproduction": validation,
        "output_paths": {"models": str(resolve(config, config["output_root"])), "results": str(result_root)},
    }
    atomic_json(result_root / "experiment_manifest.json", manifest)
    print(f"Wrote six-model aligned matrices to {result_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
