from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time

from common import atomic_json, load_config, resolve


def command_plan(method: str, root: Path, dataset: Path, model_base: Path, seed: int, iterations: int) -> list[tuple[str, list[str]]]:
    if method == "3dgs":
        train = ["python", "train.py", "-s", str(dataset), "-m", str(model_base), "--eval", "--seed", str(seed), "--iterations", str(iterations)]
        return [("train", ["/usr/bin/time", "-v", *train]), ("render", ["python", "render.py", "-m", str(model_base), "--skip_train"]), ("metrics", ["python", "metrics.py", "-m", str(model_base)])]
    if method == "ges":
        train = ["python", "train_ges.py", "-s", str(dataset), "-m", str(model_base), "--eval", "--seed", str(seed), "--iterations", str(iterations)]
        return [("train", ["/usr/bin/time", "-v", *train])]
    if method == "drk":
        common = ["-s", str(dataset), "-m", str(model_base), "--eval", "--gs_type", "DRK", "--kernel_density", "dense", "--cache_sort", "--is_unbounded", "--seed", str(seed), "--iterations", str(iterations)]
        return [("train", ["/usr/bin/time", "-v", "python", "train.py", *common]), ("metrics", ["python", "train.py", *common, "--metric", "--load_iteration", "-1"])]
    raise ValueError(method)


def discover_model_path(method: str, model_base: Path, before: set[Path]) -> Path:
    if method == "3dgs":
        candidate = model_base
    elif method == "drk":
        candidate = model_base.with_name(model_base.name + "_DRK")
    else:
        candidates = sorted((path for path in model_base.parent.glob(model_base.name + "_*") if path.is_dir() and path not in before), key=lambda path: path.stat().st_mtime)
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one new timestamped GES directory for {model_base}, found {candidates}")
        candidate = candidates[0]
    if not candidate.is_dir():
        raise FileNotFoundError(f"Expected model directory was not created: {candidate}")
    return candidate


def run_stage(name: str, command: list[str], cwd: Path, env: dict[str, str], status: dict, status_path: Path) -> None:
    record = {"stage": name, "command": command, "state": "running", "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    status["commands"].append(record)
    atomic_json(status_path, status)
    started = time.monotonic()
    completed = subprocess.run(command, cwd=cwd, env=env, check=False)
    record.update({"returncode": completed.returncode, "elapsed_seconds": time.monotonic() - started, "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "state": "complete" if completed.returncode == 0 else "failed"})
    atomic_json(status_path, status)
    if completed.returncode:
        raise RuntimeError(f"{name} failed with exit status {completed.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and officially evaluate one independent-dataset method/scene run.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--method", choices=["3dgs", "ges", "drk"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.scene not in config["scenes"]:
        raise ValueError(f"Unknown scene {args.scene}; choices={list(config['scenes'])}")
    seed = int(config["seed"])
    method_config = config["methods"][args.method]
    dataset = resolve(config, config["scenes"][args.scene]["dataset"])
    method_root = resolve(config, method_config["root"])
    manifest_path = resolve(config, config["result_root"]) / "dataset_manifest.json"
    if not args.dry_run:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Strict dataset audit manifest is required before training: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        scene_manifest = next((row for row in manifest["scenes"] if row["scene"] == args.scene), None)
        loader_manifest = next((row for row in manifest["loader_audit"] if row["method"] == args.method), None)
        if scene_manifest is None or scene_manifest.get("status") != "ready":
            raise RuntimeError(f"Dataset audit did not mark {args.scene} ready")
        if loader_manifest is None or loader_manifest.get("status") != "verified":
            raise RuntimeError(f"Loader audit did not verify {args.method}")
        method_split_hashes = scene_manifest.get("method_test_split_sha256", {})
        if set(method_split_hashes) != set(config["methods"]) or len(set(method_split_hashes.values())) != 1:
            raise RuntimeError(f"Held-out camera identities are not identical across methods: {method_split_hashes}")
    seed_root = resolve(config, config["output_root"]) / args.scene / args.method / f"seed_{seed}"
    status_path = seed_root / "run_status.json"
    if status_path.exists():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        if previous.get("state") == "complete":
            print(f"Already complete: {status_path}")
            return 0
        if not args.retry and not args.dry_run:
            raise RuntimeError(f"Existing run is {previous.get('state')}; pass --retry to preserve it and create a new attempt")
    attempts = sorted(seed_root.glob("attempt_*")) if seed_root.exists() else []
    attempt_root = seed_root / f"attempt_{len(attempts) + 1:03d}"
    model_base = attempt_root / "model"
    iterations = int(method_config["iterations"])
    plan = command_plan(args.method, method_root, dataset, model_base, seed, iterations)
    if args.dry_run:
        print(json.dumps({"scene": args.scene, "method": args.method, "dataset": str(dataset), "cwd": str(method_root), "commands": plan}, indent=2))
        return 0
    if not dataset.is_dir() or not method_root.is_dir():
        raise FileNotFoundError(f"Missing dataset or upstream checkout: {dataset}, {method_root}")
    actual_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=method_root, text=True).strip()
    if actual_commit != method_config["commit"]:
        raise RuntimeError(f"Wrong {args.method} commit: expected {method_config['commit']}, found {actual_commit}")
    attempt_root.mkdir(parents=True, exist_ok=False)
    status = {
        "experiment": config["experiment"], "scene": args.scene, "method": args.method,
        "seed": seed, "iterations": iterations, "state": "running", "dataset": str(dataset),
        "method_root": str(method_root), "upstream_commit": actual_commit,
        "model_base": str(model_base), "attempt": len(attempts) + 1, "commands": [],
        "dataset_manifest": str(manifest_path),
        "train_split_sha256": scene_manifest["train_split_sha256"],
        "test_split_sha256": scene_manifest["test_split_sha256"],
        "train_view_count": scene_manifest["train_view_count"],
        "test_view_count": scene_manifest["test_view_count"],
        "audited_camera_geometry": scene_manifest["method_camera_geometry"][args.method],
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    atomic_json(status_path, status)
    env = os.environ.copy()
    env.update({"PYTHONHASHSEED": str(seed), "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    before = set(model_base.parent.glob(model_base.name + "_*"))
    try:
        run_stage("train", plan[0][1], method_root, env, status, status_path)
        model_path = discover_model_path(args.method, model_base, before)
        status["model_path"] = str(model_path)
        atomic_json(status_path, status)
        remaining = plan[1:]
        if args.method == "ges":
            remaining = [("render", ["python", "render.py", "-m", str(model_path), "--skip_train"]), ("metrics", ["python", "metrics.py", "-m", str(model_path)])]
        for name, command in remaining:
            run_stage(name, command, method_root, env, status, status_path)
        status.update({"state": "complete", "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
        atomic_json(status_path, status)
    except BaseException as error:
        status.update({"state": "failed", "error": f"{type(error).__name__}: {error}", "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
        atomic_json(status_path, status)
        raise
    print(f"Complete: {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
