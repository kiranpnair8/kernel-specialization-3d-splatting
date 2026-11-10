from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys

from common import atomic_json, load_config, resolve_project_path


def command_plan(method: str, method_root: Path, dataset: Path, model_base: Path, seed: int) -> list[tuple[str, list[str]]]:
    if method == "3dgs":
        train = ["python", "train.py", "-s", str(dataset), "-m", str(model_base), "--eval", "--seed", str(seed)]
        evaluate = [
            ("render", ["python", "render.py", "-m", str(model_base), "--skip_train"]),
            ("metrics", ["python", "metrics.py", "-m", str(model_base)]),
        ]
    elif method == "ges":
        train = ["python", "train_ges.py", "-s", str(dataset), "-m", str(model_base), "--eval", "--seed", str(seed)]
        evaluate = []  # Filled after the timestamped model directory is discovered.
    elif method == "drk":
        common = ["-s", str(dataset), "-m", str(model_base), "--eval", "--gs_type", "DRK", "--kernel_density", "dense", "--cache_sort", "--is_unbounded", "--seed", str(seed)]
        train = ["python", "train.py", *common]
        evaluate = [("metrics", ["python", "train.py", *common, "--metric", "--load_iteration", "-1"])]
    else:
        raise ValueError(method)
    return [("train", ["/usr/bin/time", "-v", *train]), *evaluate]


def discover_model_path(method: str, model_base: Path, before: set[Path]) -> Path:
    if method == "3dgs":
        candidate = model_base
    elif method == "drk":
        candidate = model_base.with_name(model_base.name + "_DRK")
    else:
        candidates = sorted(
            [path for path in model_base.parent.glob(model_base.name + "_*") if path.is_dir() and path not in before],
            key=lambda path: path.stat().st_mtime,
        )
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one new GES model directory for {model_base}, found {candidates}")
        candidate = candidates[0]
    if not candidate.is_dir():
        raise FileNotFoundError(f"Training did not create expected model directory: {candidate}")
    return candidate


def run_command(name: str, command: list[str], cwd: Path, env: dict[str, str], status: dict, status_path: Path) -> None:
    record = {"stage": name, "command": command, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "state": "running"}
    status["commands"].append(record)
    atomic_json(status_path, status)
    completed = subprocess.run(command, cwd=cwd, env=env, check=False)
    record["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    record["returncode"] = completed.returncode
    record["state"] = "complete" if completed.returncode == 0 else "failed"
    atomic_json(status_path, status)
    if completed.returncode:
        raise RuntimeError(f"{name} failed with exit status {completed.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one isolated native-protocol Room seed and its official evaluation.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=["3dgs", "ges", "drk"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry", action="store_true", help="Create a new attempt after a failed/interrupted attempt; old attempts are preserved.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed not in [int(value) for value in config["seeds"]]:
        raise ValueError(f"Seed {args.seed} is not in configured seeds {config['seeds']}")
    method_config = config["methods"][args.method]
    project_root = Path(str(config["project_root"]))
    dataset = resolve_project_path(config, str(config["dataset"]))
    method_root = resolve_project_path(config, str(method_config["root"]))
    seed_root = resolve_project_path(config, str(config["output_root"])) / args.method / f"seed_{args.seed}"
    status_path = seed_root / "run_status.json"
    previous = None
    if status_path.exists():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        if previous.get("state") == "complete":
            print(f"Already complete: {status_path}")
            return 0
        if not args.retry and not args.dry_run:
            raise RuntimeError(f"Previous attempt is {previous.get('state')}; pass --retry to preserve it and create a new attempt")

    existing_attempts = sorted(seed_root.glob("attempt_*")) if seed_root.exists() else []
    attempt_number = len(existing_attempts) + 1
    attempt_root = seed_root / f"attempt_{attempt_number:03d}"
    model_base = attempt_root / "model"
    plan = command_plan(args.method, method_root, dataset, model_base, args.seed)
    if args.dry_run:
        print(json.dumps({"method": args.method, "seed": args.seed, "cwd": str(method_root), "model_base": str(model_base), "commands": plan}, indent=2))
        return 0

    if not dataset.is_dir() or not method_root.is_dir():
        raise FileNotFoundError(f"Missing dataset or method checkout: dataset={dataset}, method_root={method_root}")
    actual_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=method_root, text=True).strip()
    if actual_commit != str(method_config["commit"]):
        raise RuntimeError(f"Wrong {args.method} upstream commit: expected {method_config['commit']}, found {actual_commit}")
    attempt_root.mkdir(parents=True, exist_ok=False)
    status = {
        "experiment": config["experiment"],
        "method": args.method,
        "seed": args.seed,
        "attempt": attempt_number,
        "state": "running",
        "dataset": str(dataset),
        "method_root": str(method_root),
        "expected_upstream_commit": method_config["commit"],
        "actual_upstream_commit": actual_commit,
        "model_base": str(model_base),
        "commands": [],
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    atomic_json(status_path, status)
    env = os.environ.copy()
    env.update({"PYTHONHASHSEED": str(args.seed), "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    before = set(model_base.parent.glob(model_base.name + "_*"))
    try:
        run_command("train", plan[0][1], method_root, env, status, status_path)
        model_path = discover_model_path(args.method, model_base, before)
        status["model_path"] = str(model_path)
        atomic_json(status_path, status)
        if args.method == "ges":
            evaluations = [
                ("render", ["python", "render.py", "-m", str(model_path), "--skip_train"]),
                ("metrics", ["python", "metrics.py", "-m", str(model_path)]),
            ]
        else:
            evaluations = plan[1:]
        for name, command in evaluations:
            run_command(name, command, method_root, env, status, status_path)
        status["state"] = "complete"
        status["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_json(status_path, status)
    except BaseException as error:
        status["state"] = "failed"
        status["error"] = f"{type(error).__name__}: {error}"
        status["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_json(status_path, status)
        raise
    print(f"Complete: {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
