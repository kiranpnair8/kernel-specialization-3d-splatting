from __future__ import annotations

import csv
import itertools
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(config: Mapping[str, object], value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(str(config.get("project_root", REPO_ROOT))) / path


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def psnr_from_mse(value: float) -> float:
    return float("inf") if value <= 1e-12 else float(-10.0 * math.log10(value))


def expected_run_ids(seeds: Sequence[int] = (0, 1, 2)) -> list[str]:
    return [f"3dgs_s{seed}" for seed in seeds] + [f"ges_s{seed}" for seed in seeds]


def enumerate_pairs(run_ids: Sequence[str]) -> list[dict[str, object]]:
    expected = expected_run_ids()
    if list(run_ids) != expected:
        raise ValueError(f"Expected run order {expected}, found {list(run_ids)}")
    pairs = []
    for family in ("3dgs", "ges"):
        members = [run_id for run_id in run_ids if run_id.startswith(family + "_")]
        for left, right in itertools.combinations(members, 2):
            pairs.append({"group": f"within_{family}", "left": left, "right": right})
    for left in [value for value in run_ids if value.startswith("3dgs_")]:
        for right in [value for value in run_ids if value.startswith("ges_")]:
            pairs.append({"group": "cross_family", "left": left, "right": right})
    if len(pairs) != 15 or sum(pair["group"] == "cross_family" for pair in pairs) != 9:
        raise AssertionError(f"Incorrect pair enumeration: {pairs}")
    return pairs


def pair_metrics(values: np.ndarray, left_label: str, right_label: str, tie_threshold: float) -> dict[str, float | int | str]:
    errors = np.asarray(values, dtype=np.float64)
    if errors.ndim != 2 or errors.shape[1] != 2 or not np.isfinite(errors).all():
        raise ValueError("Pair errors must be finite [patch,2] values")
    left_mse, right_mse = (float(errors[:, index].mean()) for index in range(2))
    oracle = errors.min(axis=1)
    oracle_mse = float(oracle.mean())
    better_label, better_mse = (left_label, left_mse) if left_mse <= right_mse else (right_label, right_mse)
    margin = np.abs(errors[:, 0] - errors[:, 1])
    tie = margin < tie_threshold
    # Match local_compare.py's stable argsort semantics: an exact equality at
    # threshold zero is assigned to the left/first model rather than to tie.
    left_wins = (errors[:, 0] <= errors[:, 1]) & ~tie
    right_wins = (errors[:, 1] < errors[:, 0]) & ~tie
    return {
        "patch_count": len(errors), "left_model": left_label, "right_model": right_label,
        "left_mse": left_mse, "right_mse": right_mse,
        "better_single_model": better_label, "better_single_mse": better_mse,
        "mean_single_mse": 0.5 * (left_mse + right_mse),
        "oracle_mse": oracle_mse, "oracle_psnr": psnr_from_mse(oracle_mse),
        "oracle_gain_vs_left_fraction": (left_mse - oracle_mse) / left_mse,
        "oracle_gain_vs_right_fraction": (right_mse - oracle_mse) / right_mse,
        "oracle_gain_vs_better_single_fraction": (better_mse - oracle_mse) / better_mse,
        "oracle_gain_vs_mean_single_fraction": (0.5 * (left_mse + right_mse) - oracle_mse) / (0.5 * (left_mse + right_mse)),
        "left_winner_fraction": float(left_wins.mean()), "right_winner_fraction": float(right_wins.mean()),
        "tie_fraction": float(tie.mean()),
    }


def load_bundle(path: Path) -> tuple[np.ndarray, list[str], np.ndarray, dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as bundle:
        errors = np.asarray(bundle["errors"], dtype=np.float64)
        run_ids = [str(value) for value in bundle["run_ids"]]
        views = np.asarray(bundle["views"]).astype(str)
        coordinates = {name: np.asarray(bundle[name], dtype=np.int64) for name in ("x", "y", "width", "height")}
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if errors.shape != (len(views), len(run_ids)):
        raise ValueError("Bundle dimensions are inconsistent")
    return errors, run_ids, views, coordinates, metadata
