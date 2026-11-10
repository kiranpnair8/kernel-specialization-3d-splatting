from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_project_path(config: Mapping[str, object], value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    configured_root = Path(str(config.get("project_root", REPO_ROOT)))
    return configured_root / path


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
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


def winner_labels(errors: np.ndarray, labels: Sequence[str], tie_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply local_compare.py's exact best/second-best/tie semantics."""
    values = np.asarray(errors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(labels) or len(labels) < 2:
        raise ValueError("errors must have shape [patches, >=2 labels]")
    order = np.argsort(values, axis=1, kind="stable")
    margin = values[np.arange(len(values)), order[:, 1]] - values[np.arange(len(values)), order[:, 0]]
    winners = np.asarray(labels, dtype=object)[order[:, 0]]
    winners = winners.astype(object)
    winners[margin < tie_threshold] = "tie"
    return winners, margin


def fractions(winners: np.ndarray, labels: Sequence[str]) -> dict[str, float]:
    total = len(winners)
    if total == 0:
        raise ValueError("No winner labels")
    result = {f"{label}_winner_fraction": float(np.count_nonzero(winners == label) / total) for label in labels}
    result["tie_fraction"] = float(np.count_nonzero(winners == "tie") / total)
    if not math.isclose(sum(result.values()), 1.0, abs_tol=1e-12):
        raise AssertionError("Winner and tie fractions do not sum to one")
    return result


def psnr_from_mse(value: float) -> float:
    return float("inf") if value <= 1e-12 else float(-10.0 * math.log10(value))


def oracle_metrics(errors: np.ndarray, reference_indices: Iterable[int]) -> dict[str, object]:
    values = np.asarray(errors, dtype=np.float64)
    oracle_per_patch = values.min(axis=1)
    oracle_mse = float(oracle_per_patch.mean())
    references = []
    for index in reference_indices:
        reference_mse = float(values[:, index].mean())
        if oracle_mse > reference_mse + 1e-15:
            raise AssertionError("Oracle MSE exceeds reference MSE")
        references.append(
            {
                "reference_index": int(index),
                "reference_mse": reference_mse,
                "headroom_fraction": float((reference_mse - oracle_mse) / reference_mse),
            }
        )
    return {"oracle_mse": oracle_mse, "oracle_psnr": psnr_from_mse(oracle_mse), "references": references}


def cluster_bootstrap(values_by_view: Mapping[str, np.ndarray], statistic, replicates: int, seed: int) -> tuple[float, float]:
    views = sorted(values_by_view)
    if len(views) < 2:
        raise ValueError("At least two views are required for clustered bootstrap")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.choice(views, size=len(views), replace=True)
        estimates[index] = statistic(np.concatenate([values_by_view[view] for view in sampled]))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def load_error_bundle(path: Path) -> tuple[np.ndarray, list[str], np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as bundle:
        errors = np.asarray(bundle["errors"], dtype=np.float64)
        run_ids = [str(value) for value in bundle["run_ids"]]
        views = np.asarray(bundle["views"]).astype(str)
    metadata_path = path.with_suffix(".json")
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if errors.shape != (len(views), len(run_ids)):
        raise ValueError("patch_errors.npz dimensions are inconsistent")
    return errors, run_ids, views, metadata


def summarize(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }
