from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
import sys
from typing import Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate.local_compare import analyze, load_config, resolve_path


def _parse_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_floats(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _tie_label(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:.0e}".replace("-", "m")


def _flatten_summary(patch_size: int, tie_threshold: float, summary: Dict[str, object]) -> Dict[str, object]:
    oracle = summary["oracle"]
    predictors = summary["predictors"]
    row: Dict[str, object] = {
        "patch_size": patch_size,
        "tie_threshold_mse": tie_threshold,
        "patch_count": oracle.get("patch_count"),
        "decisive_patch_count": oracle.get("decisive_patch_count"),
        "oracle_patch_mse": oracle.get("oracle_patch_mse"),
        "oracle_patch_psnr": oracle.get("oracle_patch_psnr"),
        "tie_fraction": oracle.get("tie_fraction"),
        "predictor_status": predictors.get("status"),
    }

    for key, value in oracle.items():
        if key.endswith("_winner_fraction") or key.startswith("oracle_improvement_mse_vs_"):
            row[key] = value

    for model in ("logistic_regression", "random_forest"):
        for metric in (
            "view_cv_accuracy_mean",
            "view_cv_accuracy_std",
            "view_cv_balanced_accuracy_mean",
            "view_cv_auroc",
            "view_cv_folds",
        ):
            key = f"{model}_{metric}"
            if key in predictors:
                row[key] = predictors[key]
        status_key = f"{model}_status"
        if status_key in predictors:
            row[status_key] = predictors[status_key]

    return row


def _write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
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


def run_sensitivity(
    config_path: Path,
    patch_sizes: List[int],
    tie_thresholds: List[float],
    output_dir: Path | None,
) -> None:
    base_config = load_config(config_path)
    base_results_dir = resolve_path(config_path, str(base_config["results_dir"]))
    sensitivity_dir = output_dir or (base_results_dir / "sensitivity")
    rows: List[Dict[str, object]] = []
    summaries: List[Dict[str, object]] = []

    for patch_size in patch_sizes:
        for tie_threshold in tie_thresholds:
            config = deepcopy(base_config)
            config["patch_size"] = patch_size
            config["tie_threshold_mse"] = tie_threshold
            config["results_dir"] = str(
                sensitivity_dir / "runs" / f"patch_{patch_size}" / f"tie_{_tie_label(tie_threshold)}"
            )
            print(f"running sensitivity: patch_size={patch_size}, tie_threshold_mse={tie_threshold}")
            summary = analyze(config_path, config, inspect_only=False)
            if summary is None:
                raise RuntimeError("Sensitivity run unexpectedly returned no summary")
            rows.append(_flatten_summary(patch_size, tie_threshold, summary))
            summaries.append(
                {
                    "patch_size": patch_size,
                    "tie_threshold_mse": tie_threshold,
                    "results_dir": config["results_dir"],
                    "summary": summary,
                }
            )

    _write_csv(sensitivity_dir / "sensitivity_summary.csv", rows)
    with (sensitivity_dir / "sensitivity_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"runs": summaries, "summary_rows": rows}, handle, indent=2, sort_keys=True)
    print(f"wrote: {sensitivity_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local comparison sensitivity grid.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--patch-sizes", default="32,64,128")
    parser.add_argument("--tie-thresholds", default="0,1e-5,5e-5")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    try:
        run_sensitivity(
            args.config.resolve(),
            _parse_ints(args.patch_sizes),
            _parse_floats(args.tie_thresholds),
            args.output_dir.resolve() if args.output_dir else None,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
