from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional

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


def _stride_for_patch_size(patch_size: int, base_stride: int, policy: str) -> int:
    if policy == "fixed_config":
        return base_stride
    if policy == "half_patch":
        return max(1, patch_size // 2)
    raise ValueError(f"Unknown sensitivity stride policy: {policy}")


def _flatten_summary(
    patch_size: int,
    stride: int,
    tie_threshold: float,
    summary: Dict[str, object],
) -> Dict[str, object]:
    oracle = summary["oracle"]
    predictors = summary["predictors"]
    methods = list(summary["config"].get("methods", []))
    row: Dict[str, object] = {
        "patch_size": patch_size,
        "stride": stride,
        "tie_threshold_mse": tie_threshold,
        "patch_count": oracle.get("patch_count"),
        "decisive_patch_count": oracle.get("decisive_patch_count"),
        "oracle_patch_mse": oracle.get("oracle_patch_mse"),
        "oracle_patch_psnr": oracle.get("oracle_patch_psnr"),
        "tie_fraction": oracle.get("tie_fraction"),
        "predictor_status": predictors.get("status"),
    }

    non_3dgs_winner_fraction = 0.0
    for method in methods:
        row[f"{method}_winner_fraction"] = oracle.get(f"{method}_winner_fraction")
        row[f"{method}_patch_mse"] = oracle.get(f"{method}_patch_mse")
        row[f"{method}_patch_psnr"] = oracle.get(f"{method}_patch_psnr")
        improvement = oracle.get(f"oracle_improvement_mse_vs_{method}")
        row[f"oracle_improvement_mse_vs_{method}"] = improvement
        method_mse = oracle.get(f"{method}_patch_mse")
        if improvement is not None and method_mse:
            row[f"oracle_relative_improvement_pct_vs_{method}"] = 100.0 * float(improvement) / float(method_mse)
        if method != "3dgs":
            non_3dgs_winner_fraction += float(oracle.get(f"{method}_winner_fraction", 0.0) or 0.0)
    row["non_3dgs_winner_fraction"] = non_3dgs_winner_fraction

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
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def _write_aggregate_outputs(sensitivity_dir: Path, rows: List[Dict[str, object]], summaries: List[Dict[str, object]], policy: str) -> None:
    _write_csv(sensitivity_dir / "sensitivity_summary.csv", rows)
    json_path = sensitivity_dir / "sensitivity_summary.json"
    temp_json_path = json_path.with_suffix(json_path.suffix + ".tmp")
    with temp_json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "stride_policy": policy,
                "runs": summaries,
                "summary_rows": rows,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    temp_json_path.replace(json_path)


def run_sensitivity(
    config_path: Path,
    patch_sizes: List[int],
    tie_thresholds: List[float],
    output_dir: Optional[Path],
    stride_policy: Optional[str],
) -> None:
    base_config = load_config(config_path)
    base_stride = int(base_config.get("stride", int(base_config.get("patch_size", 64))))
    policy = stride_policy or str(base_config.get("sensitivity_stride_policy", "fixed_config"))
    base_results_dir = resolve_path(config_path, str(base_config["results_dir"]))
    sensitivity_dir = output_dir or (base_results_dir / "sensitivity")
    rows: List[Dict[str, object]] = []
    summaries: List[Dict[str, object]] = []
    total_conditions = len(patch_sizes) * len(tie_thresholds)

    for condition_index, (patch_size, tie_threshold) in enumerate(
        ((patch_size, tie_threshold) for patch_size in patch_sizes for tie_threshold in tie_thresholds),
        start=1,
    ):
        stride = _stride_for_patch_size(patch_size, base_stride, policy)
        config = deepcopy(base_config)
        config["patch_size"] = patch_size
        config["stride"] = stride
        config["tie_threshold_mse"] = tie_threshold
        print(
            f"running sensitivity condition {condition_index}/{total_conditions}: "
            f"patch_size={patch_size}, stride={stride}, "
            f"tie_threshold_mse={tie_threshold}, stride_policy={policy}"
        )
        summary = analyze(
            config_path,
            config,
            inspect_only=False,
            generate_maps=False,
            write_outputs=False,
            run_predictors=False,
        )
        if summary is None:
            raise RuntimeError("Sensitivity run unexpectedly returned no summary")
        rows.append(_flatten_summary(patch_size, stride, tie_threshold, summary))
        summaries.append(
            {
                "patch_size": patch_size,
                "stride": stride,
                "tie_threshold_mse": tie_threshold,
                "stride_policy": policy,
                "summary": summary,
            }
        )
        _write_aggregate_outputs(sensitivity_dir, rows, summaries, policy)
        print(f"completed sensitivity condition {condition_index}/{total_conditions}")

    print(f"wrote: {sensitivity_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local comparison sensitivity grid.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--patch-sizes", default="32,64,128")
    parser.add_argument("--tie-thresholds", default="0,1e-5,5e-5")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--stride-policy",
        choices=("fixed_config", "half_patch"),
        default=None,
        help=(
            "Stride policy for sensitivity runs. Defaults to the config's "
            "sensitivity_stride_policy, or fixed_config when unspecified."
        ),
    )
    args = parser.parse_args()

    try:
        run_sensitivity(
            args.config.resolve(),
            _parse_ints(args.patch_sizes),
            _parse_floats(args.tie_thresholds),
            args.output_dir.resolve() if args.output_dir else None,
            args.stride_policy,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
