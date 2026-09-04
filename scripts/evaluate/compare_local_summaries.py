from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def load_summary(results_dir: Path, required: bool) -> Optional[Dict[str, Any]]:
    path = results_dir / "summary.json"
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required summary is missing: {path}")
        print(f"warning: optional summary is missing: {path}")
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def config_methods(summary: Dict[str, Any]) -> List[str]:
    methods = summary.get("config", {}).get("methods")
    if not isinstance(methods, list) or not methods:
        raise ValueError("summary.json is missing config.methods")
    return [str(method) for method in methods]


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def flatten_summary(label: str, results_dir: Path, summary: Dict[str, Any]) -> Dict[str, Any]:
    methods = config_methods(summary)
    oracle = summary.get("oracle")
    if not isinstance(oracle, dict):
        raise ValueError(f"{results_dir}/summary.json is missing oracle summary")

    row: Dict[str, Any] = {
        "label": label,
        "results_dir": str(results_dir),
        "scene": summary.get("config", {}).get("scene"),
        "patch_size": summary.get("config", {}).get("patch_size"),
        "stride": summary.get("config", {}).get("stride"),
        "tie_threshold_mse": summary.get("config", {}).get("tie_threshold_mse"),
        "patch_count": oracle.get("patch_count"),
        "decisive_patch_count": oracle.get("decisive_patch_count"),
        "tie_fraction": oracle.get("tie_fraction"),
        "oracle_patch_mse": oracle.get("oracle_patch_mse"),
        "oracle_patch_psnr": oracle.get("oracle_patch_psnr"),
    }

    non_3dgs_winner_fraction = 0.0
    for method in methods:
        winner_fraction = safe_float(oracle.get(f"{method}_winner_fraction"))
        method_mse = safe_float(oracle.get(f"{method}_patch_mse"))
        improvement = safe_float(oracle.get(f"oracle_improvement_mse_vs_{method}"))
        row[f"{method}_winner_fraction"] = winner_fraction
        row[f"{method}_patch_mse"] = method_mse
        row[f"{method}_patch_psnr"] = oracle.get(f"{method}_patch_psnr")
        row[f"oracle_improvement_mse_vs_{method}"] = improvement
        if method_mse and improvement is not None:
            row[f"oracle_relative_improvement_pct_vs_{method}"] = 100.0 * improvement / method_mse
        else:
            row[f"oracle_relative_improvement_pct_vs_{method}"] = None
        if method != "3dgs" and winner_fraction is not None:
            non_3dgs_winner_fraction += winner_fraction
    row["non_3dgs_winner_fraction"] = non_3dgs_winner_fraction
    return row


def ordered_fieldnames(rows: Iterable[Dict[str, Any]]) -> List[str]:
    preferred = [
        "label",
        "results_dir",
        "scene",
        "patch_size",
        "stride",
        "tie_threshold_mse",
        "patch_count",
        "decisive_patch_count",
        "tie_fraction",
        "3dgs_winner_fraction",
        "ges_winner_fraction",
        "drk_winner_fraction",
        "non_3dgs_winner_fraction",
        "3dgs_patch_mse",
        "ges_patch_mse",
        "drk_patch_mse",
        "oracle_patch_mse",
        "oracle_patch_psnr",
        "oracle_improvement_mse_vs_3dgs",
        "oracle_improvement_mse_vs_ges",
        "oracle_improvement_mse_vs_drk",
        "oracle_relative_improvement_pct_vs_3dgs",
        "oracle_relative_improvement_pct_vs_ges",
        "oracle_relative_improvement_pct_vs_drk",
    ]
    seen = set()
    fields: List[str] = []
    for field in preferred:
        if any(field in row for row in rows):
            fields.append(field)
            seen.add(field)
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    return fields


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_deltas(baseline: Optional[Dict[str, Any]], candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    if baseline is None:
        return []
    deltas: List[Dict[str, Any]] = []
    for metric, candidate_value in candidate.items():
        if metric in {"label", "results_dir", "scene"}:
            continue
        baseline_value = baseline.get(metric)
        try:
            base = float(baseline_value)
            cand = float(candidate_value)
        except (TypeError, ValueError):
            continue
        delta = cand - base
        rel = None if base == 0.0 else 100.0 * delta / base
        deltas.append(
            {
                "metric": metric,
                "baseline": base,
                "candidate": cand,
                "delta_candidate_minus_baseline": delta,
                "relative_delta_pct": rel,
            }
        )
    return deltas


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two local-comparison summary.json files.")
    parser.add_argument("--baseline-results", required=True, type=Path)
    parser.add_argument("--candidate-results", required=True, type=Path)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    baseline_summary = load_summary(args.baseline_results, required=False)
    candidate_summary = load_summary(args.candidate_results, required=True)

    records: List[Dict[str, Any]] = []
    baseline_record: Optional[Dict[str, Any]] = None
    if baseline_summary is not None:
        baseline_record = flatten_summary(args.baseline_label, args.baseline_results, baseline_summary)
        records.append(baseline_record)
    candidate_record = flatten_summary(args.candidate_label, args.candidate_results, candidate_summary)
    records.append(candidate_record)

    deltas = build_deltas(baseline_record, candidate_record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_dir / "baseline_vs_budget250k_summary.csv"
    deltas_csv = args.output_dir / "baseline_vs_budget250k_deltas.csv"
    summary_json = args.output_dir / "baseline_vs_budget250k_summary.json"

    write_csv(summary_csv, records, ordered_fieldnames(records))
    if deltas:
        write_csv(deltas_csv, deltas, list(deltas[0].keys()))
    else:
        write_csv(deltas_csv, [], ["metric", "baseline", "candidate", "delta_candidate_minus_baseline", "relative_delta_pct"])

    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "baseline_results": str(args.baseline_results),
                "candidate_results": str(args.candidate_results),
                "baseline_available": baseline_record is not None,
                "records": records,
                "deltas": deltas,
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    print(f"wrote: {summary_csv}")
    print(f"wrote: {deltas_csv}")
    print(f"wrote: {summary_json}")


if __name__ == "__main__":
    main()
