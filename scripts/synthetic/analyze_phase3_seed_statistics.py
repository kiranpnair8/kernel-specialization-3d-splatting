#!/usr/bin/env python3
"""Seed-level Phase III-B statistics for synthetic results.

Independent synthetic scene realizations are the primary experimental units here.
Per-view bootstrap intervals from evaluate_phase3_results.py remain secondary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

METRICS = ("psnr", "ssim", "lpips", "primitive_count")
METHODS = ("3dgs", "ges", "drk")
PAIRWISE = (("ges", "3dgs"), ("drk", "3dgs"), ("drk", "ges"))
LEVEL_ORDER = {"low": 0, "medium": 1, "high": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/synthetic/phase3_controlled_pilot/evaluation"))
    parser.add_argument("--input", type=Path, default=None, help="Defaults to <results-dir>/phase3_results.csv")
    parser.add_argument("--allow-missing-seeds", action="store_true", help="Write available seed statistics before all five seeds finish.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def numeric(row: Mapping[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", "None", None):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def seed_value(row: Mapping[str, str]) -> int:
    return int(float(row.get("seed", "0") or 0))


def mean_sd(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
    mean = sum(values) / len(values)
    if len(values) > 1:
        var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        sd = math.sqrt(var)
    else:
        sd = 0.0
    return {"n": len(values), "mean": mean, "sd": sd, "min": min(values), "max": max(values)}


def condition_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["sweep_family"], row["level"], row["method"])].append(row)

    output: list[dict[str, Any]] = []
    for (family, level, method), items in sorted(grouped.items(), key=lambda x: (x[0][0], LEVEL_ORDER.get(x[0][1], 99), x[0][2])):
        seeds = sorted({seed_value(row) for row in items})
        record: dict[str, Any] = {
            "sweep_family": family,
            "level": level,
            "method": method,
            "seed_count": len(seeds),
            "seeds": ",".join(str(seed) for seed in seeds),
        }
        for metric in METRICS:
            stats = mean_sd([value for row in items if (value := numeric(row, metric)) is not None])
            for key, value in stats.items():
                record[f"{metric}_{key}"] = value
        output.append(record)
    return output


def paired_seed_delta_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key = {(row["sweep_family"], row["level"], seed_value(row), row["method"]): row for row in rows}
    deltas: list[dict[str, Any]] = []
    for family in sorted({row["sweep_family"] for row in rows}):
        for level in sorted({row["level"] for row in rows if row["sweep_family"] == family}, key=lambda item: LEVEL_ORDER.get(item, 99)):
            seeds = sorted({seed_value(row) for row in rows if row["sweep_family"] == family and row["level"] == level})
            for seed in seeds:
                for left, right in PAIRWISE:
                    left_row = by_key.get((family, level, seed, left))
                    right_row = by_key.get((family, level, seed, right))
                    if left_row is None or right_row is None:
                        continue
                    item: dict[str, Any] = {
                        "sweep_family": family,
                        "level": level,
                        "seed": seed,
                        "comparison": f"{left}_minus_{right}",
                    }
                    for metric in METRICS:
                        left_value = numeric(left_row, metric)
                        right_value = numeric(right_row, metric)
                        item[f"delta_{metric}"] = None if left_value is None or right_value is None else left_value - right_value
                    deltas.append(item)

    summary_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in deltas:
        summary_groups[(row["sweep_family"], row["level"], row["comparison"])].append(row)
    summaries: list[dict[str, Any]] = []
    for (family, level, comparison), items in sorted(summary_groups.items(), key=lambda x: (x[0][0], LEVEL_ORDER.get(x[0][1], 99), x[0][2])):
        seeds = sorted({int(row["seed"]) for row in items})
        record: dict[str, Any] = {
            "sweep_family": family,
            "level": level,
            "comparison": comparison,
            "seed_count": len(seeds),
            "seeds": ",".join(str(seed) for seed in seeds),
        }
        for metric in METRICS:
            values = [float(row[f"delta_{metric}"]) for row in items if row.get(f"delta_{metric}") is not None]
            stats = mean_sd(values)
            for key, value in stats.items():
                record[f"delta_{metric}_{key}"] = value
        summaries.append(record)
    return deltas, summaries


def write_findings(path: Path, rows: list[dict[str, str]], condition_rows: list[dict[str, Any]], delta_summary_rows: list[dict[str, Any]], complete: bool) -> None:
    seeds = sorted({seed_value(row) for row in rows})
    lines = [
        "# Phase III-B Seed-Level Statistics",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Independent scene seeds are the primary experimental units for Phase III-B. View-level bootstrap intervals remain secondary diagnostics and should not be treated as independent replication.",
        "",
        "## Coverage",
        "",
        f"- Seeds present: {', '.join(str(seed) for seed in seeds)}",
        f"- Method-scene records: {len(rows)}",
        f"- Five-seed complete: {complete}",
        "",
        "## Outputs",
        "",
        "- `phase3_seed_condition_summary.csv/json`: per condition and method mean, SD, min, and max across seeds.",
        "- `phase3_seed_paired_deltas.csv/json`: matched-seed method deltas for each condition.",
        "- `phase3_seed_paired_delta_summary.csv/json`: mean and SD of paired deltas across seeds.",
        "",
        f"Condition summary rows: {len(condition_rows)}.",
        f"Paired delta summary rows: {len(delta_summary_rows)}.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir
    input_path = args.input or (results_dir / "phase3_results.csv")
    rows = read_csv(input_path)
    expected = 5 * 9 * len(METHODS)
    complete = len(rows) >= expected and {0, 1, 2, 3, 4}.issubset({seed_value(row) for row in rows})
    if not complete and not args.allow_missing_seeds:
        raise RuntimeError(
            f"Phase III-B seed statistics require five-seed results unless --allow-missing-seeds is set. "
            f"Found {len(rows)} records; expected at least {expected}."
        )

    condition_rows = condition_summary(rows)
    delta_rows, delta_summary_rows = paired_seed_delta_rows(rows)
    write_csv(results_dir / "phase3_seed_condition_summary.csv", condition_rows)
    write_json(results_dir / "phase3_seed_condition_summary.json", condition_rows)
    write_csv(results_dir / "phase3_seed_paired_deltas.csv", delta_rows)
    write_json(results_dir / "phase3_seed_paired_deltas.json", delta_rows)
    write_csv(results_dir / "phase3_seed_paired_delta_summary.csv", delta_summary_rows)
    write_json(results_dir / "phase3_seed_paired_delta_summary.json", delta_summary_rows)
    write_findings(results_dir / "phase3_seed_findings.md", rows, condition_rows, delta_summary_rows, complete)
    print(f"Phase III-B seed statistics complete: {len(rows)} records; five-seed complete={complete}")
    print(f"Wrote {results_dir / 'phase3_seed_condition_summary.csv'}")
    print(f"Wrote {results_dir / 'phase3_seed_paired_delta_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
