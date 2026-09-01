#!/usr/bin/env python3
"""Matched seed-level interaction tests for Phase III-B synthetic results.

The independent unit is the synthetic scene realization / seed. Held-out views
are not independent replicates for this analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Mapping

FACTORS = ("edge_sharpness", "spatial_frequency", "curvature")
LEVELS = ("low", "medium", "high")
LEVEL_ORDER = {level: idx for idx, level in enumerate(LEVELS)}
METHODS = ("3dgs", "ges", "drk")
COMPARISONS = (("drk", "3dgs"), ("drk", "ges"), ("ges", "3dgs"))
METRICS = ("psnr", "ssim", "lpips")
PRIMARY_METRIC = "psnr"
EXPECTED_SEEDS = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class DeltaRecord:
    factor: str
    level: str
    seed: int
    comparison: str
    metric_values: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/synthetic/phase3_controlled_pilot/evaluation"))
    parser.add_argument("--phase3-results", type=Path, default=None, help="Defaults to <results-dir>/phase3_results.csv")
    parser.add_argument("--allow-missing-seeds", action="store_true", help="Allow exploratory output before all five seeds are available.")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def as_float(row: Mapping[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", "None", None):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def as_seed(row: Mapping[str, str]) -> int:
    return int(float(row.get("seed", "0") or 0))


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def sd(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    center = sum(values) / len(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def fmt(value: float | None, digits: int = 5) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def rank_abs(values: list[float]) -> list[float]:
    pairs = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and math.isclose(pairs[j][1], pairs[i][1], rel_tol=0.0, abs_tol=1e-15):
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for idx in range(i, j):
            ranks[pairs[idx][0]] = avg_rank
        i = j
    return ranks


def exact_wilcoxon_signed_rank(differences: list[float]) -> tuple[float, float, int]:
    nonzero = [value for value in differences if not math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1e-15)]
    n = len(nonzero)
    if n == 0:
        return 0.0, 1.0, 0
    ranks = rank_abs([abs(value) for value in nonzero])
    observed_w_plus = sum(rank for value, rank in zip(nonzero, ranks) if value > 0.0)
    total_rank = sum(ranks)
    observed_stat = min(observed_w_plus, total_rank - observed_w_plus)
    extreme = 0
    total = 2 ** n
    for signs in product((0, 1), repeat=n):
        w_plus = sum(rank for sign, rank in zip(signs, ranks) if sign)
        stat = min(w_plus, total_rank - w_plus)
        if stat <= observed_stat + 1e-12:
            extreme += 1
    return float(observed_stat), min(1.0, extreme / total), n


def paired_t_test(left: list[float], right: list[float]) -> tuple[float | None, float | None]:
    try:
        from scipy import stats
    except Exception:
        return None, None
    result = stats.ttest_rel(right, left, nan_policy="omit")
    return float(result.statistic), float(result.pvalue)


def friedman_test(level_values: dict[str, dict[int, float]]) -> tuple[float | None, float | None, float | None, int]:
    common_seeds = sorted(set.intersection(*(set(level_values[level]) for level in LEVELS)))
    if len(common_seeds) < 2:
        return None, None, None, len(common_seeds)
    arrays = [[level_values[level][seed] for seed in common_seeds] for level in LEVELS]
    try:
        from scipy import stats

        result = stats.friedmanchisquare(*arrays)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    except Exception:
        statistic = friedman_statistic(arrays)
        p_value = None
    kendall_w = statistic / (len(common_seeds) * (len(LEVELS) - 1)) if statistic is not None else None
    return statistic, p_value, kendall_w, len(common_seeds)


def friedman_statistic(arrays: list[list[float]]) -> float:
    n = len(arrays[0])
    k = len(arrays)
    rank_sums = [0.0] * k
    for seed_idx in range(n):
        values = [arrays[level_idx][seed_idx] for level_idx in range(k)]
        ranks = rank_abs(values)
        for level_idx, rank in enumerate(ranks):
            rank_sums[level_idx] += rank
    return (12.0 / (n * k * (k + 1))) * sum(rank_sum * rank_sum for rank_sum in rank_sums) - 3.0 * n * (k + 1)


def bh_fdr(p_values: list[float | None]) -> list[float | None]:
    indexed = [(idx, p) for idx, p in enumerate(p_values) if p is not None and math.isfinite(p)]
    adjusted: list[float | None] = [None] * len(p_values)
    if not indexed:
        return adjusted
    indexed.sort(key=lambda item: item[1])
    m = len(indexed)
    running = 1.0
    for rank_from_end, (idx, p) in enumerate(reversed(indexed), start=1):
        rank = m - rank_from_end + 1
        running = min(running, p * m / rank)
        adjusted[idx] = min(1.0, running)
    return adjusted


def apply_bh(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        p_value = row.get("p_value")
        if p_value is not None:
            groups[(str(row["test_kind"]), str(row["metric"]))].append(idx)
    for _, indices in groups.items():
        adjusted = bh_fdr([rows[idx].get("p_value") for idx in indices])
        for idx, p_adj in zip(indices, adjusted):
            rows[idx]["p_value_bh_fdr"] = p_adj


def validate_complete(results: list[dict[str, str]]) -> list[str]:
    present = {(row.get("sweep_family"), row.get("level"), as_seed(row), row.get("method")) for row in results}
    missing: list[str] = []
    for factor in FACTORS:
        for level in LEVELS:
            for seed in EXPECTED_SEEDS:
                for method in METHODS:
                    if (factor, level, seed, method) not in present:
                        missing.append(f"{factor}/{level}/seed{seed:04d}/{method}")
    return missing


def compute_deltas(results: list[dict[str, str]]) -> list[DeltaRecord]:
    by_key = {(row["sweep_family"], row["level"], as_seed(row), row["method"]): row for row in results}
    deltas: list[DeltaRecord] = []
    for factor in FACTORS:
        for level in LEVELS:
            seeds = sorted({seed for f, l, seed, _ in by_key if f == factor and l == level})
            for seed in seeds:
                for left, right in COMPARISONS:
                    left_row = by_key.get((factor, level, seed, left))
                    right_row = by_key.get((factor, level, seed, right))
                    if left_row is None or right_row is None:
                        continue
                    values: dict[str, float] = {}
                    for metric in METRICS:
                        left_value = as_float(left_row, metric)
                        right_value = as_float(right_row, metric)
                        if left_value is not None and right_value is not None:
                            values[metric] = left_value - right_value
                    deltas.append(DeltaRecord(factor, level, seed, f"{left}_minus_{right}", values))
    return deltas


def level_stats(values_by_level: dict[str, dict[int, float]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for level in LEVELS:
        values = [values_by_level[level][seed] for seed in sorted(values_by_level[level])]
        out[f"{level}_n"] = len(values)
        out[f"{level}_mean"] = mean(values)
        out[f"{level}_sd"] = sd(values)
        out[f"{level}_seeds"] = ",".join(str(seed) for seed in sorted(values_by_level[level]))
    return out


def paired_contrast_row(
    factor: str,
    comparison: str,
    metric: str,
    values_by_level: dict[str, dict[int, float]],
    left_level: str,
    right_level: str,
) -> dict[str, Any]:
    common_seeds = sorted(set(values_by_level[left_level]) & set(values_by_level[right_level]))
    left = [values_by_level[left_level][seed] for seed in common_seeds]
    right = [values_by_level[right_level][seed] for seed in common_seeds]
    changes = [r - l for l, r in zip(left, right)]
    change_sd = sd(changes)
    wilcoxon_stat, exact_p, nonzero_n = exact_wilcoxon_signed_rank(changes)
    t_stat, t_p = paired_t_test(left, right)
    row = {
        "factor": factor,
        "comparison": comparison,
        "metric": metric,
        "test_kind": "wilcoxon_paired_contrast",
        "contrast": f"{left_level}_vs_{right_level}",
        "contrast_direction": f"{right_level}_minus_{left_level}",
        "independent_unit": "synthetic_scene_seed",
        "n_seeds": len(common_seeds),
        "nonzero_n": nonzero_n,
        "seeds": ",".join(str(seed) for seed in common_seeds),
        "mean_change": mean(changes),
        "sd_change": change_sd,
        "effect_size_paired_dz": None if change_sd in (None, 0.0) else mean(changes) / change_sd,
        "wilcoxon_statistic": wilcoxon_stat,
        "p_value": exact_p,
        "p_value_type": "exact_two_sided_signed_rank",
        "paired_t_statistic_secondary": t_stat,
        "paired_t_p_value_secondary": t_p,
        "views_are_independent_replicates": False,
    }
    row.update(level_stats(values_by_level))
    return row


def omnibus_row(factor: str, comparison: str, metric: str, values_by_level: dict[str, dict[int, float]]) -> dict[str, Any]:
    statistic, p_value, kendall_w, n_seeds = friedman_test(values_by_level)
    row = {
        "factor": factor,
        "comparison": comparison,
        "metric": metric,
        "test_kind": "friedman_omnibus",
        "contrast": "low_vs_medium_vs_high",
        "contrast_direction": "omnibus_matched_levels",
        "independent_unit": "synthetic_scene_seed",
        "n_seeds": n_seeds,
        "nonzero_n": None,
        "seeds": ",".join(str(seed) for seed in sorted(set.intersection(*(set(values_by_level[level]) for level in LEVELS)))) if n_seeds else "",
        "mean_change": None,
        "sd_change": None,
        "effect_size_paired_dz": None,
        "friedman_statistic": statistic,
        "kendall_w": kendall_w,
        "p_value": p_value,
        "p_value_type": "friedman_chi_square_approximation" if p_value is not None else "friedman_statistic_only_scipy_unavailable",
        "paired_t_statistic_secondary": None,
        "paired_t_p_value_secondary": None,
        "views_are_independent_replicates": False,
    }
    row.update(level_stats(values_by_level))
    return row


def interaction_rows(deltas: list[DeltaRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], dict[str, dict[int, float]]] = defaultdict(lambda: {level: {} for level in LEVELS})
    for record in deltas:
        for metric, value in record.metric_values.items():
            grouped[(record.factor, record.comparison, metric)][record.level][record.seed] = value

    for factor in FACTORS:
        for comparison in [f"{left}_minus_{right}" for left, right in COMPARISONS]:
            for metric in METRICS:
                values_by_level = grouped[(factor, comparison, metric)]
                rows.append(omnibus_row(factor, comparison, metric, values_by_level))
                for left_level, right_level in (("low", "medium"), ("medium", "high"), ("low", "high")):
                    rows.append(paired_contrast_row(factor, comparison, metric, values_by_level, left_level, right_level))
    apply_bh(rows)
    return rows


def values_for(deltas: list[DeltaRecord], factor: str, comparison: str, metric: str) -> dict[str, dict[int, float]]:
    values = {level: {} for level in LEVELS}
    for record in deltas:
        if record.factor == factor and record.comparison == comparison and metric in record.metric_values:
            values[record.level][record.seed] = record.metric_values[metric]
    return values


def make_plots(deltas: list[DeltaRecord], results_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = results_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    x = list(range(len(LEVELS)))
    comparison_labels = [f"{left.upper()} - {right.upper()}" for left, right in COMPARISONS]
    comparison_names = [f"{left}_minus_{right}" for left, right in COMPARISONS]

    for factor in FACTORS:
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharex=True)
        for ax, comparison, label in zip(axes, comparison_names, comparison_labels):
            by_level = values_for(deltas, factor, comparison, PRIMARY_METRIC)
            seeds = sorted(set().union(*(set(by_level[level]) for level in LEVELS)))
            for seed in seeds:
                y = [by_level[level].get(seed, float("nan")) for level in LEVELS]
                ax.plot(x, y, color="#8a8f98", alpha=0.55, linewidth=1.1, marker="o", markersize=3)
            means = [mean([by_level[level][seed] for seed in sorted(by_level[level])]) for level in LEVELS]
            sds = [sd([by_level[level][seed] for seed in sorted(by_level[level])]) for level in LEVELS]
            ax.errorbar(
                x,
                means,
                yerr=sds,
                color="#1b1f24",
                linewidth=2.2,
                marker="o",
                markersize=5,
                capsize=4,
                label="mean +/- SD",
            )
            ax.axhline(0.0, color="#c7cbd1", linewidth=1.0, zorder=0)
            ax.set_title(label)
            ax.set_xticks(x, LEVELS)
            ax.set_xlabel("Factor level")
            ax.grid(True, axis="y", alpha=0.25)
        axes[0].set_ylabel("Delta PSNR (dB)")
        fig.suptitle(f"{factor.replace('_', ' ').title()}: Matched Seed Delta PSNR", y=1.03)
        fig.tight_layout()
        path = plot_dir / f"phase3_seed_delta_psnr_{factor}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))
    return paths


def primary_significant_rows(rows: list[dict[str, Any]], alpha: float = 0.05) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["metric"] == PRIMARY_METRIC
        and row.get("p_value_bh_fdr") is not None
        and float(row["p_value_bh_fdr"]) <= alpha
    ]


def monotonic_decrease(values: list[float | None]) -> bool | None:
    if any(value is None for value in values):
        return None
    numeric_values = [float(value) for value in values if value is not None]
    return numeric_values[0] > numeric_values[1] > numeric_values[2]


def write_findings(path: Path, rows: list[dict[str, Any]], deltas: list[DeltaRecord], plot_paths: list[str]) -> None:
    spatial = values_for(deltas, "spatial_frequency", "drk_minus_3dgs", PRIMARY_METRIC)
    spatial_means = [mean([spatial[level][seed] for seed in sorted(spatial[level])]) for level in LEVELS]
    seed_monotonic = 0
    seed_total = 0
    for seed in sorted(set().union(*(set(spatial[level]) for level in LEVELS))):
        vals = [spatial[level].get(seed) for level in LEVELS]
        result = monotonic_decrease(vals)
        if result is not None:
            seed_total += 1
            seed_monotonic += int(result)

    curvature = values_for(deltas, "curvature", "drk_minus_3dgs", PRIMARY_METRIC)
    curvature_means = [mean([curvature[level][seed] for seed in sorted(curvature[level])]) for level in LEVELS]
    significant = primary_significant_rows(rows)

    lines = [
        "# Phase III-B Matched Seed Interaction Tests",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "This analysis tests whether pairwise method performance gaps change across controlled factor levels. The independent unit is the synthetic scene realization / seed (`n=5`); held-out views are not independent replicates.",
        "",
        "## Scope",
        "",
        "- Factors: edge sharpness, spatial frequency, curvature.",
        "- Comparisons: DRK - 3DGS, DRK - GES, GES - 3DGS.",
        "- Primary metric: delta PSNR.",
        "- Secondary metrics: delta SSIM and delta LPIPS.",
        "- Paired contrasts use exact two-sided Wilcoxon signed-rank tests; paired t-tests are reported only as secondary sensitivity checks when SciPy is available.",
        "- Omnibus low/medium/high tests use Friedman tests.",
        "- BH-FDR correction is applied within each test kind and metric family.",
        "",
        "## Spatial Frequency Check",
        "",
        f"- Mean DRK-minus-3DGS delta PSNR by level: low={fmt(spatial_means[0])}, medium={fmt(spatial_means[1])}, high={fmt(spatial_means[2])}.",
        f"- Monotonic decrease in mean DRK-minus-3DGS delta PSNR from low to medium to high: {monotonic_decrease(spatial_means)}.",
        f"- Seeds with monotonic low > medium > high DRK-minus-3DGS delta PSNR: {seed_monotonic}/{seed_total}.",
        "",
        "## Curvature Check",
        "",
        f"- Mean DRK-minus-3DGS delta PSNR by level: low={fmt(curvature_means[0])}, medium={fmt(curvature_means[1])}, high={fmt(curvature_means[2])}.",
        "- Curvature is summarized without imposing monotonicity; non-monotonic relationships remain admissible and should be interpreted from the matched contrast and omnibus rows.",
        "",
        "## Primary BH-FDR Significant Rows",
        "",
    ]
    if significant:
        lines.extend(["| Factor | Comparison | Test | Contrast | p | q |", "| --- | --- | --- | --- | ---: | ---: |"])
        for row in significant:
            lines.append(
                f"| {row['factor']} | {row['comparison']} | {row['test_kind']} | {row['contrast']} | "
                f"{fmt(row.get('p_value'))} | {fmt(row.get('p_value_bh_fdr'))} |"
            )
    else:
        lines.append("No primary delta-PSNR rows pass BH-FDR q <= 0.05.")

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `phase3_seed_interaction_tests.csv/json`: matched Wilcoxon contrasts, Friedman omnibus tests, effect sizes, and BH-FDR corrected p-values.",
            "- `plots/phase3_seed_delta_psnr_<factor>.png`: seed trajectories plus mean +/- SD for delta PSNR.",
            "",
            "## Interpretation Guardrails",
            "",
            "- These tests are observational comparisons across independently trained systems on controlled synthetic stimuli; they do not establish causal kernel-family specialization by themselves.",
            "- A global method advantage does not imply local or factor-specific superiority.",
            "- Winner switching or interaction claims should be made only when supported by the matched seed tests and sensitivity checks.",
            "",
            "## Plot Files",
            "",
        ]
    )
    for plot_path in plot_paths:
        lines.append(f"- `{plot_path}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir
    phase3_results_path = args.phase3_results or (results_dir / "phase3_results.csv")
    results = read_csv(phase3_results_path)
    missing = validate_complete(results)
    if missing and not args.allow_missing_seeds:
        preview = "; ".join(missing[:12])
        suffix = "" if len(missing) <= 12 else f"; ... {len(missing) - 12} more"
        raise RuntimeError(f"Missing required five-seed method-condition records: {preview}{suffix}")

    deltas = compute_deltas(results)
    rows = interaction_rows(deltas)
    csv_path = results_dir / "phase3_seed_interaction_tests.csv"
    json_path = results_dir / "phase3_seed_interaction_tests.json"
    write_csv(csv_path, rows)
    write_json(json_path, rows)
    plot_paths = [] if args.skip_plots else make_plots(deltas, results_dir)
    write_findings(results_dir / "phase3_seed_interaction_findings.md", rows, deltas, plot_paths)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": str(phase3_results_path),
        "independent_unit": "synthetic_scene_seed",
        "expected_seeds": list(EXPECTED_SEEDS),
        "metric_count": len(METRICS),
        "row_count": len(rows),
        "plot_paths": plot_paths,
        "outputs": [str(csv_path), str(json_path), str(results_dir / "phase3_seed_interaction_findings.md")],
    }
    write_json(results_dir / "phase3_seed_interaction_manifest.json", manifest)
    print(f"Phase III-B matched interaction tests complete: {len(rows)} rows")
    print(f"Wrote {csv_path}")
    print(f"Wrote {results_dir / 'phase3_seed_interaction_findings.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
