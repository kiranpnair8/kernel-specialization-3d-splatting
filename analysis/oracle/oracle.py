from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Mapping

from analysis.local_error.metrics import psnr_from_mse


def oracle_summary(rows: Iterable[Mapping[str, object]], methods: List[str]) -> Dict[str, object]:
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot compute oracle summary with zero patch rows")

    method_sums = {method: 0.0 for method in methods}
    winner_counts: Counter[str] = Counter()
    oracle_sum = 0.0
    decisive_count = 0

    for row in rows:
        errors = {method: float(row[f"{method}_mse"]) for method in methods}
        for method, value in errors.items():
            method_sums[method] += value
        best_method = min(errors, key=errors.get)
        oracle_sum += errors[best_method]
        winner = str(row["winner"])
        winner_counts[winner] += 1
        if winner != "tie":
            decisive_count += 1

    total = len(rows)
    oracle_mse = oracle_sum / total
    summary: Dict[str, object] = {
        "patch_count": total,
        "decisive_patch_count": decisive_count,
        "oracle_patch_mse": oracle_mse,
        "oracle_patch_psnr": psnr_from_mse(oracle_mse),
    }

    for method in methods:
        method_mse = method_sums[method] / total
        summary[f"{method}_patch_mse"] = method_mse
        summary[f"{method}_patch_psnr"] = psnr_from_mse(method_mse)
        summary[f"oracle_improvement_mse_vs_{method}"] = method_mse - oracle_mse
        summary[f"{method}_winner_fraction"] = winner_counts[method] / total
    summary["tie_fraction"] = winner_counts["tie"] / total
    return summary
