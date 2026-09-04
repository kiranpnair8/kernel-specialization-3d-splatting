from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

DEFAULT_PATCHES_CSV = Path("results/room/3dgs_vs_ges_vs_drk_budget250k_p32/patches.csv")
DEFAULT_OUTPUT_CSV = Path("results/room/3dgs_vs_ges_vs_drk_budget250k_p32/figure1_candidate_views.csv")
METHODS = ("3dgs", "ges", "drk")


def require_columns(fieldnames: Iterable[str] | None, required: Iterable[str]) -> List[str]:
    if fieldnames is None:
        raise ValueError("patches.csv has no header row")
    fields = list(fieldnames)
    missing = [column for column in required if column not in fields]
    if missing:
        raise ValueError(
            "patches.csv is missing required columns "
            f"{missing}; observed columns: {fields}"
        )
    return fields


def normalized_winner_entropy(winner_counts: Mapping[str, int], decisive_count: int) -> float:
    if decisive_count <= 0:
        return 0.0
    entropy = 0.0
    for method in METHODS:
        p = winner_counts.get(method, 0) / decisive_count
        if p > 0.0:
            entropy -= p * math.log(p)
    return entropy / math.log(len(METHODS))


def fraction(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return count / total


def load_view_counts(patches_csv: Path) -> tuple[Dict[str, Counter[str]], List[str]]:
    counts: Dict[str, Counter[str]] = defaultdict(Counter)
    with patches_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = require_columns(reader.fieldnames, ["view", "winner"])
        for row_index, row in enumerate(reader, start=2):
            view = (row.get("view") or "").strip()
            winner = (row.get("winner") or "").strip().lower()
            if not view:
                raise ValueError(f"row {row_index} has an empty view value")
            if winner not in {*METHODS, "tie"}:
                raise ValueError(
                    f"row {row_index} has unexpected winner={winner!r}; "
                    f"expected one of {(*METHODS, 'tie')}"
                )
            counts[view][winner] += 1
    if not counts:
        raise ValueError(f"No patch rows found in {patches_csv}")
    return counts, fields


def summarize_views(counts: Mapping[str, Counter[str]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for view, winner_counts in counts.items():
        total = sum(winner_counts.values())
        tie_count = winner_counts.get("tie", 0)
        decisive = total - tie_count
        decisive_method_fractions = {
            method: fraction(winner_counts.get(method, 0), decisive)
            for method in METHODS
        }
        row: Dict[str, object] = {
            "view": view,
            "total_patch_count": total,
            "decisive_patch_count": decisive,
            "tie_fraction": fraction(tie_count, total),
            "3dgs_winner_fraction": fraction(winner_counts.get("3dgs", 0), total),
            "ges_winner_fraction": fraction(winner_counts.get("ges", 0), total),
            "drk_winner_fraction": fraction(winner_counts.get("drk", 0), total),
            "non_3dgs_winner_fraction": fraction(
                winner_counts.get("ges", 0) + winner_counts.get("drk", 0), total
            ),
            "3dgs_decisive_winner_fraction": decisive_method_fractions["3dgs"],
            "ges_decisive_winner_fraction": decisive_method_fractions["ges"],
            "drk_decisive_winner_fraction": decisive_method_fractions["drk"],
            "winner_diversity_score": normalized_winner_entropy(winner_counts, decisive),
        }
        rows.append(row)
    rows.sort(
        key=lambda item: (
            float(item["winner_diversity_score"]),
            float(item["non_3dgs_winner_fraction"]),
            int(item["decisive_patch_count"]),
            str(item["view"]),
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "rank",
        "view",
        "total_patch_count",
        "decisive_patch_count",
        "tie_fraction",
        "3dgs_winner_fraction",
        "ges_winner_fraction",
        "drk_winner_fraction",
        "non_3dgs_winner_fraction",
        "3dgs_decisive_winner_fraction",
        "ges_decisive_winner_fraction",
        "drk_decisive_winner_fraction",
        "winner_diversity_score",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_top(rows: List[Dict[str, object]], limit: int) -> None:
    display_fields = [
        "rank",
        "view",
        "winner_diversity_score",
        "total_patch_count",
        "decisive_patch_count",
        "tie_fraction",
        "3dgs_winner_fraction",
        "ges_winner_fraction",
        "drk_winner_fraction",
        "non_3dgs_winner_fraction",
    ]
    print(",".join(display_fields))
    for row in rows[:limit]:
        print(
            ",".join(
                str(row[field]) if not isinstance(row[field], float) else f"{row[field]:.8g}"
                for field in display_fields
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank local-comparison views by decisive winner diversity for Figure 1 candidate selection."
    )
    parser.add_argument("--patches-csv", type=Path, default=DEFAULT_PATCHES_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    counts, fields = load_view_counts(args.patches_csv)
    print(f"inspected patches schema: {fields}")
    rows = summarize_views(counts)
    write_csv(args.output_csv, rows)
    print(f"wrote: {args.output_csv}")
    print_top(rows, args.top)


if __name__ == "__main__":
    main()
