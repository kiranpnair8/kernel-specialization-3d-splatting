#!/usr/bin/env python3
"""Generate the Q2 structure-conditioned statistical audit LaTeX table.

The script reads the final trend_tests.csv and writes a booktabs/multirow
appendix table without recomputing any experimental results.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


SCENES = ("garden", "bicycle", "room")
SCENE_LABELS = {
    "garden": "Garden",
    "bicycle": "Bicycle",
    "room": "Room",
}

DESCRIPTORS = (
    "mean_gradient_magnitude",
    "edge_strength",
    "laplacian_energy",
    "local_variance",
    "high_frequency_energy",
    "entropy",
)
DESCRIPTOR_LABELS = {
    "mean_gradient_magnitude": "Mean gradient magnitude",
    "edge_strength": "Edge strength",
    "laplacian_energy": "Laplacian energy",
    "local_variance": "Local variance",
    "high_frequency_energy": "High-frequency energy",
    "entropy": "Entropy",
}

COMPARISONS = ("ges_vs_3dgs", "drk_vs_3dgs", "ges_vs_drk")
COMPARISON_LABELS = {
    "ges_vs_3dgs": "GES--3DGS",
    "drk_vs_3dgs": "DRK--3DGS",
    "ges_vs_drk": "GES--DRK",
}

REQUIRED_COLUMNS = {
    "scene",
    "descriptor",
    "comparison",
    "spearman_rho",
    "spearman_p_value",
    "spearman_p_value_bh_fdr",
    "binned_median_delta_slope",
}

CI_COLUMN_PAIRS = (
    ("binned_median_delta_slope_ci_low", "binned_median_delta_slope_ci_high"),
    ("slope_ci_low", "slope_ci_high"),
    ("trend_ci_low", "trend_ci_high"),
)

CAPTION = (
    "Complete statistical audit of structure-conditioned pairwise reconstruction "
    "error trends. Spearman rho measures the patch-level monotonic association "
    "between each ground-truth structural descriptor and pairwise Delta MSE. "
    "Raw p-values and Benjamini--Hochberg FDR-adjusted p-values are reported for "
    "completeness. Binned slopes summarize the direction of the median Delta MSE "
    "trend across descriptor quantiles. Because spatial patches overlap, statistical "
    "interpretation emphasizes view-clustered uncertainty and reproducibility across "
    "independent scenes rather than treating individual patches as independent "
    "replications."
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Generate the Q2 statistical audit appendix table from trend_tests.csv."
    )
    parser.add_argument(
        "--trend-tests",
        type=Path,
        default=root / "results" / "cross_scene" / "complexity_stratified_errors_p32" / "trend_tests.csv",
        help="Path to the final trend_tests.csv result file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "paper" / "tables" / "q2_statistical_audit.tex",
        help="LaTeX table output path.",
    )
    return parser


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required trend_tests.csv does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} is empty.")
    missing = REQUIRED_COLUMNS.difference(rows[0].keys())
    if missing:
        raise ValueError(
            f"{path} is missing required column(s): {sorted(missing)}. "
            "No values will be invented."
        )
    return rows


def find_ci_columns(columns: set[str]) -> tuple[str, str] | None:
    for low, high in CI_COLUMN_PAIRS:
        if low in columns and high in columns:
            return low, high
    return None


def numeric(value: str, column: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite value in {column}: {value}")
    return number


def format_number(value: str, column: str) -> str:
    number = numeric(value, column)
    if number == 0.0:
        return "0"
    abs_number = abs(number)
    if abs_number < 1e-3 or abs_number >= 1e4:
        return f"{number:.3e}"
    return f"{number:.6g}"


def escape_latex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def index_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    indexed: dict[tuple[str, str, str], dict[str, str]] = {}
    allowed = {
        (scene, descriptor, comparison)
        for scene in SCENES
        for descriptor in DESCRIPTORS
        for comparison in COMPARISONS
    }
    for row in rows:
        key = (row["scene"], row["descriptor"], row["comparison"])
        if key not in allowed:
            raise ValueError(f"Unexpected trend_tests.csv row outside requested audit set: {key}")
        if key in indexed:
            raise ValueError(f"Duplicate trend_tests.csv row: {key}")
        indexed[key] = row
    missing = sorted(allowed.difference(indexed))
    if missing:
        raise ValueError(f"Missing requested trend_tests.csv row(s): {missing}")
    if len(indexed) != 54:
        raise ValueError(f"Expected exactly 54 rows, found {len(indexed)}.")
    return indexed


def make_table(rows: list[dict[str, str]]) -> str:
    columns = set(rows[0].keys())
    ci_columns = find_ci_columns(columns)
    indexed = index_rows(rows)

    if ci_columns is None:
        ci_note = (
            "% No confidence-interval columns directly associated with the reported "
            "trend statistic were present in trend_tests.csv.\n"
        )
        column_spec = "lllrrrr"
        header = (
            r"\textbf{Scene} & \textbf{Descriptor} & \textbf{Comparison} & "
            r"\textbf{Spearman $\rho$} & \textbf{Raw $p$} & "
            r"\textbf{BH-FDR $q$} & \textbf{Binned slope} \\"
        )
    else:
        low_col, high_col = ci_columns
        ci_note = f"% Trend-statistic CI columns included from trend_tests.csv: {low_col}, {high_col}.\n"
        column_spec = "lllrrrrr"
        header = (
            r"\textbf{Scene} & \textbf{Descriptor} & \textbf{Comparison} & "
            r"\textbf{Spearman $\rho$} & \textbf{Raw $p$} & "
            r"\textbf{BH-FDR $q$} & \textbf{Binned slope} & \textbf{Trend CI} \\"
        )

    lines: list[str] = [
        ci_note.rstrip(),
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{" + column_spec + r"}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    for scene_index, scene in enumerate(SCENES):
        scene_row_count = len(DESCRIPTORS) * len(COMPARISONS)
        descriptor_first_in_scene = True
        for descriptor in DESCRIPTORS:
            descriptor_row_count = len(COMPARISONS)
            first_comparison = True
            for comparison in COMPARISONS:
                row = indexed[(scene, descriptor, comparison)]
                cells: list[str] = []
                if descriptor_first_in_scene:
                    cells.append(rf"\multirow{{{scene_row_count}}}{{*}}{{{SCENE_LABELS[scene]}}}")
                    descriptor_first_in_scene = False
                else:
                    cells.append("")
                if first_comparison:
                    cells.append(
                        rf"\multirow{{{descriptor_row_count}}}{{*}}{{{DESCRIPTOR_LABELS[descriptor]}}}"
                    )
                    first_comparison = False
                else:
                    cells.append("")
                cells.extend(
                    [
                        COMPARISON_LABELS[comparison],
                        format_number(row["spearman_rho"], "spearman_rho"),
                        format_number(row["spearman_p_value"], "spearman_p_value"),
                        format_number(row["spearman_p_value_bh_fdr"], "spearman_p_value_bh_fdr"),
                        format_number(row["binned_median_delta_slope"], "binned_median_delta_slope"),
                    ]
                )
                if ci_columns is not None:
                    low_col, high_col = ci_columns
                    cells.append(
                        "["
                        + format_number(row[low_col], low_col)
                        + ", "
                        + format_number(row[high_col], high_col)
                        + "]"
                    )
                lines.append(" & ".join(cells) + r" \\")
            if descriptor != DESCRIPTORS[-1]:
                lines.append(r"\addlinespace[1pt]")
        if scene_index != len(SCENES) - 1:
            lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{" + escape_latex(CAPTION) + r"}",
            r"\label{tab:q2_statistical_audit}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def verify_signs(rows: list[dict[str, str]]) -> None:
    for row in rows:
        numeric(row["spearman_rho"], "spearman_rho")
        numeric(row["binned_median_delta_slope"], "binned_median_delta_slope")


def main() -> int:
    args = build_parser().parse_args()
    rows = read_rows(args.trend_tests)
    indexed = index_rows(rows)
    ordered_rows = [
        indexed[(scene, descriptor, comparison)]
        for scene in SCENES
        for descriptor in DESCRIPTORS
        for comparison in COMPARISONS
    ]
    verify_signs(ordered_rows)
    table = make_table(ordered_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table, encoding="utf-8")
    columns = set(rows[0].keys())
    ci_columns = find_ci_columns(columns)
    if ci_columns is None:
        print("No trend-statistic CI columns found in trend_tests.csv; no CI column was written.")
    else:
        print(f"Wrote CI column from {ci_columns[0]} and {ci_columns[1]}.")
    print(f"Verified exactly {len(ordered_rows)} rows.")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
