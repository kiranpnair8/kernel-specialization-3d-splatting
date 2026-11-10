from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common import load_config, resolve_project_path, write_csv


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate rebuttal CSVs and LaTeX strictly from completed analysis outputs.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = resolve_project_path(config, str(config["result_root"]))
    within = read_csv(root / "within_family_summary.csv")
    if {row["method"] for row in within} != {"3dgs", "ges", "drk"}:
        raise ValueError("Within-family summary must contain exactly 3dgs, ges, and drk")
    cross = json.loads((root / "cross_family_summary.json").read_text(encoding="utf-8"))
    if int(cross["combination_count"]) != 27:
        raise ValueError("Cross-family summary does not contain 27 combinations")
    cross_summary = cross["summary"]

    summary_rows: list[dict[str, object]] = []
    for row in within:
        summary_rows.append({
            "comparison": f"{row['method'].upper()} within-family",
            "runs_or_groups": "3 independent seed runs",
            "global_psnr_mean": float(row["global_psnr_mean"]),
            "global_psnr_std": float(row["global_psnr_std"]),
            "oracle_mse_mean": float(row["oracle_mse"]),
            "oracle_gain_fraction_mean": float(row["oracle_headroom_mean"]),
            "oracle_gain_fraction_std": float(row["oracle_headroom_std"]),
            "comparability_note": "Within-family gain summarized across all three reference seeds.",
        })
    summary_rows.append({
        "comparison": "Cross-family (3DGS+GES+DRK)",
        "runs_or_groups": "9 models; 27 dependent seed combinations",
        "global_psnr_mean": "",
        "global_psnr_std": "",
        "oracle_mse_mean": cross_summary["oracle_mse"]["mean"],
        "oracle_gain_fraction_mean": cross_summary["oracle_gain_vs_3dgs_fraction"]["mean"],
        "oracle_gain_fraction_std": cross_summary["oracle_gain_vs_3dgs_fraction"]["std"],
        "comparability_note": "Gain is relative to each combination's selected 3DGS seed; 27 groups are not independent.",
    })
    write_csv(root / "rebuttal_multiseed_summary.csv", summary_rows)

    detailed: list[dict[str, object]] = []
    for filename, record_type in [
        ("within_family_detailed.csv", "within_family"),
        ("within_family_per_view.csv", "within_family_per_view"),
        ("cross_family_combinations.csv", "cross_family_combination"),
        ("cross_family_per_view.csv", "cross_family_per_view"),
        ("fair_oracle_comparison.csv", "same_reference_fair_comparison"),
    ]:
        for row in read_csv(root / filename):
            detailed.append({"record_type": record_type, **row})
    write_csv(root / "rebuttal_multiseed_detailed.csv", detailed)

    display = []
    for row in summary_rows:
        psnr = "--" if row["global_psnr_mean"] == "" else f"{float(row['global_psnr_mean']):.2f} $\\pm$ {float(row['global_psnr_std']):.2f}"
        display.append(
            f"{row['comparison']} & {row['runs_or_groups']} & {psnr} & {float(row['oracle_mse_mean']):.6f} & "
            f"{100 * float(row['oracle_gain_fraction_mean']):.2f} & {100 * float(row['oracle_gain_fraction_std']):.2f} \\\\"
        )
    latex = "\n".join([
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Room multi-seed control. Within-family rows select locally among three independently trained instances of one family. The cross-family row summarizes 27 dependent combinations formed from the same nine models; these combinations are not independent replicates. Cross-family oracle gain uses the selected 3DGS seed as its reference, so global PSNR is intentionally not reported for that row.}",
        r"\label{tab:rebuttal_multiseed}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Comparison & Runs / groups & Global PSNR & Oracle MSE & Gain (\%) & Gain SD (pp) \\",
        r"\midrule",
        *display,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])
    output = root / "rebuttal_multiseed_table.tex"
    output.write_text(latex, encoding="utf-8")
    print(f"Wrote {root / 'rebuttal_multiseed_summary.csv'}")
    print(f"Wrote {root / 'rebuttal_multiseed_detailed.csv'}")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
