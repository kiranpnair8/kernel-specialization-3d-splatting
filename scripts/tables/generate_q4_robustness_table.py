#!/usr/bin/env python3
"""Generate the main-paper Q4 robustness table.

The script reads existing matched-capacity and sensitivity outputs only. It
does not rerun experiments or modify result files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping


SCENES = ("garden", "bicycle", "room")
PATCH_SIZES = (32, 64, 128)
TIE_THRESHOLD = 1e-5

EXPECTED_CAPACITY = {
    "natural": {
        "non_3dgs_winner_pct": 40.3958,
        "oracle_gain_vs_3dgs_pct": 31.1564,
    },
    "matched": {
        "non_3dgs_winner_pct": 53.9209,
        "oracle_gain_vs_3dgs_pct": 29.6920,
    },
}

EXPECTED_SENSITIVITY = {
    "garden": {32: 30.0461, 64: 25.7933, 128: 21.6346},
    "bicycle": {32: 35.4703, 64: 29.9495, 128: 23.7368},
    "room": {32: 40.3958, 64: 38.6385, 128: 36.1579},
}

SCENE_LABELS = {"garden": "Garden", "bicycle": "Bicycle", "room": "Room"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--room-natural-summary",
        type=Path,
        default=root / "results" / "room" / "3dgs_vs_ges_vs_drk_p32" / "summary.json",
        help="Natural Room p32 local comparison summary.json.",
    )
    parser.add_argument(
        "--room-budget-summary",
        type=Path,
        default=root / "results" / "room" / "3dgs_vs_ges_vs_drk_budget250k_p32" / "summary.json",
        help="Matched 250k Room p32 local comparison summary.json.",
    )
    parser.add_argument(
        "--sensitivity-root",
        type=Path,
        default=root / "results",
        help="Root under which scene sensitivity_summary.csv files are searched.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "paper" / "tables" / "q4_robustness.tex",
        help="LaTeX output path.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=5e-3,
        help="Absolute percentage-point tolerance for validating extracted values.",
    )
    return parser


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required summary file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV file not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"Non-finite value for {name}: {value}")
    return out


def oracle_values(summary_path: Path) -> dict[str, float]:
    summary = read_json(summary_path)
    oracle = summary.get("oracle")
    if not isinstance(oracle, Mapping):
        raise ValueError(f"{summary_path} does not contain an oracle object.")
    ges = as_float(oracle.get("ges_winner_fraction"), "ges_winner_fraction")
    drk = as_float(oracle.get("drk_winner_fraction"), "drk_winner_fraction")
    non_3dgs_pct = 100.0 * (ges + drk)

    if "oracle_relative_improvement_pct_vs_3dgs" in oracle:
        oracle_gain_pct = as_float(
            oracle["oracle_relative_improvement_pct_vs_3dgs"],
            "oracle_relative_improvement_pct_vs_3dgs",
        )
    else:
        improvement = as_float(
            oracle.get("oracle_improvement_mse_vs_3dgs"),
            "oracle_improvement_mse_vs_3dgs",
        )
        mse_3dgs = as_float(oracle.get("3dgs_patch_mse"), "3dgs_patch_mse")
        if mse_3dgs == 0.0:
            raise ValueError(f"{summary_path} has zero 3DGS patch MSE.")
        oracle_gain_pct = 100.0 * improvement / mse_3dgs

    return {
        "non_3dgs_winner_pct": non_3dgs_pct,
        "oracle_gain_vs_3dgs_pct": oracle_gain_pct,
    }


def tie_matches(value: float, target: float = TIE_THRESHOLD) -> bool:
    return math.isclose(value, target, rel_tol=0.0, abs_tol=1e-12)


def find_sensitivity_summary(root: Path, scene: str) -> Path:
    preferred = [
        root / scene / "3dgs_vs_ges_vs_drk_p32" / "sensitivity" / "sensitivity_summary.csv",
        root / scene / "3dgs_vs_ges_vs_drk" / "sensitivity" / "sensitivity_summary.csv",
        root / scene / "3dgs_vs_ges_vs_drk_p32" / "sensitivity_summary.csv",
    ]
    for path in preferred:
        if path.exists():
            return path

    scene_root = root / scene
    candidates = sorted(scene_root.rglob("sensitivity_summary.csv")) if scene_root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No sensitivity_summary.csv found for scene {scene} under {root}")
    if len(candidates) > 1:
        formatted = "\n".join(f"  - {path}" for path in candidates)
        raise RuntimeError(
            f"Multiple sensitivity_summary.csv files found for scene {scene}; "
            f"pass a narrower --sensitivity-root or remove ambiguity:\n{formatted}"
        )
    return candidates[0]


def sensitivity_values(root: Path) -> tuple[dict[str, dict[int, float]], dict[str, Path]]:
    output: dict[str, dict[int, float]] = {}
    sources: dict[str, Path] = {}
    for scene in SCENES:
        path = find_sensitivity_summary(root, scene)
        rows = read_csv(path)
        sources[scene] = path
        scene_values: dict[int, float] = {}
        for patch_size in PATCH_SIZES:
            matches = [
                row
                for row in rows
                if int(float(row["patch_size"])) == patch_size
                and tie_matches(float(row["tie_threshold_mse"]))
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one row for scene={scene}, patch_size={patch_size}, "
                    f"tie_threshold_mse={TIE_THRESHOLD}; found {len(matches)} in {path}"
                )
            row = matches[0]
            scene_values[patch_size] = 100.0 * as_float(
                row["non_3dgs_winner_fraction"], "non_3dgs_winner_fraction"
            )
        output[scene] = scene_values
    return output, sources


def assert_close(name: str, observed: float, expected: float, tolerance: float) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{name}: observed {observed:.6f}, expected {expected:.6f}")


def validate(
    natural: dict[str, float],
    matched: dict[str, float],
    sensitivity: dict[str, dict[int, float]],
    tolerance: float,
) -> None:
    for key, expected in EXPECTED_CAPACITY["natural"].items():
        assert_close(f"natural/{key}", natural[key], expected, tolerance)
    for key, expected in EXPECTED_CAPACITY["matched"].items():
        assert_close(f"matched/{key}", matched[key], expected, tolerance)
    for scene, patch_values in EXPECTED_SENSITIVITY.items():
        for patch_size, expected in patch_values.items():
            assert_close(
                f"sensitivity/{scene}/p{patch_size}",
                sensitivity[scene][patch_size],
                expected,
                tolerance,
            )


def pct(value: float) -> str:
    return f"{value:.2f}"


def make_table(natural: dict[str, float], matched: dict[str, float], sensitivity: dict[str, dict[int, float]]) -> str:
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{5pt}",
            r"\begin{tabular}{llcc}",
            r"\toprule",
            r"\multicolumn{4}{l}{\textbf{(A) Matched representation capacity control --- Room}} \\",
            r"\midrule",
            r"Setting & Primitive budget & Non-3DGS winner (\%) & Oracle gain vs. 3DGS (\%) \\",
            r"\midrule",
            f"Natural & Natural & {pct(natural['non_3dgs_winner_pct'])} & {pct(natural['oracle_gain_vs_3dgs_pct'])} \\\\",
            f"Matched & 250k each & {pct(matched['non_3dgs_winner_pct'])} & {pct(matched['oracle_gain_vs_3dgs_pct'])} \\\\",
            r"\midrule",
            r"\multicolumn{4}{l}{\textbf{(B) Spatial analysis scale sensitivity, $\tau=10^{-5}$}} \\",
            r"\midrule",
            r"Scene & $p=32$ & $p=64$ & $p=128$ \\",
            r"\midrule",
            *[
                f"{SCENE_LABELS[scene]} & {pct(sensitivity[scene][32])} & "
                f"{pct(sensitivity[scene][64])} & {pct(sensitivity[scene][128])} \\\\"
                for scene in SCENES
            ],
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{\textbf{Robustness of local cross-family specialization to representation capacity and spatial analysis scale.} "
            r"(A) compares the natural Room setting with a strictly matched 250k-primitive budget for each family. "
            r"(B) reports the fraction of patches won by a non-3DGS family across analysis patch sizes at $\tau=10^{-5}$. "
            r"Specialization and post-hoc oracle headroom persist under matched capacity and increasingly coarse spatial analysis.}",
            r"\label{tab:q4_robustness}",
            r"\end{table}",
            "",
        ]
    )


def print_raw_values(
    natural: dict[str, float],
    matched: dict[str, float],
    sensitivity: dict[str, dict[int, float]],
    sources: dict[str, Path],
    natural_path: Path,
    matched_path: Path,
) -> None:
    print("Extracted raw values:")
    print(f"  Natural Room source: {natural_path}")
    print(
        "    non_3dgs_winner_pct="
        f"{natural['non_3dgs_winner_pct']:.4f}, "
        "oracle_gain_vs_3dgs_pct="
        f"{natural['oracle_gain_vs_3dgs_pct']:.4f}"
    )
    print(f"  Matched Room source: {matched_path}")
    print(
        "    non_3dgs_winner_pct="
        f"{matched['non_3dgs_winner_pct']:.4f}, "
        "oracle_gain_vs_3dgs_pct="
        f"{matched['oracle_gain_vs_3dgs_pct']:.4f}"
    )
    for scene in SCENES:
        values = sensitivity[scene]
        print(f"  Sensitivity {scene} source: {sources[scene]}")
        print(
            f"    p32={values[32]:.4f}, p64={values[64]:.4f}, p128={values[128]:.4f}"
        )


def main() -> int:
    args = build_parser().parse_args()
    natural = oracle_values(args.room_natural_summary)
    matched = oracle_values(args.room_budget_summary)
    sensitivity, sources = sensitivity_values(args.sensitivity_root)
    validate(natural, matched, sensitivity, args.tolerance)
    table = make_table(natural, matched, sensitivity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table, encoding="utf-8")
    print_raw_values(natural, matched, sensitivity, sources, args.room_natural_summary, args.room_budget_summary)
    print(f"Validation passed within tolerance {args.tolerance}.")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
