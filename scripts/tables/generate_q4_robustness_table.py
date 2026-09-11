#!/usr/bin/env python3
"""Generate the main-paper Q4 robustness table from saved result files.

This script does not rerun experiments and does not contain numerical result
constants. It validates the expected saved-result structure before writing the
LaTeX table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping


SCENES = ("garden", "bicycle", "room")
SCENE_LABELS = {"garden": "Garden", "bicycle": "Bicycle", "room": "Room"}
PATCH_SIZES = (32, 64, 128)
TIE_THRESHOLDS = (0.0, 1e-5, 5e-5)
SELECTED_TIE_THRESHOLD = 1e-5
EXPECTED_STRIDES = {32: 16, 64: 32, 128: 64}


class ValidationError(RuntimeError):
    """Raised when a saved result file does not match the expected schema."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matched-comparison-json",
        type=Path,
        default=(
            root
            / "results"
            / "room"
            / "3dgs_vs_ges_vs_drk_budget250k_p32"
            / "baseline_comparison"
            / "baseline_vs_budget250k_summary.json"
        ),
        help="Room natural-vs-budget250k comparison JSON.",
    )
    parser.add_argument(
        "--garden-sensitivity-csv",
        type=Path,
        default=root / "results" / "garden" / "3dgs_vs_ges_vs_drk" / "sensitivity" / "sensitivity_summary.csv",
    )
    parser.add_argument(
        "--bicycle-sensitivity-csv",
        type=Path,
        default=root / "results" / "bicycle" / "3dgs_vs_ges_vs_drk_p32" / "sensitivity" / "sensitivity_summary.csv",
    )
    parser.add_argument(
        "--room-sensitivity-csv",
        type=Path,
        default=root / "results" / "room" / "3dgs_vs_ges_vs_drk_p32" / "sensitivity" / "sensitivity_summary.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "paper" / "tables" / "q4_robustness.tex",
        help="LaTeX output path.",
    )
    return parser


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required matched-capacity JSON not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required sensitivity CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValidationError(f"Sensitivity CSV has no data rows: {path}")
    return rows


def require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"Expected {name} to be an object/mapping, got {type(value).__name__}")
    return value


def require_key(mapping: Mapping[str, Any], key: str, source: str) -> Any:
    if key not in mapping:
        raise ValidationError(f"Missing required key {key!r} in {source}")
    return mapping[key]


def as_float(value: Any, source: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"Expected finite numeric value for {source}, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValidationError(f"Expected finite numeric value for {source}, got {value!r}")
    return out


def as_int(value: Any, source: str) -> int:
    out = as_float(value, source)
    if not out.is_integer():
        raise ValidationError(f"Expected integer-like value for {source}, got {value!r}")
    return int(out)


def float_matches(value: Any, target: float, source: str, *, tol: float = 1e-12) -> bool:
    return math.isclose(as_float(value, source), target, rel_tol=0.0, abs_tol=tol)


def pct(value: float) -> str:
    return f"{value:.2f}"


def latex_row(*columns: str) -> str:
    return " & ".join(columns) + r" \\"


def label_for_matched_record(record: Mapping[str, Any]) -> str:
    raw = str(record.get("label", "")).lower()
    if "budget250k" in raw or "250k" in raw or "matched" in raw:
        return "matched"
    if "baseline" in raw or "natural" in raw:
        return "natural"
    raise ValidationError(
        "Could not classify matched-comparison record label as natural or 250k: "
        f"{record.get('label')!r}"
    )


def extract_matched_capacity(path: Path) -> dict[str, dict[str, float]]:
    payload = require_mapping(read_json(path), str(path))
    records_raw = require_key(payload, "records", str(path))
    if not isinstance(records_raw, list):
        raise ValidationError(f"Expected {path}: records to be a list")
    if len(records_raw) != 2:
        raise ValidationError(f"Expected exactly 2 matched-comparison records in {path}, got {len(records_raw)}")

    records: dict[str, dict[str, float]] = {}
    for index, raw_record in enumerate(records_raw):
        record = require_mapping(raw_record, f"{path}: records[{index}]")
        key = label_for_matched_record(record)
        if key in records:
            raise ValidationError(f"Duplicate matched-comparison record classified as {key!r} in {path}")

        patch_size = as_int(require_key(record, "patch_size", f"{path}: {key}"), f"{path}: {key}.patch_size")
        stride = as_int(require_key(record, "stride", f"{path}: {key}"), f"{path}: {key}.stride")
        tie = require_key(record, "tie_threshold_mse", f"{path}: {key}")
        if patch_size != 32:
            raise ValidationError(f"Expected {key} patch_size=32 in {path}, got {patch_size}")
        if stride != 16:
            raise ValidationError(f"Expected {key} stride=16 in {path}, got {stride}")
        if not float_matches(tie, SELECTED_TIE_THRESHOLD, f"{path}: {key}.tie_threshold_mse"):
            raise ValidationError(f"Expected {key} tie_threshold_mse=1e-5 in {path}, got {tie!r}")

        non_3dgs = as_float(
            require_key(record, "non_3dgs_winner_fraction", f"{path}: {key}"),
            f"{path}: {key}.non_3dgs_winner_fraction",
        )
        oracle_gain = as_float(
            require_key(record, "oracle_relative_improvement_pct_vs_3dgs", f"{path}: {key}"),
            f"{path}: {key}.oracle_relative_improvement_pct_vs_3dgs",
        )
        records[key] = {
            "non_3dgs_winner_pct": 100.0 * non_3dgs,
            "oracle_gain_vs_3dgs_pct": oracle_gain,
            "patch_size": float(patch_size),
            "stride": float(stride),
            "tie_threshold_mse": as_float(tie, f"{path}: {key}.tie_threshold_mse"),
        }

    missing = {"natural", "matched"} - records.keys()
    if missing:
        raise ValidationError(f"Missing matched-capacity record(s) in {path}: {sorted(missing)}")
    return records


def scene_csv_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "garden": args.garden_sensitivity_csv,
        "bicycle": args.bicycle_sensitivity_csv,
        "room": args.room_sensitivity_csv,
    }


def validate_sensitivity_grid(rows: list[Mapping[str, str]], scene: str, path: Path) -> None:
    seen: set[tuple[int, float]] = set()
    for idx, row in enumerate(rows, start=2):
        patch_size = as_int(require_key(row, "patch_size", f"{path}: line {idx}"), f"{path}: line {idx}.patch_size")
        tie = as_float(require_key(row, "tie_threshold_mse", f"{path}: line {idx}"), f"{path}: line {idx}.tie_threshold_mse")
        if patch_size not in PATCH_SIZES:
            raise ValidationError(f"Unexpected patch_size for {scene} in {path} line {idx}: {patch_size}")
        matched_ties = [target for target in TIE_THRESHOLDS if math.isclose(tie, target, rel_tol=0.0, abs_tol=1e-12)]
        if len(matched_ties) != 1:
            raise ValidationError(f"Unexpected tie_threshold_mse for {scene} in {path} line {idx}: {tie}")
        key = (patch_size, matched_ties[0])
        if key in seen:
            raise ValidationError(f"Duplicate sensitivity condition for {scene} in {path}: {key}")
        seen.add(key)

    expected = {(patch_size, tie) for patch_size in PATCH_SIZES for tie in TIE_THRESHOLDS}
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValidationError(
            f"Sensitivity grid mismatch for {scene} in {path}: missing={missing}, extra={extra}"
        )


def extract_sensitivity(paths: Mapping[str, Path]) -> dict[str, dict[int, dict[str, float]]]:
    by_scene: dict[str, dict[int, dict[str, float]]] = {}
    for scene, path in paths.items():
        rows = read_csv(path)
        validate_sensitivity_grid(rows, scene, path)
        selected: dict[int, dict[str, float]] = {}
        for patch_size in PATCH_SIZES:
            matches = [
                row
                for row in rows
                if as_int(row.get("patch_size"), f"{path}: patch_size") == patch_size
                and float_matches(row.get("tie_threshold_mse"), SELECTED_TIE_THRESHOLD, f"{path}: tie_threshold_mse")
            ]
            if len(matches) != 1:
                raise ValidationError(
                    f"Expected exactly one selected sensitivity row for scene={scene}, "
                    f"patch_size={patch_size}, tie_threshold_mse=1e-5 in {path}; found {len(matches)}"
                )
            row = matches[0]
            stride = as_int(require_key(row, "stride", f"{path}: {scene} p{patch_size}"), f"{path}: {scene} p{patch_size}.stride")
            expected_stride = EXPECTED_STRIDES[patch_size]
            if stride != expected_stride:
                raise ValidationError(
                    f"Expected stride {expected_stride} for scene={scene}, patch_size={patch_size} in {path}; got {stride}"
                )
            non_3dgs = as_float(
                require_key(row, "non_3dgs_winner_fraction", f"{path}: {scene} p{patch_size}"),
                f"{path}: {scene} p{patch_size}.non_3dgs_winner_fraction",
            )
            oracle_gain = as_float(
                require_key(row, "oracle_relative_improvement_pct_vs_3dgs", f"{path}: {scene} p{patch_size}"),
                f"{path}: {scene} p{patch_size}.oracle_relative_improvement_pct_vs_3dgs",
            )
            selected[patch_size] = {
                "stride": float(stride),
                "tie_threshold_mse": SELECTED_TIE_THRESHOLD,
                "non_3dgs_winner_pct": 100.0 * non_3dgs,
                "oracle_gain_vs_3dgs_pct": oracle_gain,
            }
        by_scene[scene] = selected
    return by_scene


def latex_table(matched: Mapping[str, Mapping[str, float]], sensitivity: Mapping[str, Mapping[int, Mapping[str, float]]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"\multicolumn{3}{l}{\textbf{(A) Matched representation-capacity control --- Room}} \\",
        r"\midrule",
        r"Setting & Non-3DGS winners (\%) & Oracle gain vs. 3DGS (\%) \\",
        r"\midrule",
        latex_row("Natural", pct(matched["natural"]["non_3dgs_winner_pct"]), pct(matched["natural"]["oracle_gain_vs_3dgs_pct"])), 
        latex_row("250k each", pct(matched["matched"]["non_3dgs_winner_pct"]), pct(matched["matched"]["oracle_gain_vs_3dgs_pct"])), 
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.45em}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\multicolumn{7}{l}{\textbf{(B) Spatial analysis scale sensitivity, $\tau=10^{-5}$}} \\",
        r"\midrule",
        r"& \multicolumn{3}{c}{Non-3DGS winners (\%)} & \multicolumn{3}{c}{Oracle gain vs. 3DGS (\%)} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Scene & $p=32$ & $p=64$ & $p=128$ & $p=32$ & $p=64$ & $p=128$ \\",
        r"\midrule",
    ]
    for scene in SCENES:
        row = sensitivity[scene]
        lines.append(
            f"{SCENE_LABELS[scene]} & "
            f"{pct(row[32]['non_3dgs_winner_pct'])} & {pct(row[64]['non_3dgs_winner_pct'])} & {pct(row[128]['non_3dgs_winner_pct'])} & "
            f"{pct(row[32]['oracle_gain_vs_3dgs_pct'])} & {pct(row[64]['oracle_gain_vs_3dgs_pct'])} & {pct(row[128]['oracle_gain_vs_3dgs_pct'])} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{\textbf{Robustness of local cross-family specialization to representation capacity and spatial analysis scale.} "
            r"(A) compares the natural Room representation with a matched 250k-primitive budget for each family. "
            r"(B) evaluates increasingly coarse local analysis at $\tau=10^{-5}$. "
            r"Non-3DGS winner fractions and post-hoc oracle improvement relative to 3DGS remain nonzero across all evaluated settings.}",
            r"\label{tab:q4_robustness}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def print_matched_values(path: Path, matched: Mapping[str, Mapping[str, float]]) -> None:
    print("Matched-capacity source:")
    print(f"  {path}")
    for key, label in (("natural", "Natural"), ("matched", "250k each")):
        row = matched[key]
        print(
            f"  {label}: patch_size={int(row['patch_size'])}, stride={int(row['stride'])}, "
            f"tie_threshold_mse={row['tie_threshold_mse']:.12g}, "
            f"non_3dgs_winner_pct={row['non_3dgs_winner_pct']:.6f}, "
            f"oracle_gain_vs_3dgs_pct={row['oracle_gain_vs_3dgs_pct']:.6f}"
        )


def print_sensitivity_values(paths: Mapping[str, Path], sensitivity: Mapping[str, Mapping[int, Mapping[str, float]]]) -> None:
    print("Sensitivity sources:")
    for scene in SCENES:
        print(f"  {SCENE_LABELS[scene]}: {paths[scene]}")
        for patch_size in PATCH_SIZES:
            row = sensitivity[scene][patch_size]
            print(
                f"    p={patch_size}: stride={int(row['stride'])}, "
                f"tie_threshold_mse={row['tie_threshold_mse']:.12g}, "
                f"non_3dgs_winner_pct={row['non_3dgs_winner_pct']:.6f}, "
                f"oracle_gain_vs_3dgs_pct={row['oracle_gain_vs_3dgs_pct']:.6f}"
            )


def main() -> int:
    args = build_parser().parse_args()
    matched = extract_matched_capacity(args.matched_comparison_json)
    sensitivity_paths = scene_csv_paths(args)
    sensitivity = extract_sensitivity(sensitivity_paths)

    print_matched_values(args.matched_comparison_json, matched)
    print_sensitivity_values(sensitivity_paths, sensitivity)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(latex_table(matched, sensitivity), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

