from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis.local_error.image_io import load_rgb
from analysis.local_error.patches import Patch
from analysis.winner_maps.visualize import save_winner_map
from common import atomic_json, cluster_bootstrap, fractions, load_config, load_error_bundle, oracle_metrics, resolve_project_path, summarize, winner_labels, write_csv


COLORS = {"seed_0": (45, 114, 210), "seed_1": (218, 87, 67), "seed_2": (62, 150, 91)}


def read_inventory(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantify three-seed within-family stochastic complementarity.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    result_root = resolve_project_path(config, str(config["result_root"]))
    figure_root = resolve_project_path(config, str(config["figure_root"]))
    errors, run_ids, views, metadata = load_error_bundle(result_root / "patch_errors.npz")
    inventory = read_inventory(result_root / "run_inventory.csv")
    inventory_by_id = {row["run_id"]: row for row in inventory}
    tie = float(config["tie_threshold_mse"])
    summary_rows: list[dict[str, object]] = []
    detailed_rows: list[dict[str, object]] = []
    per_view_rows: list[dict[str, object]] = []

    for method in config["methods"]:
        method_runs = [f"{method}_seed_{seed}" for seed in config["seeds"]]
        indices = [run_ids.index(run_id) for run_id in method_runs]
        values = errors[:, indices]
        labels = [f"seed_{seed}" for seed in config["seeds"]]
        winners, _ = winner_labels(values, labels, tie)
        winner_stats = fractions(winners, labels)
        oracle = oracle_metrics(values, range(3))
        global_psnr = [float(inventory_by_id[run_id]["psnr"]) for run_id in method_runs]
        patch_seed_means = values.mean(axis=0)
        headrooms = [record["headroom_fraction"] for record in oracle["references"]]
        row = {
            "comparison": f"{method}_within_family", "method": method, "runs_or_groups": 3,
            "global_psnr_mean": summarize(global_psnr)["mean"], "global_psnr_std": summarize(global_psnr)["std"],
            "patch_mse_mean_across_seed_means": float(patch_seed_means.mean()),
            "patch_mse_std_across_seed_means": float(patch_seed_means.std(ddof=1)),
            "oracle_mse": oracle["oracle_mse"], "oracle_psnr": oracle["oracle_psnr"],
            "oracle_headroom_mean": summarize(headrooms)["mean"], "oracle_headroom_std": summarize(headrooms)["std"],
            **winner_stats,
        }
        summary_rows.append(row)
        for seed_index, run_id in enumerate(method_runs):
            reference = oracle["references"][seed_index]
            by_view = {
                view: np.column_stack((values[views == view, seed_index], values[views == view].min(axis=1)))
                for view in sorted(set(views))
            }
            low, high = cluster_bootstrap(
                by_view,
                lambda x: float((x[:, 0].mean() - x[:, 1].mean()) / x[:, 0].mean()),
                int(config["bootstrap_replicates"]),
                int(config["bootstrap_seed"]) + seed_index,
            )
            detailed_rows.append({
                "analysis": "within_family_reference", "method": method, "reference_seed": config["seeds"][seed_index],
                "reference_mse": reference["reference_mse"], "oracle_mse": oracle["oracle_mse"],
                "headroom_fraction": reference["headroom_fraction"], "view_cluster_bootstrap_ci_low": low,
                "view_cluster_bootstrap_ci_high": high,
            })
        for left, right in combinations(range(3), 2):
            difference = values[:, left] - values[:, right]
            detailed_rows.append({
                "analysis": "within_family_pairwise_disagreement", "method": method,
                "left_seed": config["seeds"][left], "right_seed": config["seeds"][right],
                "disagreement_fraction": float(np.mean(np.abs(difference) >= tie)),
                "left_wins_fraction": float(np.mean(difference <= -tie)),
                "right_wins_fraction": float(np.mean(difference >= tie)),
                "tie_fraction": float(np.mean(np.abs(difference) < tie)),
            })
        for view in sorted(set(views)):
            mask = views == view
            view_oracle = values[mask].min(axis=1)
            view_row: dict[str, object] = {"method": method, "view": view, "patch_count": int(mask.sum()), "oracle_mse": float(view_oracle.mean())}
            for seed_index, seed in enumerate(config["seeds"]):
                reference_mse = float(values[mask, seed_index].mean())
                view_row[f"seed_{seed}_mse"] = reference_mse
                view_row[f"seed_{seed}_headroom_fraction"] = float((reference_mse - view_oracle.mean()) / reference_mse)
            per_view_rows.append(view_row)

        if method == "3dgs":
            diagnostic = str(config["diagnostic_view"])
            if diagnostic not in set(views):
                diagnostic = sorted(set(views))[0]
            mask = views == diagnostic
            with np.load(result_root / "patch_errors.npz", allow_pickle=False) as bundle:
                patches = [Patch(int(x), int(y), int(w), int(h)) for x, y, w, h in zip(bundle["x"][mask], bundle["y"][mask], bundle["width"][mask], bundle["height"][mask])]
            gt = load_rgb(resolve_project_path(config, str(config["reference_gt"])) / diagnostic)
            figure_root.mkdir(parents=True, exist_ok=True)
            save_winner_map(figure_root / f"within_3dgs_{Path(diagnostic).stem}_winner.png", gt.shape[:2], patches, winners[mask], COLORS)

    write_csv(result_root / "within_family_summary.csv", summary_rows)
    write_csv(result_root / "within_family_detailed.csv", detailed_rows)
    write_csv(result_root / "within_family_per_view.csv", per_view_rows)
    atomic_json(result_root / "within_family_summary.json", {"summary": summary_rows, "notes": {
        "independent_training_units": "Three seeds per family.",
        "patch_dependence": "Overlapping patches are not treated as independent replicates; uncertainty resamples complete held-out views.",
        "disagreement": "A seed pair disagrees decisively when absolute patch-MSE difference is at least the configured tie threshold.",
    }})
    print(f"Wrote within-family analysis to {result_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
