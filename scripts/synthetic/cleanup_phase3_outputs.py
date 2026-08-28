#!/usr/bin/env python3
"""Safely list or remove non-canonical Phase-III synthetic pilot outputs."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import inventory_phase3_outputs as inventory

LOG_SUFFIXES = {".out", ".err"}


@dataclass
class CleanupCandidate:
    kind: str
    path: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/synthetic/phase3_controlled_pilot"),
        help="Phase-III synthetic dataset root containing manifest.csv/json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/synthetic/phase3_controlled_pilot"),
        help="Only directories under this Phase-III output root are eligible for cleanup.",
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("jobs/logs"),
        help="Root to scan recursively for obsolete Phase-III .out/.err logs.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/synthetic/phase3_controlled_pilot"),
        help="Directory where cleanup_plan.{md,json} will be written.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete listed candidates. Default is dry-run only.",
    )
    parser.add_argument(
        "--skip-logs",
        action="store_true",
        help="Only identify output directories; do not scan obsolete Slurm logs.",
    )
    return parser.parse_args()


def resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def find_selected_outputs(project_root: Path, dataset_root: Path, output_root: Path) -> tuple[set[Path], list[dict[str, Any]]]:
    manifest = inventory.load_manifest(dataset_root)
    selected: set[Path] = set()
    records: list[dict[str, Any]] = []

    for row in manifest:
        scene_id = str(row["scene_id"])
        expected_test_views = inventory.as_int(row, "test_view_count", 8)
        for method in inventory.METHODS:
            expected_iteration = inventory.EXPECTED_ITERATIONS[method]
            method_root = output_root / method
            candidate_paths = inventory.find_candidates(method_root, scene_id)
            statuses = [
                inventory.candidate_status(path, method, expected_iteration, expected_test_views)
                for path in candidate_paths
            ]
            chosen = max(statuses, key=inventory.candidate_sort_key) if statuses else None
            if chosen and chosen.path:
                chosen_path = Path(chosen.path).resolve()
                selected.add(chosen_path)
                selected_output = rel(chosen_path, project_root)
            else:
                selected_output = None
            records.append(
                {
                    "scene_id": scene_id,
                    "method": method,
                    "selected_output": selected_output,
                    "status": chosen.status if chosen else "missing",
                    "final_iteration": chosen.final_iteration if chosen else None,
                    "render_count": chosen.render_count if chosen else 0,
                    "gt_count": chosen.gt_count if chosen else 0,
                    "metrics_found": chosen.metrics_found if chosen else False,
                }
            )
    return selected, records


def find_output_cleanup_candidates(project_root: Path, output_root: Path, selected: set[Path]) -> list[CleanupCandidate]:
    candidates: list[CleanupCandidate] = []
    output_root_resolved = output_root.resolve()
    dataset_roots = [project_root / "datasets", project_root / "data"]

    for method in inventory.METHODS:
        method_root = output_root / method
        if not method_root.exists():
            continue
        for child in sorted(method_root.iterdir()):
            if not child.is_dir():
                continue
            child_resolved = child.resolve()
            if child_resolved in selected:
                continue
            if not within(child_resolved, output_root_resolved):
                continue
            if any(within(child_resolved, dataset_root) for dataset_root in dataset_roots if dataset_root.exists()):
                continue
            candidates.append(
                CleanupCandidate(
                    kind="output_dir",
                    path=rel(child_resolved, project_root),
                    reason="not selected as canonical by current Phase-III inventory",
                )
            )
    return candidates


def find_obsolete_log_candidates(project_root: Path, logs_root: Path) -> list[CleanupCandidate]:
    candidates: list[CleanupCandidate] = []
    if not logs_root.exists():
        return candidates

    keep_job_ids = set(inventory.FINAL_SUCCESSFUL_ARRAYS.values()) | {inventory.LOADER_CHECK_JOB}
    obsolete_job_ids = set(inventory.SUPERSEDED_ARRAYS)

    for path in sorted(logs_root.rglob("*")):
        if not path.is_file() or path.suffix not in LOG_SUFFIXES:
            continue
        text = str(path)
        if any(job_id in text for job_id in keep_job_ids):
            continue
        matched = [job_id for job_id in obsolete_job_ids if job_id in text]
        if matched:
            candidates.append(
                CleanupCandidate(
                    kind="log_file",
                    path=rel(path, project_root),
                    reason="obsolete Phase-III failed/mixed Slurm array log: " + ",".join(matched),
                )
            )
    return candidates


def write_reports(results_dir: Path, project_root: Path, payload: dict[str, Any]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "cleanup_plan.json"
    md_path = results_dir / "cleanup_plan.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Phase-III Cleanup Plan",
        "",
        f"Generated: {payload['generated_at']}",
        f"Mode: {'DELETE' if payload['delete_requested'] else 'DRY-RUN'}",
        "",
        "This plan never includes selected canonical outputs from the current Phase-III inventory and never scans dataset directories.",
        "",
        "## Summary",
        "",
        f"- Output directories: {payload['summary']['output_dir_count']}",
        f"- Obsolete log files: {payload['summary']['log_file_count']}",
        f"- Total candidates: {payload['summary']['total_count']}",
        "",
        "## Candidates",
        "",
        "| Kind | Path | Reason |",
        "| --- | --- | --- |",
    ]
    for item in payload["candidates"]:
        lines.append(f"| {item['kind']} | `{item['path']}` | {item['reason']} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {rel(md_path, project_root)}")
    print(f"Wrote {rel(json_path, project_root)}")


def delete_candidates(project_root: Path, output_root: Path, logs_root: Path, candidates: list[CleanupCandidate]) -> None:
    selected_root = output_root.resolve()
    logs_root_resolved = logs_root.resolve()
    for candidate in candidates:
        path = (project_root / candidate.path).resolve()
        if candidate.kind == "output_dir":
            if not within(path, selected_root):
                raise RuntimeError(f"Refusing to delete output outside Phase-III output root: {path}")
            if not path.is_dir():
                print(f"Skipping missing output directory: {path}")
                continue
            print(f"Deleting output directory: {path}")
            shutil.rmtree(path)
        elif candidate.kind == "log_file":
            if not within(path, logs_root_resolved):
                raise RuntimeError(f"Refusing to delete log outside logs root: {path}")
            if not path.is_file():
                print(f"Skipping missing log file: {path}")
                continue
            print(f"Deleting log file: {path}")
            path.unlink()
        else:
            raise RuntimeError(f"Unknown cleanup candidate kind: {candidate.kind}")


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    dataset_root = resolve_path(project_root, args.dataset_root)
    output_root = resolve_path(project_root, args.output_root)
    logs_root = resolve_path(project_root, args.logs_root)
    results_dir = resolve_path(project_root, args.results_dir)

    if within(output_root, dataset_root) or within(dataset_root, output_root):
        raise RuntimeError("Output root and dataset root overlap; refusing to build cleanup plan.")

    selected_outputs, inventory_records = find_selected_outputs(project_root, dataset_root, output_root)
    output_candidates = find_output_cleanup_candidates(project_root, output_root, selected_outputs)
    log_candidates = [] if args.skip_logs else find_obsolete_log_candidates(project_root, logs_root)
    candidates = output_candidates + log_candidates

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "delete_requested": bool(args.delete),
        "project_root": str(project_root),
        "dataset_root": rel(dataset_root, project_root),
        "output_root": rel(output_root, project_root),
        "logs_root": rel(logs_root, project_root),
        "results_dir": rel(results_dir, project_root),
        "selected_outputs": sorted(rel(path, project_root) for path in selected_outputs),
        "inventory_records": inventory_records,
        "summary": {
            "canonical_output_count": len(selected_outputs),
            "output_dir_count": len(output_candidates),
            "log_file_count": len(log_candidates),
            "total_count": len(candidates),
        },
        "candidates": [asdict(candidate) for candidate in candidates],
    }

    print("Phase-III cleanup candidate plan")
    print(f"Mode: {'DELETE' if args.delete else 'DRY-RUN'}")
    print(f"Canonical outputs protected: {len(selected_outputs)}")
    print(f"Output directories listed: {len(output_candidates)}")
    print(f"Obsolete log files listed: {len(log_candidates)}")
    print("")
    for candidate in candidates:
        action = "would remove" if not args.delete else "will remove"
        print(f"{action} {candidate.kind}: {candidate.path} ({candidate.reason})")
    if not candidates:
        print("No cleanup candidates found.")

    write_reports(results_dir, project_root, payload)

    if args.delete:
        delete_candidates(project_root, output_root, logs_root, candidates)
    else:
        print("Dry-run only. Re-run with --delete to remove exactly the listed candidates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
