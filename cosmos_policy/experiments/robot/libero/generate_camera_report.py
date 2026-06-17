"""Generate condition/level breakdown report from LIBERO-Plus camera eval results.

Reads per-task JSONL files (one per shard, or merged), joins with task_classification.json
to get condition (C1/C2/C3) and difficulty level (1-5), then outputs:
  - results.csv  — per-condition-level aggregated SR with Wilson 95% CI
  - summary.md   — markdown table matching the original view-invariant_vla report format

Condition definitions (same as original):
  C1: scale_factor != 100    (zoom perturbation)
  C2: horizon_view != 0 or vertical_view != 0   (pan/tilt perturbation)
  C3: end_point_rot != 0 or end_point_vertical != 0   (endpoint rotation)

Can also parse existing log files (--log_files) to extract per-task results without JSONL.

Usage:
    python generate_camera_report.py \
        --jsonl_files shard_00/tasks.jsonl shard_01/tasks.jsonl ... \
        --task_classification /path/to/task_classification.json \
        --output_dir outputs/phase0/report

    # Or from logs directly (for already-running evals):
    python generate_camera_report.py \
        --log_files stdout/camera_full_spatial.log stdout/camera_full_object.log ... \
        --task_classification /path/to/task_classification.json \
        --output_dir outputs/phase0/report
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re
import sys
from collections import defaultdict
from typing import Any


CAMERA_TASK_RE = re.compile(
    r"(?P<base>.+)_view_(?P<hv>-?\d+)_(?P<vv>-?\d+)_(?P<scale>\d+)_(?P<rotz>-?\d+)_(?P<roty>-?\d+)_initstate_\d+$"
)


def _classify_condition(task_name: str) -> str | None:
    m = CAMERA_TASK_RE.match(task_name)
    if m is None:
        return None
    hv, vv = int(m.group("hv")), int(m.group("vv"))
    scale = int(m.group("scale"))
    rotz, roty = int(m.group("rotz")), int(m.group("roty"))
    if scale != 100:
        return "C1"
    if rotz != 0 or roty != 0:
        return "C3"
    if hv != 0 or vv != 0:
        return "C2"
    return None  # nominal (hv=vv=0, scale=100, rotz=roty=0) — skip


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def load_task_classification(path: str | pathlib.Path) -> dict[str, dict]:
    """Return {task_name -> {difficulty_level, suite_name}} from task_classification.json."""
    raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    mapping: dict[str, dict] = {}
    for suite_name, entries in raw.items():
        for entry in entries:
            if entry.get("category") == "Camera Viewpoints":
                mapping[entry["name"]] = {
                    "difficulty_level": int(entry["difficulty_level"]),
                    "suite_name": suite_name,
                }
    return mapping


def parse_jsonl_files(paths: list[str | pathlib.Path]) -> list[dict]:
    """Read per-task records from one or more JSONL files."""
    records = []
    for p in paths:
        p = pathlib.Path(p)
        if not p.exists():
            print(f"[warn] JSONL not found: {p}", file=sys.stderr)
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_log_files(paths: list[str | pathlib.Path]) -> list[dict]:
    """Extract per-task results by parsing eval log files.

    Looks for pairs:
        Camera task: <task_name>
        ...
        Camera task SR: <succ>/<total> (...)
    """
    records = []
    task_sr_re = re.compile(r"Camera task SR:\s*(\d+)/(\d+)")
    task_name_re = re.compile(r"Camera task:\s*(\S+)")

    for p in paths:
        p = pathlib.Path(p)
        if not p.exists():
            print(f"[warn] log not found: {p}", file=sys.stderr)
            continue
        current_task = None
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            m = task_name_re.search(line)
            if m:
                current_task = m.group(1).strip()
            m = task_sr_re.search(line)
            if m and current_task:
                succ, total = int(m.group(1)), int(m.group(2))
                records.append({"task_name": current_task, "successes": succ, "trials": total})
                current_task = None  # reset; next SR line would be a new task
    return records


def aggregate(
    records: list[dict],
    task_meta: dict[str, dict],
) -> tuple[list[dict], list[dict]]:
    """Aggregate per-task records into per-(condition, level) and per-condition rows."""
    # Deduplicate: if same task_name appears multiple times (e.g. shard overlap), keep last
    seen: dict[str, dict] = {}
    for r in records:
        seen[r["task_name"]] = r

    by_cond_level: dict[tuple[str, int], list[dict]] = defaultdict(list)
    by_cond: dict[str, list[dict]] = defaultdict(list)
    by_suite: dict[str, list[dict]] = defaultdict(list)
    all_rows: list[dict] = []

    skipped = 0
    for task_name, r in seen.items():
        cond = _classify_condition(task_name)
        if cond is None:
            skipped += 1
            continue
        meta = task_meta.get(task_name, {})
        level = int(meta.get("difficulty_level", -1))
        suite = meta.get("suite_name", "unknown")
        by_cond_level[(cond, level)].append(r)
        by_cond[cond].append(r)
        by_suite[suite].append(r)
        all_rows.append(r)

    if skipped:
        print(f"[info] skipped {skipped} tasks with no classifiable condition (nominal?)", file=sys.stderr)

    agg_rows: list[dict] = []

    # Per (condition, level)
    for (cond, level), rows in sorted(by_cond_level.items()):
        n_total = sum(r["trials"] for r in rows)
        n_succ = sum(r["successes"] for r in rows)
        ci_lo, ci_hi = wilson_ci(n_succ, n_total)
        agg_rows.append({
            "condition": cond, "level": level, "suite": "all",
            "n_tasks": len(rows), "n_success": n_succ, "n_total": n_total,
            "success_rate": n_succ / n_total if n_total > 0 else 0.0,
            "ci_low": ci_lo, "ci_high": ci_hi,
        })

    # Per condition aggregate
    for cond, rows in sorted(by_cond.items()):
        n_total = sum(r["trials"] for r in rows)
        n_succ = sum(r["successes"] for r in rows)
        ci_lo, ci_hi = wilson_ci(n_succ, n_total)
        agg_rows.append({
            "condition": cond, "level": "Aggregate", "suite": "all",
            "n_tasks": len(rows), "n_success": n_succ, "n_total": n_total,
            "success_rate": n_succ / n_total if n_total > 0 else 0.0,
            "ci_low": ci_lo, "ci_high": ci_hi,
        })

    # Per suite aggregate
    suite_rows_list: list[dict] = []
    for suite_name, rows in sorted(by_suite.items()):
        n_total = sum(r["trials"] for r in rows)
        n_succ = sum(r["successes"] for r in rows)
        ci_lo, ci_hi = wilson_ci(n_succ, n_total)
        suite_rows_list.append({
            "condition": "Overall", "level": suite_name, "suite": suite_name,
            "n_tasks": len(rows), "n_success": n_succ, "n_total": n_total,
            "success_rate": n_succ / n_total if n_total > 0 else 0.0,
            "ci_low": ci_lo, "ci_high": ci_hi,
        })

    # Grand total
    n_total_all = sum(r["trials"] for r in all_rows)
    n_succ_all = sum(r["successes"] for r in all_rows)
    ci_lo, ci_hi = wilson_ci(n_succ_all, n_total_all)
    suite_rows_list.append({
        "condition": "Overall", "level": "Grand Total", "suite": "all",
        "n_tasks": len(seen), "n_success": n_succ_all, "n_total": n_total_all,
        "success_rate": n_succ_all / n_total_all if n_total_all > 0 else 0.0,
        "ci_low": ci_lo, "ci_high": ci_hi,
    })

    return agg_rows, suite_rows_list


def write_csv(rows: list[dict], path: pathlib.Path) -> None:
    fieldnames = ["condition", "level", "suite", "n_tasks", "n_success", "n_total",
                  "success_rate", "ci_low", "ci_high"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_markdown(agg_rows: list[dict], suite_rows: list[dict], path: pathlib.Path) -> None:
    lines = [
        "# LIBERO-Plus Camera Evaluation",
        "",
        "## Breakdown by Condition × Difficulty Level",
        "",
        "| Condition | Level | SR (95% Wilson CI) | n_success / n_trials | n_tasks |",
        "|---|---:|---:|---:|---:|",
    ]

    def _level_sort_key(v: Any) -> tuple[int, str]:
        return (0, f"{v:02d}") if isinstance(v, int) else (1, str(v))

    for row in sorted(agg_rows, key=lambda r: (r["condition"], _level_sort_key(r["level"]))):
        sr = float(row["success_rate"])
        lo, hi = float(row["ci_low"]), float(row["ci_high"])
        lines.append(
            f"| {row['condition']} | {row['level']} | "
            f"{sr*100:.1f}% ([{lo*100:.1f}, {hi*100:.1f}]) | "
            f"{row['n_success']} / {row['n_total']} | {row['n_tasks']} |"
        )

    lines += [
        "",
        "## Breakdown by Suite",
        "",
        "| Suite | SR (95% Wilson CI) | n_success / n_trials | n_tasks |",
        "|---|---:|---:|---:|",
    ]
    for row in suite_rows:
        sr = float(row["success_rate"])
        lo, hi = float(row["ci_low"]), float(row["ci_high"])
        lines.append(
            f"| {row['level']} | "
            f"{sr*100:.1f}% ([{lo*100:.1f}, {hi*100:.1f}]) | "
            f"{row['n_success']} / {row['n_total']} | {row['n_tasks']} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--jsonl_files", nargs="+", help="Per-task JSONL files from --per_task_jsonl")
    g.add_argument("--log_files", nargs="+", help="Eval stdout log files to parse")
    p.add_argument(
        "--task_classification",
        default=str(
            pathlib.Path("~/.cache/huggingface/hub/datasets--Sylvest--libero_plus_data_4suite/task_classification.json").expanduser()
        ),
        help="Path to LIBERO-Plus task_classification.json",
    )
    p.add_argument("--output_dir", default="outputs/phase0/camera_report", help="Directory for output files")
    args = p.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_meta = load_task_classification(args.task_classification)
    print(f"[info] loaded metadata for {len(task_meta)} camera tasks from task_classification.json")

    if args.jsonl_files:
        records = parse_jsonl_files(args.jsonl_files)
    else:
        records = parse_log_files(args.log_files)
    print(f"[info] loaded {len(records)} per-task records")

    agg_rows, suite_rows = aggregate(records, task_meta)

    csv_path = output_dir / "results.csv"
    md_path = output_dir / "summary.md"
    write_csv(agg_rows + suite_rows, csv_path)
    write_markdown(agg_rows, suite_rows, md_path)

    print(f"\n[output] CSV:      {csv_path}")
    print(f"[output] Markdown: {md_path}")
    print()
    # Print summary to stdout
    for row in suite_rows:
        sr = float(row["success_rate"])
        print(f"  {row['level']:30s}  SR={sr*100:.1f}%  ({row['n_success']}/{row['n_total']})")


if __name__ == "__main__":
    main()
