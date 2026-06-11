"""Aggregate standard LIBERO suite eval JSONL records.

This complements ``generate_camera_report.py``.  The input records are the
``per_task_jsonl`` rows emitted by ``run_libero_eval.py`` in standard
task-index mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
from collections import defaultdict
from typing import Any


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def parse_jsonl_files(paths: list[str | pathlib.Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_path in paths:
        path = pathlib.Path(raw_path)
        if not path.exists():
            print(f"[warn] JSONL not found: {path}", file=sys.stderr)
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def aggregate(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Deduplicate within repeated report generation; keep the last record for a
    # suite/task pair, matching the camera report behavior.
    by_task: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        if record.get("mode") not in {None, "standard"}:
            continue
        suite = str(record["task_suite_name"])
        task_id = int(record["task_id"])
        by_task[(suite, task_id)] = record

    per_task_rows: list[dict[str, Any]] = []
    by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (suite, task_id), record in sorted(by_task.items()):
        successes = int(record["successes"])
        trials = int(record["trials"])
        ci_low, ci_high = wilson_ci(successes, trials)
        row = {
            "suite": suite,
            "task_id": task_id,
            "task_name": record.get("task_name", ""),
            "task_description": record.get("task_description", ""),
            "n_success": successes,
            "n_total": trials,
            "success_rate": successes / trials if trials else 0.0,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
        per_task_rows.append(row)
        by_suite[suite].append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for suite, rows in sorted(by_suite.items()):
        n_success = sum(int(row["n_success"]) for row in rows)
        n_total = sum(int(row["n_total"]) for row in rows)
        ci_low, ci_high = wilson_ci(n_success, n_total)
        aggregate_rows.append(
            {
                "suite": suite,
                "n_tasks": len(rows),
                "n_success": n_success,
                "n_total": n_total,
                "success_rate": n_success / n_total if n_total else 0.0,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    n_success = sum(int(row["n_success"]) for row in aggregate_rows)
    n_total = sum(int(row["n_total"]) for row in aggregate_rows)
    ci_low, ci_high = wilson_ci(n_success, n_total)
    aggregate_rows.append(
        {
            "suite": "Grand Total",
            "n_tasks": sum(int(row["n_tasks"]) for row in aggregate_rows),
            "n_success": n_success,
            "n_total": n_total,
            "success_rate": n_success / n_total if n_total else 0.0,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    )
    return per_task_rows, aggregate_rows


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: pathlib.Path, aggregate_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# LIBERO Standard Evaluation",
        "",
        "| Suite | Success Rate (95% Wilson CI) | n_success / n_total | Tasks |",
        "|---|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            "| "
            f"{row['suite']} | "
            f"{100 * float(row['success_rate']):.1f}% "
            f"([{100 * float(row['ci_low']):.1f}, {100 * float(row['ci_high']):.1f}]) | "
            f"{row['n_success']} / {row['n_total']} | "
            f"{row['n_tasks']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl_files", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    records = parse_jsonl_files(args.jsonl_files)
    per_task_rows, aggregate_rows = aggregate(records)

    write_csv(output_dir / "per_task_results.csv", per_task_rows)
    write_csv(output_dir / "aggregate_results.csv", aggregate_rows)
    write_summary(output_dir / "summary.md", aggregate_rows)
    (output_dir / "merged_metadata.json").write_text(
        json.dumps(
            {
                "jsonl_files": [str(path) for path in args.jsonl_files],
                "num_input_records": len(records),
                "num_per_task_records": len(per_task_rows),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[report] Wrote standard LIBERO report to {output_dir}")


if __name__ == "__main__":
    main()
