"""Summarize E1.0 dominance audit per-task JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default="outputs/phase1/e1_dominance_audit/expanded_3task_5ep")
    p.add_argument("--output_dir", default="")
    return p.parse_args()


def main() -> None:
    ns = parse_args()
    input_dir = pathlib.Path(ns.input_dir)
    output_dir = pathlib.Path(ns.output_dir) if ns.output_dir else input_dir / "report"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for path in sorted(input_dir.glob("*/*/per_task.jsonl")):
        label = path.parent.name
        suite = path.parent.parent.name
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            rec["suite"] = suite
            rec["condition"] = label
            rec["sr"] = rec["successes"] / rec["trials"] if rec["trials"] else 0.0
            rows.append(rec)

    by_condition: dict[tuple[str, str], dict[str, float]] = {}
    for rec in rows:
        key = (rec["suite"], rec["condition"])
        agg = by_condition.setdefault(key, {"successes": 0, "trials": 0, "n_tasks": 0})
        agg["successes"] += rec["successes"]
        agg["trials"] += rec["trials"]
        agg["n_tasks"] += 1

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["suite", "condition", "successes", "trials", "sr", "n_tasks"])
        writer.writeheader()
        for (suite, condition), agg in sorted(by_condition.items()):
            trials = int(agg["trials"])
            successes = int(agg["successes"])
            writer.writerow(
                {
                    "suite": suite,
                    "condition": condition,
                    "successes": successes,
                    "trials": trials,
                    "sr": successes / trials if trials else 0.0,
                    "n_tasks": int(agg["n_tasks"]),
                }
            )

    md_lines = ["# E1.0 Dominance Audit", "", "| Suite | Condition | SR | Successes / Trials | Tasks |", "|---|---|---:|---:|---:|"]
    for (suite, condition), agg in sorted(by_condition.items()):
        trials = int(agg["trials"])
        successes = int(agg["successes"])
        sr = successes / trials if trials else 0.0
        md_lines.append(f"| {suite} | {condition} | {sr * 100:.1f}% | {successes} / {trials} | {int(agg['n_tasks'])} |")
    (output_dir / "summary.md").write_text("\n".join(md_lines) + "\n")
    print(f"Wrote {csv_path} and {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
