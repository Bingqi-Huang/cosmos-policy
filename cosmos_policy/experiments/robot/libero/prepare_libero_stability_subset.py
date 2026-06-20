"""Prepare the fixed LIBERO validation subset for baseline stability checks.

The subset is intentionally small enough for repeated checkpoint selection, but
still covers every standard LIBERO suite and every task in those suites. It is
not a replacement for the final 50-trial evaluation.
"""

from __future__ import annotations

import argparse
import json
import pathlib


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TASK_IDS = list(range(10))


def write_json(path: pathlib.Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="outputs/phase1/libero_validation_subsets/stability_600_v1",
        help="Directory for task-index files and manifest.",
    )
    parser.add_argument(
        "--trials-per-task",
        type=int,
        default=15,
        help="Rollouts per task used by the stability validation launcher.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Protocol seed recorded in the manifest.")
    args = parser.parse_args()

    output_root = pathlib.Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    task_files: dict[str, str] = {}
    for suite in SUITES:
        path = output_root / f"{suite}_task_indices.json"
        write_json(path, TASK_IDS)
        task_files[suite] = str(path)

    total_rollouts = len(SUITES) * len(TASK_IDS) * args.trials_per_task
    manifest = {
        "name": output_root.name,
        "purpose": "checkpoint_selection_and_training_stability_validation_only",
        "standard_suites": list(SUITES),
        "task_indices": TASK_IDS,
        "task_files": task_files,
        "trials_per_task": args.trials_per_task,
        "total_tasks": len(SUITES) * len(TASK_IDS),
        "total_rollouts": total_rollouts,
        "seed": args.seed,
        "final_eval_replacement": False,
        "notes": [
            "Uses all 10 tasks from each standard LIBERO suite.",
            "Uses the first N default initial states per task through the standard LIBERO evaluator.",
            "Use only to select one frozen checkpoint and diagnose instability.",
        ],
    }
    write_json(output_root / "manifest.json", manifest)

    lines = [
        "# LIBERO Stability Validation Subset",
        "",
        f"- Suites: {', '.join(SUITES)}",
        f"- Tasks per suite: {len(TASK_IDS)}",
        f"- Trials per task: {args.trials_per_task}",
        f"- Total rollouts: {total_rollouts}",
        "- Scope: validation-only checkpoint selection, not final benchmark reporting.",
        "",
        "Task-index files:",
    ]
    for suite, path in task_files.items():
        lines.append(f"- `{suite}`: `{path}`")
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[subset] wrote {output_root}")
    print(f"[subset] total_rollouts={total_rollouts}")


if __name__ == "__main__":
    main()
