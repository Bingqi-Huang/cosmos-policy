"""Prepare LIBERO-Plus camera subset task-name files for Cosmos eval.

The source format is the v4-plus ``subset_tasks.jsonl`` file.  The output is
compatible with ``run_libero_camera_parallel.py``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter, defaultdict
from typing import Any


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[6]
DEFAULT_SUBSET_JSONL = (
    PROJECT_ROOT / "view-invariant_vla" / "code" / "results" / "libero_plus_subsets" / "screen_120_s7" / "subset_tasks.jsonl"
)


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset_jsonl",
        default=str(DEFAULT_SUBSET_JSONL),
    )
    parser.add_argument("--output_dir", default="outputs/phase1/libero_plus_subsets/screen_120_s7")
    args = parser.parse_args()

    subset_path = pathlib.Path(args.subset_jsonl)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(subset_path)
    by_suite: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_suite[str(row["suite_name"])].append(str(row["task_name"]))

    for suite, names in sorted(by_suite.items()):
        (output_dir / f"camera_task_names_{suite}.json").write_text(
            json.dumps(names, indent=2) + "\n",
            encoding="utf-8",
        )
    (output_dir / "camera_task_names_all.json").write_text(
        json.dumps([str(row["task_name"]) for row in rows], indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "subset_tasks.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    condition_counts = Counter(str(row["condition"]) for row in rows)
    suite_counts = Counter(str(row["suite_name"]) for row in rows)
    level_counts = Counter(str(row["level"]) for row in rows)
    metadata = {
        "source_subset_jsonl": str(subset_path),
        "num_tasks": len(rows),
        "condition_counts": dict(sorted(condition_counts.items())),
        "suite_counts": dict(sorted(suite_counts.items())),
        "level_counts": dict(sorted(level_counts.items())),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LIBERO-Plus Screen Subset",
        "",
        f"- Source: `{subset_path}`",
        f"- Tasks: {len(rows)}",
        "",
        "## By Suite",
        "",
    ]
    for suite, count in sorted(suite_counts.items()):
        lines.append(f"- {suite}: {count}")
    lines.extend(["", "## By Condition", ""])
    for condition, count in sorted(condition_counts.items()):
        lines.append(f"- {condition}: {count}")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[subset] wrote {len(rows)} tasks to {output_dir}")


if __name__ == "__main__":
    main()
