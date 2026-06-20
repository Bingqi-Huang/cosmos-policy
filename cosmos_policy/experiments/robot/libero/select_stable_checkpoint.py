"""Select one deployable checkpoint from stability-validation reports.

This consumes per-checkpoint ``report_final/aggregate_results.csv`` artifacts
created by ``run_scene_only_stability_eval.sh``. It never averages checkpoints;
the output is one selected checkpoint directory plus a diagnostic report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re
import sys
from dataclasses import asdict, dataclass


ITER_RE = re.compile(r"iter_(\d{9})$")


@dataclass
class Candidate:
    iter_name: str
    iteration: int
    checkpoint_path: str
    total_success_rate: float
    libero_10_success_rate: float
    max_neighbor_delta_pp: float | None
    passes_floor: bool
    passes_neighbor_check: bool

    @property
    def selectable(self) -> bool:
        return self.passes_floor and self.passes_neighbor_check


def read_aggregate(report_csv: pathlib.Path) -> dict[str, float]:
    rows: dict[str, float] = {}
    with report_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[str(row["suite"])] = float(row["success_rate"])
    required = {"Grand Total", "libero_10"}
    missing = sorted(required - set(rows))
    if missing:
        raise ValueError(f"{report_csv} missing aggregate rows: {missing}")
    return rows


def discover_iters(eval_root: pathlib.Path, explicit_iters: list[int] | None) -> list[tuple[int, str, pathlib.Path]]:
    discovered: list[tuple[int, str, pathlib.Path]] = []
    if explicit_iters:
        for iteration in sorted(explicit_iters):
            iter_name = f"iter_{iteration:09d}"
            discovered.append((iteration, iter_name, eval_root / iter_name))
        return discovered

    for path in sorted(eval_root.iterdir() if eval_root.exists() else []):
        if not path.is_dir():
            continue
        match = ITER_RE.match(path.name)
        if match:
            discovered.append((int(match.group(1)), path.name, path))
    return discovered


def checkpoint_path_for(ckpt_root: str, iter_name: str, iter_dir: pathlib.Path) -> str:
    if ckpt_root:
        return str(pathlib.Path(ckpt_root) / iter_name)
    metadata = iter_dir / "checkpoint_path.txt"
    if metadata.exists():
        return metadata.read_text(encoding="utf-8").strip()
    return ""


def build_candidates(
    eval_root: pathlib.Path,
    ckpt_root: str,
    iters: list[tuple[int, str, pathlib.Path]],
    min_libero10: float,
    max_neighbor_delta_pp: float,
) -> list[Candidate]:
    raw: dict[int, tuple[str, pathlib.Path, float, float]] = {}
    for iteration, iter_name, iter_dir in iters:
        report_csv = iter_dir / "report_final" / "aggregate_results.csv"
        if not report_csv.exists():
            print(f"[warn] missing report: {report_csv}", file=sys.stderr)
            continue
        rates = read_aggregate(report_csv)
        raw[iteration] = (
            iter_name,
            iter_dir,
            rates["Grand Total"],
            rates["libero_10"],
        )

    candidates: list[Candidate] = []
    ordered_iters = sorted(raw)
    for pos, iteration in enumerate(ordered_iters):
        iter_name, iter_dir, total_rate, libero10_rate = raw[iteration]
        neighbor_deltas: list[float] = []
        if pos > 0:
            prev_rate = raw[ordered_iters[pos - 1]][3]
            neighbor_deltas.append(abs(libero10_rate - prev_rate) * 100.0)
        if pos + 1 < len(ordered_iters):
            next_rate = raw[ordered_iters[pos + 1]][3]
            neighbor_deltas.append(abs(libero10_rate - next_rate) * 100.0)
        max_delta = max(neighbor_deltas) if neighbor_deltas else None
        candidates.append(
            Candidate(
                iter_name=iter_name,
                iteration=iteration,
                checkpoint_path=checkpoint_path_for(ckpt_root, iter_name, iter_dir),
                total_success_rate=total_rate,
                libero_10_success_rate=libero10_rate,
                max_neighbor_delta_pp=max_delta,
                passes_floor=libero10_rate >= min_libero10,
                passes_neighbor_check=max_delta is None or max_delta <= max_neighbor_delta_pp,
            )
        )
    return candidates


def write_outputs(eval_root: pathlib.Path, candidates: list[Candidate], selected: Candidate | None, args) -> None:
    rows = [asdict(candidate) | {"selectable": candidate.selectable} for candidate in candidates]
    (eval_root / "checkpoint_selection.json").write_text(
        json.dumps(
            {
                "selected": asdict(selected) if selected else None,
                "criteria": {
                    "min_libero_10_success_rate": args.min_libero10,
                    "max_neighbor_delta_pp": args.max_neighbor_delta_pp,
                },
                "candidates": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    csv_path = eval_root / "checkpoint_selection.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "iter_name",
            "iteration",
            "checkpoint_path",
            "total_success_rate",
            "libero_10_success_rate",
            "max_neighbor_delta_pp",
            "passes_floor",
            "passes_neighbor_check",
            "selectable",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [
        "# Stability Checkpoint Selection",
        "",
        "This report selects one checkpoint. It does not average checkpoints.",
        "",
        f"- Minimum LIBERO-10 validation success rate: {100 * args.min_libero10:.1f}%",
        f"- Maximum adjacent LIBERO-10 jump: {args.max_neighbor_delta_pp:.1f} percentage points",
        "",
    ]
    if selected:
        lines.extend(
            [
                f"Selected: `{selected.iter_name}`",
                f"Checkpoint: `{selected.checkpoint_path}`",
                f"Grand total validation success: {100 * selected.total_success_rate:.1f}%",
                f"LIBERO-10 validation success: {100 * selected.libero_10_success_rate:.1f}%",
                "",
            ]
        )
    else:
        lines.extend(["Selected: none", "", "No checkpoint passed the stability criteria.", ""])

    lines.extend(
        [
            "| Checkpoint | Grand Total | LIBERO-10 | Max Neighbor Jump | Selectable |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for candidate in candidates:
        delta = "-" if candidate.max_neighbor_delta_pp is None else f"{candidate.max_neighbor_delta_pp:.1f} pp"
        lines.append(
            f"| `{candidate.iter_name}` | "
            f"{100 * candidate.total_success_rate:.1f}% | "
            f"{100 * candidate.libero_10_success_rate:.1f}% | "
            f"{delta} | "
            f"{'yes' if candidate.selectable else 'no'} |"
        )
    (eval_root / "checkpoint_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True, help="Root containing iter_*/report_final artifacts.")
    parser.add_argument("--ckpt-root", default="", help="Optional checkpoint root used to fill selected path.")
    parser.add_argument("--iters", nargs="*", type=int, help="Explicit checkpoint iterations to consider.")
    parser.add_argument(
        "--min-libero10",
        type=float,
        default=0.10,
        help="Minimum LIBERO-10 validation success rate for a selectable checkpoint.",
    )
    parser.add_argument(
        "--max-neighbor-delta-pp",
        type=float,
        default=25.0,
        help="Maximum allowed adjacent-checkpoint LIBERO-10 jump in percentage points.",
    )
    args = parser.parse_args()

    eval_root = pathlib.Path(args.eval_root)
    discovered = discover_iters(eval_root, args.iters)
    if not discovered:
        raise SystemExit(f"No checkpoint eval directories found under {eval_root}")

    candidates = build_candidates(
        eval_root=eval_root,
        ckpt_root=args.ckpt_root,
        iters=discovered,
        min_libero10=args.min_libero10,
        max_neighbor_delta_pp=args.max_neighbor_delta_pp,
    )
    if not candidates:
        raise SystemExit("No candidates had complete aggregate reports")

    selectable = [candidate for candidate in candidates if candidate.selectable]
    selected = max(
        selectable,
        key=lambda c: (c.total_success_rate, c.libero_10_success_rate, c.iteration),
        default=None,
    )
    write_outputs(eval_root, candidates, selected, args)

    if selected:
        print(f"[select] {selected.iter_name} {selected.checkpoint_path}")
        return

    print(f"[select][none] wrote diagnostics under {eval_root}", file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
