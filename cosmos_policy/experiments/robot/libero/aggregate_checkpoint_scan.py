"""Cross-checkpoint aggregator for standard LIBERO suite evals.

Reads the raw ``per_task.jsonl`` shards produced by
``run_libero_standard_parallel.py`` for a set of checkpoint iterations and
emits an iter x suite success table plus the mean +/- std summary that is the
new official paper-number convention (default: over the last 3 iterations at
5K spacing; configurable).

This is intentionally decoupled from ``generate_standard_report.py``'s CSV
schema -- it re-parses the authoritative jsonl rows (fields: task_suite_name /
task_name, successes, trials, task_id) so it cannot silently break if the
report columns change.

Usage:
    python aggregate_checkpoint_scan.py \
        --scan_root outputs/.../scan \
        --iters 6000 9000 12000 15000 \
        --suites libero_10 \
        --final_window 3 \
        --label "ArmA demo-only"
Layout expected under scan_root:
    scan_root/iter_000006000/<suite>/shards/shard_*/per_task.jsonl
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import math
import pathlib


def load_suite_task_counts(jsonl_glob: str) -> dict[str, dict[str, list[int]]]:
    """suite -> task_name -> [successes, trials]."""
    out: dict[str, dict[str, list[int]]] = collections.defaultdict(
        lambda: collections.defaultdict(lambda: [0, 0])
    )
    for f in sorted(glob.glob(jsonl_glob)):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                suite = str(d.get("task_suite_name", "?"))
                task = str(d.get("task_name", d.get("task_id")))
                out[suite][task][0] += int(d["successes"])
                out[suite][task][1] += int(d["trials"])
    return out


def suite_rate(task_counts: dict[str, list[int]]) -> tuple[int, int, float | None]:
    s = sum(v[0] for v in task_counts.values())
    t = sum(v[1] for v in task_counts.values())
    return s, t, (100.0 * s / t if t else None)


def mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(values) / n
    if n == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (n - 1)  # sample std
    return m, math.sqrt(var)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan_root", required=True)
    ap.add_argument("--iters", nargs="+", type=int, required=True)
    ap.add_argument("--suites", nargs="+", required=True)
    ap.add_argument(
        "--final_window",
        type=int,
        default=3,
        help="Number of trailing iters to use for the official mean+/-std (default 3).",
    )
    ap.add_argument("--label", default="")
    ap.add_argument("--out_name", default="checkpoint_scan_report")
    args = ap.parse_args()

    scan_root = pathlib.Path(args.scan_root)
    iters = sorted(args.iters)

    # iter -> suite -> (s, t, rate)
    per_iter: dict[int, dict[str, tuple[int, int, float | None]]] = {}
    # iter -> suite -> {task: rate%}
    per_iter_tasks: dict[int, dict[str, dict[str, float | None]]] = {}
    for it in iters:
        idir = scan_root / f"iter_{it:09d}"
        counts = load_suite_task_counts(str(idir / "*/shards/shard_*/per_task.jsonl"))
        per_iter[it] = {}
        per_iter_tasks[it] = {}
        for suite in args.suites:
            tc = counts.get(suite, {})
            per_iter[it][suite] = suite_rate(tc)
            per_iter_tasks[it][suite] = {
                task: (100.0 * v[0] / v[1] if v[1] else None) for task, v in sorted(tc.items())
            }

    lines: list[str] = []
    title = "# Checkpoint scan" + (f" — {args.label}" if args.label else "")
    lines += [title, ""]
    lines.append(f"Iters: {', '.join(str(i) for i in iters)}")
    lines.append("")

    # --- suite x iter table -------------------------------------------------
    lines.append("## Suite success rate by checkpoint")
    lines.append("")
    header = "| Suite | " + " | ".join(f"{i}" for i in iters) + f" | mean±std (last {args.final_window}) | mean±std (all) |"
    lines.append(header)
    lines.append("|---|" + "---:|" * (len(iters) + 2))
    for suite in args.suites:
        cells = []
        rates_all = []
        for it in iters:
            s, t, r = per_iter[it][suite]
            cells.append(f"{r:.1f}% ({s}/{t})" if r is not None else "-")
            if r is not None:
                rates_all.append(r)
        window_iters = iters[-args.final_window :]
        rates_win = [per_iter[it][suite][2] for it in window_iters if per_iter[it][suite][2] is not None]
        mw, sw = mean_std(rates_win)
        ma, sa = mean_std(rates_all)
        cells.append(f"**{mw:.1f}±{sw:.1f}**" if rates_win else "-")
        cells.append(f"{ma:.1f}±{sa:.1f}" if rates_all else "-")
        lines.append(f"| {suite} | " + " | ".join(cells) + " |")
    lines.append("")

    # --- per-task breakdown (volatility visibility) -------------------------
    for suite in args.suites:
        tasks = sorted({t for it in iters for t in per_iter_tasks[it][suite]})
        if not tasks:
            continue
        lines.append(f"## Per-task — {suite}")
        lines.append("")
        lines.append("| Task | " + " | ".join(str(i) for i in iters) + " |")
        lines.append("|---|" + "---:|" * len(iters))
        for task in tasks:
            row = []
            for it in iters:
                r = per_iter_tasks[it][suite].get(task)
                row.append(f"{r:.0f}%" if r is not None else "-")
            short = task if len(task) <= 48 else task[:45] + "..."
            lines.append(f"| {short} | " + " | ".join(row) + " |")
        lines.append("")

    report_md = scan_root / f"{args.out_name}.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # CSV: one row per (suite, iter)
    with open(scan_root / f"{args.out_name}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["suite", "iter", "successes", "trials", "success_rate_pct"])
        for suite in args.suites:
            for it in iters:
                s, t, r = per_iter[it][suite]
                w.writerow([suite, it, s, t, f"{r:.3f}" if r is not None else ""])

    print("\n".join(lines))
    print(f"\n[report] {report_md}")


if __name__ == "__main__":
    main()
