"""Side-by-side pilot comparison: Arm A (demo-only) vs Arm B (success-only
mixture) vs current recipe (0.5/0.5), on libero_10.

Archived judgement discipline: compare pilot arms by multi-point diagnostics,
not by one noisy endpoint. This script does not define the formal policy
protocol; final policy numbers must come from one frozen checkpoint.
"""
from __future__ import annotations

import argparse
import pathlib

from aggregate_checkpoint_scan import load_suite_task_counts, mean_std, suite_rate

SUITE = "libero_10"


def scan_rates(scan_root: str, iters: list[int]) -> dict[int, tuple[int, int, float | None]]:
    root = pathlib.Path(scan_root)
    out: dict[int, tuple[int, int, float | None]] = {}
    for it in iters:
        idir = root / f"iter_{it:09d}"
        counts = load_suite_task_counts(str(idir / "*/shards/shard_*/per_task.jsonl"))
        tc = counts.get(SUITE, {})
        if tc:
            out[it] = suite_rate(tc)
    return out


def fmt(cell: tuple[int, int, float | None] | None) -> str:
    if cell is None:
        return "-"
    s, t, r = cell
    return f"{r:.1f}% ({s}/{t})" if r is not None else "-"


def ms(rates: list[float]) -> str:
    if not rates:
        return "-"
    m, sd = mean_std(rates)
    return f"{m:.1f}±{sd:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--armA", required=True)
    ap.add_argument("--armB", required=True)
    ap.add_argument("--current_scan", required=True)
    ap.add_argument("--iters", nargs="+", type=int, default=[6000, 9000, 12000, 15000])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    iters = sorted(args.iters)
    a = scan_rates(args.armA, iters)
    b = scan_rates(args.armB, iters)
    # current recipe: whatever it has at the pilot iters, plus its own extended scan
    cur_iters = sorted(set(iters) | {18000, 21000, 24000, 27000})
    c = scan_rates(args.current_scan, cur_iters)

    a_rates = [a[i][2] for i in iters if i in a and a[i][2] is not None]
    b_rates = [b[i][2] for i in iters if i in b and b[i][2] is not None]
    c_overlap = [c[i][2] for i in iters if i in c and c[i][2] is not None]
    c_all = [c[i][2] for i in cur_iters if i in c and c[i][2] is not None]

    L: list[str] = []
    L.append("# Pilot comparison — recipe discrimination (libero_10)")
    L.append("")
    L.append("> **Archived diagnostic rule:** compare arms by multi-point stability diagnostics.")
    L.append("> This file reports measurements only; it is not the formal policy protocol.")
    L.append("")
    L.append("## libero_10 success by checkpoint")
    L.append("")
    L.append("| iter | Arm A (demo-only) | Arm B (success-only mix) | current recipe (0.5/0.5) |")
    L.append("|---|---:|---:|---:|")
    for it in cur_iters:
        L.append(
            f"| {it} | {fmt(a.get(it))} | {fmt(b.get(it))} | {fmt(c.get(it))} |"
        )
    L.append("")
    L.append("## Multi-point diagnostic mean±std")
    L.append("")
    L.append("| arm | window | mean±std |")
    L.append("|---|---|---:|")
    L.append(f"| Arm A demo-only | {iters} | **{ms(a_rates)}** |")
    L.append(f"| Arm B success-only mix | {iters} | **{ms(b_rates)}** |")
    L.append(f"| current recipe | overlap {iters} | {ms(c_overlap)} |")
    L.append(f"| current recipe | extended 12–27K | {ms(c_all)} |")
    L.append("")
    L.append("## Reading guide")
    L.append("")
    L.append("- If Arm A and Arm B mean±std are both clearly above the current recipe's "
             "overlap mean → failure-replay leakage confirmed; the project should drop failures.")
    L.append("- If Arm A > Arm B → action-supervision density (demo share) matters beyond just "
             "removing failures; consider full demo-only.")
    L.append("- If A ≈ B ≈ current → leakage is not the dominant driver; the ceiling is a "
             "sample-budget / BC-OOD problem → invest in variance governance (Task 3) and more steps,"
             " keep the official recipe.")
    L.append("- Std magnitude is itself a result: a recipe that halves the std is worth a lot even "
             "at equal mean (the oscillation is half the current pain).")
    L.append("")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
