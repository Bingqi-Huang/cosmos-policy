"""E0 Tier-1 -- exact numerical verification of the 1/(1+4 lambda) shrinkage law.

paper_outline LOCKED DECISION 11 / Section 6.2:
  "Tier 1 -- exact simulation: numerically reproduce 1/(1+4 lambda) under the
   proposition's exact assumptions (independent per-pair pointwise minimizers);
   a sanity check of the derivation; near-zero cost. ... doubles as instrumentation
   debugging for the at-scale shrinkage measurement."

Exact assumptions (Prop. 2 / Lemma 1):
  Objective per covariant-block pair, minimized pointwise over (D_0, D_p):
    L = 1/2||D_0 - u_0||^2 + 1/2||D_p - u_p||^2 + lambda * ||D_0 - D_p||^2
  First-order conditions give the closed form
    D_bar*       = (u_0 + u_p) / 2                         (mean preserved)
    (D_0 - D_p)* = (u_0 - u_p) / (1 + 4 lambda)            (view-residual shrinkage)
    D_v* - u_v   = -/+ (2 lambda / (1 + 4 lambda)) (u_0 - u_p)   (per-branch bias)

What this script checks, for lambda in a grid:
  1. CLOSED FORM reproduces (D_0 - D_p)* = (u_0 - u_p)/(1+4 lambda) to ~machine eps,
     plus D_bar* = u_bar and the bias formula.
  2. GRADIENT DESCENT on the (convex quadratic) L converges to the same minimizer,
     confirming the law is the actual optimum, not just an algebraic identity.
  3. The REAL at-scale instrumentation (ShrinkageStats from evaluate_scvc_shrinkage)
     fed with these per-pair norms returns global_shrinkage_ratio == 1/(1+4 lambda),
     and the normalized R(lambda)/R(0) == 1/(1+4 lambda) (R(0)=1). This is the
     instrumentation-debug half of Tier-1: it exercises the frozen 5th-percentile
     denominator floor + Pi_Cov projection + RMS norm-pooled ratio exactly as the
     2B-model evaluator will.

Pi_Cov: an optional random orthonormal projection applied identically to predictions
and targets. The law holds per-coordinate, so the ratio is invariant to Pi_Cov; we
include it to mirror the real measurement (which selects the future-scene latent slice)
and to assert that projection does not perturb the law.

Exit 0 = all assertions pass within tolerance; 1 = any failure. CPU only, < 1s.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from evaluate_scvc_shrinkage import ShrinkageStats  # noqa: E402  (sibling module, CPU-light)


def closed_form(u0: np.ndarray, up: np.ndarray, lam: float):
    ubar = 0.5 * (u0 + up)
    delta_pred = (u0 - up) / (1.0 + 4.0 * lam)   # = (D_0 - D_p)*
    d0 = ubar + 0.5 * delta_pred
    dp = ubar - 0.5 * delta_pred
    return d0, dp


def gradient_descent(u0: np.ndarray, up: np.ndarray, lam: float, lr: float, steps: int):
    """Minimize L = 1/2||D0-u0||^2 + 1/2||Dp-up||^2 + lambda||D0-Dp||^2 from a cold start."""
    d0 = np.zeros_like(u0)
    dp = np.zeros_like(up)
    for _ in range(steps):
        g0 = (d0 - u0) + 2.0 * lam * (d0 - dp)
        gp = (dp - up) - 2.0 * lam * (d0 - dp)
        d0 = d0 - lr * g0
        dp = dp - lr * gp
    return d0, dp


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.1, 0.5, 2.0])
    ap.add_argument("--n-pairs", type=int, default=4096)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--proj-dim", type=int, default=0, help="Pi_Cov output dim; 0 = identity (no projection)")
    ap.add_argument("--gd-lr", type=float, default=0.05)
    ap.add_argument("--gd-steps", type=int, default=4000)
    ap.add_argument("--floor-percentile", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--closed-form-tol", type=float, default=1e-9)
    ap.add_argument("--gd-tol", type=float, default=1e-3)
    ap.add_argument("--output-json", default="")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    # Synthetic covariant-block targets: a shared component plus per-view content,
    # so u_0 != u_p with a non-degenerate ||u_0 - u_p|| distribution.
    shared = rng.standard_normal((args.n_pairs, args.dim))
    u0 = shared + 0.7 * rng.standard_normal((args.n_pairs, args.dim))
    up = shared + 0.7 * rng.standard_normal((args.n_pairs, args.dim))

    # Optional Pi_Cov: random orthonormal projection applied to preds and targets alike.
    if args.proj_dim and args.proj_dim < args.dim:
        G = rng.standard_normal((args.dim, args.proj_dim))
        Q, _ = np.linalg.qr(G)            # (dim, proj_dim), orthonormal columns
        proj = lambda X: X @ Q            # noqa: E731
    else:
        proj = lambda X: X                # noqa: E731

    # Frozen denominator floor: 5th percentile of ||Pi_Cov(u_0 - u_p)||, computed once.
    target_diff = proj(u0 - up)
    den_all = np.linalg.norm(target_diff, axis=1)
    floor = float(np.percentile(den_all, args.floor_percentile))

    rows = []
    R0 = None
    all_pass = True
    for lam in args.lambdas:
        d0_cf, dp_cf = closed_form(u0, up, lam)
        d0_gd, dp_gd = gradient_descent(u0, up, lam, args.gd_lr, args.gd_steps)

        ideal = 1.0 / (1.0 + 4.0 * lam)

        # --- direct closed-form checks (per-pair, pre-instrumentation) ---
        pred_diff_cf = (d0_cf - dp_cf)
        # law: (D0-Dp)* == (u0-up)/(1+4lam)
        law_err = np.abs(pred_diff_cf - (u0 - up) * ideal).max()
        # mean preserved: D_bar* == u_bar
        mean_err = np.abs(0.5 * (d0_cf + dp_cf) - 0.5 * (u0 + up)).max()
        # bias: D0*-u0 == -(2lam/(1+4lam))(u0-up)
        bias_err = np.abs((d0_cf - u0) - (-(2.0 * lam / (1.0 + 4.0 * lam)) * (u0 - up))).max()

        # --- GD vs closed form ---
        gd_err = max(np.abs(d0_gd - d0_cf).max(), np.abs(dp_gd - dp_cf).max())

        # --- real instrumentation (ShrinkageStats) on projected norms ---
        num_cf = np.linalg.norm(proj(d0_cf - dp_cf), axis=1)
        den_cf = np.linalg.norm(proj(u0 - up), axis=1)
        stats = ShrinkageStats(checkpoint="tier1", sigma=1.0)
        stats.add(torch.from_numpy(num_cf), torch.from_numpy(den_cf), floor=floor)
        row = stats.row(lambda_cv=lam)
        global_ratio = row["global_shrinkage_ratio"]
        if abs(lam) < 1e-12:
            R0 = global_ratio
        norm_ratio = global_ratio / R0 if R0 else float("nan")

        instr_err = abs(global_ratio - ideal)
        norm_err = abs(norm_ratio - ideal)

        passed = (
            law_err < args.closed_form_tol
            and mean_err < args.closed_form_tol
            and bias_err < args.closed_form_tol
            and gd_err < args.gd_tol
            and instr_err < 1e-6
            and norm_err < 1e-6
        )
        all_pass = all_pass and passed
        rows.append({
            "lambda": lam,
            "ideal_1_over_1_plus_4lambda": round(ideal, 6),
            "instrument_global_ratio": round(global_ratio, 8),
            "instrument_R_over_R0": round(norm_ratio, 8),
            "closed_form_law_err": float(law_err),
            "mean_preserved_err": float(mean_err),
            "bias_formula_err": float(bias_err),
            "gd_vs_closed_form_err": float(gd_err),
            "instrument_vs_ideal_err": float(instr_err),
            "normalized_vs_ideal_err": float(norm_err),
            "num_kept": row["num_kept"],
            "num_excluded": row["num_excluded"],
            "pass": bool(passed),
        })

    report = {
        "verdict": "PASS" if all_pass else "FAIL",
        "config": {
            "n_pairs": args.n_pairs, "dim": args.dim, "proj_dim": args.proj_dim,
            "gd_lr": args.gd_lr, "gd_steps": args.gd_steps,
            "floor_percentile": args.floor_percentile, "denominator_floor": floor,
            "seed": args.seed,
        },
        "rows": rows,
        "notes": (
            "Closed form and GD both reproduce (D0-Dp)*=(u0-up)/(1+4L); the real "
            "ShrinkageStats instrument returns global_ratio==1/(1+4L) and R(L)/R(0)==1/(1+4L). "
            "Tier-1 confirms the derivation and the at-scale measurement code; it does NOT "
            "test whether SGD on a shared network reaches this optimum (that is Tier-2)."
        ),
    }
    print(json.dumps(report, indent=2))
    if args.output_json:
        out = pathlib.Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
