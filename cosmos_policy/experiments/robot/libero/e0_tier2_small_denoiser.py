"""E0 Tier-2 -- controlled small denoiser: does SGD on a SHARED network reach the
1/(1+4 lambda) wrong-coordinate shrinkage optimum at convergence?

paper_outline LOCKED DECISION 11 / Section 6.2 (binding design):
  state z ~ N(0, I); view nuisance c; invariant target u_I = f(z); covariant target
  u_Cov = g(z) + B phi(c) with controllable ||u_Cov(c0)-u_Cov(cp)||; observation
  o = h(z, c) that does NOT directly leak u_Cov; denoising input x_sigma = u + sigma*n,
  x0-prediction, same sigma grid and paired shared-noise evaluation as at-scale.
  Shared-trunk MLP / small-DiT (the coupling test). lambda in {0,0.1,0.5,2.0}, trained
  to convergence; R(lambda)/R(0) on HELD-OUT (z,c); success = within 10% of 1/(1+4 lambda)
  at every lambda on the shared-trunk model.

The question Tier-1's algebra cannot answer: a shared network sees both views through
the SAME weights; the per-pair pointwise optimum is only *achievable*, not guaranteed,
by SGD. Tier-2 trains it and checks the covariant-block prediction difference contracts
by 1/(1+4 lambda) (normalized by the lambda=0 model, which already contracts toward the
posterior mean as sigma grows -- that is why we normalize).

Consistency is applied ONLY to the covariant block (the A2 wrong-coordinate case); the
invariant block is FM-supervised only and shares the trunk -> this is also the coupling
stress test (does an attached invariant block corrupt the covariant law?).

Outputs CSV/JSON; exit 0 if every lambda passes the 10% criterion on the low-sigma panel.
Single GPU; the default config converges in minutes. Reuses the real ShrinkageStats.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from evaluate_scvc_shrinkage import ShrinkageStats  # noqa: E402


# --------------------------------------------------------------------------------------
# Synthetic generative process (fixed random maps; one ground-truth world).
# --------------------------------------------------------------------------------------
class World:
    def __init__(self, d_z, d_c, m_inv, m_cov, d_obs, cov_view_scale, seed, device):
        import torch
        g = torch.Generator(device="cpu").manual_seed(seed)

        def rnd(*shape):
            return torch.randn(*shape, generator=g)

        self.d_z, self.d_c, self.m_inv, self.m_cov, self.d_obs = d_z, d_c, m_inv, m_cov, d_obs
        self.cov_view_scale = cov_view_scale
        # invariant target u_I = f(z) = tanh(z W_f1) W_f2
        self.Wf1 = rnd(d_z, 64) / math.sqrt(d_z)
        self.Wf2 = rnd(64, m_inv) / math.sqrt(64)
        # covariant target u_C = g(z) + B phi(c)
        self.Wg1 = rnd(d_z, 64) / math.sqrt(d_z)
        self.Wg2 = rnd(64, m_cov) / math.sqrt(64)
        self.Wphi = rnd(d_c, 64) / math.sqrt(d_c)
        self.B = rnd(64, m_cov) / math.sqrt(64)
        # observation o = h(z, c) = tanh([z;c] P) ; carries z and c but is not the target
        self.P = rnd(d_z + d_c, d_obs) / math.sqrt(d_z + d_c)
        self.device = device
        for k, v in list(self.__dict__.items()):
            if torch.is_tensor(v):
                setattr(self, k, v.to(device))

    def sample(self, n, generator):
        import torch
        z = torch.randn(n, self.d_z, generator=generator, device="cpu").to(self.device)
        c = torch.randn(n, self.d_c, generator=generator, device="cpu").to(self.device)
        return z, c

    def targets(self, z, c):
        import torch
        u_I = torch.tanh(z @ self.Wf1) @ self.Wf2
        phi = torch.tanh(c @ self.Wphi)
        u_C = (torch.tanh(z @ self.Wg1) @ self.Wg2) + self.cov_view_scale * (phi @ self.B)
        return u_I, u_C

    def obs(self, z, c):
        import torch
        return torch.tanh(torch.cat([z, c], dim=1) @ self.P)


# --------------------------------------------------------------------------------------
# Shared-trunk denoiser: predicts x0 = (u_I, u_C) from (x_sigma, o, log-sigma embedding).
# --------------------------------------------------------------------------------------
def build_model(world, width, depth, n_fourier, device):
    import torch
    import torch.nn as nn

    m = world.m_inv + world.m_cov
    in_dim = m + world.d_obs + 2 * n_fourier

    class Denoiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("freqs", torch.randn(n_fourier) * 2.0)
            layers = [nn.Linear(in_dim, width), nn.SiLU()]
            for _ in range(depth - 1):
                layers += [nn.Linear(width, width), nn.SiLU()]
            layers += [nn.Linear(width, m)]
            self.net = nn.Sequential(*layers)

        def forward(self, x_sigma, o, sigma):
            # EDM-style input scaling keeps magnitudes sane across the sigma grid.
            sigma_data = 1.0
            c_in = 1.0 / torch.sqrt(sigma**2 + sigma_data**2)
            ls = torch.log(sigma).clamp(-10, 10)
            emb = torch.cat([torch.sin(ls * self.freqs), torch.cos(ls * self.freqs)], dim=1)
            h = torch.cat([x_sigma * c_in, o, emb], dim=1)
            return self.net(h)

    return Denoiser().to(device)


def sample_sigma(n, sigma_min, sigma_max, generator, device):
    import torch
    # log-uniform sigma sampling over the training grid.
    u = torch.rand(n, 1, generator=generator, device="cpu").to(device)
    log = math.log(sigma_min) + u * (math.log(sigma_max) - math.log(sigma_min))
    return torch.exp(log)


def train_one(world, lam, args, device, seed):
    import torch

    model = build_model(world, args.width, args.depth, args.n_fourier, device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mI = world.m_inv

    model.train()
    for step in range(args.steps):
        z, c0 = world.sample(args.batch, gen)
        _, cp = world.sample(args.batch, gen)  # second view shares the SAME state z
        uI0, uC0 = world.targets(z, c0)
        uI1, uC1 = world.targets(z, cp)
        u0 = torch.cat([uI0, uC0], dim=1)
        up = torch.cat([uI1, uC1], dim=1)
        o0, op = world.obs(z, c0), world.obs(z, cp)
        sigma = sample_sigma(args.batch, args.sigma_min, args.sigma_max, gen, device)
        n = torch.randn(args.batch, u0.shape[1], generator=gen, device="cpu").to(device)  # SHARED noise
        x0 = u0 + sigma * n
        xp = up + sigma * n
        D0 = model(x0, o0, sigma)
        Dp = model(xp, op, sigma)
        # FM anchor (1/2 convention so nominal lambda == Prop-2 lambda), EDM-ish weight.
        w = (sigma**2 + 1.0) / (sigma**2)            # de-emphasise easy low-sigma a bit
        w = w.clamp(max=args.weight_clip)
        fm = 0.5 * (w * (D0 - u0) ** 2).mean() + 0.5 * (w * (Dp - up) ** 2).mean()
        # CV on the COVARIANT block only (wrong-coordinate); same w convention.
        cv = lam * (w * (D0[:, mI:] - Dp[:, mI:]) ** 2).mean()
        loss = fm + cv
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return model


def evaluate(model, world, args, device, sigmas, floor_by_sigma, seed):
    import torch
    model.eval()
    mI = world.m_inv
    gen = torch.Generator(device="cpu").manual_seed(seed)
    # one fixed held-out pair set, reused across all sigmas
    z, c0 = world.sample(args.eval_pairs, gen)
    _, cp = world.sample(args.eval_pairs, gen)
    uI0, uC0 = world.targets(z, c0)
    uI1, uC1 = world.targets(z, cp)
    u0 = torch.cat([uI0, uC0], dim=1)
    up = torch.cat([uI1, uC1], dim=1)
    o0, op = world.obs(z, c0), world.obs(z, cp)
    stats = {s: ShrinkageStats(checkpoint="tier2", sigma=s) for s in sigmas}
    inv_resid = []  # Prop-1 sanity: invariant block residual should be ~0
    with torch.no_grad():
        for s in sigmas:
            sigma = torch.full((args.eval_pairs, 1), s, device=device)
            gen_n = torch.Generator(device="cpu").manual_seed(seed + 1)
            n = torch.randn(args.eval_pairs, u0.shape[1], generator=gen_n, device="cpu").to(device)
            D0 = model(u0 + sigma * n, o0, sigma)
            Dp = model(up + sigma * n, op, sigma)
            num = torch.linalg.vector_norm((D0[:, mI:] - Dp[:, mI:]).float(), dim=1)
            den = torch.linalg.vector_norm((u0[:, mI:] - up[:, mI:]).float(), dim=1)
            stats[s].add(num, den, floor=floor_by_sigma[s])
            inv_resid.append(float(torch.linalg.vector_norm((D0[:, :mI] - Dp[:, :mI]).float(), dim=1).mean()))
    return stats, float(np.mean(inv_resid))


def compute_floor(world, args, device, sigmas, seed, pct):
    import torch
    gen = torch.Generator(device="cpu").manual_seed(seed)
    z, c0 = world.sample(args.eval_pairs, gen)
    _, cp = world.sample(args.eval_pairs, gen)
    _, uC0 = world.targets(z, c0)
    _, uC1 = world.targets(z, cp)
    den = torch.linalg.vector_norm((uC0 - uC1).float(), dim=1).cpu().numpy()
    floor = float(np.percentile(den, pct))
    return {s: floor for s in sigmas}  # target diff is sigma-independent


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.1, 0.5, 2.0])
    ap.add_argument("--d-z", type=int, default=16)
    ap.add_argument("--d-c", type=int, default=8)
    ap.add_argument("--m-inv", type=int, default=16)
    ap.add_argument("--m-cov", type=int, default=16)
    ap.add_argument("--d-obs", type=int, default=48)
    ap.add_argument("--cov-view-scale", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--n-fourier", type=int, default=16)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-clip", type=float, default=100.0)
    ap.add_argument("--sigma-min", type=float, default=0.02)
    ap.add_argument("--sigma-max", type=float, default=20.0)
    ap.add_argument("--eval-sigmas", type=float, nargs="+",
                    default=[0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
    ap.add_argument("--low-sigma-cutoff", type=float, default=0.2,
                    help="low-sigma primary panel: sigmas <= cutoff (law-identification regime)")
    ap.add_argument("--eval-pairs", type=int, default=8192)
    ap.add_argument("--floor-percentile", type=float, default=5.0)
    ap.add_argument("--tol-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    ap.add_argument("--output-json", default="")
    ap.add_argument("--output-csv", default="")
    args = ap.parse_args()

    device = args.device
    world = World(args.d_z, args.d_c, args.m_inv, args.m_cov, args.d_obs,
                  args.cov_view_scale, args.seed, device)
    floor_by_sigma = compute_floor(world, args, device, args.eval_sigmas, args.seed + 7, args.floor_percentile)

    # Aggregate over the low-sigma primary panel (law-identification regime).
    low_sigmas = [s for s in args.eval_sigmas if s <= args.low_sigma_cutoff]

    def panel_ratio(stats):
        num_sq = sum(stats[s].numerator_sq for s in low_sigmas)
        den_sq = sum(stats[s].denominator_sq for s in low_sigmas)
        return math.sqrt(num_sq / den_sq) if den_sq > 0 else float("nan")

    results = {}
    per_sigma_rows = []
    R0_panel = None
    R0_stats = None
    for lam in args.lambdas:
        model = train_one(world, lam, args, device, seed=args.seed + int(round(lam * 1000)))
        stats, inv_resid = evaluate(model, world, args, device, args.eval_sigmas, floor_by_sigma, args.seed + 99)
        results[lam] = (stats, inv_resid)
        if abs(lam) < 1e-12:
            R0_panel = panel_ratio(stats)
            R0_stats = stats
        for s in args.eval_sigmas:
            per_sigma_rows.append({"lambda": lam, "sigma": s, **stats[s].row(lambda_cv=lam)})

    rows = []
    all_pass = True
    for lam in args.lambdas:
        stats, inv_resid = results[lam]
        ideal = 1.0 / (1.0 + 4.0 * lam)
        R_panel = panel_ratio(stats)
        norm = R_panel / R0_panel if R0_panel else float("nan")
        # raw per-low-sigma normalized by R0 at the SAME sigma (for the figure)
        rel_err = abs(norm - ideal) / ideal
        passed = rel_err <= args.tol_frac or abs(lam) < 1e-12
        all_pass = all_pass and passed
        rows.append({
            "lambda": lam,
            "ideal_1_over_1_plus_4lambda": round(ideal, 6),
            "R_lambda_panel_lowsigma": round(R_panel, 6),
            "R0_panel_lowsigma": round(R0_panel, 6),
            "normalized_R_over_R0": round(norm, 6),
            "rel_err_vs_ideal": round(rel_err, 4),
            "invariant_block_residual_mean": round(inv_resid, 5),
            "pass_within_10pct": bool(passed),
        })

    # Monotonicity of R(lambda)/R(0) in lambda (Spearman sign over positive lambdas).
    pos = [r for r in rows if r["lambda"] > 0]
    mono = all(pos[i]["normalized_R_over_R0"] >= pos[i + 1]["normalized_R_over_R0"] for i in range(len(pos) - 1))

    report = {
        "verdict": "PASS" if (all_pass and mono) else "FAIL",
        "monotonic_in_lambda": bool(mono),
        "low_sigma_panel": low_sigmas,
        "config": vars(args),
        "rows": rows,
        "per_sigma": per_sigma_rows,
        "notes": (
            "PASS => SGD on a shared-trunk denoiser reaches the per-pair 1/(1+4L) optimum "
            "at convergence on the low-sigma panel, normalized by the lambda=0 posterior-mean "
            "contraction. This is the missing rung between Tier-1 algebra and the 2B model. "
            "If FAIL at convergence, the law is wrong at network scale -> LOCKED DECISION 4 matrix."
        ),
    }
    print(json.dumps({k: report[k] for k in ["verdict", "monotonic_in_lambda", "low_sigma_panel", "rows"]}, indent=2))
    if args.output_json:
        out = pathlib.Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.output_csv:
        import csv
        outc = pathlib.Path(args.output_csv)
        outc.parent.mkdir(parents=True, exist_ok=True)
        with outc.open("w", newline="") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(per_sigma_rows[0].keys()))
            wcsv.writeheader()
            wcsv.writerows(per_sigma_rows)
    sys.exit(0 if (all_pass and mono) else 1)


if __name__ == "__main__":
    main()
