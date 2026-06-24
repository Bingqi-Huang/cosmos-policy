"""Held-out invariant-block cross-view disagreement (Row4-vs-Row3 early go/no-go).

Purpose
-------
SCVC's training CV loss on the *invariant* block (action / value / future-proprio)
necessarily decays toward ~0, because both views share the same target and the FM
loss alone already pulls them together (Lemma 1: the FM-only residual coefficient is
1; SCVC raises it to 1+4*lambda). The train CV magnitude therefore *cannot* tell us
whether SCVC produced a more view-invariant action readout than the matched
lambda_cv=0 control (Row3). The binding quantity is the held-out cross-view
*disagreement* of the predicted invariant block: does Row4 predict more nearly the
same action/value/proprio across two cameras than Row3 does, on eval-disjoint pairs?

This script measures exactly that. For each checkpoint it runs paired forward passes
on the frozen holdout manifest with deterministic, *shared* (sigma, epsilon) across
the two views (and identical noise across checkpoints, so Row3/Row4 are compared on
the same draws), and reports the RMS of (D_0 - D_p) on each invariant latent frame.
Lower = more view-invariant. The decision number is the Row4/Row3 ratio: clearly
below 1 means the SCVC mechanism is engaging beyond pair-FM augmentation; ~1 means
the current lambda_cv=0.1 + 10%-warmup recipe is too gentle to beat the strong Row3
baseline and is not worth finishing to 10k.

Two-sided read. Besides the invariant disagreement (which should drop), the same
forward pass also reports a covariant guard: the held-out future-scene cross-view
ratio, which must NOT collapse for Row4 (the selective method leaves the future
scene per-view supervised, so its view-spread should stay ~unchanged vs Row3; a
collapse toward 0 means the dream got ghosted). The decision keys on the ACTION
frame specifically, since a falling aggregate driven by value/proprio while action
stays put would help camera robustness little.

This is a latent-space diagnostic only (no rollouts, no decoding); pair it with a
quick camera-subset SR readout for the success-level half of the go/no-go.

Reuses the proven dataset / paired-forward scaffolding from
``evaluate_scvc_shrinkage`` (the A2 covariant evaluator) so the forward contract is
identical; this script keeps its own invariant-only model-load semantics, swaps
the covariant future-scene frame for the invariant frames, and turns the
single-checkpoint shrinkage measurement into a two-checkpoint comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Any

# Absolute imports: this is a leaf script (nothing imports it), so living in a
# subfolder is safe and does not affect the import stability of the flat evaluators.
# NOTE: we deliberately do NOT import make_paired_batch. That helper does a shallow
# dict(batch) copy, so the paired view shares every non-video tensor with the base
# batch; get_data_and_condition mutates shared tensors in place, which corrupts the
# second branch's invariant frames (verified: shallow paths report spurious ~1.0
# invariant target diffs that vanish under full isolation). We build fully cloned,
# independent view batches instead.
from cosmos_policy.experiments.robot.libero.evaluate_scvc_shrinkage import (
    build_dataset,
    future_scene_norms,
    move_batch_to_device,
    run_branch,
)


def _isolated_views(batch: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Two fully independent per-view batches that share no tensor storage.

    view0 uses ``video``; viewp uses ``video_pair``. Cloning every tensor prevents
    in-place mutation in ``get_data_and_condition`` from leaking across the two
    forward passes (the root cause of spurious cross-view invariant-target diffs).
    """
    import copy

    import torch

    def clone_dict(b: dict[str, Any]) -> dict[str, Any]:
        return {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in b.items()}

    view0 = clone_dict(batch)
    viewp = clone_dict(batch)
    viewp["video"] = batch["video_pair"].clone()
    return view0, viewp

# Invariant block under the wrist-excluded P2 layout: cv_frame_set=action+value+fproprio.
# Each entry maps a human label to the per-sample latent-frame-index field on the batch.
INVARIANT_FRAME_FIELDS = {
    "action": "action_latent_idx",
    "value": "value_latent_idx",
    "future_proprio": "future_proprio_latent_idx",
}

DEFAULT_SIGMA_GRID = "0.02,0.05,0.1,0.2,0.5,1.0,2.0,5.0"


@dataclass
class DisagreementAccumulator:
    """Running sum of per-pair MSE of (D_0 - D_p) on a single invariant frame."""

    sum_sq: float = 0.0  # sum over valid pairs of mean_{C,H,W} (D_0 - D_p)^2
    count: int = 0  # number of valid pairs contributing

    def add(self, per_pair_mse, valid) -> None:
        sel = per_pair_mse[valid]
        if sel.numel() == 0:
            return
        self.sum_sq += float(sel.sum().item())
        self.count += int(sel.numel())

    @property
    def mse(self) -> float:
        return self.sum_sq / self.count if self.count else float("nan")

    @property
    def rms(self) -> float:
        # RMS per latent unit: sqrt(mean over elements of squared view-difference).
        m = self.mse
        return math.sqrt(m) if m == m else float("nan")  # m==m guards NaN


def invariant_disagreement(out0: dict[str, Any], outp: dict[str, Any], batch: dict[str, Any]):
    """Per-pair MSE of (pred_0 - pred_p) for each invariant frame, plus the valid mask.

    Returns ``{label: (per_pair_mse[B], valid[B])}``. Mirrors the validity logic of
    ``evaluate_scvc_shrinkage.future_scene_norms`` (demo rows only, both arms valid,
    frame present) but on the invariant latent frames instead of the future scene.
    """
    import torch

    pred0 = out0["model_pred"].x0
    predp = outp["model_pred"].x0
    device = pred0.device
    bsz = pred0.shape[0]
    bidx = torch.arange(bsz, device=device)
    pair_valid = batch.get("pair_valid", torch.ones_like(batch["rollout_data_mask"])).to(device)
    is_demo = batch["rollout_data_mask"].to(device) == 0

    out: dict[str, Any] = {}
    for label, field_name in INVARIANT_FRAME_FIELDS.items():
        if field_name not in batch:
            continue
        idx = batch[field_name].to(device).long().reshape(bsz)
        valid = is_demo.reshape(bsz) & (pair_valid.reshape(bsz) > 0) & (idx != -1)
        safe_idx = idx.clamp_min(0)
        diff = pred0[bidx, :, safe_idx, :, :] - predp[bidx, :, safe_idx, :, :]
        per_pair_mse = (diff.float() ** 2).mean(dim=(1, 2, 3))  # (B,)
        out[label] = (per_pair_mse, valid)
    return out


def check_invariant_targets_equal(x0, x0p, batch: dict[str, Any]) -> None:
    """Fail fast if the paired holdout does not share invariant targets."""
    import torch

    if x0.shape != x0p.shape:
        raise AssertionError(f"paired branch latent shapes differ: {tuple(x0.shape)} vs {tuple(x0p.shape)}")

    device = x0.device
    bsz = x0.shape[0]
    bidx = torch.arange(bsz, device=device)
    pair_valid = batch.get("pair_valid", torch.ones_like(batch["rollout_data_mask"])).to(device)
    valid = (batch["rollout_data_mask"].to(device) == 0).reshape(bsz) & (pair_valid.reshape(bsz) > 0)

    if not torch.any(valid):
        return

    for label, field_name in INVARIANT_FRAME_FIELDS.items():
        if field_name not in batch:
            continue
        idx = batch[field_name].to(device).long().reshape(bsz)
        good = valid & (idx != -1)
        if not torch.any(good):
            continue
        safe_idx = idx.clamp_min(0)
        target0 = x0[bidx, :, safe_idx, :, :][good]
        targetp = x0p[bidx, :, safe_idx, :, :][good]
        if not torch.allclose(target0.float(), targetp.float(), rtol=0.0, atol=1e-5):
            max_err = (target0.float() - targetp.float()).abs().max().item()
            raise AssertionError(
                f"paired invariant target {label} differs across views (max |delta|={max_err:.3e}); "
                "check the holdout manifest or latent injection path"
            )


def load_model(args, checkpoint: str):
    from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint

    model, _config = load_model_from_checkpoint(
        experiment_name=args.experiment,
        s3_checkpoint_dir=checkpoint,
        config_file=args.config_file,
        enable_fsdp=False,
        load_ema_to_reg=False,
        instantiate_ema=False,
        experiment_opts=[
            "model.config.fsdp_shard_size=1",
            "model.config.cv_frame_set=action+value+fproprio",
            f"model.config.cv_total_steps={args.cv_total_steps}",
        ],
        to_device=args.device,
    )
    model.eval()
    if not hasattr(model, "_branch_loss"):
        raise TypeError(
            "Invariant disagreement evaluation requires an SCVCPolicyVideo2WorldModel checkpoint/config "
            "because it reuses the paired branch-loss path."
        )
    return model


def evaluate_checkpoint(model, dataloader, sigmas: list[float], args) -> dict[str, Any]:
    import torch
    from tqdm import tqdm

    # accumulators[sigma][frame_label]
    accs: dict[float, dict[str, DisagreementAccumulator]] = {
        s: {lbl: DisagreementAccumulator() for lbl in INVARIANT_FRAME_FIELDS} for s in sigmas
    }
    # Covariant guard (future scene): aggregate ||Pi_Cov(D0-Dp)|| / ||Pi_Cov(u0-up)||.
    # This must NOT drop for Row4 -- the selective method leaves future-scene per-view
    # supervised, so its held-out cross-view spread should stay ~unchanged vs Row3 (and
    # near the train covar ratio ~0.95). A collapse toward 0 would mean the dream got
    # ghosted. NOTE: monitoring guard only (norm-aggregated, floor-free); the *binding*
    # A2 shrinkage number is evaluate_scvc_shrinkage.py with the frozen 5th-pct floor.
    cov_pred_sq = 0.0
    cov_tgt_sq = 0.0
    device = torch.device(args.device)
    generator = torch.Generator(device=device)
    max_batches = getattr(args, "max_batches", 0) or 0
    checked_targets = False
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="pairs", dynamic_ncols=True)):
            if max_batches and batch_idx >= max_batches:
                break
            batch = move_batch_to_device(batch, args.device)
            view0, viewp = _isolated_views(batch)
            _, x0_raw, condition = model.get_data_and_condition(view0)
            _, x0p_raw, conditionp = model.get_data_and_condition(viewp)
            for sigma_value in sigmas:
                # Checkpoint-independent seed => Row3 and Row4 see identical (sigma, epsilon).
                generator.manual_seed(args.seed + batch_idx * 100_000 + int(round(sigma_value * 10_000)))
                sigma = torch.full(
                    (x0_raw.shape[0], x0_raw.shape[2]),
                    float(sigma_value),
                    device=args.device,
                    dtype=torch.float32,
                )
                epsilon = torch.randn(x0_raw.shape, device=args.device, dtype=x0_raw.dtype, generator=generator)
                # Network weights are bf16 while latents are fp32; run the branch forward
                # under autocast (matches the training mixed-precision contract).
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out0 = run_branch(model, x0_raw, condition, epsilon, sigma, view0)
                    outp = run_branch(model, x0p_raw, conditionp, epsilon, sigma, viewp)
                # Sanity guard on the POST-injection targets (out["x0"]), not the raw VAE
                # latents: the 3D temporal VAE mixes scene info into the invariant frame
                # positions, so x0_raw legitimately differs across views (~0.8); the
                # invariant labels are injected after the encode, so out["x0"] is the
                # stage where the two views must match (mirrors the model's own
                # _first_step_contract_check). Verified equal to 0.0 on held-out pairs.
                if not checked_targets:
                    check_invariant_targets_equal(out0["x0"], outp["x0"], batch)
                    checked_targets = True
                per_frame = invariant_disagreement(out0, outp, batch)
                for lbl, (per_pair_mse, valid) in per_frame.items():
                    accs[sigma_value][lbl].add(per_pair_mse.detach(), valid.detach())
                # Covariant guard on the future-scene frame (same forward, same pairs).
                cov_pred, cov_tgt = future_scene_norms(out0, outp, batch)
                cov_pred_sq += float((cov_pred.float() ** 2).sum().item())
                cov_tgt_sq += float((cov_tgt.float() ** 2).sum().item())

    # Collapse to a per-sigma + overall summary.
    summary: dict[str, Any] = {"per_sigma": {}, "overall": {}}
    overall = {lbl: DisagreementAccumulator() for lbl in INVARIANT_FRAME_FIELDS}
    for s in sigmas:
        per: dict[str, Any] = {}
        for lbl in INVARIANT_FRAME_FIELDS:
            a = accs[s][lbl]
            per[lbl] = {"rms": a.rms, "mse": a.mse, "num_pairs": a.count}
            overall[lbl].sum_sq += a.sum_sq
            overall[lbl].count += a.count
        # block RMS = sqrt(mean MSE across the invariant frames present, pair-weighted)
        tot_sq = sum(accs[s][lbl].sum_sq for lbl in INVARIANT_FRAME_FIELDS)
        tot_n = sum(accs[s][lbl].count for lbl in INVARIANT_FRAME_FIELDS)
        per["block"] = {"rms": math.sqrt(tot_sq / tot_n) if tot_n else float("nan"), "num_pairs": tot_n}
        summary["per_sigma"][str(s)] = per
    for lbl in INVARIANT_FRAME_FIELDS:
        summary["overall"][lbl] = {"rms": overall[lbl].rms, "mse": overall[lbl].mse, "num_pairs": overall[lbl].count}
    tot_sq = sum(overall[lbl].sum_sq for lbl in INVARIANT_FRAME_FIELDS)
    tot_n = sum(overall[lbl].count for lbl in INVARIANT_FRAME_FIELDS)
    summary["overall"]["block"] = {
        "rms": math.sqrt(tot_sq / tot_n) if tot_n else float("nan"),
        "num_pairs": tot_n,
    }
    summary["covariant_guard"] = {
        "future_scene_ratio": math.sqrt(cov_pred_sq / cov_tgt_sq) if cov_tgt_sq else float("nan"),
        "note": "monitoring guard (norm-aggregated, floor-free); should stay ~unchanged vs Row3 and not collapse toward 0",
    }
    return summary


def parse_sigma_grid(spec: str) -> list[float]:
    return [float(x) for x in spec.split(",") if x.strip()]


def _format_report(results: dict[str, Any], labels: list[str]) -> str:
    lines: list[str] = []
    lines.append(
        "# Held-out invariant-block cross-view disagreement "
        "(RMS per latent unit; lower = more view-invariant)\n"
    )
    frame_order = list(INVARIANT_FRAME_FIELDS) + ["block"]
    # Invariant disagreement table (the part SCVC should drive DOWN).
    lines.append("label | " + " | ".join(frame_order) + " | covar_guard | num_terms(block)")
    lines.append("|".join(["---"] * (len(frame_order) + 3)))
    for label in labels:
        ov = results[label]["overall"]
        cells = [f"{ov[f]['rms']:.5f}" for f in frame_order]
        cov = results[label]["covariant_guard"]["future_scene_ratio"]
        lines.append(f"{label} | " + " | ".join(cells) + f" | {cov:.4f} | {ov['block']['num_pairs']}")

    if len(labels) == 2:
        a, b = labels  # a = baseline (e.g. row3 lambda=0), b = treatment (e.g. row4 SCVC)
        lines.append(f"\n## Decision read  {b} / {a}")
        ratio_cells = []
        for f in frame_order:
            ra = results[a]["overall"][f]["rms"]
            rb = results[b]["overall"][f]["rms"]
            ratio_cells.append(f"{f}={rb / ra:.3f}" if ra and ra == ra else f"{f}=nan")
        lines.append("invariant disagreement ratio (lower = SCVC working): " + "  ".join(ratio_cells))

        # Action is the load-bearing frame for camera robustness: aggregate can fall
        # because value/proprio fell while action did not, which helps little. Key on action.
        ra_act = results[a]["overall"]["action"]["rms"]
        rb_act = results[b]["overall"]["action"]["rms"]
        action_ratio = (rb_act / ra_act) if (ra_act and ra_act == ra_act) else float("nan")
        cov_a = results[a]["covariant_guard"]["future_scene_ratio"]
        cov_b = results[b]["covariant_guard"]["future_scene_ratio"]

        lines.append("")
        lines.append(f"PRIMARY (action-frame ratio {b}/{a}) = {action_ratio:.3f}")
        lines.append("\nper-sigma action read:")
        lines.append("sigma | " + f"{a}_action | {b}_action | ratio")
        lines.append("---|---:|---:|---:")
        for sigma_key in results[a]["per_sigma"]:
            ra_sigma = results[a]["per_sigma"][sigma_key]["action"]["rms"]
            rb_sigma = results[b]["per_sigma"][sigma_key]["action"]["rms"]
            ratio = rb_sigma / ra_sigma if ra_sigma and ra_sigma == ra_sigma else float("nan")
            lines.append(f"{float(sigma_key):.5g} | {ra_sigma:.5f} | {rb_sigma:.5f} | {ratio:.3f}")
        if action_ratio == action_ratio:  # not NaN
            if action_ratio <= 0.80:
                verdict = "GO  -> mechanism clearly engaging; let it finish to 10k."
            elif action_ratio >= 0.95:
                verdict = "NO-GO -> essentially null; stop and relaunch stronger (lambda=0.5, ~0 warmup)."
            else:
                verdict = (
                    "AMBIGUOUS -> inspect per-frame breakdown + a very small camera subset; "
                    "consider waiting for iter_2000 before deciding."
                )
            lines.append("VERDICT: " + verdict)
        # Covariant guard: Row4 future-scene ratio must not collapse vs Row3.
        guard = "OK"
        if cov_a and cov_a == cov_a and cov_b == cov_b and cov_b < 0.85 * cov_a:
            guard = "WARN -> Row4 future-scene spread dropped vs Row3; selective method may be ghosting the dream."
        lines.append(f"COVARIANT GUARD: {a}={cov_a:.3f}  {b}={cov_b:.3f}  -> {guard}")
        lines.append(
            "\nNote: latent-space early read only. Confirm with a small held-out camera-subset SR "
            "before any final claim; the binding A2 number is evaluate_scvc_shrinkage.py (frozen floor)."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--checkpoints", nargs="+", required=True, help="One or more checkpoint dirs (compared in order)."
    )
    parser.add_argument(
        "--labels", nargs="+", default=None, help="Human labels per checkpoint (e.g. row3_1k row4_1k)."
    )
    parser.add_argument("--holdout-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment", default="cosmos_predict2_2b_480p_libero_scvc_scene_only")
    parser.add_argument("--config-file", default="cosmos_policy/config/config.py")
    parser.add_argument("--data-dir", default="LIBERO-Cosmos-Policy/success_only")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--t5-text-embeddings-path", default="LIBERO-Cosmos-Policy/success_only/t5_embeddings.pkl")
    parser.add_argument("--sigma-grid", default=DEFAULT_SIGMA_GRID)
    parser.add_argument("--cv-total-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Cap batches per checkpoint for a fast read (0 = full holdout). A few hundred pairs suffice for the aggregate RMS.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    labels = args.labels or [pathlib.Path(c).parent.parent.name or pathlib.Path(c).name for c in args.checkpoints]
    if len(labels) != len(args.checkpoints):
        raise ValueError("--labels must match --checkpoints in length")

    sigmas = parse_sigma_grid(args.sigma_grid)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )

    results: dict[str, Any] = {}
    for label, checkpoint in zip(labels, args.checkpoints):
        model = load_model(args, checkpoint)
        summary = evaluate_checkpoint(model, dataloader, sigmas, args)
        summary["checkpoint"] = checkpoint
        results[label] = summary
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    report = _format_report(results, labels)
    print(report)
    (output_dir / "invariant_view_disagreement.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_dir / "invariant_view_disagreement.md").write_text(report, encoding="utf-8")
    print(f"\nWrote: {output_dir / 'invariant_view_disagreement.json'}")


if __name__ == "__main__":
    main()
