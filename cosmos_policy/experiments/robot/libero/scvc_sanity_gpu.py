"""SCVC GPU sanity gates before the main grid (single GPU, official checkpoint).

Gate 1 -- lambda_cv=0 fixed-batch equivalence:
    On ONE fixed pair batch, run training_step at lambda_cv=0 and assert the CV term
    does not enter the loss: total loss == FM-only loss to ~fp tolerance, and the
    logged scvc_lambda_cv==0. This is the precondition for Row 3 to be a clean R(0)
    anchor (FM-only on the same paired data).

Gate 2 -- 100-pair overfit (mechanism alive):
    Take a fixed small set of valid demo pairs, run a few dozen optimizer steps at
    lambda_cv>0, and assert the unscaled CV loss decreases substantially from its
    first-step value -- i.e. the cross-view consistency objective is actually being
    optimized (not silently masked/dead).

Single GPU, official released checkpoint (config load_path). Exit 0 iff both gates pass.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def load_scvc_model(experiment, config_file, device, lambda_cv, cv_total_steps, checkpoint):
    from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint

    model, _ = load_model_from_checkpoint(
        experiment_name=experiment,
        s3_checkpoint_dir=checkpoint,  # official released policy ckpt (hf://...)
        config_file=config_file,
        enable_fsdp=False,
        load_ema_to_reg=False,
        instantiate_ema=False,
        experiment_opts=[
            "model.config.fsdp_shard_size=1",
            f"model.config.lambda_cv={lambda_cv}",
            f"model.config.cv_total_steps={cv_total_steps}",
            # no warmup ramp during sanity: hold lambda at its nominal value from step 0
            "model.config.cv_warmup_start_fraction=0.0",
            "model.config.cv_warmup_end_fraction=0.0",
        ],
        to_device=device,
    )
    return model


def build_pair_dataset(manifest, device):
    from cosmos_policy.datasets.libero_pair_dataset import LIBEROPairDataset

    return LIBEROPairDataset(
        data_dir="LIBERO-Cosmos-Policy/success_only",
        pair_manifest_path=manifest,
        repo_root=".",
        t5_text_embeddings_path="LIBERO-Cosmos-Policy/success_only/t5_embeddings.pkl",
        rollout_data_dir="LIBERO-Cosmos-Policy/all_episodes",
    )


def collate(samples, device):
    import torch
    from torch.utils.data._utils.collate import default_collate

    batch = default_collate(samples)
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def make_batch(dataset, indices, device):
    return collate([dataset[i] for i in indices], device)


def gate1_equivalence(args, device):
    import torch

    print("\n===== Gate 1: lambda_cv=0 fixed-batch equivalence =====", flush=True)
    model = load_scvc_model(args.experiment, args.config_file, device, lambda_cv=0.0,
                            cv_total_steps=args.cv_total_steps, checkpoint=args.checkpoint)
    model.eval()
    dataset = build_pair_dataset(args.manifest, device)
    # use the first valid demo-pair indices [0, bs)
    batch = make_batch(dataset, list(range(args.bs)), device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out, loss = model.training_step(batch, iteration=0)
    fm = float(out["scvc_fm_loss_unscaled"])
    cv = float(out["scvc_cv_loss_unscaled"])
    lam = float(out["scvc_lambda_cv"])
    scale = float(getattr(model, "loss_scale", 1.0))
    total = float(loss)
    expected = fm * scale  # at lambda=0, total = (fm + 0*cv)*scale
    rel = abs(total - expected) / (abs(expected) + 1e-12)
    passed = (lam == 0.0) and (rel < args.tol)
    print(json.dumps({
        "lambda_cv": lam, "fm_unscaled": fm, "cv_unscaled": cv,
        "loss_scale": scale, "total_loss": total, "expected_fm_only": expected,
        "rel_err": rel, "tol": args.tol, "pass": passed,
    }, indent=2), flush=True)
    del model
    torch.cuda.empty_cache()
    return passed


def gate2_overfit(args, device):
    import torch

    print("\n===== Gate 2: 100-pair overfit (CV loss must drop) =====", flush=True)
    model = load_scvc_model(args.experiment, args.config_file, device, lambda_cv=args.lambda_cv,
                            cv_total_steps=args.overfit_steps, checkpoint=args.checkpoint)
    model.train()
    dataset = build_pair_dataset(args.manifest, device)
    n = min(args.overfit_pairs, len(dataset))
    batch = make_batch(dataset, list(range(min(args.bs, n))), device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.overfit_lr)

    cv_first = None
    cv_last = None
    fm_curve = []
    cv_curve = []
    for step in range(args.overfit_steps):
        # training_step normalizes video in-place and consumes batch keys, so feed a
        # fresh shallow-cloned batch each step (same fixed data, uncorrupted).
        step_batch = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, loss = model.training_step(step_batch, iteration=step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        cv = float(out["scvc_cv_loss_unscaled"])
        fm = float(out["scvc_fm_loss_unscaled"])
        if cv_first is None:
            cv_first = cv
        cv_last = cv
        if step % max(1, args.overfit_steps // 10) == 0 or step == args.overfit_steps - 1:
            fm_curve.append(round(fm, 5)); cv_curve.append(round(cv, 6))
            print(f"  step {step:3d}: fm={fm:.5f} cv={cv:.6f}", flush=True)
    drop = (cv_first - cv_last) / (abs(cv_first) + 1e-12) if cv_first else 0.0
    passed = cv_first is not None and cv_last < cv_first and drop >= args.min_cv_drop
    print(json.dumps({
        "cv_first": cv_first, "cv_last": cv_last, "relative_drop": drop,
        "min_required_drop": args.min_cv_drop, "fm_curve": fm_curve, "cv_curve": cv_curve,
        "pass": passed,
    }, indent=2), flush=True)
    del model
    torch.cuda.empty_cache()
    return passed


def main():
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", default="cosmos_predict2_2b_480p_libero_scvc_scene_only")
    ap.add_argument("--config-file", default="cosmos_policy/config/config.py")
    ap.add_argument("--checkpoint", default="hf://nvidia/Cosmos-Policy-LIBERO-Predict2-2B/Cosmos-Policy-LIBERO-Predict2-2B.pt")
    ap.add_argument("--manifest", default="outputs/phase2/pair_future_frames/libero_pair_future_manifest_train.jsonl")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--lambda-cv", type=float, default=0.1)
    ap.add_argument("--overfit-pairs", type=int, default=100)
    ap.add_argument("--overfit-steps", type=int, default=60)
    ap.add_argument("--overfit-lr", type=float, default=1e-4)
    ap.add_argument("--min-cv-drop", type=float, default=0.3)
    ap.add_argument("--cv-total-steps", type=int, default=10000)
    ap.add_argument("--gates", default="1,2")
    ap.add_argument("--output-json", default="outputs/phase3/scvc_sanity/gpu_gates.json")
    args = ap.parse_args()

    device = "cuda"
    results = {}
    gates = set(args.gates.split(","))
    if "1" in gates:
        results["gate1_lambda0_equivalence"] = gate1_equivalence(args, device)
    if "2" in gates:
        results["gate2_100pair_overfit"] = gate2_overfit(args, device)

    all_pass = all(results.values())
    report = {"results": results, "all_pass": all_pass}
    out = pathlib.Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\n===== SANITY SUMMARY =====")
    print(json.dumps(report, indent=2))
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
