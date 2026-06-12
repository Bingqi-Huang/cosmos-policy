"""Evaluate the A2 latent shrinkage law on a frozen pair holdout set.

This is the paper_outline LOCKED DECISION 8 measurement path: fixed held-out
pairs, deterministic shared (sigma, epsilon) across views, latent-space
future-scene ratio, per-sigma bins, and a frozen denominator floor.

The script is intentionally not a training smoke.  It loads one checkpoint at a
time, runs paired forward passes on the holdout manifest, and writes CSV/JSON
artifacts that can be compared against the ideal 1/(1+4*lambda) curve.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Any


DEFAULT_SIGMA_GRID = "0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,80,200"


@dataclass
class ShrinkageStats:
    checkpoint: str
    sigma: float
    num_pairs: int = 0
    num_excluded: int = 0
    numerator_sq: float = 0.0
    denominator_sq: float = 0.0
    ratio_sum: float = 0.0
    ratio_sq_sum: float = 0.0

    def add(self, numerator, denominator, floor: float) -> None:
        keep = denominator >= floor
        self.num_pairs += int(denominator.numel())
        self.num_excluded += int((~keep).sum().item())
        if not bool(keep.any()):
            return
        kept_num = numerator[keep].float()
        kept_den = denominator[keep].float()
        ratio = kept_num / kept_den.clamp_min(1e-12)
        self.numerator_sq += float((kept_num**2).sum().item())
        self.denominator_sq += float((kept_den**2).sum().item())
        self.ratio_sum += float(ratio.sum().item())
        self.ratio_sq_sum += float((ratio**2).sum().item())

    @property
    def num_kept(self) -> int:
        return self.num_pairs - self.num_excluded

    def row(self, lambda_cv: float | None = None) -> dict[str, Any]:
        global_ratio = math.sqrt(self.numerator_sq / self.denominator_sq) if self.denominator_sq > 0 else float("nan")
        mean_ratio = self.ratio_sum / self.num_kept if self.num_kept else float("nan")
        variance = self.ratio_sq_sum / self.num_kept - mean_ratio**2 if self.num_kept else float("nan")
        ideal = 1.0 / (1.0 + 4.0 * lambda_cv) if lambda_cv is not None else float("nan")
        return {
            "checkpoint": self.checkpoint,
            "sigma": self.sigma,
            "num_pairs": self.num_pairs,
            "num_kept": self.num_kept,
            "num_excluded": self.num_excluded,
            "exclusion_rate": self.num_excluded / self.num_pairs if self.num_pairs else 0.0,
            "global_shrinkage_ratio": global_ratio,
            "mean_pair_ratio": mean_ratio,
            "std_pair_ratio": math.sqrt(max(0.0, variance)) if self.num_kept else float("nan"),
            "ideal_ratio_1_over_1_plus_4lambda": ideal,
            "ratio_over_ideal": global_ratio / ideal if ideal and not math.isnan(ideal) else float("nan"),
        }


def parse_sigma_grid(raw: str) -> list[float]:
    sigmas = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not sigmas:
        raise ValueError("sigma grid is empty")
    if any(sigma <= 0 for sigma in sigmas):
        raise ValueError(f"all sigma values must be positive: {sigmas}")
    return sigmas


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: pathlib.Path, rows: list[dict[str, Any]], floor_meta: dict[str, Any]) -> None:
    lines = [
        "# A2 Latent Shrinkage Evaluation",
        "",
        f"- Denominator floor: {floor_meta['denominator_floor']:.8g}",
        f"- Floor percentile: {floor_meta['floor_percentile']}",
        f"- Holdout manifest: `{floor_meta['holdout_manifest']}`",
        "",
        "| Checkpoint | sigma | global ratio | ideal | kept / total | exclusion |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['checkpoint']} | "
            f"{float(row['sigma']):.5g} | "
            f"{float(row['global_shrinkage_ratio']):.4f} | "
            f"{float(row['ideal_ratio_1_over_1_plus_4lambda']):.4f} | "
            f"{row['num_kept']} / {row['num_pairs']} | "
            f"{100 * float(row['exclusion_rate']):.2f}% |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def make_paired_batch(batch: dict[str, Any]) -> dict[str, Any]:
    paired = dict(batch)
    paired["video"] = batch["video_pair"]
    return paired


def run_branch(model, x0_raw, condition, epsilon, sigma, batch):
    x0, cond, eps, sig = model.broadcast_split_for_model_parallelsim(x0_raw, condition, epsilon, sigma)
    return model._branch_loss(x0, cond, eps, sig, batch)[0]


def future_scene_norms(out0: dict[str, Any], outp: dict[str, Any], batch: dict[str, Any]):
    import torch

    pred0 = out0["model_pred"].x0
    predp = outp["model_pred"].x0
    target0 = out0["x0"]
    targetp = outp["x0"]
    idx = batch["future_image_latent_idx"].to(pred0.device).long()
    pair_valid = batch.get("pair_valid", torch.ones_like(batch["rollout_data_mask"])).to(pred0.device)
    valid = (batch["rollout_data_mask"].to(pred0.device) == 0) & (pair_valid > 0) & (idx != -1)
    bidx = torch.arange(pred0.shape[0], device=pred0.device)
    safe_idx = idx.clamp_min(0)
    pred_diff = pred0[bidx, :, safe_idx, :, :] - predp[bidx, :, safe_idx, :, :]
    target_diff = target0[bidx, :, safe_idx, :, :] - targetp[bidx, :, safe_idx, :, :]
    pred_norm = torch.linalg.vector_norm(pred_diff.float().flatten(1), dim=1)
    target_norm = torch.linalg.vector_norm(target_diff.float().flatten(1), dim=1)
    return pred_norm[valid], target_norm[valid]


def future_scene_target_norms(target0, targetp, batch: dict[str, Any]):
    import torch

    idx = batch["future_image_latent_idx"].to(target0.device).long()
    pair_valid = batch.get("pair_valid", torch.ones_like(batch["rollout_data_mask"])).to(target0.device)
    valid = (batch["rollout_data_mask"].to(target0.device) == 0) & (pair_valid > 0) & (idx != -1)
    bidx = torch.arange(target0.shape[0], device=target0.device)
    safe_idx = idx.clamp_min(0)
    target_diff = target0[bidx, :, safe_idx, :, :] - targetp[bidx, :, safe_idx, :, :]
    target_norm = torch.linalg.vector_norm(target_diff.float().flatten(1), dim=1)
    return target_norm[valid]


def build_dataset(args):
    from cosmos_policy.datasets.libero_pair_dataset import LIBEROPairDataset

    return LIBEROPairDataset(
        data_dir=args.data_dir,
        pair_manifest_path=args.holdout_manifest,
        repo_root=args.repo_root,
        t5_text_embeddings_path=args.t5_text_embeddings_path,
        rollout_data_dir="",
        use_image_aug=False,
        use_stronger_image_aug=False,
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
            f"model.config.fsdp_shard_size=1",
            f"model.config.cv_frame_set=invariant_plus_fscene",
            f"model.config.cv_total_steps={args.cv_total_steps}",
        ],
        to_device=args.device,
    )
    model.eval()
    if not hasattr(model, "_branch_loss"):
        raise TypeError(
            "A2 shrinkage evaluation requires an SCVCPolicyVideo2WorldModel checkpoint/config "
            "because it reuses the paired branch-loss path."
        )
    return model


def collect_denominators(model, dataloader, args) -> list[float]:
    import torch
    from tqdm import tqdm

    denominators: list[float] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="denominator", dynamic_ncols=True):
            batch = move_batch_to_device(batch, args.device)
            paired = make_paired_batch(batch)
            _, x0_raw, condition = model.get_data_and_condition(batch)
            _, x0p_raw, conditionp = model.get_data_and_condition(paired)
            sigma = torch.ones((x0_raw.shape[0], x0_raw.shape[2]), device=args.device, dtype=torch.float32)
            sigmap = torch.ones((x0p_raw.shape[0], x0p_raw.shape[2]), device=args.device, dtype=torch.float32)
            x0, _, _, _ = model.broadcast_split_for_model_parallelsim(
                x0_raw, condition, torch.zeros_like(x0_raw), sigma
            )
            x0p, _, _, _ = model.broadcast_split_for_model_parallelsim(
                x0p_raw, conditionp, torch.zeros_like(x0p_raw), sigmap
            )
            den = future_scene_target_norms(x0, x0p, batch)
            denominators.extend(float(value) for value in den.detach().cpu())
    return denominators


def freeze_denominator_floor(model, dataloader, args) -> dict[str, Any]:
    import numpy as np

    denominators = collect_denominators(model, dataloader, args)
    if not denominators:
        raise RuntimeError("No valid pair denominators found; check holdout manifest and pair_valid fields.")
    floor = float(np.percentile(np.asarray(denominators, dtype=np.float64), args.floor_percentile))
    meta = {
        "holdout_manifest": str(args.holdout_manifest),
        "num_valid_pairs": len(denominators),
        "floor_percentile": args.floor_percentile,
        "denominator_floor": floor,
    }
    out = pathlib.Path(args.denominator_floor_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def evaluate_checkpoint(model, dataloader, checkpoint: str, sigmas: list[float], floor: float, args) -> list[dict[str, Any]]:
    import torch
    from tqdm import tqdm

    rows: list[dict[str, Any]] = []
    stats_by_sigma = {sigma: ShrinkageStats(checkpoint=checkpoint, sigma=sigma) for sigma in sigmas}
    generator = torch.Generator(device=args.device)
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=pathlib.Path(checkpoint).name, dynamic_ncols=True)):
            batch = move_batch_to_device(batch, args.device)
            paired = make_paired_batch(batch)
            _, x0_raw, condition = model.get_data_and_condition(batch)
            _, x0p_raw, conditionp = model.get_data_and_condition(paired)
            for sigma_value in sigmas:
                generator.manual_seed(args.seed + batch_idx * 100_000 + int(round(sigma_value * 10_000)))
                sigma = torch.full(
                    (x0_raw.shape[0], x0_raw.shape[2]),
                    float(sigma_value),
                    device=args.device,
                    dtype=torch.float32,
                )
                epsilon = torch.randn(x0_raw.shape, device=args.device, dtype=x0_raw.dtype, generator=generator)
                out0 = run_branch(model, x0_raw, condition, epsilon, sigma, batch)
                outp = run_branch(model, x0p_raw, conditionp, epsilon, sigma, paired)
                numerator, denominator = future_scene_norms(out0, outp, batch)
                stats_by_sigma[sigma_value].add(numerator.detach(), denominator.detach(), floor=floor)
    for sigma in sigmas:
        rows.append(stats_by_sigma[sigma].row(lambda_cv=args.lambda_cv))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--holdout-manifest", required=True)
    parser.add_argument("--denominator-floor-json", required=True)
    parser.add_argument("--freeze-denominator-floor", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment", default="cosmos_predict2_2b_480p_libero_scvc_scene_only")
    parser.add_argument("--config-file", default="cosmos_policy/config/config.py")
    parser.add_argument("--data-dir", default="LIBERO-Cosmos-Policy/success_only")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--t5-text-embeddings-path", default="LIBERO-Cosmos-Policy/success_only/t5_embeddings.pkl")
    parser.add_argument("--sigma-grid", default=DEFAULT_SIGMA_GRID)
    parser.add_argument("--lambda-cv", type=float, default=None)
    parser.add_argument("--cv-total-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--floor-percentile", type=float, default=5.0)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

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

    # The floor is checkpoint-independent for a frozen holdout/model tokenizer, so compute it once
    # with the first checkpoint only, then reuse it for every measured checkpoint.
    floor_path = pathlib.Path(args.denominator_floor_json)
    first_model = load_model(args, args.checkpoints[0])
    if args.freeze_denominator_floor or not floor_path.exists():
        floor_meta = freeze_denominator_floor(first_model, dataloader, args)
    else:
        floor_meta = json.loads(floor_path.read_text(encoding="utf-8"))
    floor = float(floor_meta["denominator_floor"])

    all_rows: list[dict[str, Any]] = []
    for idx, checkpoint in enumerate(args.checkpoints):
        model = first_model if idx == 0 else load_model(args, checkpoint)
        rows = evaluate_checkpoint(model, dataloader, checkpoint, sigmas, floor, args)
        all_rows.extend(rows)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    write_csv(output_dir / "shrinkage_by_sigma.csv", all_rows)
    aggregate_rows = []
    for checkpoint in args.checkpoints:
        subset = [row for row in all_rows if row["checkpoint"] == checkpoint]
        kept = sum(int(row["num_kept"]) for row in subset)
        total = sum(int(row["num_pairs"]) for row in subset)
        # Sigma bins where the floor excluded every pair report a NaN ratio; their weight is 0
        # but NaN*0 is still NaN, so they must be skipped, not just down-weighted.
        weighted = sum(
            float(row["global_shrinkage_ratio"]) * int(row["num_kept"]) for row in subset if int(row["num_kept"]) > 0
        )
        aggregate_rows.append(
            {
                "checkpoint": checkpoint,
                "num_sigma_bins": len(subset),
                "num_kept_sigma_pairs": kept,
                "num_total_sigma_pairs": total,
                "mean_global_shrinkage_ratio": weighted / kept if kept else float("nan"),
            }
        )
    write_csv(output_dir / "aggregate_shrinkage.csv", aggregate_rows)
    write_summary(output_dir / "summary.md", all_rows, floor_meta)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoints": args.checkpoints,
                "sigma_grid": sigmas,
                "lambda_cv": args.lambda_cv,
                "denominator_floor": floor_meta,
                "protocol": "paper_outline LOCKED DECISION 8",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[shrinkage] wrote report to {output_dir}")


if __name__ == "__main__":
    main()
