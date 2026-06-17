"""Camera-conditioned excess-FID for the E1-main dissociation study (video-side metric).

Researcher decision (2026-06-17): the Cosmos scene-only WAM emits a SINGLE predicted
future-scene frame per query (cosmos_utils.get_future_images_from_generated_samples ->
future_image), not a video. So the LD13 "excess-FVD" is realised at the FRAME level as
excess-FID (Frechet Inception Distance), the honest distributional fidelity metric for
single-frame predictions:

    excess-FID(c) = FID(model future frames @ c, realised-future frames @ c)
                    - FID(realised split A @ c, realised split B @ c)   # oracle floor
    Delta(c)      = (excess-FID(c) - excess-FID(nom)) / (excess-FID(nom) + eps)

State-matched GT (key correctness fix): the "realised-future frames" are the rollout's OWN
observations under the SAME perturbed camera (captured during the eval), NOT a separate
demo-state render. Model prediction and GT therefore come from the same states / camera /
rollout — no policy-vs-demo state-distribution confound.

Features: torchvision InceptionV3 pool (2048-d), the standard FID backbone. The Frechet
math, excess-FID arithmetic, and the per-cell aggregation are CPU-unit-testable; only the
Inception forward needs a GPU at scale. Weights load from an in-repo file (offline boxes).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from cosmos_policy.experiments.robot.libero.e1_main_fvd import relative_degradation


def frechet_distance(feats_a: np.ndarray, feats_b: np.ndarray, eps: float = 1e-6) -> float:
    """Canonical Frechet distance between two Gaussians fit to feature sets [N, D].

        FD = ||mu_a - mu_b||^2 + Tr(Cov_a + Cov_b - 2*(Cov_a Cov_b)^{1/2}).

    The matrix square root is computed with scipy.linalg.sqrtm on the (generally NON-symmetric)
    product Cov_a @ Cov_b, exactly as in the reference FID implementation (pytorch-fid / TTUR).
    A naive eigendecomposition of a symmetrised product is mathematically wrong because the
    product of two symmetric PSD matrices need not be symmetric, so it is NOT used here.
    """
    from scipy import linalg

    feats_a = np.asarray(feats_a, dtype=np.float64)
    feats_b = np.asarray(feats_b, dtype=np.float64)
    if feats_a.ndim != 2 or feats_b.ndim != 2:
        raise ValueError("feature sets must be 2-D [N, D]")
    if feats_a.shape[0] < 2 or feats_b.shape[0] < 2:
        # A Gaussian covariance is undefined for <2 samples; return NaN rather than fabricate.
        return float("nan")
    if feats_a.shape[1] != feats_b.shape[1]:
        raise ValueError("feature dimensions differ between the two sets")

    mu_a, mu_b = feats_a.mean(axis=0), feats_b.mean(axis=0)
    cov_a = np.atleast_2d(np.cov(feats_a, rowvar=False))
    cov_b = np.atleast_2d(np.cov(feats_b, rowvar=False))
    diff = mu_a - mu_b

    # Product matrix square root (reference FID recipe), with a PSD offset retry if singular.
    covmean, _ = linalg.sqrtm(cov_a @ cov_b, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(cov_a.shape[0]) * eps
        covmean = linalg.sqrtm((cov_a + offset) @ (cov_b + offset))
    if np.iscomplexobj(covmean):
        # sqrtm of a real matrix can carry a tiny imaginary residue; assert it is numerical.
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"FID sqrtm imaginary component too large: {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real

    fd = float(diff @ diff + np.trace(cov_a) + np.trace(cov_b) - 2.0 * np.trace(covmean))
    return max(fd, 0.0)  # FD is non-negative by construction; clamp numerical undershoot.


# Frames are [H, W, 3] uint8 (or float). A frame-set is an array [N, H, W, 3] or a list of them.
_DEFAULT_INCEPTION = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "assets", "fid", "inception_v3_imagenet.pth"
)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class InceptionFeatureExtractor:
    """torchvision InceptionV3 pool features (2048-d) for FID. Loaded lazily; offline-safe.

    Input: an array [N, H, W, 3] (uint8 [0,255] or float [0,1]). Output: [N, 2048].
    """

    def __init__(self, ckpt_path: str | None = None, device: str = "auto", batch_size: int = 64):
        self.ckpt_path = ckpt_path or os.environ.get("E1_INCEPTION_CKPT", "") or os.path.abspath(_DEFAULT_INCEPTION)
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        import torch as _torch

        if self.device in (None, "auto"):
            self.device = "cuda" if _torch.cuda.is_available() else "cpu"
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(
                f"Inception weights not found at {self.ckpt_path}. Save them with torchvision "
                "(assets/fid/inception_v3_imagenet.pth) and rsync to the eval box."
            )
        import torch
        import torch.nn as nn
        from torchvision.models import inception_v3

        m = inception_v3(weights=None, transform_input=False, aux_logits=True, init_weights=False)
        m.load_state_dict(torch.load(self.ckpt_path, map_location=self.device))
        m.fc = nn.Identity()  # expose the 2048-d pooled features
        self._model = m.eval().to(self.device)

    def __call__(self, frames) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        arr = np.asarray(frames)
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError(f"frames must be [N, H, W, 3], got {arr.shape}")
        if arr.shape[0] == 0:
            return np.zeros((0, 1), dtype=np.float64)  # frechet_distance guards on <2 rows
        self._ensure_model()
        arr = arr.astype(np.float32)
        if arr.size and arr.max() > 1.0:
            arr = arr / 255.0  # uint8 [0,255] -> [0,1]
        mean = torch.tensor(_IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        feats: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, arr.shape[0], self.batch_size):
                batch = torch.from_numpy(arr[i : i + self.batch_size]).to(self.device)
                batch = batch.permute(0, 3, 1, 2)  # [B, 3, H, W]
                batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
                batch = (batch - mean) / std
                feat = self._model(batch)
                feats.append(feat.reshape(feat.shape[0], -1).cpu().numpy())
        return np.concatenate(feats, axis=0)


def save_episode_frames(
    model_future_frames: Sequence[Optional[np.ndarray]],
    gt_scene_frames: Optional[np.ndarray],
    name: str,
    condition: Optional[str],
    out_dir: str,
    manifest_path: str,
    clip_id: Optional[str] = None,
) -> Optional[str]:
    """Persist one rollout episode's model future-frame set + state-matched GT scene set.

    Drop-in hook for run_camera_task. ``model_future_frames`` are the per-query predicted
    future scenes (cosmos future_image); ``gt_scene_frames`` are the rollout's OWN realised-
    future scene observations under the SAME perturbed camera, already temporally aligned by
    the caller (prediction k <-> replay_images[(k+1)*stride]) — the realised future, state/
    camera-matched to the predictions, no separate render.
    Manifest rows: {name, condition, model_path, gt_path}. Returns the model path or None.
    """
    model = [np.asarray(f) for f in model_future_frames if f is not None]
    if not model or gt_scene_frames is None or len(gt_scene_frames) == 0:
        return None
    model_arr = np.stack(model, axis=0).astype(np.uint8)  # [n, H, W, 3]
    gt_arr = np.asarray(gt_scene_frames).astype(np.uint8)  # [m, H, W, 3]
    safe = (clip_id or name).replace("/", "_")
    d = Path(out_dir) / (condition or "nominal") / safe
    d.mkdir(parents=True, exist_ok=True)
    model_path, gt_path = d / "model_frames.npy", d / "gt_frames.npy"
    np.save(model_path, model_arr)
    np.save(gt_path, gt_arr)
    with open(manifest_path, "a", encoding="utf-8") as fh:
        fh.write(
            json.dumps({"name": name, "condition": condition, "model_path": str(model_path), "gt_path": str(gt_path)})
            + "\n"
        )
    return str(model_path)


@dataclass(frozen=True)
class BinFrames:
    """Frame-sets needed to score one camera bin's excess-FID (each [N, H, W, 3])."""

    model: np.ndarray
    gt: np.ndarray
    gt_split_a: np.ndarray
    gt_split_b: np.ndarray


def fid(frames_a, frames_b, extractor, eps: float = 1e-6) -> float:
    return frechet_distance(extractor(frames_a), extractor(frames_b), eps=eps)


def excess_fid(bin_frames: BinFrames, extractor, eps: float = 1e-6) -> dict[str, float]:
    fid_model_gt = fid(bin_frames.model, bin_frames.gt, extractor, eps=eps)
    fid_oracle = fid(bin_frames.gt_split_a, bin_frames.gt_split_b, extractor, eps=eps)
    return {
        "fid_model_vs_gt": fid_model_gt,
        "fid_oracle": fid_oracle,
        "excess_fid": fid_model_gt - fid_oracle,
    }


def split_frames(frames: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic disjoint two-way split of a frame-set (the oracle A/B floor)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(frames.shape[0])
    mid = frames.shape[0] // 2
    return frames[idx[:mid]], frames[idx[mid:]]


# ---------------------------------------------------------------------------
# CLI: per-cell excess-FID from saved per-episode frame manifests (GPU box)
# ---------------------------------------------------------------------------
def _load_manifest(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _cell_key_for(name, condition, task_classification):
    meta = task_classification.get(name)
    if condition is None or condition == "nominal" or meta is None:
        return None
    return f"{condition}_L{int(meta['difficulty_level'])}"


def main() -> None:
    """Compute per-cell excess-FID from a manifest of per-episode model + GT frame arrays.

    Manifest rows: {name, condition, model_path, gt_path}, each .npy of shape [n, H, W, 3].
    Frames are pooled per (condition, difficulty-level) cell. Writes {cell_key: excess_fid,
    "_nominal": <nominal value>}. Requires the Inception weights + a GPU at scale.
    """
    from cosmos_policy.experiments.robot.libero.generate_camera_report import load_task_classification

    ap = argparse.ArgumentParser(description="E1-main per-cell excess-FID")
    ap.add_argument("--manifest", required=True, help="JSONL of per-episode {name,condition,model_path,gt_path}")
    ap.add_argument("--task_classification", required=True)
    ap.add_argument("--nominal_manifest", default="", help="optional JSONL for the nominal-camera bin")
    ap.add_argument("--inception_ckpt", default="", help="Inception weights (else E1_INCEPTION_CKPT / in-repo)")
    ap.add_argument("--min_frames", type=int, default=50, help="min GT frames per cell for a stable FID + oracle split")
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    task_cls = load_task_classification(args.task_classification)
    extractor = InceptionFeatureExtractor(ckpt_path=args.inception_ckpt or None)

    def pool_cells(manifest_path, force_key=None):
        """Pool per-episode frame arrays into {cell_key: (model[N,...], gt[M,...])}."""
        model_lists: dict[str, list] = defaultdict(list)
        gt_lists: dict[str, list] = defaultdict(list)
        for row in _load_manifest(manifest_path):
            ck = force_key or _cell_key_for(row["name"], row.get("condition"), task_cls)
            if ck is None:
                continue
            model_lists[ck].append(np.load(row["model_path"]))
            gt_lists[ck].append(np.load(row["gt_path"]))
        cells = {}
        for ck in model_lists:
            cells[ck] = (
                np.concatenate(model_lists[ck], axis=0),
                np.concatenate(gt_lists[ck], axis=0),
            )
        return cells

    out: dict[str, float] = {}
    skipped: dict[str, str] = {}

    def score(ck, model_frames, gt_frames, is_nominal=False):
        if model_frames.shape[0] < 2 or gt_frames.shape[0] < args.min_frames:
            skipped[ck] = f"model={model_frames.shape[0]}, gt={gt_frames.shape[0]} (need model>=2, gt>={args.min_frames})"
            return
        a, b = split_frames(gt_frames, seed=args.split_seed)
        try:
            stats = excess_fid(BinFrames(model_frames, gt_frames, a, b), extractor)
        except Exception as exc:  # noqa: BLE001 - record + continue; one bad cell never aborts
            skipped[ck] = f"error: {exc}"
            return
        if stats["excess_fid"] != stats["excess_fid"]:  # NaN
            skipped[ck] = "excess-FID is NaN (degenerate features)"
            return
        out[ck] = stats["excess_fid"]
        if is_nominal:
            # The oracle FID (GT-vs-GT floor, always > 0) is the natural positive scale for
            # normalizing the dissociation Delta — excess-FID can be <= 0, so the LD13 ratio
            # over excess(nom) is ill-posed. The report uses this as the Delta denominator.
            out["_nominal_oracle"] = stats["fid_oracle"]
        print(f"  {ck}: excess-FID = {stats['excess_fid']:.3f}  (model={model_frames.shape[0]}, gt={gt_frames.shape[0]})")

    for ck, (mf, gf) in sorted(pool_cells(args.manifest).items()):
        score(ck, mf, gf)

    if args.nominal_manifest:
        nom = pool_cells(args.nominal_manifest, force_key="_nominal")
        if "_nominal" in nom:
            mf, gf = nom["_nominal"]
            score("_nominal", mf, gf, is_nominal=True)

    # Record the perturbed tasks the video side actually saw (those with a valid cell key).
    # The report restricts the action-side SR to this same task universe, so action and video
    # are always measured on the SAME tasks — critical for subset/smoke runs (where the frozen
    # action JSONL covers all 1599 tasks but the video manifest covers only a subset). On a
    # full run this is the complete set, so the restriction is a no-op.
    out["_measured_tasks"] = sorted(
        {
            row["name"]
            for row in _load_manifest(args.manifest)
            if _cell_key_for(row["name"], row.get("condition"), task_cls) is not None
        }
    )

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"excess-FID: {len(out)} cells written, {len(skipped)} skipped -> {args.out}")
    for ck, why in sorted(skipped.items()):
        print(f"  [skip] {ck}: {why}")


if __name__ == "__main__":
    main()
