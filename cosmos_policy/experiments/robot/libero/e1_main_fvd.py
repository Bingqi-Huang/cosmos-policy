"""Camera-conditioned excess-FVD for the E1-main dissociation study.

Implements the video-side primary metric of E1-main exactly as pre-registered in
paper_outline LOCKED DECISION 13:

    excess-FVD(c) = FVD(model-predicted futures @ c, GT replays @ c)
                    - FVD(GT-replay split A @ c, GT-replay split B @ c)   # oracle floor
    Delta(c)      = (excess-FVD(c) - excess-FVD(nom)) / (excess-FVD(nom) + eps)

The oracle term removes the fact that the GT *reference distribution itself* changes
with the camera, so raw FVD is not comparable across camera conditions; the subtraction
measures the metric's floor under that camera distribution.

Design notes
------------
* The Frechet-distance math, the excess-FVD arithmetic, and the Delta aggregation are
  pure-numpy and fully CPU-unit-testable with any feature extractor (see the dummy
  extractor used by the test harness).
* The only GPU-bound piece is the I3D feature extraction; it is isolated behind the
  ``VideoFeatureExtractor`` protocol and the ``I3DFeatureExtractor`` loader so the
  measurement logic can be validated without a GPU or the I3D checkpoint.
* I3D checkpoint: set ``E1_I3D_CKPT`` to a TorchScript I3D (the standard FVD backbone,
  e.g. the Kinetics-400 ``i3d_torchscript.pt``). This is only needed at measurement time
  on the GPU box, never for the CPU logic tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import numpy as np


# A clip is a float array [T, H, W, 3] in [0, 1]. A clip-set is a sequence of clips
# (clips may have different T; the extractor is responsible for any temporal handling).
Clip = np.ndarray
ClipSet = Sequence[Clip]


class VideoFeatureExtractor(Protocol):
    """Maps a set of clips to a feature matrix [N, D] (one row per clip)."""

    def __call__(self, clips: ClipSet) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Frechet distance (pure numpy; CPU-testable)
# ---------------------------------------------------------------------------
def _matrix_sqrt(mat: np.ndarray) -> np.ndarray:
    """Symmetric PSD matrix square root via eigendecomposition (stable, real)."""
    # Symmetrize to kill numerical asymmetry, then clamp tiny negative eigenvalues.
    mat = (mat + mat.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.clip(eigvals, 0.0, None)
    return (eigvecs * np.sqrt(eigvals)) @ eigvecs.T


def frechet_distance(feats_a: np.ndarray, feats_b: np.ndarray, eps: float = 1e-6) -> float:
    """Frechet distance between two Gaussians fit to feature sets [N, D].

    FD = ||mu_a - mu_b||^2 + Tr(Cov_a + Cov_b - 2*(Cov_a Cov_b)^{1/2}).
    """
    feats_a = np.asarray(feats_a, dtype=np.float64)
    feats_b = np.asarray(feats_b, dtype=np.float64)
    if feats_a.ndim != 2 or feats_b.ndim != 2:
        raise ValueError("feature sets must be 2-D [N, D]")
    if feats_a.shape[1] != feats_b.shape[1]:
        raise ValueError("feature dimensions differ between the two sets")

    mu_a, mu_b = feats_a.mean(axis=0), feats_b.mean(axis=0)
    # rowvar=False -> variables are columns (feature dims); needs N>=2 per set.
    cov_a = np.cov(feats_a, rowvar=False)
    cov_b = np.cov(feats_b, rowvar=False)
    cov_a = np.atleast_2d(cov_a)
    cov_b = np.atleast_2d(cov_b)

    diff = mu_a - mu_b
    offset = np.eye(cov_a.shape[0]) * eps
    covmean = _matrix_sqrt((cov_a + offset) @ (cov_b + offset))
    fd = float(diff @ diff + np.trace(cov_a + cov_b - 2.0 * covmean))
    # Numerical guard: FD is non-negative by construction.
    return max(fd, 0.0)


def fvd(clips_a: ClipSet, clips_b: ClipSet, extractor: VideoFeatureExtractor, eps: float = 1e-6) -> float:
    """FVD = Frechet distance between I3D features of two clip-sets."""
    return frechet_distance(extractor(clips_a), extractor(clips_b), eps=eps)


# ---------------------------------------------------------------------------
# excess-FVD and the dissociation Delta aggregation (pure logic; CPU-testable)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BinClips:
    """The four clip-sets needed to score one camera bin's excess-FVD.

    ``gt_split_a`` / ``gt_split_b`` are a disjoint two-way split of the GT replays for
    this bin (the oracle floor); ``gt`` is the full GT-replay set the model is scored
    against (typically the union of the two splits).
    """

    model: ClipSet
    gt: ClipSet
    gt_split_a: ClipSet
    gt_split_b: ClipSet


def excess_fvd(bin_clips: BinClips, extractor: VideoFeatureExtractor, eps: float = 1e-6) -> dict[str, float]:
    """excess-FVD for a single camera bin, with the two component FVDs reported too."""
    fvd_model_gt = fvd(bin_clips.model, bin_clips.gt, extractor, eps=eps)
    fvd_oracle = fvd(bin_clips.gt_split_a, bin_clips.gt_split_b, extractor, eps=eps)
    return {
        "fvd_model_vs_gt": fvd_model_gt,
        "fvd_oracle": fvd_oracle,
        "excess_fvd": fvd_model_gt - fvd_oracle,
    }


def relative_degradation(
    excess_by_bin: dict[str, float],
    nominal_key: str,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Delta(c) = (excess-FVD(c) - excess-FVD(nom)) / (excess-FVD(nom) + eps).

    Returns Delta per non-nominal bin. A small Delta means fidelity is preserved under
    that perturbation (the dissociation signature, when action success has collapsed).
    """
    if nominal_key not in excess_by_bin:
        raise KeyError(f"nominal bin {nominal_key!r} missing from excess-FVD map")
    nominal = excess_by_bin[nominal_key]
    out: dict[str, float] = {}
    for key, val in excess_by_bin.items():
        if key == nominal_key:
            continue
        out[key] = (val - nominal) / (nominal + eps)
    return out


# ---------------------------------------------------------------------------
# I3D feature extractor (GPU-bound; isolated, not exercised by CPU tests)
# ---------------------------------------------------------------------------
# Default location for the FVD I3D backbone (the StyleGAN-V `i3d_torchscript.pt`); the
# launcher/rsync put it here so the extractor works with no env var or code edit.
_DEFAULT_I3D_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "assets", "fvd", "i3d_torchscript.pt")


class I3DFeatureExtractor:
    """TorchScript I3D backbone for FVD. Loaded lazily; runs on the configured device.

    Uses the standard StyleGAN-V ``i3d_torchscript.pt`` contract: input ``[B, 3, T, H, W]``
    in [-1, 1], called as ``model(x, rescale=False, resize=False, return_features=True)`` →
    ``[B, 400]`` features (verified against the checkpoint). Clips arrive as ``[T, H, W, 3]``
    in [0, 1]; we permute, resize to 224, and rescale to [-1, 1] here. The checkpoint path
    comes from ``ckpt_path`` → ``E1_I3D_CKPT`` → the in-repo ``assets/fvd/`` default.
    """

    def __init__(self, ckpt_path: str | None = None, device: str = "cuda", target_size: int = 224):
        self.ckpt_path = ckpt_path or os.environ.get("E1_I3D_CKPT", "") or os.path.abspath(_DEFAULT_I3D_PATH)
        self.device = device
        self.target_size = target_size
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        if not self.ckpt_path or not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(
                "I3D checkpoint not found. Set E1_I3D_CKPT to a TorchScript I3D "
                f"(got {self.ckpt_path!r}). Required only for the GPU measurement run."
            )
        import torch  # local import keeps the module importable without torch on CPU-only paths

        self._model = torch.jit.load(self.ckpt_path, map_location=self.device).eval()

    def __call__(self, clips: ClipSet) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        self._ensure_model()
        feats: list[np.ndarray] = []
        with torch.no_grad():
            for clip in clips:
                arr = np.asarray(clip, dtype=np.float32)  # [T, H, W, 3] in [0, 1]
                t = torch.from_numpy(arr).to(self.device)
                t = t.permute(3, 0, 1, 2).unsqueeze(0)  # [1, 3, T, H, W]
                t = F.interpolate(
                    t, size=(t.shape[2], self.target_size, self.target_size), mode="trilinear", align_corners=False
                )
                t = t * 2.0 - 1.0  # [0,1] -> [-1,1]
                feat = self._model(t, rescale=False, resize=False, return_features=True)
                feats.append(feat.reshape(feat.shape[0], -1).cpu().numpy())
        return np.concatenate(feats, axis=0)


def save_model_future_clip(
    future_primary_image_predictions: Sequence[Optional[np.ndarray]],
    name: str,
    condition: Optional[str],
    out_dir: str,
    manifest_path: str,
    clip_id: Optional[str] = None,
) -> Optional[str]:
    """Persist one episode's predicted future frames as a clip + append a manifest row.

    Drop-in hook for the camera eval: called right after the future-prediction video save
    in run_libero_eval.run_camera_task. ``name`` is the camera task name (used for the
    report's cell classification); ``clip_id`` makes the on-disk path unique per episode
    (multiple trials per camera task) — defaults to ``name``. Returns the clip path, or
    None when the episode produced no usable future frames. Manifest rows match the format
    e1_main_fvd.main / e1_main_render_gt_futures emit: {name, condition, clip_path}.
    """
    import json
    from pathlib import Path

    frames = [np.asarray(f) for f in future_primary_image_predictions if f is not None]
    if not frames:
        return None
    clip = np.stack(frames, axis=0).astype(np.uint8)  # [T, H, W, 3]
    safe = (clip_id or name).replace("/", "_")
    clip_dir = Path(out_dir) / (condition or "nominal") / safe
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_path = clip_dir / "clip.npy"
    np.save(clip_path, clip)
    with open(manifest_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"name": name, "condition": condition, "clip_path": str(clip_path)}) + "\n")
    return str(clip_path)


def build_extractor(kind: str = "i3d", **kwargs) -> VideoFeatureExtractor:
    if kind == "i3d":
        return I3DFeatureExtractor(**kwargs)
    raise ValueError(f"unknown extractor kind {kind!r}")


# ---------------------------------------------------------------------------
# CLI driver: per-cell excess-FVD from saved clip manifests (runs on the GPU box)
# ---------------------------------------------------------------------------
def _load_manifest(path):
    import json
    from pathlib import Path

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
    """Compute per-cell excess-FVD from a model-futures manifest + a GT-futures manifest.

    Each manifest is JSONL with rows {name, condition, clip_path}. ``clip_path`` is a .npy
    of shape [T, H, W, 3]. Writes {cell_key: excess_fvd, "_nominal": <nominal cell value>}.
    Requires the I3D checkpoint (E1_I3D_CKPT) and a GPU — run on the 5090 box.
    """
    import argparse
    import json
    from collections import defaultdict
    from pathlib import Path

    from cosmos_policy.experiments.robot.libero.generate_camera_report import load_task_classification

    ap = argparse.ArgumentParser(description="E1-main per-cell excess-FVD")
    ap.add_argument("--model_manifest", required=True, help="JSONL of model-predicted future clips")
    ap.add_argument("--gt_manifest", required=True, help="JSONL of GT-replay future clips")
    ap.add_argument("--task_classification", required=True)
    ap.add_argument("--nominal_cell", default="_nominal", help="cell key to treat as the nominal bin")
    ap.add_argument("--i3d_ckpt", default="", help="TorchScript I3D (else uses E1_I3D_CKPT)")
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    task_cls = load_task_classification(args.task_classification)
    extractor = build_extractor("i3d", ckpt_path=args.i3d_ckpt or None)

    def group(manifest):
        cells = defaultdict(list)
        for row in _load_manifest(manifest):
            ck = _cell_key_for(row["name"], row.get("condition"), task_cls)
            if ck is not None:
                cells[ck].append(row["clip_path"])
        return cells

    model_cells = group(args.model_manifest)
    gt_cells = group(args.gt_manifest)

    def load_clips(paths):
        return [np.load(p) for p in paths]

    from cosmos_policy.experiments.robot.libero.e1_main_render_gt_futures import split_clip_ids

    out: dict[str, float] = {}
    for ck in sorted(set(model_cells) & set(gt_cells)):
        gt_paths = gt_cells[ck]
        a_ids, b_ids = split_clip_ids(gt_paths, seed=args.split_seed)
        bin_clips = BinClips(
            model=load_clips(model_cells[ck]),
            gt=load_clips(gt_paths),
            gt_split_a=load_clips(a_ids),
            gt_split_b=load_clips(b_ids),
        )
        out[ck] = excess_fvd(bin_clips, extractor)["excess_fvd"]
        print(f"  {ck}: excess-FVD = {out[ck]:.3f}")

    if args.nominal_cell in out:
        out["_nominal"] = out[args.nominal_cell]
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"excess-FVD per cell written: {args.out}")


if __name__ == "__main__":
    main()


__all__ = [
    "Clip",
    "ClipSet",
    "VideoFeatureExtractor",
    "BinClips",
    "frechet_distance",
    "fvd",
    "excess_fvd",
    "relative_degradation",
    "I3DFeatureExtractor",
    "build_extractor",
]
