"""a0 -- assert the action target is a viewpoint-invariant base-frame delta-EE command.

SCVC forces the nominal-camera and perturbed-camera branches to regress the SAME
action target. That cross-view consistency is only well-posed if the ground-truth
action is independent of the camera: i.e. a robot-base-frame end-effector delta, NOT
a camera-frame / eye-in-hand quantity. If the action lived in the camera frame, the
correct action would change when the scene camera moves, and matched-pair consistency
would be the A2 *wrong-coordinate* pathology rather than the invariant target.

This script verifies that premise empirically from the recorded LIBERO-Cosmos data,
making no appeal to "LIBERO uses OSC_POSE" as an article of faith:

  (1) Frame test. Over many recorded steps, least-squares-fit M with
        d_ee_pos[t] (base/world frame, from obs/ee_pos) ~= action[t, :3] @ M.
      If actions are base-frame translational deltas, M is near-diagonal and
      positive (a per-axis scale). If they were camera-frame, M would carry the
      camera->base rotation (large off-diagonal mass). We assert:
        - per-axis Pearson corr(action_i, d_ee_i) > CORR_MIN for i in x,y,z
        - off-diagonal energy of the normalized M is small (< OFFDIAG_MAX)
        - positive diagonal (no axis flips)
  (2) Channel test. gripper channel is bang-bang in {-1, +1}; rotation channels
      (3:6) are small relative to translation -> 7-dim OSC_POSE delta layout.
  (3) Camera-invariance by construction. The renderer loads ONE action array per
      demo and reuses it for every sampled camera, so a given (demo, t) has a single
      action target shared across all camera variants. We assert the renderer source
      reads actions once per demo (outside the camera loop). When a merged manifest
      exists, the same fact is machine-checkable via identical action_chunk_hash
      across camera labels for one (demo, t); that check is deferred to
      validate_pair_future_data.py.

Exit code 0 = all assertions pass; 1 = any failure. CPU only.

Usage:
  .venv/bin/python cosmos_policy/experiments/robot/libero/assert_action_coordinate_frame.py \
      --max-demos-per-task 6 --output-json outputs/phase2/a0_action_frame/report.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[4]
DEFAULT_ROOT = REPO / "LIBERO-Cosmos-Policy" / "success_only"
RENDERER = REPO / "cosmos_policy" / "experiments" / "robot" / "libero" / "render_libero_pair_future_frames.py"

# Thresholds.
# CORR_MIN is a SECONDARY sanity bound. action[:3] is the *commanded* OSC_POSE
# setpoint delta; d_ee is the *realized* motion under an impedance controller, so
# corr < 1 even when the frame is correct (controller lag, action saturation,
# contact, gripper dynamics). Pooled over grasp/contact phases this sits ~0.90,
# vs ~0.98 on a single clean reach. The PRIMARY frame discriminator is the
# orientation of the least-squares map M (rotation angle + off-diagonal energy):
# a base frame gives a scaled identity (angle ~0 deg), a camera frame carries the
# camera->base rotation (tens of degrees, large off-diagonal mass).
CORR_MIN = 0.85
OFFDIAG_MAX = 0.20        # fraction of |M| energy off the diagonal
ROT_ANGLE_MAX_DEG = 15.0  # rotation angle of M's orthogonal (polar) factor
ROT_TRANS_RATIO_MAX = 0.5  # median |rot| / median |trans| of action channels


def _rotation_angle_deg(M: np.ndarray) -> float:
    """Rotation angle of the orthogonal factor of M via polar decomposition (M = R S)."""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:  # reflect to a proper rotation for angle read-out
        U[:, -1] *= -1
        R = U @ Vt
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _iter_task_files(root: pathlib.Path, suites: list[str]) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for suite in suites:
        sdir = root / suite
        if sdir.is_dir():
            files.extend(sorted(sdir.glob("*_demo.hdf5")))
    return files


def _collect(root: pathlib.Path, suites: list[str], max_tasks: int, max_demos: int):
    import h5py

    actions_all: list[np.ndarray] = []
    dee_all: list[np.ndarray] = []
    rot_mag: list[float] = []
    trans_mag: list[float] = []
    gripper_vals: set[float] = set()
    n_demos = 0
    files = _iter_task_files(root, suites)[:max_tasks]
    if not files:
        raise FileNotFoundError(f"No *_demo.hdf5 under {root} for suites {suites}")
    for path in files:
        with h5py.File(path, "r") as f:
            data = f["data"]
            keys = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))[:max_demos]
            for k in keys:
                g = data[k]
                act = g["actions"][:].astype(np.float64)            # (T,7)
                ee = g["obs"]["ee_pos"][:].astype(np.float64)       # (T,3) base/world
                if len(act) < 3:
                    continue
                dee = np.diff(ee, axis=0)
                actions_all.append(act[:-1, :3])
                dee_all.append(dee)
                rot_mag.append(float(np.median(np.abs(act[:, 3:6]))))
                trans_mag.append(float(np.median(np.abs(act[:, 0:3]))))
                gripper_vals.update(np.round(np.unique(act[:, 6]), 2).tolist())
                n_demos += 1
    A = np.concatenate(actions_all, axis=0)   # (N,3) commanded delta
    D = np.concatenate(dee_all, axis=0)        # (N,3) realized base-frame delta
    return A, D, np.array(rot_mag), np.array(trans_mag), sorted(gripper_vals), n_demos, [p.name for p in files]


def _frame_report(A: np.ndarray, D: np.ndarray) -> dict[str, Any]:
    # Per-axis Pearson correlation.
    corr = [float(np.corrcoef(A[:, i], D[:, i])[0, 1]) for i in range(3)]
    # Least-squares M: D ~= A @ M  -> M = pinv(A) @ D  (3x3).
    M, *_ = np.linalg.lstsq(A, D, rcond=None)
    Mabs = np.abs(M)
    diag = np.diag(Mabs)
    offdiag_energy = float((Mabs.sum() - diag.sum()) / (Mabs.sum() + 1e-12))
    diag_signs = np.sign(np.diag(M)).tolist()
    return {
        "per_axis_corr": corr,
        "lstsq_M": M.round(4).tolist(),
        "offdiag_energy_frac": round(offdiag_energy, 4),
        "rotation_angle_deg": round(_rotation_angle_deg(M), 3),
        "diag_signs": diag_signs,
        "diag_values": np.diag(M).round(4).tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--suites", nargs="+", default=[
        "libero_spatial_regen", "libero_object_regen",
        "libero_goal_regen", "libero_10_regen",
    ])
    ap.add_argument("--max-tasks-per-suite", type=int, default=3)
    ap.add_argument("--max-demos-per-task", type=int, default=6)
    ap.add_argument("--output-json", default="")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    # Spread max-tasks across suites by collecting per suite then merging.
    A_list, D_list, rot_list, trans_list, grip_set, total_demos, files = [], [], [], [], set(), 0, []
    for suite in args.suites:
        try:
            A, D, rot, trans, grip, nd, fs = _collect(root, [suite], args.max_tasks_per_suite, args.max_demos_per_task)
        except FileNotFoundError:
            continue
        A_list.append(A); D_list.append(D); rot_list.append(rot); trans_list.append(trans)
        grip_set.update(grip); total_demos += nd; files.extend(fs)
    if not A_list:
        print("FAIL: no data collected", file=sys.stderr)
        sys.exit(1)
    A = np.concatenate(A_list); D = np.concatenate(D_list)
    rot_mag = np.concatenate(rot_list); trans_mag = np.concatenate(trans_list)
    gripper_vals = sorted(grip_set)

    frame = _frame_report(A, D)
    rot_trans_ratio = float(np.median(rot_mag) / (np.median(trans_mag) + 1e-12))

    # (3) renderer camera-invariance by construction: actions read once per demo,
    # outside the per-camera loop.
    src = RENDERER.read_text(encoding="utf-8")
    actions_read = len(re.findall(r'demo\["actions"\]', src))
    # camera sampling happens inside the timestep loop; actions are loaded in the demo loop.
    actions_before_camera_loop = bool(re.search(
        r'actions\s*=\s*demo\["actions"\][\s\S]{0,1200}?for\s+t\s+in\s+sampled_t[\s\S]{0,400}?sample_camera',
        src))

    checks = {
        "per_axis_corr_min": min(frame["per_axis_corr"]),
        "per_axis_corr_pass": all(c > CORR_MIN for c in frame["per_axis_corr"]),
        "rotation_angle_deg": frame["rotation_angle_deg"],
        "rotation_angle_pass": frame["rotation_angle_deg"] < ROT_ANGLE_MAX_DEG,
        "offdiag_energy_frac": frame["offdiag_energy_frac"],
        "offdiag_pass": frame["offdiag_energy_frac"] < OFFDIAG_MAX,
        "diag_positive_pass": all(s > 0 for s in frame["diag_signs"]),
        "gripper_bang_bang_pass": set(gripper_vals).issubset({-1.0, 1.0}) and len(gripper_vals) > 0,
        "rot_trans_ratio": round(rot_trans_ratio, 4),
        "rot_small_pass": rot_trans_ratio < ROT_TRANS_RATIO_MAX,
        "renderer_actions_read_count": actions_read,
        "renderer_actions_loaded_per_demo_pass": actions_read == 1 and actions_before_camera_loop,
    }
    passed = all(v for k, v in checks.items() if k.endswith("_pass"))

    report = {
        "verdict": "PASS" if passed else "FAIL",
        "interpretation": (
            "action target is a viewpoint-invariant base-frame delta-EE command "
            "(OSC_POSE-style); cross-view action consistency is well-posed"
        ) if passed else "action frame premise NOT confirmed -- investigate before any SCVC run",
        "n_steps": int(len(A)),
        "n_demos": total_demos,
        "n_task_files": len(files),
        "gripper_values": gripper_vals,
        "frame_fit": frame,
        "checks": checks,
        "thresholds": {"CORR_MIN": CORR_MIN, "OFFDIAG_MAX": OFFDIAG_MAX, "ROT_TRANS_RATIO_MAX": ROT_TRANS_RATIO_MAX},
    }
    print(json.dumps(report, indent=2))
    if args.output_json:
        out = pathlib.Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
