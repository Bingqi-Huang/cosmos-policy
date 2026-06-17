"""GT-replay future-clip renderer for E1-main's camera-conditioned excess-FVD.

For each (demo, benchmark camera) the model is scored against, this renders the demo's
GROUND-TRUTH future-scene evolution at the SAME benchmark (eval) camera the rollout used
— a short clip of frames at successive recorded MuJoCo states. These GT clips form, per
(condition, level) cell, the reference distribution and the two oracle splits (A/B) that
e1_main_fvd consumes:

    excess-FVD(c) = FVD(model futures @ c, GT replays @ c) - FVD(GT split A @ c, GT split B @ c)

Reuse: all camera math + sim/render machinery is imported from the audited pair renderer
(render_libero_pair_future_frames) so GT pixels are orientation-identical to the SCVC pair
data and to Cosmos training pixels (flipud=True). Unlike the pair renderer, cameras here
come from the LIBERO-Plus BENCHMARK task names (the actual eval poses), NOT the
train-disjoint sampler.

GPU note: the render loop uses EGL offscreen rendering and runs on the GPU box (5090).
The name-parsing / classification / split helpers are pure and CPU-unit-testable.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np

from cosmos_policy.experiments.robot.libero.generate_camera_report import _classify_condition
from cosmos_policy.experiments.robot.libero.render_libero_pair_future_frames import (
    REPO_ROOT,
    compute_camera,
    ensure_dir,
)

# Task-name suffix: ``..._view_<horizon>_<vertical>_<scale>_<end_rot>_<end_vert>_initstate_<k>``
CAMERA_NAME_RE = re.compile(r"_view_(-?\d+)_(-?\d+)_(\d+)_(-?\d+)_(-?\d+)_initstate_(\d+)$")


def parse_camera_task_name(name: str) -> Optional[dict[str, Any]]:
    """Parse a LIBERO-Plus camera task name into its camera params + base task + initstate.

    Returns None for names that do not carry a camera suffix (pure CPU logic).
    """
    m = CAMERA_NAME_RE.search(name)
    if not m:
        return None
    horizon, vertical, scale, end_rot, end_vert, initstate = (int(g) for g in m.groups())
    return {
        "base": name[: m.start()],
        "horizon": horizon,
        "vertical": vertical,
        "scale": scale,
        "end_rot": end_rot,
        "end_vert": end_vert,
        "initstate": initstate,
        "condition": _classify_condition(name),  # "C1"|"C2"|"C3"|None(nominal)
    }


def camera_params(parsed: dict[str, Any]) -> dict[str, int]:
    """The 5 fields compute_camera() needs."""
    return {k: int(parsed[k]) for k in ("horizon", "vertical", "scale", "end_rot", "end_vert")}


def split_clip_ids(clip_ids: list[str], seed: int = 0) -> tuple[list[str], list[str]]:
    """Deterministic disjoint two-way split of a cell's clip ids (the oracle A/B floor)."""
    rng = np.random.default_rng(seed)
    order = list(clip_ids)
    rng.shuffle(order)
    mid = len(order) // 2
    return sorted(order[:mid]), sorted(order[mid:])


# ---------------------------------------------------------------------------
# GPU render loop (runs on the 5090; not exercised by CPU tests)
# ---------------------------------------------------------------------------
def _render_gt_future_clip(
    env,
    cam_id: int,
    nom_pos: np.ndarray,
    nom_quat: np.ndarray,
    states: np.ndarray,
    t0: int,
    params: dict[str, int],
    img_size: int,
    n_frames: int,
    stride: int,
) -> np.ndarray:
    from cosmos_policy.experiments.robot.libero.render_libero_pair_future_frames import _render

    pert_pos, pert_quat = compute_camera(
        nom_pos,
        nom_quat,
        horizon=params["horizon"],
        vertical=params["vertical"],
        scale_pct=params["scale"],
        end_rot=params["end_rot"],
        end_vert=params["end_vert"],
    )
    n_steps = len(states)
    # Render the demo's natural future evolution at chunk cadence (stride), stopping at the
    # end of the demo rather than padding with duplicates. ``n_frames`` is a max cap so the
    # GT clip length tracks the model's per-query predicted-future sequence (one frame/chunk).
    indices = list(range(t0, n_steps, max(1, stride)))[:n_frames]
    frames = []
    for idx in indices:
        env.sim.set_state_from_flattened(states[idx])
        env.sim.forward()
        raw = _render(env, cam_id, pert_pos, pert_quat, img_size)
        frames.append(np.flipud(raw).astype(np.uint8))  # match Cosmos / pair-data orientation
    env.sim.model.cam_pos[cam_id][:] = nom_pos
    env.sim.model.cam_quat[cam_id][:] = nom_quat
    return np.stack(frames, axis=0)


def main() -> None:
    """Render GT future clips for a set of camera tasks. GPU/EGL; run on the 5090 box."""
    import h5py
    from tqdm import tqdm

    from cosmos_policy.experiments.robot.libero.render_libero_pair_future_frames import (
        build_env,
        default_bddl_root,
        get_agentview_camera,
        hdf5_to_bddl,
    )

    ap = argparse.ArgumentParser(description="E1-main GT-replay future-clip renderer (GPU/EGL)")
    ap.add_argument("--camera_tasks_file", required=True, help="JSON list of LIBERO-Plus camera task names")
    ap.add_argument("--libero_root", default=str(REPO_ROOT / "LIBERO-Cosmos-Policy" / "success_only"))
    ap.add_argument("--suite", required=True, help="suite dir name, e.g. libero_spatial")
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "outputs" / "phase1" / "e1_main" / "gt_futures"))
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--n_frames", type=int, default=32, help="max GT clip length (cap)")
    ap.add_argument("--stride", type=int, default=16, help="state stride = policy chunk cadence (num_open_loop_steps)")
    ap.add_argument("--gpu_device_id", type=int, default=0)
    ap.add_argument("--t0", type=int, default=16, help="start state index (~first future target, t+chunk)")
    args = ap.parse_args()

    names = json.loads(Path(args.camera_tasks_file).read_text(encoding="utf-8"))
    out_root = ensure_dir(Path(args.out_dir))
    manifest_rows: list[dict[str, Any]] = []

    # Group camera tasks by their base task so one env build serves many cameras.
    by_base: dict[str, list[dict[str, Any]]] = {}
    for name in names:
        parsed = parse_camera_task_name(name)
        if parsed is None:
            continue
        by_base.setdefault(parsed["base"], []).append({"name": name, **parsed})

    # The LIBERO-Cosmos-Policy success_only suites are stored as "<suite>_regen" (re-rendered
    # demos); the eval suite name passed in (e.g. libero_spatial) has no suffix. Resolve to
    # whichever directory actually exists.
    suite_dir = args.suite
    if not (Path(args.libero_root) / suite_dir).is_dir() and (Path(args.libero_root) / f"{args.suite}_regen").is_dir():
        suite_dir = f"{args.suite}_regen"

    bddl_root = default_bddl_root()
    for base, cams in tqdm(by_base.items(), desc="tasks"):
        hdf5_path = Path(args.libero_root) / suite_dir / f"{base}_demo.hdf5"
        if not hdf5_path.exists():
            print(f"  skip (missing demo): {hdf5_path}")
            continue
        bddl_file = hdf5_to_bddl(hdf5_path, bddl_root)
        env = build_env(bddl_file, args.img_size, args.gpu_device_id)
        env.reset()
        cam_id, nom_pos, nom_quat = get_agentview_camera(env)
        with h5py.File(hdf5_path, "r") as f:
            data = f["data"]
            demo_keys = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))
            for cam in cams:
                params = camera_params(cam)
                k = int(cam["initstate"]) % len(demo_keys)
                states = data[demo_keys[k]]["states"][:]
                clip = _render_gt_future_clip(
                    env, cam_id, nom_pos, nom_quat, states, args.t0, params,
                    args.img_size, args.n_frames, args.stride,
                )
                cond = cam["condition"] or "nominal"
                clip_dir = ensure_dir(out_root / cond / cam["name"])
                np.save(clip_dir / "clip.npy", clip)
                manifest_rows.append({
                    "name": cam["name"], "base": base, "suite": args.suite,
                    "condition": cond, "initstate": k, "clip_path": str(clip_dir / "clip.npy"),
                    "n_frames": int(clip.shape[0]),
                })
        env.close()

    manifest = out_root / f"gt_futures_manifest_{args.suite}.jsonl"
    with manifest.open("w", encoding="utf-8") as fh:
        for row in manifest_rows:
            fh.write(json.dumps(row) + "\n")
    print(f"GT future clips rendered: {len(manifest_rows)} -> {manifest}")


if __name__ == "__main__":
    main()
