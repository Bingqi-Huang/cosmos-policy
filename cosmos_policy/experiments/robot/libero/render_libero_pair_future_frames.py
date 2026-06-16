"""Render same-state LIBERO pair frames for scene-only Cosmos SCVC.

For each sampled demo timestep this script renders four scene-camera images:

* branch A current frame at ``t`` under the nominal camera
* branch B current frame at ``t`` under a sampled perturbed camera
* branch A future frame at ``min(t + chunk_size, T - 1)`` under the nominal camera
* branch B future frame at the same future state under the same perturbed camera

The future state is taken from the recorded HDF5 demo state.  This matches the
Cosmos dataset target index exactly and avoids simulator drift from replaying
controller actions.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LIBERO_ROOT = REPO_ROOT / "LIBERO-Cosmos-Policy" / "success_only"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "phase2" / "pair_future_frames" / "images"
DEFAULT_RESULTS_DIR = REPO_ROOT / "outputs" / "phase2" / "pair_future_frames"
DEFAULT_DEBUG_DIR = REPO_ROOT / "outputs" / "phase2" / "pair_future_frames" / "debug"
DEFAULT_SUITES = ("libero_spatial_regen", "libero_object_regen", "libero_goal_regen", "libero_10_regen")
_SCALE_PIVOT = np.array([0.0, 0.0, 0.8])
C2_HORIZON_POOL = list(range(1, 76)) + list(range(285, 360))
C2_VERTICAL_POOL = [0, 15]
C3_ROT_POOL = [2, 4, 6, 8, 10, 350, 352, 354, 356, 358]
CATEGORY_PROBS = {"C1": 0.196, "C2": 0.620, "C3": 0.184}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, default=json_default) + "\n")


def hash_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()[:24]


def stable_id(*parts: Any, length: int = 24) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return digest.hexdigest()[:length]


def rel_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def suite_base_name(suite_dir_name: str) -> str:
    return suite_dir_name.removesuffix("_regen")


def command_from_hdf5_path(hdf5_path: Path) -> str:
    words = hdf5_path.stem.removesuffix("_demo").split("_")
    command = ""
    for word in words:
        if "SCENE" in word:
            command = ""
            continue
        command += word + " "
    return command.strip()


def _rot_z(quat: np.ndarray | None = None, pos: np.ndarray | None = None, deg: float = 0.0) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    r = Rotation.from_euler("z", deg, degrees=True)
    if quat is not None:
        q = np.asarray(quat, dtype=float)
        orig = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        xyzw = (r * orig).as_quat()
        result["new_quat"] = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
    if pos is not None:
        result["new_pos"] = r.apply(np.asarray(pos, dtype=float))
    return result


def _rot_y_pivot(
    quat: np.ndarray | None = None,
    pos: np.ndarray | None = None,
    deg: float = 0.0,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    r = Rotation.from_rotvec(np.radians(-deg) * np.array([0.0, 1.0, 0.0]))
    if quat is not None:
        q = np.asarray(quat, dtype=float)
        orig = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        xyzw = (r * orig).as_quat()
        result["new_quat"] = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
    if pos is not None:
        p = np.asarray(pos, dtype=float)
        result["new_pos"] = r.apply(p - _SCALE_PIVOT) + _SCALE_PIVOT
    return result


def _scale_dist(
    quat: np.ndarray | None = None,
    pos: np.ndarray | None = None,
    factor: float = 1.0,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    if quat is not None:
        result["new_quat"] = np.asarray(quat, dtype=float).copy()
    if pos is not None:
        p = np.asarray(pos, dtype=float)
        result["new_pos"] = _SCALE_PIVOT + (p - _SCALE_PIVOT) * factor
    return result


def compute_camera(
    nom_pos: np.ndarray,
    nom_quat: np.ndarray,
    *,
    horizon: int,
    vertical: int,
    scale_pct: int,
    end_rot: int,
    end_vert: int,
) -> tuple[np.ndarray, np.ndarray]:
    pos = nom_pos.copy().astype(float)
    quat = nom_quat.copy().astype(float)
    if vertical != 0:
        result = _rot_y_pivot(quat=quat, pos=pos, deg=float(vertical))
        pos, quat = result["new_pos"], result["new_quat"]
    if horizon != 0:
        result = _rot_z(quat=quat, pos=pos, deg=float(horizon))
        pos, quat = result["new_pos"], result["new_quat"]
    if scale_pct != 100:
        result = _scale_dist(quat=quat, pos=pos, factor=scale_pct / 100.0)
        pos, quat = result["new_pos"], result["new_quat"]
    if end_rot != 0:
        quat = _rot_z(quat=quat, deg=float(end_rot))["new_quat"]
    if end_vert != 0:
        quat = _rot_y_pivot(quat=quat, deg=float(end_vert))["new_quat"]
    return pos, quat


def load_benchmark_camera_tuples(json_paths: list[Path]) -> set[tuple[int, int, int, int, int]]:
    """Parse LIBERO-Plus camera task names into (horizon, vertical, scale, end_rot, end_vert) tuples.

    Task-name suffix format: ``_view_<hv>_<vv>_<scale>_<rotz>_<roty>_initstate_<k>``. Training-pair
    cameras must never collide with these benchmark poses (execution_plan standing rule 4:
    train/eval camera-pose disjointness).
    """
    pattern = re.compile(r"_view_(-?\d+)_(-?\d+)_(\d+)_(-?\d+)_(-?\d+)_initstate_")
    tuples: set[tuple[int, int, int, int, int]] = set()
    for json_path in json_paths:
        names = json.loads(Path(json_path).read_text(encoding="utf-8"))
        for name in names:
            match = pattern.search(name)
            if match:
                tuples.add(tuple(int(g) for g in match.groups()))  # type: ignore[arg-type]
    return tuples


def sample_camera(
    rng: np.random.Generator,
    category: str | None = None,
    exclude: set[tuple[int, int, int, int, int]] | None = None,
) -> dict[str, Any]:
    if exclude:
        for _ in range(1000):
            params = sample_camera(rng, category=category, exclude=None)
            key = (
                int(params["horizon"]),
                int(params["vertical"]),
                int(params["scale"]),
                int(params["end_rot"]),
                int(params["end_vert"]),
            )
            if key not in exclude:
                return params
        raise RuntimeError("Could not sample an eval-disjoint camera in 1000 tries; check exclusion set size.")
    if category is None:
        cats = list(CATEGORY_PROBS)
        category = str(rng.choice(cats, p=[CATEGORY_PROBS[c] for c in cats]))
    if category == "C1":
        return {
            "horizon": 0,
            "vertical": 0,
            "scale": int(rng.integers(115, 201)),
            "end_rot": 0,
            "end_vert": 0,
            "category": "C1",
        }
    if category == "C2":
        return {
            "horizon": int(rng.choice(C2_HORIZON_POOL)),
            "vertical": int(rng.choice(C2_VERTICAL_POOL)),
            "scale": 100,
            "end_rot": 0,
            "end_vert": 0,
            "category": "C2",
        }
    if category != "C3":
        raise ValueError("category must be one of C1, C2, C3, or None")
    return {
        "horizon": 0,
        "vertical": 0,
        "scale": 100,
        "end_rot": int(rng.choice(C3_ROT_POOL)),
        "end_vert": int(rng.choice(C3_ROT_POOL)),
        "category": "C3",
    }


def camera_label(params: dict[str, Any]) -> str:
    if params["category"] == "C1":
        return f"C1_s{params['scale']}"
    if params["category"] == "C2":
        return f"C2_h{params['horizon']}_v{params['vertical']}"
    return f"C3_er{params['end_rot']}_ev{params['end_vert']}"


def default_bddl_root() -> Path:
    from libero.libero import get_libero_path  # noqa: PLC0415

    return Path(get_libero_path("bddl_files"))


def hdf5_to_bddl(hdf5_path: Path, bddl_root: Path) -> Path:
    suite = suite_base_name(hdf5_path.parent.name)
    task_stem = hdf5_path.stem.removesuffix("_demo")
    return bddl_root / suite / f"{task_stem}.bddl"


def build_env(bddl_file: Path, img_size: int, gpu_device_id: int):
    from libero.libero.envs import OffScreenRenderEnv  # noqa: PLC0415

    kwargs = {
        "bddl_file_name": str(bddl_file),
        "camera_heights": img_size,
        "camera_widths": img_size,
        "camera_names": ["agentview"],
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "use_camera_obs": True,
        "ignore_done": True,
        "hard_reset": True,
    }
    if gpu_device_id >= 0:
        kwargs["render_gpu_device_id"] = gpu_device_id
    return OffScreenRenderEnv(**kwargs)


def get_agentview_camera(env) -> tuple[int, np.ndarray, np.ndarray]:
    cam_id = env.sim.model.camera_name2id("agentview")
    return cam_id, env.sim.model.cam_pos[cam_id].copy(), env.sim.model.cam_quat[cam_id].copy()


def _render(env, cam_id: int, pos: np.ndarray, quat: np.ndarray, img_size: int) -> np.ndarray:
    env.sim.model.cam_pos[cam_id][:] = pos
    env.sim.model.cam_quat[cam_id][:] = quat
    env.sim.forward()
    return env.sim.render(width=img_size, height=img_size, camera_name="agentview")


def render_state_pair(
    env,
    cam_id: int,
    nom_pos: np.ndarray,
    nom_quat: np.ndarray,
    state_vec: np.ndarray,
    camera_params: dict[str, Any],
    img_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    env.sim.set_state_from_flattened(state_vec)
    env.sim.forward()
    nom_img = _render(env, cam_id, nom_pos, nom_quat, img_size)
    pert_pos, pert_quat = compute_camera(
        nom_pos,
        nom_quat,
        horizon=int(camera_params["horizon"]),
        vertical=int(camera_params["vertical"]),
        scale_pct=int(camera_params["scale"]),
        end_rot=int(camera_params["end_rot"]),
        end_vert=int(camera_params["end_vert"]),
    )
    pert_img = _render(env, cam_id, pert_pos, pert_quat, img_size)
    env.sim.model.cam_pos[cam_id][:] = nom_pos
    env.sim.model.cam_quat[cam_id][:] = nom_quat
    return nom_img, pert_img, pert_pos, pert_quat


def save_png(path: Path, raw_img: np.ndarray, *, flipud: bool = True) -> None:
    import imageio.v2 as imageio  # noqa: PLC0415

    ensure_dir(path.parent)
    image = np.flipud(raw_img) if flipud else raw_img
    imageio.imwrite(str(path), image.astype(np.uint8))


def process_hdf5(
    hdf5_path: Path,
    bddl_root: Path,
    output_dir: Path,
    rng: np.random.Generator,
    *,
    timestep_sample_rate: float,
    views_per_state: int,
    max_pairs: int | None,
    img_size: int,
    gpu_device_id: int,
    val_demo_fraction: float,
    chunk_size: int,
    save_flipud: bool,
    benchmark_exclude: set[tuple[int, int, int, int, int]] | None = None,
) -> list[dict[str, Any]]:
    bddl = hdf5_to_bddl(hdf5_path, bddl_root)
    if not bddl.exists():
        print(f"[SKIP] BDDL not found: {bddl}", flush=True)
        return []

    suite_dir = hdf5_path.parent.name
    suite = suite_base_name(suite_dir)
    task_name = hdf5_path.stem.removesuffix("_demo")
    task_out = output_dir / suite / task_name
    command = command_from_hdf5_path(hdf5_path)

    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted(f["data"].keys(), key=lambda key: int(key.split("_")[1]))
    n_val = max(1, int(len(demo_keys) * val_demo_fraction))
    val_demo_ids = set(range(len(demo_keys) - n_val, len(demo_keys)))

    env = build_env(bddl, img_size=img_size, gpu_device_id=gpu_device_id)
    env.reset()
    cam_id, nom_pos, nom_quat = get_agentview_camera(env)

    rows: list[dict[str, Any]] = []
    pair_count = 0
    with h5py.File(hdf5_path, "r") as f:
        data_group = f["data"]
        for demo_idx, demo_key in enumerate(tqdm(demo_keys, desc="demos", leave=False, dynamic_ncols=True)):
            if max_pairs is not None and pair_count >= max_pairs:
                break
            demo = data_group[demo_key]
            states = demo["states"][:]
            actions = demo["actions"][:].astype(np.float32)
            robot_states = demo["robot_states"][:].astype(np.float32)
            n_steps = len(states)
            split = "val" if demo_idx in val_demo_ids else "train"

            n_sample = max(1, int(n_steps * timestep_sample_rate))
            sampled_t = sorted(rng.choice(n_steps, size=min(n_sample, n_steps), replace=False).tolist())
            for t in sampled_t:
                if max_pairs is not None and pair_count >= max_pairs:
                    break
                future_t = min(t + chunk_size, n_steps - 1)
                # Guarantee the views of THIS state use DISTINCT perturbed cameras:
                # pair_id keys on the camera label (not the view index), so a within-state
                # collision would collapse two views into one duplicate pair (and silently
                # give the state fewer distinct views than requested). Resample on collision.
                used_labels: set[str] = set()
                for _view_idx in range(views_per_state):
                    cam_params = sample_camera(rng, exclude=benchmark_exclude)
                    label = camera_label(cam_params)
                    _tries = 0
                    while label in used_labels and _tries < 50:
                        cam_params = sample_camera(rng, exclude=benchmark_exclude)
                        label = camera_label(cam_params)
                        _tries += 1
                    used_labels.add(label)
                    current_nom, current_pert, pert_pos, pert_quat = render_state_pair(
                        env, cam_id, nom_pos, nom_quat, states[t], cam_params, img_size
                    )
                    future_nom, future_pert, _, _ = render_state_pair(
                        env, cam_id, nom_pos, nom_quat, states[future_t], cam_params, img_size
                    )

                    base_dir = task_out / f"demo_{demo_idx}" / f"t{t:06d}_ft{future_t:06d}_{label}"
                    current_a = base_dir / "current_nominal.png"
                    current_b = base_dir / "current_perturbed.png"
                    future_a = base_dir / "future_nominal.png"
                    future_b = base_dir / "future_perturbed.png"
                    if not current_a.exists():
                        save_png(current_a, current_nom, flipud=save_flipud)
                    if not current_b.exists():
                        save_png(current_b, current_pert, flipud=save_flipud)
                    if not future_a.exists():
                        save_png(future_a, future_nom, flipud=save_flipud)
                    if not future_b.exists():
                        save_png(future_b, future_pert, flipud=save_flipud)

                    pair_id = stable_id(suite, task_name, demo_idx, t, future_t, label)
                    row = {
                        "pair_id": pair_id,
                        "source_type": "libero_recorded_state_future_rerender",
                        "suite": suite,
                        "suite_dir": suite_dir,
                        "task_name": task_name,
                        "hdf5_path": str(hdf5_path),
                        "demo_key": demo_key,
                        "demo_id": demo_idx,
                        "timestep": int(t),
                        "future_timestep": int(future_t),
                        "chunk_size": int(chunk_size),
                        "split": split,
                        "language": command,
                        "state_hash": hash_array(states[t]),
                        "future_state_hash": hash_array(states[future_t]),
                        "action_chunk_hash": hash_array(actions[t : min(t + chunk_size, n_steps)]),
                        "robot_state_hash": hash_array(robot_states[t]),
                        "future_robot_state_hash": hash_array(robot_states[future_t]),
                        "pair_confidence": 1.0,
                        "pair_type": "matched",
                        "action_equivalence": "same HDF5 demo, same timestep, exact recorded MuJoCo states",
                        "future_source": "recorded_hdf5_state_at_min_t_plus_chunk",
                        "img_size": int(img_size),
                        "save_flipud": bool(save_flipud),
                        "current_img_a_path": rel_to_repo(current_a),
                        "current_img_b_path": rel_to_repo(current_b),
                        "future_img_a_path": rel_to_repo(future_a),
                        "future_img_b_path": rel_to_repo(future_b),
                        "cam_pos_a": nom_pos.tolist(),
                        "cam_quat_a": nom_quat.tolist(),
                        "cam_pos_b": pert_pos.tolist(),
                        "cam_quat_b": pert_quat.tolist(),
                        "camera_params_a": {
                            "horizon": 0,
                            "vertical": 0,
                            "scale": 100,
                            "end_rot": 0,
                            "end_vert": 0,
                        },
                        "camera_params_b": {
                            key: int(cam_params[key])
                            for key in ("horizon", "vertical", "scale", "end_rot", "end_vert")
                        },
                        "camera_category": cam_params["category"],
                    }
                    rows.append(row)
                    pair_count += 1

    env.close()
    gc.collect()
    return rows


def build_wrong_state_controls(rows: list[dict[str, Any]], rng: np.random.Generator) -> list[dict[str, Any]]:
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_category.setdefault(str(row["camera_category"]), []).append(row)
    wrong: list[dict[str, Any]] = []
    for category_rows in by_category.values():
        if len(category_rows) < 2:
            continue
        indices = list(range(len(category_rows)))
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        shuffled = shuffled[1:] + shuffled[:1]
        for src_idx, dst_idx in zip(indices, shuffled):
            src = category_rows[src_idx]
            dst = category_rows[dst_idx]
            if src["state_hash"] == dst["state_hash"]:
                continue
            out = dict(src)
            out["pair_id"] = "wrong_" + str(src["pair_id"])
            out["pair_type"] = "wrong_state"
            out["pair_confidence"] = 0.0
            out["state_hash_b"] = dst["state_hash"]
            out["future_state_hash_b"] = dst["future_state_hash"]
            out["current_img_b_path"] = dst["current_img_b_path"]
            out["future_img_b_path"] = dst["future_img_b_path"]
            out["cam_pos_b"] = dst["cam_pos_b"]
            out["cam_quat_b"] = dst["cam_quat_b"]
            out["camera_params_b"] = dst["camera_params_b"]
            wrong.append(out)
    return wrong


def write_audit_report(path: Path, rows: list[dict[str, Any]], elapsed_sec: float) -> None:
    by_suite: dict[str, int] = {}
    by_category: dict[str, int] = {}
    by_split: dict[str, int] = {}
    for row in rows:
        by_suite[row["suite"]] = by_suite.get(row["suite"], 0) + 1
        by_category[row["camera_category"]] = by_category.get(row["camera_category"], 0) + 1
        by_split[row["split"]] = by_split.get(row["split"], 0) + 1
    lines = [
        "# LIBERO Pair Future-Frame Render Audit",
        "",
        f"- Total matched pairs: {len(rows)}",
        f"- Elapsed seconds: {elapsed_sec:.1f}",
        "",
        "## By Split",
        "",
    ]
    for key, value in sorted(by_split.items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## By Suite", ""])
    for key, value in sorted(by_suite.items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## By Camera Category", ""])
    for key in ("C1", "C2", "C3"):
        value = by_category.get(key, 0)
        pct = 100 * value / len(rows) if rows else 0.0
        lines.append(f"- {key}: {value} ({pct:.1f}%)")
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--libero-root", type=Path, default=DEFAULT_LIBERO_ROOT)
    parser.add_argument("--suite", nargs="*", dest="suites", default=list(DEFAULT_SUITES))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--debug-dir", type=Path, default=DEFAULT_DEBUG_DIR)
    parser.add_argument("--bddl-root", type=Path, default=None)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--timestep-sample-rate", type=float, default=1.0)
    parser.add_argument("--views-per-state", type=int, default=1)
    parser.add_argument("--max-pairs-per-suite", type=int, default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--val-demo-fraction", type=float, default=0.10)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--no-save-flipud", action="store_true")
    parser.add_argument(
        "--benchmark-camera-jsons",
        nargs="*",
        type=Path,
        default=sorted((REPO_ROOT / "outputs" / "phase0" / "libero_plus_camera_eval").glob("camera_task_names_*.json")),
        help="LIBERO-Plus camera task-name JSONs; sampled training cameras are rejection-sampled "
        "against these benchmark poses (standing rule 4: train/eval camera disjointness). "
        "Pass an empty list to disable (NOT allowed for training data).",
    )
    args = parser.parse_args()

    if args.shard_idx >= args.n_shards:
        raise ValueError(f"--shard-idx {args.shard_idx} must be < --n-shards {args.n_shards}")
    benchmark_exclude = load_benchmark_camera_tuples(list(args.benchmark_camera_jsons))
    if benchmark_exclude:
        print(f"[disjointness] excluding {len(benchmark_exclude)} benchmark camera poses from sampling")
    else:
        print("[disjointness] WARNING: empty benchmark exclusion set — training cameras may collide with eval poses")
    bddl_root = args.bddl_root if args.bddl_root is not None else default_bddl_root()
    ensure_dir(args.output_dir)
    ensure_dir(args.results_dir)
    ensure_dir(args.debug_dir)

    all_work: list[Path] = []
    for suite_name in args.suites:
        suite_dir = args.libero_root / suite_name
        if not suite_dir.exists():
            print(f"[SKIP] suite dir not found: {suite_dir}")
            continue
        hdf5_files = sorted(suite_dir.glob("*.hdf5"))
        if args.max_tasks is not None:
            hdf5_files = hdf5_files[: args.max_tasks]
        all_work.extend(hdf5_files)
    if args.n_shards > 1:
        all_work = [path for idx, path in enumerate(all_work) if idx % args.n_shards == args.shard_idx]
        print(f"[shard {args.shard_idx}/{args.n_shards}] owns {len(all_work)} HDF5 files")

    started = time.time()
    rng = np.random.default_rng(args.seed + args.shard_idx)
    all_rows: list[dict[str, Any]] = []
    suite_counts: dict[str, int] = {}
    for hdf5_path in tqdm(all_work, desc=f"shard{args.shard_idx}", dynamic_ncols=True):
        suite = suite_base_name(hdf5_path.parent.name)
        remaining = None
        if args.max_pairs_per_suite is not None:
            remaining = args.max_pairs_per_suite - suite_counts.get(suite, 0)
            if remaining <= 0:
                continue
        rows = process_hdf5(
            hdf5_path,
            bddl_root,
            args.output_dir,
            rng,
            timestep_sample_rate=args.timestep_sample_rate,
            views_per_state=args.views_per_state,
            max_pairs=remaining,
            img_size=args.img_size,
            gpu_device_id=args.gpu_device_id,
            val_demo_fraction=args.val_demo_fraction,
            chunk_size=args.chunk_size,
            save_flipud=not args.no_save_flipud,
            benchmark_exclude=benchmark_exclude,
        )
        all_rows.extend(rows)
        suite_counts[suite] = suite_counts.get(suite, 0) + len(rows)
        tqdm.write(f"{hdf5_path.parent.name}/{hdf5_path.name}: +{len(rows)} pairs")

    shard_tag = f"_shard{args.shard_idx:02d}" if args.n_shards > 1 else ""
    train_rows = [row for row in all_rows if row["split"] == "train"]
    val_rows = [row for row in all_rows if row["split"] == "val"]
    write_jsonl(args.results_dir / f"libero_pair_future_manifest_train{shard_tag}.jsonl", train_rows)
    write_jsonl(args.results_dir / f"libero_pair_future_manifest_val{shard_tag}.jsonl", val_rows)
    write_jsonl(
        args.results_dir / f"libero_wrong_pair_future_manifest_train{shard_tag}.jsonl",
        build_wrong_state_controls(train_rows, rng),
    )
    write_jsonl(
        args.results_dir / f"libero_wrong_pair_future_manifest_val{shard_tag}.jsonl",
        build_wrong_state_controls(val_rows, rng),
    )
    write_audit_report(
        args.results_dir / f"libero_pair_future_audit_report{shard_tag}.md",
        all_rows,
        time.time() - started,
    )
    print(f"[done] shard={args.shard_idx} matched_pairs={len(all_rows)}")


if __name__ == "__main__":
    main()
