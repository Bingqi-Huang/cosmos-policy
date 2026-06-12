"""Validate Phase-2 pair future-frame manifests without using GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
from collections import Counter
from typing import Any

import numpy as np


def _hash_array(array: np.ndarray) -> str:
    """Exact replica of render_libero_pair_future_frames.hash_array — must stay in sync."""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()[:24]


def _check_pair_type_semantics(rows: list[dict[str, Any]]) -> list[str]:
    """Per-record same-state guarantee, manifest-level half.

    A 'matched' record holds ONE (hdf5, demo, t) label source for both views, so both views share
    actions/proprio/value by construction — any per-view ``*_hash_b`` field must therefore be absent
    or equal. A 'wrong_state' (A1 data-level control) record must carry a ``state_hash_b`` that
    actually differs, otherwise the control is silently diluted with correct-state pairs.
    """
    violations: list[str] = []
    for row in rows:
        pair_id = str(row.get("pair_id"))
        pair_type = str(row.get("pair_type"))
        if pair_type == "matched":
            for key in ("state_hash_b", "future_state_hash_b"):
                if key in row and row[key] != row.get(key.removesuffix("_b")):
                    violations.append(f"{pair_id}: matched pair has differing {key}")
        elif pair_type == "wrong_state":
            if "state_hash_b" not in row:
                violations.append(f"{pair_id}: wrong_state pair missing state_hash_b")
            elif row["state_hash_b"] == row.get("state_hash"):
                violations.append(f"{pair_id}: wrong_state pair has IDENTICAL state_hash_b (diluted control)")
            if float(row.get("pair_confidence", 1.0)) != 0.0:
                violations.append(f"{pair_id}: wrong_state pair must have pair_confidence=0")
        else:
            violations.append(f"{pair_id}: unknown pair_type {pair_type!r}")
    return violations


def _check_state_hashes_against_hdf5(
    rows: list[dict[str, Any]], repo_root: pathlib.Path, num_rows: int, seed: int = 0
) -> dict[str, Any]:
    """Recompute the manifest's label hashes from the HDF5 source (sampled rows).

    This is the end-to-end form of "assert value_0 == value_p": the pair dataset reads
    actions/proprio/value from (hdf5_path, demo_key, timestep), so verifying that the manifest's
    state/action/robot-state hashes match the HDF5 contents proves both views of every checked
    record share exactly the recorded (s, q, a) the images were rendered from.
    """
    import h5py  # local import: keep the validator importable without the training env

    if num_rows == 0:
        return {"num_rows_checked": 0, "mismatches": [], "skipped": True}
    rng = np.random.default_rng(seed)
    candidates = [row for row in rows if str(row.get("pair_type")) == "matched"]
    if 0 < num_rows < len(candidates):
        picked = [candidates[i] for i in rng.choice(len(candidates), size=num_rows, replace=False)]
    else:
        picked = candidates

    mismatches: list[str] = []
    by_file: dict[str, list[dict[str, Any]]] = {}
    for row in picked:
        by_file.setdefault(str(row["hdf5_path"]), []).append(row)
    for hdf5_path, file_rows in by_file.items():
        path = _resolve(repo_root, hdf5_path)
        if not path.exists():
            mismatches.extend(f"{row['pair_id']}: missing HDF5 {path}" for row in file_rows)
            continue
        with h5py.File(path, "r") as f:
            data_group = f["data"]
            for row in file_rows:
                pair_id = str(row["pair_id"])
                demo = data_group[str(row["demo_key"])]
                t = int(row["timestep"])
                future_t = int(row["future_timestep"])
                chunk_size = int(row["chunk_size"])
                # dtype handling replicates the renderer exactly: states raw, actions/robot_states float32.
                states = demo["states"][:]
                actions = demo["actions"][:].astype(np.float32)
                robot_states = demo["robot_states"][:].astype(np.float32)
                expected = {
                    "state_hash": _hash_array(states[t]),
                    "future_state_hash": _hash_array(states[future_t]),
                    "action_chunk_hash": _hash_array(actions[t : min(t + chunk_size, len(states))]),
                    "robot_state_hash": _hash_array(robot_states[t]),
                    "future_robot_state_hash": _hash_array(robot_states[future_t]),
                }
                for key, value in expected.items():
                    if str(row.get(key)) != value:
                        mismatches.append(f"{pair_id}: {key} manifest={row.get(key)} hdf5={value}")
    return {"num_rows_checked": len(picked), "mismatches": mismatches, "skipped": False}


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _resolve(path: pathlib.Path, raw: str) -> pathlib.Path:
    candidate = pathlib.Path(raw)
    if candidate.is_absolute():
        return candidate
    return path / candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--max-missing-to-print", type=int, default=20)
    parser.add_argument(
        "--benchmark-camera-jsons",
        nargs="*",
        default=None,
        help="LIBERO-Plus camera_task_names_*.json files; defaults to auto-discovery under "
        "outputs/phase0/libero_plus_camera_eval. Audits train/eval camera-pose disjointness "
        "(execution_plan standing rule 4) and fails on any collision.",
    )
    parser.add_argument(
        "--state-hash-check-rows",
        type=int,
        default=256,
        help="Recompute state/action/robot-state hashes from HDF5 for this many sampled matched "
        "rows and compare with the manifest (same-state pair guarantee). 0 disables, -1 checks all rows.",
    )
    args = parser.parse_args()

    repo_root = pathlib.Path(args.repo_root).resolve()
    manifest = pathlib.Path(args.manifest)
    rows = _read_jsonl(manifest)
    required_image_keys = [
        "current_img_a_path",
        "current_img_b_path",
        "future_img_a_path",
        "future_img_b_path",
    ]

    missing: list[str] = []
    duplicate_ids = [pair_id for pair_id, count in Counter(row["pair_id"] for row in rows).items() if count > 1]
    by_split = Counter(str(row.get("split")) for row in rows)
    by_suite = Counter(str(row.get("suite")) for row in rows)
    by_category = Counter(str(row.get("camera_category")) for row in rows)
    bad_future = [
        row["pair_id"]
        for row in rows
        if int(row["future_timestep"]) < int(row["timestep"])
        or int(row["future_timestep"]) - int(row["timestep"]) > int(row["chunk_size"])
    ]
    for row in rows:
        for key in required_image_keys:
            path = _resolve(repo_root, str(row[key]))
            if not path.exists():
                missing.append(str(path))

    # Train/eval camera-pose disjointness audit (execution_plan standing rule 4).
    benchmark_jsons = args.benchmark_camera_jsons
    if benchmark_jsons is None:
        benchmark_jsons = sorted(
            str(p)
            for p in (repo_root / "outputs" / "phase0" / "libero_plus_camera_eval").glob("camera_task_names_*.json")
        )
    benchmark_tuples: set[tuple[int, ...]] = set()
    view_pattern = re.compile(r"_view_(-?\d+)_(-?\d+)_(\d+)_(-?\d+)_(-?\d+)_initstate_")
    for json_path in benchmark_jsons:
        for name in json.loads(pathlib.Path(json_path).read_text(encoding="utf-8")):
            match = view_pattern.search(name)
            if match:
                benchmark_tuples.add(tuple(int(g) for g in match.groups()))
    camera_collisions: list[str] = []
    if benchmark_tuples:
        for row in rows:
            params = row.get("camera_params_b") or {}
            key = (
                int(params.get("horizon", 0)),
                int(params.get("vertical", 0)),
                int(params.get("scale", 100)),
                int(params.get("end_rot", 0)),
                int(params.get("end_vert", 0)),
            )
            if key in benchmark_tuples:
                camera_collisions.append(str(row["pair_id"]))

    # Same-state pair guarantees (the data-layer form of "assert value_0 == value_p").
    pair_type_violations = _check_pair_type_semantics(rows)
    state_hash_checks = _check_state_hashes_against_hdf5(
        rows, repo_root=repo_root, num_rows=args.state_hash_check_rows
    )

    report = {
        "manifest": str(manifest),
        "num_rows": len(rows),
        "by_split": dict(sorted(by_split.items())),
        "by_suite": dict(sorted(by_suite.items())),
        "by_camera_category": dict(sorted(by_category.items())),
        "num_duplicate_pair_ids": len(duplicate_ids),
        "num_bad_future_indices": len(bad_future),
        "num_missing_images": len(missing),
        "sample_missing_images": missing[: args.max_missing_to_print],
        "num_benchmark_camera_poses": len(benchmark_tuples),
        "num_eval_camera_collisions": len(camera_collisions),
        "sample_eval_camera_collisions": camera_collisions[: args.max_missing_to_print],
        "num_pair_type_violations": len(pair_type_violations),
        "sample_pair_type_violations": pair_type_violations[: args.max_missing_to_print],
        "state_hash_checks": {
            "num_rows_checked": state_hash_checks["num_rows_checked"],
            "skipped": state_hash_checks["skipped"],
            "num_mismatches": len(state_hash_checks["mismatches"]),
            "sample_mismatches": state_hash_checks["mismatches"][: args.max_missing_to_print],
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if duplicate_ids or bad_future or missing or camera_collisions or pair_type_violations or state_hash_checks["mismatches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
