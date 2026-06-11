"""Validate Phase-2 pair future-frame manifests without using GPUs."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import Counter
from typing import Any


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
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if duplicate_ids or bad_future or missing or camera_collisions:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
