"""CPU checks for the SCVC pair-data and batch contract.

This is the first rung of the Phase-3 sanity ladder.  It intentionally avoids
model imports and CUDA.  The goal is to catch schema, layout, and
inference-contract errors before a GPU memory smoke is allowed to run.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter
from typing import Any


REQUIRED_MANIFEST_KEYS = {
    "pair_id",
    "hdf5_path",
    "demo_key",
    "timestep",
    "future_timestep",
    "chunk_size",
    "split",
    "language",
    "current_img_a_path",
    "current_img_b_path",
    "future_img_a_path",
    "future_img_b_path",
    "camera_params_b",
    "camera_category",
    "pair_type",
}

P2_LATENT_LAYOUT = {
    "action_latent_idx": 3,
    "current_proprio_latent_idx": 1,
    "current_wrist_image_latent_idx": -1,
    "current_image_latent_idx": 2,
    "future_proprio_latent_idx": 4,
    "future_wrist_image_latent_idx": -1,
    "future_image_latent_idx": 5,
    "value_latent_idx": 6,
}

MODEL_INPUT_KEYS_ALLOWED = {
    "video",
    "video_pair",
    "pair_valid",
    "actions",
    "t5_text_embeddings",
    "t5_text_mask",
    "fps",
    "padding_mask",
    "image_size",
    "proprio",
    "future_proprio",
    "__key__",
    "rollout_data_mask",
    "rollout_data_success_mask",
    "world_model_sample_mask",
    "value_function_sample_mask",
    "global_rollout_idx",
    "action_latent_idx",
    "value_latent_idx",
    "current_proprio_latent_idx",
    "current_wrist_image_latent_idx",
    "current_image_latent_idx",
    "future_proprio_latent_idx",
    "future_wrist_image_latent_idx",
    "future_image_latent_idx",
    "value_function_return",
    "next_action_chunk",
    "next_value_function_return",
    # Audit-only strings that default PyTorch collate keeps outside tensor inputs.
    "pair_id",
    "pair_type",
    "camera_category",
}

FORBIDDEN_MODEL_KEY_FRAGMENTS = ("camera_params", "cam_pos", "cam_quat", "extrinsic", "intrinsic", "depth")


def read_jsonl(path: pathlib.Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


def resolve(repo_root: pathlib.Path, raw: str) -> pathlib.Path:
    path = pathlib.Path(raw)
    return path if path.is_absolute() else repo_root / path


def conjugated_cycle_derangement(batch_size: int, sigma: list[int]) -> list[int]:
    """Pure-Python replica of SCVCPolicyVideo2WorldModel._derangement's fallback:
    perm[sigma] = sigma[(identity + 1) % B].  Must stay in sync with the model code."""
    if batch_size < 2:
        raise ValueError("batch_size must be >=2")
    perm = [0] * batch_size
    for i in range(batch_size):
        perm[sigma[i]] = sigma[(i + 1) % batch_size]
    return perm


def check_manifest(rows: list[dict[str, Any]], repo_root: pathlib.Path, check_images: bool) -> dict[str, Any]:
    missing_keys: Counter[str] = Counter()
    missing_images: list[str] = []
    bad_future: list[str] = []
    forbidden_manifest_keys: Counter[str] = Counter()
    pair_ids = Counter(str(row.get("pair_id")) for row in rows)
    for row in rows:
        missing = REQUIRED_MANIFEST_KEYS - set(row)
        for key in missing:
            missing_keys[key] += 1
        for key in row:
            if any(fragment in key for fragment in FORBIDDEN_MODEL_KEY_FRAGMENTS):
                # These are allowed in the manifest for audit; they must not enter model inputs.
                forbidden_manifest_keys[key] += 1
        if "timestep" in row and "future_timestep" in row and "chunk_size" in row:
            dt = int(row["future_timestep"]) - int(row["timestep"])
            if dt < 0 or dt > int(row["chunk_size"]):
                bad_future.append(str(row.get("pair_id", "<missing>")))
        if check_images:
            for key in ("current_img_a_path", "current_img_b_path", "future_img_a_path", "future_img_b_path"):
                if key in row:
                    path = resolve(repo_root, str(row[key]))
                    if not path.exists():
                        missing_images.append(str(path))
    duplicate_pair_ids = [pair_id for pair_id, count in pair_ids.items() if count > 1]
    return {
        "num_rows_checked": len(rows),
        "missing_required_keys": dict(missing_keys),
        "duplicate_pair_ids": duplicate_pair_ids[:20],
        "num_duplicate_pair_ids": len(duplicate_pair_ids),
        "bad_future_indices": bad_future[:20],
        "num_bad_future_indices": len(bad_future),
        "num_missing_images": len(missing_images),
        "sample_missing_images": missing_images[:20],
        "manifest_audit_camera_fields_seen": dict(forbidden_manifest_keys),
    }


def check_dataset_sample(args) -> dict[str, Any]:
    from cosmos_policy.datasets.libero_pair_dataset import LIBEROPairDataset

    dataset = LIBEROPairDataset(
        data_dir=args.data_dir,
        pair_manifest_path=args.manifest,
        repo_root=args.repo_root,
        t5_text_embeddings_path=args.t5_text_embeddings_path,
        rollout_data_dir=args.rollout_data_dir,
        use_image_aug=False,
        use_stronger_image_aug=False,
    )
    item = dataset[0]
    unexpected_keys = sorted(set(item) - MODEL_INPUT_KEYS_ALLOWED)
    forbidden_model_keys = [
        key for key in item if any(fragment in key for fragment in FORBIDDEN_MODEL_KEY_FRAGMENTS)
    ]
    layout_errors = {
        key: {"expected": expected, "actual": int(item.get(key, 999))}
        for key, expected in P2_LATENT_LAYOUT.items()
        if int(item.get(key, 999)) != expected
    }
    video_shape = tuple(int(v) for v in item["video"].shape)
    video_pair_shape = tuple(int(v) for v in item["video_pair"].shape)
    return {
        "dataset_length": len(dataset),
        "sample_video_shape": video_shape,
        "sample_video_pair_shape": video_pair_shape,
        "pair_valid": int(item["pair_valid"]),
        "rollout_data_mask": int(item["rollout_data_mask"]),
        "layout_errors": layout_errors,
        "unexpected_model_keys": unexpected_keys,
        "forbidden_model_keys": forbidden_model_keys,
    }


def check_derangement(num_random_sigmas: int = 200, seed: int = 0) -> dict[str, Any]:
    """Check the model's actual derangement construction, not a strawman.

    The model first rejection-samples random permutations (correct by construction when it
    returns: it only accepts fixed-point-free draws), then falls back to a conjugated cyclic
    shift.  The fallback formula is the part that can silently be wrong, so it is replicated
    here in pure Python and exercised over many random conjugating permutations sigma.
    """
    import random

    rng = random.Random(seed)
    failures = []
    for batch_size in range(2, 65):
        for _ in range(num_random_sigmas):
            sigma = list(range(batch_size))
            rng.shuffle(sigma)
            perm = conjugated_cycle_derangement(batch_size, sigma)
            if sorted(perm) != list(range(batch_size)) or any(i == p for i, p in enumerate(perm)):
                failures.append({"batch_size": batch_size, "sigma": sigma})
                break
    return {
        "batch_sizes_checked": [2, 64],
        "random_sigmas_per_batch_size": num_random_sigmas,
        "failures": failures,
    }


def check_valid_subset_derangement() -> dict[str, Any]:
    """Pure-Python contract for A1: derange valid demo-pair rows without CV dilution."""

    failures = []
    cases = [
        [1, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    for valid in cases:
        batch_size = len(valid)
        valid_indices = [i for i, v in enumerate(valid) if v]
        if len(valid_indices) >= 2:
            subperm = conjugated_cycle_derangement(len(valid_indices), list(range(len(valid_indices))))
            perm = list(range(batch_size))
            for src_pos, dst_pos in enumerate(subperm):
                perm[valid_indices[src_pos]] = valid_indices[dst_pos]
            active = [bool(valid[i] and valid[perm[i]]) for i in range(batch_size)]
            if sum(active) != len(valid_indices) or any(i == perm[i] for i in valid_indices):
                failures.append({"case": valid, "perm": perm, "active": active})
        else:
            # With fewer than 2 valid rows, a wrong-state valid comparison is impossible;
            # the model falls back to a full-batch derangement so active CV must be zero.
            perm = conjugated_cycle_derangement(batch_size, list(range(batch_size)))
            active = [bool(valid[i] and valid[perm[i]]) for i in range(batch_size)]
            if any(active):
                failures.append({"case": valid, "perm": perm, "active": active})
    return {"cases_checked": len(cases), "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--check-images", action="store_true")
    parser.add_argument("--dataset-sample", action="store_true")
    parser.add_argument("--data-dir", default="LIBERO-Cosmos-Policy/success_only")
    parser.add_argument("--rollout-data-dir", default="")
    parser.add_argument("--t5-text-embeddings-path", default="LIBERO-Cosmos-Policy/success_only/t5_embeddings.pkl")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    repo_root = pathlib.Path(args.repo_root).resolve()
    rows = read_jsonl(pathlib.Path(args.manifest), max_rows=args.max_rows)
    report = {
        "manifest": args.manifest,
        "repo_root": str(repo_root),
        "manifest_checks": check_manifest(rows, repo_root=repo_root, check_images=args.check_images),
        "derangement_checks": check_derangement(),
        "valid_subset_derangement_checks": check_valid_subset_derangement(),
    }
    if args.dataset_sample:
        report["dataset_sample_checks"] = check_dataset_sample(args)

    failures = []
    m = report["manifest_checks"]
    if m["missing_required_keys"] or m["num_duplicate_pair_ids"] or m["num_bad_future_indices"] or m["num_missing_images"]:
        failures.append("manifest")
    d = report["derangement_checks"]
    if d["failures"]:
        failures.append("derangement")
    vd = report["valid_subset_derangement_checks"]
    if vd["failures"]:
        failures.append("valid_subset_derangement")
    if "dataset_sample_checks" in report:
        s = report["dataset_sample_checks"]
        if s["layout_errors"] or s["forbidden_model_keys"]:
            failures.append("dataset_sample")
    report["status"] = "fail" if failures else "pass"
    report["failures"] = failures

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        out = pathlib.Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
