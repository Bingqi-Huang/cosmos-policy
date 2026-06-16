"""Validate the Phase-3/4 SCVC experiment registry without importing CUDA code."""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter, defaultdict
from typing import Any


# 'invariant_plus_fscene' (= invariant block ∪ {future-scene}) is the A2 wrong-coordinates set;
# it was formerly (misleadingly) named 'full' — blank/conditioning frames are never in any CV set.
ALLOWED_FRAME_SETS = {"action", "action+value", "action+value+fproprio", "invariant_plus_fscene"}
ALLOWED_PAIR_MODES = {"matched", "derangement"}
MAIN_RUN_TYPES = {"main_10k"}
FORBIDDEN_ENV_FRAGMENTS = ("camera", "camera_params", "extrinsic", "intrinsic", "depth", "cam_pos", "cam_quat")


def load_registry(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_run(run: dict[str, Any], common: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    run_id = run.get("run_id", "<missing>")
    launchable = bool(run.get("launchable", False))
    run_type = str(run.get("run_type", ""))
    max_iter = int(run.get("max_iter", common.get("max_iter", 0)))
    save_iter = int(run.get("save_iter", common.get("save_iter", 0)))
    pair_batch_size = int(run.get("pair_batch_size", common.get("pair_batch_size", 0)))
    cv_num_samples = int(run.get("cv_num_samples", common.get("cv_num_samples", 0)))
    cv_total_steps = int(run.get("cv_total_steps", common.get("cv_total_steps", 0)))
    fsdp_shard_size = int(run.get("fsdp_shard_size", common.get("fsdp_shard_size", 0)))
    num_gpus = int(run.get("num_gpus", common.get("num_gpus", 0)))
    grad_accum_iter = int(run.get("grad_accum_iter", common.get("grad_accum_iter", 0)))
    effective_batch = int(run.get("effective_batch", common.get("effective_batch", 0)))
    budget = int(run.get("budget_sample_presentations", common.get("budget_sample_presentations", 0)))

    require("run_id" in run, "run missing run_id", errors)
    require("job_name" in run, f"{run_id}: missing job_name", errors)
    # Seed is wired into the generated launcher as trainer.seed / sampler.seed overrides;
    # a missing seed would silently collapse multi-seed rows into identical runs.
    require(isinstance(run.get("seed"), int), f"{run_id}: seed must be an integer", errors)
    require(str(run.get("cv_frame_set")) in ALLOWED_FRAME_SETS, f"{run_id}: invalid cv_frame_set", errors)
    require(str(run.get("cv_pair_mode")) in ALLOWED_PAIR_MODES, f"{run_id}: invalid cv_pair_mode", errors)
    require(isinstance(run.get("cv_noise_shared"), bool), f"{run_id}: cv_noise_shared must be bool", errors)
    require(float(run.get("lambda_cv", -1.0)) >= 0.0, f"{run_id}: lambda_cv must be non-negative", errors)

    if run_type in MAIN_RUN_TYPES:
        require(launchable, f"{run_id}: main_10k run must be launchable", errors)
        require(max_iter == 10000, f"{run_id}: main_10k max_iter must be 10000", errors)
        require(cv_total_steps == 10000, f"{run_id}: main_10k cv_total_steps must be 10000", errors)
        require(save_iter == 1000, f"{run_id}: save_iter must be 1000", errors)
        require(pair_batch_size == 10, f"{run_id}: pair_batch_size must be 10 until memory smoke changes registry", errors)
        require(cv_num_samples == 2, f"{run_id}: cv_num_samples must be 2", errors)
        require(num_gpus == 6, f"{run_id}: num_gpus must be 6", errors)
        require(fsdp_shard_size == 6, f"{run_id}: fsdp_shard_size must be 6", errors)
        # Budget arithmetic: grad_accum x bs x gpus = effective batch; x max_iter = sample
        # presentations. Must equal the frozen 7.2M budget, else the run is silently off-budget
        # (the SCVC config default grad_accum=1 would yield eff batch 60 = 1/12 of the budget).
        require(grad_accum_iter > 0, f"{run_id}: grad_accum_iter must be set (>0)", errors)
        require(
            grad_accum_iter * pair_batch_size * num_gpus == effective_batch,
            f"{run_id}: effective_batch {effective_batch} != grad_accum {grad_accum_iter} x bs "
            f"{pair_batch_size} x gpus {num_gpus} = {grad_accum_iter * pair_batch_size * num_gpus}",
            errors,
        )
        require(
            effective_batch * max_iter == budget == 7_200_000,
            f"{run_id}: sample presentations {effective_batch * max_iter} (eff {effective_batch} x "
            f"iter {max_iter}) must equal the frozen budget 7,200,000 (registry budget={budget})",
            errors,
        )
    elif run_type == "planned_not_launchable":
        require(not launchable, f"{run_id}: planned_not_launchable must not be launchable", errors)
        require("blocked_by" in run, f"{run_id}: planned_not_launchable needs blocked_by", errors)
    else:
        errors.append(f"{run_id}: unknown run_type {run_type!r}")

    if launchable:
        env_keys = run.get("extra_env", {})
        if isinstance(env_keys, dict):
            for key in env_keys:
                lowered = str(key).lower()
                if any(fragment in lowered for fragment in FORBIDDEN_ENV_FRAGMENTS):
                    errors.append(f"{run_id}: forbidden camera/depth env key {key!r}")
        else:
            errors.append(f"{run_id}: extra_env must be an object if present")

    if run.get("cv_frame_set") == "invariant_plus_fscene" and run.get("group") != "E3_A2_wrong_coordinates":
        warnings.append(f"{run_id}: invariant_plus_fscene frame-set outside A2")
    if run.get("group") == "E3_A2_wrong_coordinates" and run.get("cv_frame_set") != "invariant_plus_fscene":
        errors.append(f"{run_id}: A2 wrong-coordinates run must use cv_frame_set=invariant_plus_fscene")
    if run.get("cv_pair_mode") == "derangement" and run.get("group") != "E3_A1_wrong_states":
        warnings.append(f"{run_id}: derangement outside A1")
    if run.get("cv_noise_shared") is False and run.get("group") != "E3_A5_wrong_noise":
        warnings.append(f"{run_id}: independent noise outside A5")


def validate_registry(registry: dict[str, Any]) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    common = registry.get("common", {})
    runs = registry.get("runs", [])

    require(registry.get("schema_version") == 1, "schema_version must be 1", errors)
    require(isinstance(common, dict), "common must be an object", errors)
    require(isinstance(runs, list) and runs, "runs must be a non-empty list", errors)
    require("Cosmos-Policy-LIBERO-Predict2-2B" in str(common.get("base_checkpoint", "")), "base checkpoint must be LIBERO policy checkpoint", errors)
    require(str(common.get("protocol")) == "P2 scene-only, wrist excluded", "protocol must be P2 scene-only, wrist excluded", errors)

    run_ids = Counter(str(run.get("run_id")) for run in runs)
    job_names = Counter(str(run.get("job_name")) for run in runs)
    for run_id, count in run_ids.items():
        if count > 1:
            errors.append(f"duplicate run_id: {run_id}")
    for job_name, count in job_names.items():
        if job_name and job_name != "None" and count > 1:
            errors.append(f"duplicate job_name: {job_name}")

    by_group: dict[str, int] = defaultdict(int)
    launchable = 0
    for run in runs:
        validate_run(run, common, errors, warnings)
        by_group[str(run.get("group"))] += 1
        if run.get("launchable"):
            launchable += 1

    require(by_group.get("E3_A2_wrong_coordinates", 0) == 6, "A2 must contain 3 lambdas x 2 seeds = 6 runs", errors)
    require(by_group.get("E2_row3_pair_fm_only", 0) >= 2, "row3 must have at least 2 seeds", errors)
    require(by_group.get("E2_row4_scvc", 0) >= 2, "row4 must have at least 2 seeds", errors)
    require(by_group.get("E3_A1_wrong_states", 0) >= 1, "A1 must have at least 1 run", errors)
    require(by_group.get("E3_A5_wrong_noise", 0) >= 2, "A5 must have at least 2 registry entries", errors)

    summary = {
        "num_runs": len(runs),
        "num_launchable": launchable,
        "by_group": dict(sorted(by_group.items())),
    }
    return errors, warnings, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="cosmos_policy/experiments/robot/libero/phase3_run_registry.json")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    registry_path = pathlib.Path(args.registry)
    registry = load_registry(registry_path)
    errors, warnings, summary = validate_registry(registry)
    report = {
        "registry": str(registry_path),
        "status": "fail" if errors else "pass",
        "errors": errors,
        "warnings": warnings,
        "summary": summary,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        out = pathlib.Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
