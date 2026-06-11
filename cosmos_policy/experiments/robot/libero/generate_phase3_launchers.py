"""Generate shell launchers from the Phase-3/4 SCVC run registry.

This script writes launch scripts only.  It does not start training and is safe
to run while GPUs are occupied.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shlex
from typing import Any

from validate_phase3_registry import validate_registry


def shell_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def script_for_run(run: dict[str, Any], common: dict[str, Any]) -> str:
    launcher = common["launcher"]
    env = {
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5",
        "NUM_GPUS": run.get("num_gpus", common["num_gpus"]),
        "FSDP_SHARD_SIZE": run.get("fsdp_shard_size", common["fsdp_shard_size"]),
        "PAIR_BATCH_SIZE": run.get("pair_batch_size", common["pair_batch_size"]),
        "PAIR_MANIFEST_PATH": common["pair_manifest_train"],
        "JOB_NAME": run["job_name"],
        "LAMBDA_CV": run["lambda_cv"],
        "CV_FRAME_SET": run["cv_frame_set"],
        "CV_NOISE_SHARED": run["cv_noise_shared"],
        "CV_PAIR_MODE": run["cv_pair_mode"],
        "CV_NUM_SAMPLES": run.get("cv_num_samples", common["cv_num_samples"]),
        "CV_TOTAL_STEPS": run.get("cv_total_steps", common["cv_total_steps"]),
        "MAX_ITER": run.get("max_iter", common["max_iter"]),
        "SAVE_ITER": run.get("save_iter", common["save_iter"]),
        "WANDB_MODE": run.get("wandb_mode", common["wandb_mode"]),
    }
    env.update(run.get("extra_env", {}))
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# Generated from phase3_run_registry.json for {run['run_id']}.",
        "# Review GPU availability before executing.",
        "",
    ]
    for key, value in env.items():
        lines.append(f"export {key}={shlex.quote(shell_value(value))}")
    lines.extend(
        [
            "",
            f"bash {shlex.quote(launcher)} \"$@\"",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="cosmos_policy/experiments/robot/libero/phase3_run_registry.json")
    parser.add_argument("--output-dir", default="outputs/phase3/launchers")
    parser.add_argument("--include-planned", action="store_true", help="Also emit commented placeholders for planned-not-launchable rows")
    args = parser.parse_args()

    registry_path = pathlib.Path(args.registry)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    errors, warnings, summary = validate_registry(registry)
    if errors:
        raise SystemExit("Registry validation failed; run validate_phase3_registry.py for details.")
    for warning in warnings:
        print(f"[registry warning] {warning}")

    common = registry["common"]
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for run in registry["runs"]:
        if not run.get("launchable"):
            if args.include_planned:
                path = output_dir / f"{run['run_id']}.sh"
                path.write_text(
                    "#!/usr/bin/env bash\n"
                    "set -euo pipefail\n\n"
                    f"echo {shlex.quote('Not launchable: ' + run.get('blocked_by', 'planned row'))}\n"
                    "exit 2\n",
                    encoding="utf-8",
                )
                path.chmod(0o755)
            continue
        path = output_dir / f"{run['run_id']}.sh"
        path.write_text(script_for_run(run, common), encoding="utf-8")
        path.chmod(0o755)
        manifest_rows.append({"run_id": run["run_id"], "job_name": run["job_name"], "launcher": str(path)})

    manifest = {
        "registry": str(registry_path),
        "output_dir": str(output_dir),
        "num_launchers": len(manifest_rows),
        "registry_summary": summary,
        "launchers": manifest_rows,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[launchers] wrote {len(manifest_rows)} launchers to {output_dir}")


if __name__ == "__main__":
    main()
