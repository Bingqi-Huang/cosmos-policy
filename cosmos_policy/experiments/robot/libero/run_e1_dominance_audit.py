"""Launcher for the E1.0 wrist/scene dominance audit.

Runs standard LIBERO task-index evals for:
  - no mask
  - scene/primary mask in black, gray, noise
  - wrist mask in black, gray, noise

Each job uses one GPU. Jobs are queued across the provided GPU ids.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time


REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
EVAL_SCRIPT = pathlib.Path(__file__).resolve().parent / "run_libero_eval.py"

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MASK_JOBS = (
    ("none", "none", "none"),
    ("primary_black", "black", "none"),
    ("primary_gray", "gray", "none"),
    ("primary_noise", "noise", "none"),
    ("wrist_black", "none", "black"),
    ("wrist_gray", "none", "gray"),
    ("wrist_noise", "none", "noise"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="outputs/phase1/e1_dominance_audit/expanded_3task_5ep")
    p.add_argument("--gpu_ids", default="0,1,2,3,4,5")
    p.add_argument("--task_ids", default="0,1,2")
    p.add_argument("--num_trials_per_task", type=int, default=5)
    p.add_argument("--ckpt_path", default="nvidia/Cosmos-Policy-LIBERO-Predict2-2B")
    p.add_argument("--config", default="cosmos_predict2_2b_480p_libero__inference_only")
    p.add_argument("--dataset_stats_path", default="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json")
    p.add_argument("--t5_text_embeddings_path", default="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl")
    return p.parse_args()


def running_count(workers: list[tuple[str, subprocess.Popen, pathlib.Path]]) -> int:
    return sum(1 for _, proc, _ in workers if proc.poll() is None)


def main() -> None:
    ns = parse_args()
    output_dir = pathlib.Path(ns.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_dir = output_dir / "stdout"
    stdout_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = [g.strip() for g in ns.gpu_ids.split(",") if g.strip()]
    task_ids = [int(x.strip()) for x in ns.task_ids.split(",") if x.strip()]
    task_indices_file = output_dir / "task_indices.json"
    task_indices_file.write_text(json.dumps(task_ids, indent=2) + "\n")

    jobs: list[tuple[str, str, str, str, str]] = []
    for suite in SUITES:
        for label, primary_mask, wrist_mask in MASK_JOBS:
            jobs.append((suite, label, primary_mask, wrist_mask, gpu_ids[len(jobs) % len(gpu_ids)]))

    manifest = {
        "suites": list(SUITES),
        "mask_jobs": [
            {"label": label, "primary_image_mask_mode": primary, "wrist_image_mask_mode": wrist}
            for label, primary, wrist in MASK_JOBS
        ],
        "task_ids": task_ids,
        "num_trials_per_task": ns.num_trials_per_task,
        "jobs": [
            {"suite": suite, "label": label, "gpu": gpu}
            for suite, label, _primary, _wrist, gpu in jobs
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    workers: list[tuple[str, subprocess.Popen, pathlib.Path]] = []
    pending = list(jobs)
    started_at = time.time()

    while pending or running_count(workers) > 0:
        busy_gpus = {
            name.split("|")[-1]
            for name, proc, _log in workers
            if proc.poll() is None
        }
        available = [g for g in gpu_ids if g not in busy_gpus]

        while pending and available:
            suite, label, primary_mask, wrist_mask, gpu = pending.pop(0)
            if gpu not in available:
                pending.append((suite, label, primary_mask, wrist_mask, gpu))
                break
            available.remove(gpu)

            job_dir = output_dir / suite / label
            job_dir.mkdir(parents=True, exist_ok=True)
            log_path = stdout_dir / f"{suite}__{label}.log"
            result_json = job_dir / "results.json"
            per_task_jsonl = job_dir / "per_task.jsonl"
            run_id_note = f"e1_expand_{suite}_{label}"

            cmd = [
                sys.executable,
                str(EVAL_SCRIPT),
                "--task_suite_name", suite,
                "--task_indices_file", str(task_indices_file),
                "--num_trials_per_task", str(ns.num_trials_per_task),
                "--shard_results_json", str(result_json),
                "--per_task_jsonl", str(per_task_jsonl),
                "--available_gpus", gpu,
                "--local_log_dir", str(job_dir / "logs"),
                "--run_id_note", run_id_note,
                "--ckpt_path", ns.ckpt_path,
                "--config", ns.config,
                "--dataset_stats_path", ns.dataset_stats_path,
                "--t5_text_embeddings_path", ns.t5_text_embeddings_path,
                "--use_wrist_image", "True",
                "--use_proprio", "True",
                "--normalize_proprio", "True",
                "--unnormalize_actions", "True",
                "--trained_with_image_aug", "True",
                "--chunk_size", "16",
                "--num_open_loop_steps", "16",
                "--flip_images", "True",
                "--use_jpeg_compression", "True",
                "--num_denoising_steps_action", "5",
                "--num_denoising_steps_future_state", "1",
                "--num_denoising_steps_value", "1",
                "--ar_future_prediction", "False",
                "--ar_value_prediction", "False",
                "--deterministic", "True",
                "--randomize_seed", "False",
                "--seed", "195",
                "--primary_image_mask_mode", primary_mask,
                "--wrist_image_mask_mode", wrist_mask,
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["MUJOCO_GL"] = "egl"
            log_f = log_path.open("a", encoding="utf-8")
            proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=log_f, stderr=subprocess.STDOUT, text=True)
            name = f"{suite}/{label}|{gpu}"
            workers.append((name, proc, log_path))
            print(f"[launch] {suite}/{label} gpu={gpu} pid={proc.pid} log={log_path}", flush=True)

        done = 0
        failed = []
        for name, proc, log_path in workers:
            rc = proc.poll()
            if rc is None:
                continue
            done += 1
            if rc != 0:
                failed.append((name, rc, log_path))

        elapsed_min = (time.time() - started_at) / 60.0
        print(
            f"[status] done={done}/{len(jobs)} running={running_count(workers)} "
            f"pending={len(pending)} elapsed={elapsed_min:.1f}m",
            flush=True,
        )
        if failed:
            for name, rc, log_path in failed:
                print(f"[ERROR] {name} exited {rc}; see {log_path}", flush=True)
            raise SystemExit(1)

        if pending or running_count(workers) > 0:
            time.sleep(30)

    print(f"[complete] all jobs finished under {output_dir}", flush=True)


if __name__ == "__main__":
    main()
