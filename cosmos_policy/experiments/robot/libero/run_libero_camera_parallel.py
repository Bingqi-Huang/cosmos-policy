"""Parallel launcher for LIBERO-Plus camera eval across multiple GPUs.

Splits camera_tasks_file into N shards (one per GPU) via task_index % num_shards == shard_index,
spawns one worker per GPU, monitors progress, then merges shard results into a final SR.

Example (6 GPUs, libero_spatial):
    python run_libero_camera_parallel.py \
        --ckpt_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
        --config cosmos_predict2_2b_480p_libero__inference_only \
        --dataset_stats_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json \
        --t5_text_embeddings_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl \
        --task_suite_name libero_spatial \
        --camera_tasks_file outputs/phase0/libero_plus_camera_eval/camera_task_names_libero_spatial.json \
        --gpu_ids 0,1,2,3,4,5 \
        --num_trials_per_task 3 \
        --output_dir outputs/phase0/libero_plus_camera_eval/parallel_spatial
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time


REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]  # cosmos-policy/
EVAL_SCRIPT = pathlib.Path(__file__).resolve().parent / "run_libero_eval.py"


def _format_eta(seconds: float | None) -> str:
    if seconds is None or seconds <= 0 or not seconds < float("inf"):
        return "unknown"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _count_in_log(log_path: pathlib.Path, pattern: str) -> int:
    if not log_path.exists():
        return 0
    count = 0
    try:
        with log_path.open("r", errors="replace") as f:
            for line in f:
                if pattern in line:
                    count += 1
    except OSError:
        pass
    return count


def _count_jsonl_lines(path: pathlib.Path) -> int:
    """Count completed tasks = non-empty lines in a shard's per_task.jsonl (one line per task).

    Authoritative progress signal. The old approach grepped the worker log for
    "Camera task SR:", which is emitted twice per task, so progress read ~200%
    (e.g. 44/22). per_task.jsonl has exactly one line per finished task.
    """
    if not path.exists():
        return 0
    count = 0
    try:
        with path.open("r", errors="replace") as f:
            for line in f:
                if line.strip():
                    count += 1
    except OSError:
        pass
    return count


def _parse_pass_through_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Split our launcher-specific args from the pass-through eval flags."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpu_ids", default="0,1,2,3,4,5")
    p.add_argument("--output_dir", default="outputs/camera_parallel")
    p.add_argument("--task_suite_name", default="libero_spatial")
    p.add_argument("--camera_tasks_file", required=True)
    p.add_argument("--num_trials_per_task", type=int, default=3)
    ns, remaining = p.parse_known_args(argv)
    return ns, remaining


def main() -> None:
    ns, extra_flags = _parse_pass_through_args(sys.argv[1:])

    gpu_ids = [g.strip() for g in ns.gpu_ids.split(",") if g.strip()]
    num_shards = len(gpu_ids)
    output_dir = pathlib.Path(ns.output_dir)
    shard_root = output_dir / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)

    # Load task list to know total tasks per shard
    task_names = json.loads(pathlib.Path(ns.camera_tasks_file).read_text())
    total_tasks = len(task_names)
    tasks_per_shard = [(total_tasks + i) // num_shards for i in range(num_shards)]  # balanced split

    python = sys.executable

    workers: list[tuple[int, str, subprocess.Popen, pathlib.Path, pathlib.Path, pathlib.Path]] = []
    started_at = time.time()

    for shard_idx, gpu_id in enumerate(gpu_ids):
        shard_dir = shard_root / f"shard_{shard_idx:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        log_path = shard_dir / "worker.log"
        results_json = shard_dir / "results.json"

        per_task_jsonl = shard_dir / "per_task.jsonl"
        cmd = [
            python,
            str(EVAL_SCRIPT),
            "--task_suite_name", ns.task_suite_name,
            "--camera_tasks_file", ns.camera_tasks_file,
            "--num_trials_per_task", str(ns.num_trials_per_task),
            "--num_shards", str(num_shards),
            "--shard_index", str(shard_idx),
            "--shard_results_json", str(results_json),
            "--per_task_jsonl", str(per_task_jsonl),
            "--available_gpus", gpu_id,
            "--local_log_dir", str(shard_dir / "logs"),
            "--run_id_note", f"shard{shard_idx:02d}",
        ] + extra_flags

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["MUJOCO_GL"] = "egl"

        log_file = log_path.open("a", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        workers.append((shard_idx, gpu_id, proc, log_path, results_json, per_task_jsonl))
        print(f"[launch] shard {shard_idx}/{num_shards} GPU={gpu_id} pid={proc.pid} log={log_path}")

    completed: set[int] = set()
    failed: set[int] = set()

    print(f"\n[info] {num_shards} shards launched. Total tasks: {total_tasks} (~{total_tasks//num_shards}/shard)\n")

    while len(completed) + len(failed) < num_shards:
        total_done = 0
        shard_summaries = []

        for shard_idx, gpu_id, proc, log_path, results_json, per_task_jsonl in workers:
            done_ep = _count_jsonl_lines(per_task_jsonl)
            shard_total = tasks_per_shard[shard_idx]
            total_done += done_ep
            pct = f"{done_ep*100//shard_total}%" if shard_total > 0 else "?"
            shard_summaries.append(f"s{shard_idx}(GPU{gpu_id}):{done_ep}/{shard_total}={pct}")

            rc = proc.poll()
            if rc is not None and shard_idx not in completed and shard_idx not in failed:
                if rc == 0:
                    completed.add(shard_idx)
                    print(f"[done]  shard {shard_idx} finished OK")
                else:
                    failed.add(shard_idx)
                    print(f"[ERROR] shard {shard_idx} exited with code {rc} — see {log_path}")

        elapsed = time.time() - started_at
        eta = None
        if total_done > 0:
            rate = total_done / elapsed
            eta = (total_tasks - total_done) / rate if rate > 0 else None

        print(
            f"\r[progress] {total_done}/{total_tasks} tasks  "
            f"elapsed={_format_eta(elapsed)}  eta={_format_eta(eta)}  |  "
            + "  ".join(shard_summaries),
            end="", flush=True,
        )

        if len(completed) + len(failed) < num_shards:
            time.sleep(15)

    print()  # newline after \r progress

    if failed:
        print(f"\n[ERROR] {len(failed)} shard(s) failed: {sorted(failed)}")
        sys.exit(1)

    # Merge shard results
    total_ep, total_succ = 0, 0
    for shard_idx, gpu_id, proc, log_path, results_json, per_task_jsonl in workers:
        if results_json.exists():
            r = json.loads(results_json.read_text())
            total_ep += r["total_episodes"]
            total_succ += r["total_successes"]
        else:
            print(f"[WARN] shard {shard_idx} results JSON not found: {results_json}")

    sr = total_succ / total_ep if total_ep > 0 else 0.0
    elapsed_total = time.time() - started_at

    merged = {
        "task_suite_name": ns.task_suite_name,
        "total_tasks": total_tasks,
        "num_shards": num_shards,
        "total_episodes": total_ep,
        "total_successes": total_succ,
        "success_rate": sr,
        "elapsed_seconds": elapsed_total,
    }
    merged_path = output_dir / "merged_results.json"
    merged_path.write_text(json.dumps(merged, indent=2))

    print(f"\n{'='*60}")
    print(f"  Suite:    {ns.task_suite_name}")
    print(f"  Tasks:    {total_tasks}  |  Episodes: {total_ep}")
    print(f"  Successes: {total_succ}  |  SR: {sr*100:.1f}%")
    print(f"  Elapsed:  {_format_eta(elapsed_total)}")
    print(f"  Results:  {merged_path}")
    print(f"{'='*60}\n")

    # Generate condition/level breakdown report
    jsonl_files = [str(shard_root / f"shard_{i:02d}" / "per_task.jsonl") for i in range(num_shards)]
    report_dir = output_dir / "report"
    report_script = pathlib.Path(__file__).parent / "generate_camera_report.py"
    report_cmd = [python, str(report_script), "--jsonl_files"] + jsonl_files + ["--output_dir", str(report_dir)]
    print(f"[report] Generating condition/level breakdown → {report_dir}")
    subprocess.run(report_cmd, check=False)  # non-fatal if report fails


if __name__ == "__main__":
    main()
