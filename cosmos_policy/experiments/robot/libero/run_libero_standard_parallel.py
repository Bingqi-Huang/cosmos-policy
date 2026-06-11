"""Parallel launcher for standard LIBERO suite evaluation.

This is the standard-task counterpart of ``run_libero_camera_parallel.py``.
It shards task IDs across GPUs, launches ``run_libero_eval.py`` workers, and
aggregates their per-task JSONL output into the house report schema.
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
REPORT_SCRIPT = pathlib.Path(__file__).resolve().parent / "generate_standard_report.py"
TASK_COUNTS = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}


def _format_eta(seconds: float | None) -> str:
    if seconds is None or seconds <= 0 or not seconds < float("inf"):
        return "unknown"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
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


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu_ids", default="0,1,2,3,4,5")
    parser.add_argument("--output_dir", default="outputs/standard_parallel")
    parser.add_argument("--task_suite_name", required=True)
    parser.add_argument("--task_indices_file", default="")
    parser.add_argument("--num_trials_per_task", type=int, default=50)
    ns, remaining = parser.parse_known_args(argv)
    return ns, remaining


def _task_ids_for_suite(task_suite_name: str, task_indices_file: str) -> list[int]:
    if task_indices_file:
        return [int(x) for x in json.loads(pathlib.Path(task_indices_file).read_text(encoding="utf-8"))]
    if task_suite_name not in TASK_COUNTS:
        known = ", ".join(sorted(TASK_COUNTS))
        raise ValueError(f"Unknown task suite {task_suite_name!r}. Known: {known}")
    return list(range(TASK_COUNTS[task_suite_name]))


def main() -> None:
    ns, extra_flags = _parse_args(sys.argv[1:])
    gpu_ids = [g.strip() for g in ns.gpu_ids.split(",") if g.strip()]
    if not gpu_ids:
        raise ValueError("--gpu_ids must contain at least one GPU id")

    task_ids = _task_ids_for_suite(ns.task_suite_name, ns.task_indices_file)
    num_shards = len(gpu_ids)
    output_dir = pathlib.Path(ns.output_dir)
    shard_root = output_dir / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)

    workers: list[tuple[int, str, subprocess.Popen, pathlib.Path, pathlib.Path, int]] = []
    started_at = time.time()
    python = sys.executable

    for shard_idx, gpu_id in enumerate(gpu_ids):
        shard_task_ids = [task_id for pos, task_id in enumerate(task_ids) if pos % num_shards == shard_idx]
        shard_dir = shard_root / f"shard_{shard_idx:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_tasks_file = shard_dir / "task_indices.json"
        shard_tasks_file.write_text(json.dumps(shard_task_ids) + "\n", encoding="utf-8")

        log_path = shard_dir / "worker.log"
        results_json = shard_dir / "results.json"
        per_task_jsonl = shard_dir / "per_task.jsonl"
        cmd = [
            python,
            str(EVAL_SCRIPT),
            "--task_suite_name",
            ns.task_suite_name,
            "--task_indices_file",
            str(shard_tasks_file),
            "--num_trials_per_task",
            str(ns.num_trials_per_task),
            "--shard_results_json",
            str(results_json),
            "--per_task_jsonl",
            str(per_task_jsonl),
            "--available_gpus",
            gpu_id,
            "--local_log_dir",
            str(shard_dir / "logs"),
            "--run_id_note",
            f"standard_{ns.task_suite_name}_shard{shard_idx:02d}",
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
        workers.append((shard_idx, gpu_id, proc, log_path, results_json, len(shard_task_ids)))
        print(
            f"[launch] suite={ns.task_suite_name} shard={shard_idx}/{num_shards} "
            f"GPU={gpu_id} tasks={len(shard_task_ids)} pid={proc.pid} log={log_path}"
        )

    completed: set[int] = set()
    failed: set[int] = set()
    total_tasks = len(task_ids)
    while len(completed) + len(failed) < num_shards:
        total_done = 0
        parts = []
        for shard_idx, gpu_id, proc, log_path, _results_json, shard_total in workers:
            done_tasks = _count_in_log(log_path, "Current task success rate:")
            total_done += done_tasks
            pct = f"{done_tasks * 100 // shard_total}%" if shard_total else "100%"
            parts.append(f"s{shard_idx}(GPU{gpu_id}):{done_tasks}/{shard_total}={pct}")
            rc = proc.poll()
            if rc is not None and shard_idx not in completed and shard_idx not in failed:
                if rc == 0:
                    completed.add(shard_idx)
                    print(f"\n[done] shard {shard_idx} finished OK")
                else:
                    failed.add(shard_idx)
                    print(f"\n[ERROR] shard {shard_idx} exited with code {rc} — see {log_path}")

        elapsed = time.time() - started_at
        eta = None
        if total_done:
            rate = total_done / elapsed
            eta = (total_tasks - total_done) / rate if rate > 0 else None
        print(
            f"\r[progress] {total_done}/{total_tasks} tasks elapsed={_format_eta(elapsed)} "
            f"eta={_format_eta(eta)} | " + "  ".join(parts),
            end="",
            flush=True,
        )
        if len(completed) + len(failed) < num_shards:
            time.sleep(15)
    print()

    if failed:
        print(f"[ERROR] {len(failed)} shard(s) failed: {sorted(failed)}")
        sys.exit(1)

    total_ep = 0
    total_succ = 0
    for shard_idx, _gpu_id, _proc, _log_path, results_json, _shard_total in workers:
        if not results_json.exists():
            print(f"[WARN] shard {shard_idx} results JSON not found: {results_json}")
            continue
        result = json.loads(results_json.read_text(encoding="utf-8"))
        total_ep += int(result["total_episodes"])
        total_succ += int(result["total_successes"])

    merged = {
        "task_suite_name": ns.task_suite_name,
        "total_tasks": total_tasks,
        "num_shards": num_shards,
        "total_episodes": total_ep,
        "total_successes": total_succ,
        "success_rate": total_succ / total_ep if total_ep else 0.0,
        "elapsed_seconds": time.time() - started_at,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "merged_results.json").write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")

    jsonl_files = [str(shard_root / f"shard_{i:02d}" / "per_task.jsonl") for i in range(num_shards)]
    report_dir = output_dir / "report"
    report_cmd = [python, str(REPORT_SCRIPT), "--jsonl_files", *jsonl_files, "--output_dir", str(report_dir)]
    print(f"[report] Generating standard report -> {report_dir}")
    subprocess.run(report_cmd, check=False)
    print(
        f"[summary] suite={ns.task_suite_name} episodes={total_ep} "
        f"successes={total_succ} sr={100 * merged['success_rate']:.1f}%"
    )


if __name__ == "__main__":
    main()
