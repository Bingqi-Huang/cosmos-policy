"""ONE-OFF resume script — Phase 0.B camera baseline eval.

Picks up where the interrupted 4-GPU serial runs left off.
Distributes REMAINING tasks across 18 shards (6 GPU × 3 shards):
  GPU 0: libero_spatial  (3 shards, num_shards=3)
  GPU 1: libero_object   (3 shards, num_shards=3)
  GPU 2: libero_goal     (3 shards, num_shards=3)
  GPU 3-5: libero_10     (9 shards, num_shards=9, 3 per GPU)

After all shards finish: merges old + new JSONL → generates final C1/C2/C3 × level report.

DO NOT USE FOR FUTURE EVALS.  Future path:
  run_libero_camera_parallel.py  (one suite at a time, any GPU count)
"""
from __future__ import annotations
import json, os, pathlib, re, subprocess, sys, time

COSMOS_ROOT = pathlib.Path(__file__).resolve().parents[3]   # cosmos-policy/
EVAL_SCRIPT  = COSMOS_ROOT / "cosmos_policy/experiments/robot/libero/run_libero_eval.py"
REPORT_SCRIPT= COSMOS_ROOT / "cosmos_policy/experiments/robot/libero/generate_camera_report.py"
BASE         = pathlib.Path(__file__).parent
SHARD_ROOT   = BASE / "resume_shards"
PYTHON       = sys.executable

COMMON_FLAGS = [
    "--ckpt_path", "nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
    "--config", "cosmos_predict2_2b_480p_libero__inference_only",
    "--dataset_stats_path", "nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json",
    "--t5_text_embeddings_path", "nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl",
    "--num_trials_per_task", "3",
    "--use_wrist_image", "True", "--use_proprio", "True", "--normalize_proprio", "True",
    "--unnormalize_actions", "True", "--trained_with_image_aug", "True",
    "--chunk_size", "16", "--num_open_loop_steps", "16", "--flip_images", "True",
    "--use_jpeg_compression", "True", "--num_denoising_steps_action", "5",
    "--num_denoising_steps_future_state", "1", "--num_denoising_steps_value", "1",
    "--ar_future_prediction", "False", "--ar_value_prediction", "False",
    "--deterministic", "True", "--randomize_seed", "False", "--seed", "195",
]

# shard plan: list of (suite, num_shards_for_suite, shard_index, gpu_id)
SHARD_PLAN = (
    # spatial on GPU 0
    [("libero_spatial", 3, i, "0") for i in range(3)] +
    # object on GPU 1
    [("libero_object",  3, i, "1") for i in range(3)] +
    # goal on GPU 2
    [("libero_goal",    3, i, "2") for i in range(3)] +
    # libero_10 on GPUs 3,4,5 (3 shards per GPU)
    [("libero_10",      9, i, str(3 + i // 3)) for i in range(9)]
)


def _count_sr_in_log(log_path: pathlib.Path) -> int:
    """Count completed camera tasks (each has exactly one Camera task SR: line pair)."""
    if not log_path.exists():
        return 0
    pattern = re.compile(r"Camera task SR:")
    task_name_re = re.compile(r"Camera task:\s*\S+")
    seen = set()
    current = None
    count = 0
    try:
        for line in log_path.read_text(errors="replace").splitlines():
            m = task_name_re.search(line)
            if m:
                current = line.split("Camera task:")[-1].strip()
            if pattern.search(line) and current and current not in seen:
                seen.add(current)
                count += 1
                current = None
    except OSError:
        pass
    return count


def _format_eta(s):
    if s is None or s <= 0:
        return "?"
    h, r = divmod(int(s), 3600)
    m, sc = divmod(r, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{sc:02d}s" if m else f"{sc}s")


def main():
    SHARD_ROOT.mkdir(parents=True, exist_ok=True)

    # expected tasks per suite shard
    suite_rem = {}
    for suite in ["libero_spatial","libero_object","libero_goal","libero_10"]:
        suite_rem[suite] = json.loads((BASE / f"remaining_tasks_{suite}.json").read_text())

    workers = []   # (suite, num_shards, shard_idx, gpu, proc, log_path, jsonl_path, results_json)
    started_at = time.time()

    for suite, num_shards, shard_idx, gpu in SHARD_PLAN:
        rem_file = BASE / f"remaining_tasks_{suite}.json"
        shard_dir = SHARD_ROOT / f"{suite}__s{num_shards}_{shard_idx:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        log_path    = shard_dir / "worker.log"
        jsonl_path  = shard_dir / "per_task.jsonl"
        results_json= shard_dir / "results.json"

        cmd = [
            PYTHON, str(EVAL_SCRIPT),
            "--task_suite_name", suite,
            "--camera_tasks_file", str(rem_file),
            "--num_shards", str(num_shards),
            "--shard_index", str(shard_idx),
            "--per_task_jsonl", str(jsonl_path),
            "--shard_results_json", str(results_json),
            "--available_gpus", gpu,
            "--local_log_dir", str(shard_dir / "logs"),
            "--run_id_note", f"resume_{suite}_s{num_shards}_{shard_idx:02d}",
        ] + COMMON_FLAGS

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["MUJOCO_GL"] = "egl"

        log_file = log_path.open("a", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=str(COSMOS_ROOT), env=env,
                                stdout=log_file, stderr=subprocess.STDOUT, text=True)
        workers.append((suite, num_shards, shard_idx, gpu, proc, log_path, jsonl_path, results_json))
        print(f"[launch] {suite} shard {shard_idx}/{num_shards} GPU={gpu} pid={proc.pid}")

    total_remaining = sum(len(v) for v in suite_rem.values())
    print(f"\n18 shards launched. Total remaining tasks: {total_remaining}\n")

    completed_set: set[int] = set()
    failed_set: set[int] = set()

    while len(completed_set) + len(failed_set) < len(workers):
        total_done = 0
        summaries = []
        for i, (suite, num_shards, shard_idx, gpu, proc, log_path, jsonl_path, results_json) in enumerate(workers):
            done = _count_sr_in_log(log_path)
            shard_total = len([t for j,t in enumerate(suite_rem[suite]) if j % num_shards == shard_idx])
            total_done += done
            summaries.append(f"{suite[:8]}/s{shard_idx}:{done}/{shard_total}")
            rc = proc.poll()
            if rc is not None and i not in completed_set and i not in failed_set:
                if rc == 0:
                    completed_set.add(i)
                    print(f"\n[done] {suite} shard {shard_idx}")
                else:
                    failed_set.add(i)
                    print(f"\n[FAIL] {suite} shard {shard_idx} rc={rc} → {log_path}")

        elapsed = time.time() - started_at
        eta = (total_remaining - total_done) / (total_done / elapsed) if total_done > 0 else None
        print(f"\r[{_format_eta(elapsed)}] {total_done}/{total_remaining} remaining tasks done  eta={_format_eta(eta)}",
              end="", flush=True)

        if len(completed_set) + len(failed_set) < len(workers):
            time.sleep(20)

    print()
    if failed_set:
        print(f"[ERROR] {len(failed_set)} shard(s) failed: {sorted(failed_set)}")

    # ── Merge all JSONL (old + new) and generate report ──────────────────────
    print("\n[merge] Combining old + new per-task records …")
    all_jsonl = []
    # old records from interrupted serial runs
    for suite in ["libero_spatial","libero_object","libero_goal","libero_10"]:
        old_path = BASE / f"completed_from_old_run_{suite}.jsonl"
        if old_path.exists():
            all_jsonl.append(str(old_path))
    # new shard records
    for _, _, _, _, _, _, jsonl_path, _ in workers:
        if jsonl_path.exists():
            all_jsonl.append(str(jsonl_path))

    report_dir = BASE / "report_final"
    report_dir.mkdir(exist_ok=True)
    result = subprocess.run(
        [PYTHON, str(REPORT_SCRIPT), "--jsonl_files"] + all_jsonl + ["--output_dir", str(report_dir)],
        cwd=str(COSMOS_ROOT), capture_output=False,
    )
    if result.returncode == 0:
        print(f"\n[report] Final report → {report_dir}/summary.md")
    else:
        print("[warn] Report generation failed; JSONL files preserved for manual re-run.")


if __name__ == "__main__":
    main()
