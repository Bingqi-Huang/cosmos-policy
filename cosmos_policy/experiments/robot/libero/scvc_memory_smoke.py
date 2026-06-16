"""Two-branch SCVC memory smoke: sweep per-branch batch size, measure peak GPU + host RAM.

SCVC residency is ~2*K_s*bs sample-forwards (both branches x all K_s noise draws stay
live until the single backward), so peak GPU memory scales as ~4*bs at K_s=2 -- NOT 2*bs.
This sweeps bs and reports the largest that fits under a GPU ceiling, with a host-RAM
fuse that kills the run group if available RAM drops too low (6 ranks x 8 workers can
balloon host memory).

For each bs it launches a real ~4-step SCVC training run (grad_accum=1, since grad_accum
does not change peak memory -- micro-batches are sequential; periodic save disabled), polls
nvidia-smi + /proc/meminfo, records peak per-GPU NVML and min available RAM, and detects
CUDA OOM. Pick the largest bs with peak <= the chosen ceiling, then set the real-run
grad_accum = 720 / (6*bs) to preserve the 7.2M sample-presentation budget.

Usage:
  .venv/bin/python cosmos_policy/experiments/robot/libero/scvc_memory_smoke.py \
      --batch-sizes 6 8 10 --max-iter 4 --ram-floor-gb 30 \
      --output-json outputs/phase2/scvc_mem_smoke/report.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import signal
import subprocess
import time

REPO = pathlib.Path(__file__).resolve().parents[4]
LAUNCHER = "cosmos_policy/experiments/robot/libero/launchers/scvc/run_scene_only_scvc_train_formal_6gpu.sh"


def gpu_used_mib() -> list[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        )
        return [int(x) for x in out.split()]
    except Exception:
        return []


def mem_available_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    return float("inf")


def run_one(bs: int, args) -> dict:
    job = f"scvc_mem_smoke_bs{bs}"
    log_path = REPO / "outputs" / "phase2" / "scvc_mem_smoke" / f"{job}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "PAIR_BATCH_SIZE": str(bs),
        "GRAD_ACCUM_ITER": "1",          # peak is grad_accum-independent; 1 is fastest
        "MAX_ITER": str(args.max_iter),
        "SAVE_ITER": "99999999",          # never save a checkpoint during the smoke
        "CV_NUM_SAMPLES": str(args.cv_num_samples),
        "JOB_NAME": job,                  # run/output name only; CLI job.name= override below
        "WANDB_MODE": "disabled",
        "PAIR_MANIFEST_PATH": args.manifest,
    })
    # source the local runtime env (BASE_DATASETS_DIR / HF settings), then run the launcher.
    cmd = f"source bin/libero_local_env.sh && bash {shlex.quote(LAUNCHER)}"
    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        ["bash", "-lc", cmd], cwd=str(REPO), env=env,
        stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True,
    )
    peak_gpu = 0
    min_ram = float("inf")
    ram_abort = False
    t0 = time.time()
    while proc.poll() is None:
        used = gpu_used_mib()
        if used:
            peak_gpu = max(peak_gpu, max(used))
        avail = mem_available_gb()
        min_ram = min(min_ram, avail)
        if avail < args.ram_floor_gb:
            ram_abort = True
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            break
        time.sleep(args.poll_sec)
    rc = proc.wait()
    log_f.close()
    elapsed = time.time() - t0
    log_text = log_path.read_text(errors="replace")
    oom = ("out of memory" in log_text.lower()) or ("CUDA out of memory" in log_text)
    # crude steady-step throughput if logged
    fit = (rc == 0) and not oom and not ram_abort
    return {
        "bs": bs,
        "cv_num_samples": args.cv_num_samples,
        "grad_accum_for_720": (720 // (6 * bs)) if (6 * bs) and 720 % (6 * bs) == 0 else None,
        "peak_gpu_mib": peak_gpu,
        "peak_gpu_gb": round(peak_gpu / 1024, 1),
        "min_ram_available_gb": round(min_ram, 1) if min_ram != float("inf") else None,
        "exit_code": rc,
        "cuda_oom": oom,
        "ram_abort": ram_abort,
        "fit": fit,
        "elapsed_sec": round(elapsed, 1),
        "log": str(log_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[6, 8, 10])
    ap.add_argument("--cv-num-samples", type=int, default=2)
    ap.add_argument("--max-iter", type=int, default=4)
    ap.add_argument("--ram-floor-gb", type=float, default=30.0)
    ap.add_argument("--gpu-ceiling-gb", type=float, default=92.0)
    ap.add_argument("--poll-sec", type=float, default=0.25)
    ap.add_argument("--manifest", default="outputs/phase2/pair_future_frames/libero_pair_future_manifest_train.jsonl")
    ap.add_argument("--output-json", default="outputs/phase2/scvc_mem_smoke/report.json")
    args = ap.parse_args()

    rows = []
    for bs in sorted(args.batch_sizes):  # ascending: safest first
        print(f"\n===== memory smoke bs={bs}, K_s={args.cv_num_samples} (grad_accum=1) =====", flush=True)
        r = run_one(bs, args)
        print(json.dumps(r, indent=2), flush=True)
        rows.append(r)
        if r["ram_abort"]:
            print("[STOP] host RAM fuse tripped; aborting the sweep.", flush=True)
            break
        if r["cuda_oom"]:
            print(f"[note] bs={bs} hit CUDA OOM; larger bs will too — stopping sweep.", flush=True)
            break

    fits = [r for r in rows if r["fit"] and r["peak_gpu_gb"] <= args.gpu_ceiling_gb]
    best = max(fits, key=lambda r: r["bs"]) if fits else None
    report = {
        "ceiling_gb": args.gpu_ceiling_gb,
        "ram_floor_gb": args.ram_floor_gb,
        "rows": rows,
        "recommended_bs": best["bs"] if best else None,
        "recommended_grad_accum": best["grad_accum_for_720"] if best else None,
        "note": (
            "Largest bs whose peak GPU <= ceiling with K_s fixed. Real runs use "
            "grad_accum = 720/(6*bs) to preserve the 7.2M sample-presentation budget."
        ),
    }
    out = pathlib.Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\n===== SUMMARY =====")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
