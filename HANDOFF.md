# SCVC-for-WAMs Handoff Document

## Project goal

Train a view-invariant Cosmos-Policy-2B (Selective Cross-View Consistency, SCVC) on LIBERO without ever feeding camera labels to the model. Baseline: released `nvidia/Cosmos-Policy-LIBERO-Predict2-2B`. Target: match or exceed standard SR while closing the gap on perturbed-camera SR.

---

## How to run LIBERO-Plus camera evaluation

### Standard path (use this for all future evals)

**Step 1 — Generate camera task name lists** (already done; skip if files exist)

```bash
cd cosmos-policy
python3 - <<'EOF'
import json, pathlib, re, sys
sys.path.insert(0, '/nvme02/bingqi/Project/view-invariant_vla/code/LIBERO-plus')
from libero.libero.benchmark.task_classification import ...
# See outputs/phase0/libero_plus_camera_eval/camera_task_names_*.json for reference
EOF
```

The task name JSON files already exist at:
- `outputs/phase0/libero_plus_camera_eval/camera_task_names_libero_spatial.json` (376 tasks)
- `outputs/phase0/libero_plus_camera_eval/camera_task_names_libero_object.json` (396 tasks)
- `outputs/phase0/libero_plus_camera_eval/camera_task_names_libero_goal.json` (408 tasks)
- `outputs/phase0/libero_plus_camera_eval/camera_task_names_libero_10.json` (419 tasks)

**Step 2 — Launch parallel eval (one suite at a time)**

```bash
cd cosmos-policy

uv run --extra cu128 --group libero --python 3.10 \
python cosmos_policy/experiments/robot/libero/run_libero_camera_parallel.py \
  --gpu_ids 0,1,2,3,4,5 \
  --task_suite_name libero_spatial \
  --camera_tasks_file outputs/phase0/libero_plus_camera_eval/camera_task_names_libero_spatial.json \
  --output_dir outputs/<run_name>/parallel_spatial \
  --num_trials_per_task 3 \
  --ckpt_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
  --config cosmos_predict2_2b_480p_libero__inference_only \
  --dataset_stats_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json \
  --t5_text_embeddings_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl \
  --use_wrist_image True --use_proprio True --normalize_proprio True \
  --unnormalize_actions True --trained_with_image_aug True \
  --chunk_size 16 --num_open_loop_steps 16 --flip_images True \
  --use_jpeg_compression True --num_denoising_steps_action 5 \
  --deterministic True --randomize_seed False --seed 195
```

Repeat for each of the 4 suites (change `--task_suite_name` and `--output_dir`).

The launcher:
- Spawns `N_gpus` worker processes (one per GPU listed in `--gpu_ids`)
- Each worker handles `1/N_gpus` of the tasks via `task_index % N_gpus == shard_index`
- Writes `per_task.jsonl` per shard (one JSON line per completed task)
- Auto-generates a condition/level report at `<output_dir>/report/` when all shards finish

**Step 3 — Generate / refresh the report manually**

```bash
python cosmos_policy/experiments/robot/libero/generate_camera_report.py \
  --jsonl_files outputs/<run_name>/parallel_spatial/shards/shard_*/per_task.jsonl \
              outputs/<run_name>/parallel_object/shards/shard_*/per_task.jsonl \
              outputs/<run_name>/parallel_goal/shards/shard_*/per_task.jsonl \
              outputs/<run_name>/parallel_libero10/shards/shard_*/per_task.jsonl \
  --output_dir outputs/<run_name>/report_merged
```

Or from log files (for already-completed runs without JSONL):

```bash
python cosmos_policy/experiments/robot/libero/generate_camera_report.py \
  --log_files outputs/<run_name>/stdout/*.log \
  --output_dir outputs/<run_name>/report_from_logs
```

### Report format

`summary.md` contains:
- **Condition × Level table**: C1 (scale), C2 (pan/tilt), C3 (rotation) × difficulty 1–5, with 95% Wilson CI
- **Suite breakdown table**: per-suite SR

**Condition definitions:**
| Cond | Perturbation | Parameter |
|---|---|---|
| C1 | Zoom (scale ≠ 100%) | `scale_factor_percent` |
| C2 | Pan / tilt | `horizon_view` or `vertical_view` ≠ 0 |
| C3 | Endpoint rotation | `end_point_rot` or `end_point_vertical` ≠ 0 |

---

## Key infrastructure files

| File | Purpose |
|---|---|
| `cosmos_policy/experiments/robot/libero/run_libero_eval.py` | Core eval script; `--camera_tasks_file` enables LIBERO-Plus camera mode; `--num_shards`/`--shard_index` enable sharding |
| `cosmos_policy/experiments/robot/libero/run_libero_camera_parallel.py` | **Parallel launcher** — use this for all future evals |
| `cosmos_policy/experiments/robot/libero/generate_camera_report.py` | Post-processing: JSONL/log → CSV + Markdown report |
| `outputs/phase0/libero_plus_camera_eval/camera_task_names_*.json` | Pre-built camera task name lists (do not regenerate) |

## LIBERO-Plus env dependency

Camera task eval requires the **git HEAD** version of LIBERO-Plus envs:
```
/nvme02/bingqi/Project/view-invariant_vla/code/LIBERO-plus/libero/libero/envs/
```
If any eval crashes with `TypeError: ManipulationEnv got unexpected kwarg horizon_view`, restore:
```bash
cd /nvme02/bingqi/Project/view-invariant_vla/code/LIBERO-plus/libero
git checkout libero/libero/envs/
```

---

## Phase 0.B baseline results — G0 camera baseline (P1 released ckpt)

Model: `nvidia/Cosmos-Policy-LIBERO-Predict2-2B` (released, no finetuning)
Seed: 195, deterministic=True, 3 trials/task, 1599 tasks × 3 trials = 4797 episodes

Final report: `outputs/phase0/libero_plus_camera_eval/report_final/summary.md`

| | C1 (scale) | C2 (pan/tilt) | C3 (rotation) | Grand Total |
|---|---:|---:|---:|---:|
| SR | 59.0% | 83.3% | 88.5% | **79.5%** |
| 95% CI | [55.8, 62.1] | [81.9, 84.6] | [86.3, 90.5] | [78.3, 80.6] |
| n_success/n_trials | 554/939 | 2479/2976 | 781/882 | 3814/4797 |
| n_tasks | 313 | 992 | 294 | 1599 |

Suite breakdown:

| Suite | SR |
|---|---:|
| libero_spatial | 84.1% |
| libero_object | 87.4% |
| libero_goal | 83.1% |
| libero_10 | 64.4% |

Eval cost: ~1h16m resume (18 shards, 6 GPUs) for 1083 remaining tasks.
Full from-scratch estimate on 6 GPUs × 3 shards: ~1.5–2h.

---

## Phase roadmap

| Phase | Gate | Status |
|---|---|---|
| 0.B | G0: camera baseline SR established | **running** |
| E1.0 | Dominance audit: mask scene vs wrist | pending G0 |
| E1 | SCVC training + camera eval | pending E1.0 |
| E2 | Ablations (A1/A2/A5) | pending E1 |
