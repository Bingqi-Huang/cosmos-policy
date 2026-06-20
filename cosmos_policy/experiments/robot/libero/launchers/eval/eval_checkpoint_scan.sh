#!/usr/bin/env bash
set -euo pipefail
# -----------------------------------------------------------------------------
# Reusable standard-suite checkpoint-scan evaluator (archived 2026-06-12 from
# the temporary tmp_scan_libero10_checkpoints.sh).  Evaluates a list of
# checkpoint iterations of a run over one or more standard LIBERO suites with
# the P2 scene-only inference protocol, writes the per-iter standard report
# artifacts (aggregate/per_task CSV + summary.md) and a cross-iter diagnostic
# report.
#
# Current rule: final policy numbers must come from one frozen checkpoint.
# Cross-checkpoint summaries from this script are diagnostic only.
#
# Usage:
#   CKPT_ROOT=.../checkpoints ITERS="6000 9000 12000 15000" SUITES="libero_10" \
#   NUM_TRIALS=10 OUT_ROOT=.../scan LABEL="ArmA demo-only" \
#   bash eval_checkpoint_scan.sh
# -----------------------------------------------------------------------------
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)" 2>/dev/null || true

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

CKPT_ROOT="${CKPT_ROOT:?set CKPT_ROOT to the checkpoints dir}"
ITERS=(${ITERS:?set ITERS like 6000 9000 12000 15000})
SUITES=(${SUITES:-libero_10})
NUM_TRIALS="${NUM_TRIALS:-10}"
# All 6 GPUs, 3 shards each, for throughput on a single checkpoint at a time.
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5}"
OUT_ROOT="${OUT_ROOT:?set OUT_ROOT}"
SEED="${SEED:-42}"
FINAL_WINDOW="${FINAL_WINDOW:-3}"
LABEL="${LABEL:-}"
CONFIG="${CONFIG:-cosmos_predict2_2b_480p_libero_scene_only__inference_only}"

COMMON_FLAGS=(
  --config "${CONFIG}"
  --dataset_stats_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json
  --t5_text_embeddings_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl
  --use_wrist_image False
  --use_third_person_image True
  --use_proprio True
  --normalize_proprio True
  --unnormalize_actions True
  --trained_with_image_aug True
  --chunk_size 16
  --num_open_loop_steps 16
  --flip_images True
  --use_jpeg_compression True
  --num_denoising_steps_action 5
  --num_denoising_steps_future_state 1
  --num_denoising_steps_value 1
  --ar_future_prediction False
  --ar_value_prediction False
  --deterministic True
  --randomize_seed False
  --seed "${SEED}"
)

mkdir -p "${OUT_ROOT}"
echo "[scan] ckpt_root=${CKPT_ROOT}"
echo "[scan] iters=${ITERS[*]} suites=${SUITES[*]} trials=${NUM_TRIALS} out=${OUT_ROOT}"

for ITER in "${ITERS[@]}"; do
  ITER_DIR=$(printf "iter_%09d" "${ITER}")
  CKPT_PATH="${CKPT_ROOT}/${ITER_DIR}"
  if [[ ! -d "${CKPT_PATH}" ]]; then
    echo "[scan][skip] missing checkpoint ${CKPT_PATH}" >&2
    continue
  fi
  for SUITE in "${SUITES[@]}"; do
    OUT_DIR="${OUT_ROOT}/${ITER_DIR}/${SUITE}"
    if [[ -f "${OUT_DIR}/.done" ]]; then
      echo "[scan][skip-done] ${ITER_DIR}/${SUITE}"
      continue
    fi
    mkdir -p "${OUT_DIR}"
    echo "[scan][run] ${ITER_DIR} ${SUITE} trials=${NUM_TRIALS}"
    uv run --extra cu128 --group libero --python 3.10 \
      python cosmos_policy/experiments/robot/libero/run_libero_standard_parallel.py \
        --task_suite_name "${SUITE}" \
        --gpu_ids "${GPU_IDS}" \
        --num_trials_per_task "${NUM_TRIALS}" \
        --output_dir "${OUT_DIR}" \
        "${COMMON_FLAGS[@]}" \
        --ckpt_path "${CKPT_PATH}" \
        --run_id_note "scan_${ITER_DIR}_${SUITE}"
    touch "${OUT_DIR}/.done"
  done
  # Per-iter standard report (artifact schema) across the suites evaluated.
  mapfile -t ITER_JSONL < <(find "${OUT_ROOT}/${ITER_DIR}" -path "*/shards/shard_*/per_task.jsonl" | sort)
  if [[ ${#ITER_JSONL[@]} -gt 0 ]]; then
    uv run --extra cu128 --group libero --python 3.10 \
      python cosmos_policy/experiments/robot/libero/generate_standard_report.py \
        --jsonl_files "${ITER_JSONL[@]}" \
        --output_dir "${OUT_ROOT}/${ITER_DIR}/report_final" || true
  fi
done

# Cross-iter diagnostic report (pure stdlib -> system python3 to avoid uv env re-resolution).
python3 cosmos_policy/experiments/robot/libero/aggregate_checkpoint_scan.py \
    --scan_root "${OUT_ROOT}" \
    --iters "${ITERS[@]}" \
    --suites "${SUITES[@]}" \
    --final_window "${FINAL_WINDOW}" \
    --label "${LABEL}"

echo "[scan][done] report: ${OUT_ROOT}/checkpoint_scan_report.md"
