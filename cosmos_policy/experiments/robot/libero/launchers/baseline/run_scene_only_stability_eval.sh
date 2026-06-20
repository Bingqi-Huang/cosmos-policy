#!/usr/bin/env bash
set -euo pipefail

# Validation-only evaluator for selecting one scene-only baseline checkpoint.
# It runs a fixed 600-rollout LIBERO subset by default and writes one selected
# checkpoint. It does not average policy checkpoints.

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

bool_arg() {
  case "${1,,}" in
    1|true|yes|y|on) echo "True" ;;
    *) echo "False" ;;
  esac
}

SCRIPT_DIR="cosmos_policy/experiments/robot/libero"
SUBSET_SCRIPT="${SCRIPT_DIR}/prepare_libero_stability_subset.py"
SELECT_SCRIPT="${SCRIPT_DIR}/select_stable_checkpoint.py"

VALIDATION_ROOT="${VALIDATION_ROOT:-outputs/phase1/libero_validation_subsets/stability_600_v1}"
TRIALS_PER_TASK="${TRIALS_PER_TASK:-15}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5}"
OUT_ROOT="${OUT_ROOT:-outputs/phase1/scene_only_stability_eval}"
SEED="${SEED:-42}"
LOAD_EMA_TO_REG="${LOAD_EMA_TO_REG:-1}"
SAVE_ROLLOUT_VIDEOS="${SAVE_ROLLOUT_VIDEOS:-0}"
RUN_SELECTION="${RUN_SELECTION:-1}"
MIN_LIBERO10="${MIN_LIBERO10:-0.10}"
MAX_NEIGHBOR_DELTA_PP="${MAX_NEIGHBOR_DELTA_PP:-25}"
CONFIG="${CONFIG:-cosmos_predict2_2b_480p_libero_scene_only__inference_only}"
SUITES=(${SUITES:-libero_spatial libero_object libero_goal libero_10})

if [[ ! -f "${VALIDATION_ROOT}/manifest.json" ]]; then
  python3 "${SUBSET_SCRIPT}" \
    --output-root "${VALIDATION_ROOT}" \
    --trials-per-task "${TRIALS_PER_TASK}" \
    --seed "${SEED}"
fi

LOAD_EMA_BOOL="$(bool_arg "${LOAD_EMA_TO_REG}")"
SAVE_VIDEO_BOOL="$(bool_arg "${SAVE_ROLLOUT_VIDEOS}")"

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
  --load_ema_to_reg "${LOAD_EMA_BOOL}"
  --save_rollout_videos "${SAVE_VIDEO_BOOL}"
)

mkdir -p "${OUT_ROOT}"
echo "[stability-eval] validation_root=${VALIDATION_ROOT}"
echo "[stability-eval] suites=${SUITES[*]} trials_per_task=${TRIALS_PER_TASK} gpus=${GPU_IDS}"
echo "[stability-eval] load_ema_to_reg=${LOAD_EMA_BOOL} save_rollout_videos=${SAVE_VIDEO_BOOL}"

ITER_ARGS=()
if [[ -n "${CKPT_PATH:-}" ]]; then
  if [[ ! -d "${CKPT_PATH}" ]]; then
    echo "[ERROR] CKPT_PATH does not exist: ${CKPT_PATH}" >&2
    exit 2
  fi
  ITER_ITEMS=("direct:${CKPT_PATH}")
elif [[ -n "${CKPT_ROOT:-}" && -n "${ITERS:-}" ]]; then
  read -r -a RAW_ITERS <<< "${ITERS}"
  ITER_ITEMS=()
  for ITER in "${RAW_ITERS[@]}"; do
    ITER_DIR="$(printf "iter_%09d" "${ITER}")"
    ITER_ITEMS+=("${ITER_DIR}:${CKPT_ROOT}/${ITER_DIR}")
    ITER_ARGS+=("${ITER}")
  done
else
  echo "[ERROR] set either CKPT_PATH, or both CKPT_ROOT and ITERS" >&2
  exit 2
fi

for ITEM in "${ITER_ITEMS[@]}"; do
  ITER_NAME="${ITEM%%:*}"
  CHECKPOINT_PATH="${ITEM#*:}"
  if [[ ! -d "${CHECKPOINT_PATH}" ]]; then
    echo "[stability-eval][skip] missing checkpoint ${CHECKPOINT_PATH}" >&2
    continue
  fi

  ITER_OUT="${OUT_ROOT}/${ITER_NAME}"
  mkdir -p "${ITER_OUT}"
  printf '%s\n' "${CHECKPOINT_PATH}" > "${ITER_OUT}/checkpoint_path.txt"
  echo "[stability-eval][run] ${ITER_NAME} checkpoint=${CHECKPOINT_PATH}"

  for SUITE in "${SUITES[@]}"; do
    TASK_FILE="${VALIDATION_ROOT}/${SUITE}_task_indices.json"
    if [[ ! -f "${TASK_FILE}" ]]; then
      echo "[ERROR] missing task file: ${TASK_FILE}" >&2
      exit 2
    fi
    OUT_DIR="${ITER_OUT}/${SUITE}"
    if [[ -f "${OUT_DIR}/.done" ]]; then
      echo "[stability-eval][skip-done] ${ITER_NAME}/${SUITE}"
      continue
    fi
    mkdir -p "${OUT_DIR}"
    uv run --extra cu128 --group libero --python 3.10 \
      python "${SCRIPT_DIR}/run_libero_standard_parallel.py" \
        --task_suite_name "${SUITE}" \
        --task_indices_file "${TASK_FILE}" \
        --gpu_ids "${GPU_IDS}" \
        --num_trials_per_task "${TRIALS_PER_TASK}" \
        --output_dir "${OUT_DIR}" \
        "${COMMON_FLAGS[@]}" \
        --ckpt_path "${CHECKPOINT_PATH}" \
        --run_id_note "stability_${ITER_NAME}_${SUITE}"
    touch "${OUT_DIR}/.done"
  done

  mapfile -t ITER_JSONL < <(find "${ITER_OUT}" -path "*/shards/shard_*/per_task.jsonl" | sort)
  if [[ ${#ITER_JSONL[@]} -eq 0 ]]; then
    echo "[ERROR] no per-task JSONL files found under ${ITER_OUT}" >&2
    exit 3
  fi
  uv run --extra cu128 --group libero --python 3.10 \
    python "${SCRIPT_DIR}/generate_standard_report.py" \
      --jsonl_files "${ITER_JSONL[@]}" \
      --output_dir "${ITER_OUT}/report_final"
done

if [[ "${RUN_SELECTION}" == "1" && ${#ITER_ARGS[@]} -gt 1 ]]; then
  python3 "${SELECT_SCRIPT}" \
    --eval-root "${OUT_ROOT}" \
    --ckpt-root "${CKPT_ROOT}" \
    --iters "${ITER_ARGS[@]}" \
    --min-libero10 "${MIN_LIBERO10}" \
    --max-neighbor-delta-pp "${MAX_NEIGHBOR_DELTA_PP}"
fi

echo "[stability-eval][done] outputs under ${OUT_ROOT}"
