#!/usr/bin/env bash
set -euo pipefail

# Full evaluation launcher for the stable scene-only baseline.
# Start it only after the training job has been stopped.

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

bool_arg() {
  case "${1,,}" in
    1|true|yes|y|on) echo "True" ;;
    *) echo "False" ;;
  esac
}

CKPT_ITER="${CKPT_ITER:-10000}"
ITER_NAME="$(printf "iter_%09d" "${CKPT_ITER}")"
JOB_NAME="${JOB_NAME:-phase1_scene_only_stable_original_ema_b24_gacc5_lr5e5_seed42_15k}"
CKPT_ROOT="${CKPT_ROOT:-outputs/phase1/cosmos_policy/cosmos_v2_finetune/${JOB_NAME}/checkpoints}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5}"
CKPT_PATH="${CKPT_PATH:-${CKPT_ROOT}/${ITER_NAME}}"
OUT_ROOT="${OUT_ROOT:-outputs/phase1/scene_only_stable_eval/${ITER_NAME}}"
ID_NUM_TRIALS="${ID_NUM_TRIALS:-50}"
CAMERA_NUM_TRIALS="${CAMERA_NUM_TRIALS:-3}"
SEED="${SEED:-42}"
RUN_ID_NOTE="${RUN_ID_NOTE:-scene_only_stable_${ITER_NAME}}"
RUN_ID="${RUN_ID:-all}"
LOAD_EMA_TO_REG="${LOAD_EMA_TO_REG:-1}"

RUN_ID_EVAL="${RUN_ID_EVAL:-1}"
RUN_CAMERA_EVAL="${RUN_CAMERA_EVAL:-1}"

if [[ ! -d "${CKPT_PATH}" ]]; then
  echo "[ERROR] CKPT_PATH does not exist: ${CKPT_PATH}" >&2
  exit 2
fi

COMMON_FLAGS=(
  --ckpt_path "${CKPT_PATH}"
  --config cosmos_predict2_2b_480p_libero_scene_only__inference_only
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
  --load_ema_to_reg "$(bool_arg "${LOAD_EMA_TO_REG}")"
)

mkdir -p "${OUT_ROOT}"
echo "[preflight] checkpoint: ${CKPT_PATH}"
echo "[preflight] output:     ${OUT_ROOT}"
echo "[preflight] GPUs:       ${GPU_IDS}"
echo "[preflight] load_ema_to_reg: $(bool_arg "${LOAD_EMA_TO_REG}")"

if [[ "${RUN_ID_EVAL}" == "1" ]]; then
  for SUITE in libero_spatial libero_object libero_goal libero_10; do
    uv run --extra cu128 --group libero --python 3.10 \
      python cosmos_policy/experiments/robot/libero/run_libero_standard_parallel.py \
        --task_suite_name "${SUITE}" \
        --gpu_ids "${GPU_IDS}" \
        --num_trials_per_task "${ID_NUM_TRIALS}" \
        --output_dir "${OUT_ROOT}/id_full/${SUITE}" \
        "${COMMON_FLAGS[@]}" \
        --run_id_note "${RUN_ID_NOTE}_id_${SUITE}_${RUN_ID}"
  done

  mapfile -t ID_JSONL < <(find "${OUT_ROOT}/id_full" -path "*/shards/shard_*/per_task.jsonl" | sort)
  uv run --extra cu128 --group libero --python 3.10 \
    python cosmos_policy/experiments/robot/libero/generate_standard_report.py \
      --jsonl_files "${ID_JSONL[@]}" \
      --output_dir "${OUT_ROOT}/id_full/report_final"
fi

if [[ "${RUN_CAMERA_EVAL}" == "1" ]]; then
  for SUITE in libero_spatial libero_object libero_goal libero_10; do
    TASK_FILE="outputs/phase0/libero_plus_camera_eval/camera_task_names_${SUITE}.json"
    uv run --extra cu128 --group libero --python 3.10 \
      python cosmos_policy/experiments/robot/libero/run_libero_camera_parallel.py \
        --task_suite_name "${SUITE}" \
        --camera_tasks_file "${TASK_FILE}" \
        --gpu_ids "${GPU_IDS}" \
        --num_trials_per_task "${CAMERA_NUM_TRIALS}" \
        --output_dir "${OUT_ROOT}/camera_full/${SUITE}" \
        "${COMMON_FLAGS[@]}" \
        --run_id_note "${RUN_ID_NOTE}_camera_${SUITE}_${RUN_ID}"
  done

  mapfile -t CAMERA_JSONL < <(find "${OUT_ROOT}/camera_full" -path "*/shards/shard_*/per_task.jsonl" | sort)
  uv run --extra cu128 --group libero --python 3.10 \
    python cosmos_policy/experiments/robot/libero/generate_camera_report.py \
      --jsonl_files "${CAMERA_JSONL[@]}" \
      --output_dir "${OUT_ROOT}/camera_full/report_final"
fi

echo "[done] Row-1 eval launcher finished. Reports under ${OUT_ROOT}"
