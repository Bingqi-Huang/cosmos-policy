#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

CKPT_PATH="${CKPT_PATH:-outputs/phase1/cosmos_policy/cosmos_v2_finetune/phase1_scene_only_train_smoke_1gpu_b1_gacc8/checkpoints/iter_000000002}"
OUT_DIR="${OUT_DIR:-outputs/phase1/scene_only_eval_smoke}"
TASK_INDICES_FILE="${TASK_INDICES_FILE:-outputs/phase1/scene_only_eval_smoke/task_indices.json}"
NUM_TRIALS="${NUM_TRIALS:-1}"

mkdir -p "$(dirname "${TASK_INDICES_FILE}")" "${OUT_DIR}"
if [[ ! -f "${TASK_INDICES_FILE}" ]]; then
  printf '[0]\n' > "${TASK_INDICES_FILE}"
fi

uv run --extra cu128 --group libero --python 3.10 \
  python cosmos_policy/experiments/robot/libero/run_libero_eval.py \
  --task_suite_name libero_spatial \
  --task_indices_file "${TASK_INDICES_FILE}" \
  --num_trials_per_task "${NUM_TRIALS}" \
  --per_task_jsonl "${OUT_DIR}/per_task.jsonl" \
  --shard_results_json "${OUT_DIR}/results.json" \
  --local_log_dir "${OUT_DIR}/logs" \
  --run_id_note scene_only_eval_smoke \
  --ckpt_path "${CKPT_PATH}" \
  --config cosmos_predict2_2b_480p_libero_scene_only__inference_only \
  --dataset_stats_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json \
  --t5_text_embeddings_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl \
  --use_wrist_image False \
  --use_proprio True \
  --normalize_proprio True \
  --unnormalize_actions True \
  --trained_with_image_aug True \
  --chunk_size 16 \
  --num_open_loop_steps 16 \
  --flip_images True \
  --use_jpeg_compression True \
  --num_denoising_steps_action 5 \
  --num_denoising_steps_future_state 1 \
  --num_denoising_steps_value 1 \
  --ar_future_prediction False \
  --ar_value_prediction False \
  --deterministic True \
  --randomize_seed False \
  --seed 195
