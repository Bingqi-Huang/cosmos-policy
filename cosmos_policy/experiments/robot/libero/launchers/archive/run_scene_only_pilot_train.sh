#!/usr/bin/env bash
set -euo pipefail
# -----------------------------------------------------------------------------
# Recipe-discrimination pilot (2026-06-12, researcher-approved).
#
# Two arms, run SERIALLY on the same 6-GPU node so effective batch (bs30/card)
# is identical to the current 30K scene-only run -- the pilot varies ONLY the
# data mixture, nothing else (lr/scheduler/batch untouched).  Both arms start
# fresh from the official LIBERO checkpoint (the scene_only experiment config
# already sets checkpoint.load_path to it; a new job.name with no existing
# checkpoint dir guarantees a fresh load_path init rather than a resume).
#
#   Arm A (demo_only)      : pure successful demos, no rollout data at all.
#   Arm B (success_only)   : official 50/25/25 structure but rollout pool is
#                            success-only (failures excluded); value learning
#                            preserved for E5.
#
# Usage:
#   bash run_scene_only_pilot_train.sh demo_only
#   bash run_scene_only_pilot_train.sh success_only
#
# train.py instantiates dataloader_train.dataset ONCE and wraps it in the
# sampler (see train.py:60-81), so dataloader_train.sampler.dataset is dead
# config -- overriding dataloader_train.dataset.* alone is sufficient.
# -----------------------------------------------------------------------------

ARM="${1:-${ARM:-}}"
if [[ "${ARM}" != "demo_only" && "${ARM}" != "success_only" ]]; then
  echo "Usage: $0 <demo_only|success_only>" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export BASE_DATASETS_DIR="${BASE_DATASETS_DIR:-.}"
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-outputs/phase1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"

NUM_GPUS="${NUM_GPUS:-6}"
FSDP_SHARD_SIZE="${FSDP_SHARD_SIZE:-6}"
MAX_ITER="${MAX_ITER:-15000}"
SAVE_ITER="${SAVE_ITER:-3000}"

if [[ "${ARM}" == "demo_only" ]]; then
  JOB_NAME="${JOB_NAME:-phase1_pilot_demo_only_6gpu_b30_15k}"
  ARM_OVERRIDES=(
    'dataloader_train.dataset.rollout_data_dir=""'
    dataloader_train.dataset.demonstration_sampling_prob=1.0
  )
else
  JOB_NAME="${JOB_NAME:-phase1_pilot_success_only_mix_6gpu_b30_15k}"
  ARM_OVERRIDES=(
    dataloader_train.dataset.success_rollout_sampling_prob=1.0
  )
fi

echo "[pilot] arm=${ARM} job=${JOB_NAME} max_iter=${MAX_ITER} save_iter=${SAVE_ITER}"
echo "[pilot] overrides: ${ARM_OVERRIDES[*]}"

uv run --extra cu128 --group libero --python 3.10 \
  python -m torch.distributed.run --nproc_per_node="${NUM_GPUS}" cosmos_policy/scripts/train.py \
  --config=cosmos_policy/config/config.py -- \
  experiment=cosmos_predict2_2b_480p_libero_scene_only \
  job.name="${JOB_NAME}" \
  job.wandb_mode="${WANDB_MODE}" \
  model.config.fsdp_shard_size="${FSDP_SHARD_SIZE}" \
  trainer.max_iter="${MAX_ITER}" \
  checkpoint.save_iter="${SAVE_ITER}" \
  "${ARM_OVERRIDES[@]}" \
  "${@:2}"
