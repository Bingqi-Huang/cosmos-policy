#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export BASE_DATASETS_DIR="${BASE_DATASETS_DIR:-.}"
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-outputs/phase1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

JOB_NAME="${JOB_NAME:-phase1_scene_only_train_step_dryrun_1gpu_b1_gacc4}"

uv run --extra cu128 --group libero --python 3.10 \
  python -m torch.distributed.run --nproc_per_node=1 cosmos_policy/scripts/train.py \
  --config=cosmos_policy/config/config.py -- \
  experiment=cosmos_predict2_2b_480p_libero_scene_only \
  job.name="${JOB_NAME}" \
  job.wandb_mode="${WANDB_MODE}" \
  trainer.max_iter=1 \
  trainer.grad_accum_iter=4 \
  trainer.logging_iter=1 \
  checkpoint.save_iter=999999 \
  model.config.fsdp_shard_size=1 \
  dataloader_train.batch_size=1 \
  dataloader_train.num_workers=2 \
  dataloader_train.persistent_workers=False
