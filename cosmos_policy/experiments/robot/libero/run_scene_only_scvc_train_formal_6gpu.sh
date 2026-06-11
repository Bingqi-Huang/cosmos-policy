#!/usr/bin/env bash
set -euo pipefail

# Prepared launcher for scene-only P2 SCVC / A1 / A2 / A5 training.
# Do not run until Phase-2 pair manifests and the two-branch memory smoke are done.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export BASE_DATASETS_DIR="${BASE_DATASETS_DIR:-.}"
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-outputs/phase3}"
export WANDB_MODE="${WANDB_MODE:-online}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

# NCCL transport (2026-06-11): nvidia-nccl-cu12==2.30.7 makes default transports stable
# and ~1.75x faster on this Blackwell host (see AGENT/training_speed_optimization.md).
# The 2.26.2-era conservative workaround is kept behind NCCL_CONSERVATIVE=1 for rollback.
if [[ "${NCCL_CONSERVATIVE:-0}" == "1" ]]; then
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
  export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
  export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
  export NCCL_MNNVL_ENABLE="${NCCL_MNNVL_ENABLE:-0}"
  export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
  export NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"
  export NCCL_DMABUF_ENABLE="${NCCL_DMABUF_ENABLE:-0}"
  export NCCL_ALGO="${NCCL_ALGO:-Ring}"
  export NCCL_PROTO="${NCCL_PROTO:-Simple}"
fi
# Gloo is still used for DCP checkpoint metadata collectives (harmless under 2.30.7).
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"

NUM_GPUS="${NUM_GPUS:-6}"
FSDP_SHARD_SIZE="${FSDP_SHARD_SIZE:-6}"
PAIR_BATCH_SIZE="${PAIR_BATCH_SIZE:-10}"
PAIR_MANIFEST_PATH="${PAIR_MANIFEST_PATH:-outputs/phase2/pair_future_frames/libero_pair_future_manifest_train.jsonl}"
JOB_NAME="${JOB_NAME:-phase3_scene_only_scvc_6gpu_b10_lambda010}"

export PAIR_BATCH_SIZE
export PAIR_MANIFEST_PATH
export JOB_NAME
export LAMBDA_CV="${LAMBDA_CV:-0.1}"
export CV_FRAME_SET="${CV_FRAME_SET:-action+value+fproprio}"
export CV_NOISE_SHARED="${CV_NOISE_SHARED:-true}"
export CV_PAIR_MODE="${CV_PAIR_MODE:-matched}"
export CV_NUM_SAMPLES="${CV_NUM_SAMPLES:-2}"
export CV_TOTAL_STEPS="${CV_TOTAL_STEPS:-${MAX_ITER:-10000}}"
export CV_WARMUP_START_FRACTION="${CV_WARMUP_START_FRACTION:-0.0}"
export CV_WARMUP_END_FRACTION="${CV_WARMUP_END_FRACTION:-0.1}"

uv run --extra cu128 --group libero --python 3.10 \
  python -m torch.distributed.run --nproc_per_node="${NUM_GPUS}" cosmos_policy/scripts/train.py \
  --config=cosmos_policy/config/config.py -- \
  experiment=cosmos_predict2_2b_480p_libero_scvc_scene_only \
  trainer.max_iter="${MAX_ITER:-10000}" \
  checkpoint.save_iter="${SAVE_ITER:-1000}" \
  job.name="${JOB_NAME}" \
  job.wandb_mode="${WANDB_MODE}" \
  model.config.fsdp_shard_size="${FSDP_SHARD_SIZE}" \
  "$@"
