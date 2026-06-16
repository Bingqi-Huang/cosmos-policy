#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export BASE_DATASETS_DIR="${BASE_DATASETS_DIR:-.}"
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-outputs/phase1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

# NCCL transport (2026-06-11): with nvidia-nccl-cu12==2.30.7 the default transports
# (P2P/SHM/CUMEM) are stable on this Blackwell host and ~1.75x faster than the old
# socket-loopback workaround (3.4 vs 5.9 s/iter; see AGENT/training_speed_optimization.md).
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
JOB_NAME="${JOB_NAME:-phase1_scene_only_formal_6gpu_b30_from_libero_ckpt}"
FSDP_SHARD_SIZE="${FSDP_SHARD_SIZE:-6}"

uv run --extra cu128 --group libero --python 3.10 \
  python -m torch.distributed.run --nproc_per_node="${NUM_GPUS}" cosmos_policy/scripts/train.py \
  --config=cosmos_policy/config/config.py -- \
  experiment=cosmos_predict2_2b_480p_libero_scene_only \
  job.name="${JOB_NAME}" \
  job.wandb_mode="${WANDB_MODE}" \
  model.config.fsdp_shard_size="${FSDP_SHARD_SIZE}" \
  "$@"
