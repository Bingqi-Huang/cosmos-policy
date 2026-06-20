#!/usr/bin/env bash
set -euo pipefail

# Stable scene-only baseline candidate.
# Recipe: original full LIBERO recipe from the scene-only config; no failure-data filtering.
# Variance controls: EMA, effective batch 720, lower LR.

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export BASE_DATASETS_DIR="${BASE_DATASETS_DIR:-.}"
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-outputs/phase1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

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
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"

NUM_GPUS="${NUM_GPUS:-6}"
FSDP_SHARD_SIZE="${FSDP_SHARD_SIZE:-6}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-24}"
GRAD_ACCUM="${GRAD_ACCUM:-5}"
LR="${LR:-5e-5}"
MAX_ITER="${MAX_ITER:-15000}"
SAVE_ITER="${SAVE_ITER:-1000}"
SEED="${SEED:-42}"
JOB_NAME="${JOB_NAME:-phase1_scene_only_stable_original_ema_b24_gacc5_lr5e5_seed${SEED}_15k}"

EFFECTIVE_BATCH=$((LOCAL_BATCH_SIZE * NUM_GPUS * GRAD_ACCUM))

echo "[stable-train] job=${JOB_NAME}"
echo "[stable-train] recipe=original_full_data"
echo "[stable-train] num_gpus=${NUM_GPUS} local_batch=${LOCAL_BATCH_SIZE} grad_accum=${GRAD_ACCUM} effective_batch=${EFFECTIVE_BATCH}"
echo "[stable-train] lr=${LR} ema=true seed=${SEED} max_iter=${MAX_ITER} save_iter=${SAVE_ITER}"

uv run --extra cu128 --group libero --python 3.10 \
  python -m torch.distributed.run --nproc_per_node="${NUM_GPUS}" cosmos_policy/scripts/train.py \
  --config=cosmos_policy/config/config.py -- \
  experiment=cosmos_predict2_2b_480p_libero_scene_only \
  job.name="${JOB_NAME}" \
  job.wandb_mode="${WANDB_MODE}" \
  model.config.fsdp_shard_size="${FSDP_SHARD_SIZE}" \
  model.config.ema.enabled=True \
  checkpoint.load_ema_to_reg=False \
  dataloader_train.batch_size="${LOCAL_BATCH_SIZE}" \
  trainer.grad_accum_iter="${GRAD_ACCUM}" \
  trainer.seed="${SEED}" \
  trainer.max_iter="${MAX_ITER}" \
  checkpoint.save_iter="${SAVE_ITER}" \
  optimizer.lr="${LR}" \
  "$@"
