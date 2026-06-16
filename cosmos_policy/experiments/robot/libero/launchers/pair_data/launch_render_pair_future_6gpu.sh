#!/usr/bin/env bash
set -euo pipefail

# Prepared launcher for Phase-2 same-state pair future-frame rendering.
# Do not run while training owns the GPUs.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
PYTHON="${PYTHON:-${REPO}/.venv/bin/python}"
SCRIPT="${REPO}/cosmos_policy/experiments/robot/libero/render_libero_pair_future_frames.py"

LIBERO_ROOT="${LIBERO_ROOT:-${REPO}/LIBERO-Cosmos-Policy/success_only}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO}/outputs/phase2/pair_future_frames/images}"
RESULTS_DIR="${RESULTS_DIR:-${REPO}/outputs/phase2/pair_future_frames}"
LOG_DIR="${LOG_DIR:-${REPO}/outputs/phase2/pair_future_frames/logs}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5}"
TIMESTEP_SAMPLE_RATE="${TIMESTEP_SAMPLE_RATE:-1.0}"
VIEWS_PER_STATE="${VIEWS_PER_STATE:-2}"
MAX_PAIRS_PER_SUITE="${MAX_PAIRS_PER_SUITE:-}"

mkdir -p "${LOG_DIR}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
N_SHARDS="${#GPU_ARRAY[@]}"

# Suite selection. Default is the full 4-suite set; set SUITE_OVERRIDE to a
# space-separated subset (e.g. SUITE_OVERRIDE="libero_spatial_regen") to run a
# genuine single-suite pilot. Word-splitting on SUITE_OVERRIDE is intentional.
DEFAULT_SUITES="libero_spatial_regen libero_object_regen libero_goal_regen libero_10_regen"
read -r -a SUITE_ARRAY <<< "${SUITE_OVERRIDE:-${DEFAULT_SUITES}}"
echo "[launch] suites: ${SUITE_ARRAY[*]}"

COMMON_ARGS=(
  --libero-root "${LIBERO_ROOT}"
  --suite "${SUITE_ARRAY[@]}"
  --output-dir "${OUTPUT_DIR}"
  --results-dir "${RESULTS_DIR}"
  --img-size 256
  --timestep-sample-rate "${TIMESTEP_SAMPLE_RATE}"
  --views-per-state "${VIEWS_PER_STATE}"
  --val-demo-fraction 0.10
  --chunk-size 16
  --seed 42
  --n-shards "${N_SHARDS}"
)

if [[ -n "${MAX_PAIRS_PER_SUITE}" ]]; then
  COMMON_ARGS+=(--max-pairs-per-suite "${MAX_PAIRS_PER_SUITE}")
fi

PIDS=()
for SHARD in "${!GPU_ARRAY[@]}"; do
  GPU_ID="${GPU_ARRAY[$SHARD]}"
  LOG="${LOG_DIR}/render_pair_future_shard_${SHARD}.log"
  echo "[launch] GPU ${GPU_ID} shard ${SHARD}/${N_SHARDS} -> ${LOG}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" MUJOCO_GL=egl "${PYTHON}" "${SCRIPT}" \
    "${COMMON_ARGS[@]}" \
    --gpu-device-id 0 \
    --shard-idx "${SHARD}" \
    > "${LOG}" 2>&1 &
  PIDS+=($!)
done

printf '%s\n' "${PIDS[@]}" > "${LOG_DIR}/render_pair_future_pids.txt"
echo "[launch] PIDs: ${PIDS[*]}"
echo "[next] after all finish: bash cosmos_policy/experiments/robot/libero/launchers/pair_data/merge_pair_future_shards.sh"
