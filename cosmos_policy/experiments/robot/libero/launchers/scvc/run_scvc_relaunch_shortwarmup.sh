#!/usr/bin/env bash
# Fixed-recipe SCVC relaunch: short/zero warmup + stronger lambda_cv.
#
# Rationale (handoff: 2026-06-20 Row4 diagnosis): on the invariant-only block the
# cross-view residual collapses within ~40 steps (FM alone shares the target), but
# the default 10% linear warmup only reaches full lambda at step 1000 -- so the
# (1+4*lambda) residual amplification arrives after the signal it should act on is
# gone. And lambda_cv=0.1 is a CoRL/VLA transfer default; in a WAM the FM budget is
# dominated by future-scene video reconstruction, so the same lambda applies weaker
# pressure to the (small) invariant block here than it did on a VLA. This relaunch
# (a) front-loads the warmup and (b) raises lambda so 1+4*lambda is 3.0 (lambda=0.5)
# instead of 1.4 (lambda=0.1).
#
# NOT a registry row yet -- this is the diagnostic-ablation pivot prepared so we can
# relaunch immediately if the iter_1000 invariant-disagreement read comes back null.
# If adopted as a mainline row, fold it into phase3_run_registry.json + regenerate.
#
# Sweep example:
#   for L in 0.5 2.0; do LAMBDA_CV=$L JOB_SUFFIX=lambda0${L/./} bash THIS_SCRIPT; done
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${SCRIPT_DIR}/../../../../../.." && pwd -P))"

# --- knobs (overridable) ---
export LAMBDA_CV="${LAMBDA_CV:-0.5}"                       # 1+4*lambda = 3.0
export CV_WARMUP_START_FRACTION="${CV_WARMUP_START_FRACTION:-0.0}"
export CV_WARMUP_END_FRACTION="${CV_WARMUP_END_FRACTION:-0.02}"  # full lambda by step 200 (of 10k)
SEED="${SEED:-42}"
LAMBDA_TAG="${LAMBDA_CV/./}"
JOB_SUFFIX="${JOB_SUFFIX:-lambda${LAMBDA_TAG}_warm${CV_WARMUP_END_FRACTION/./}_seed${SEED}}"

# --- fixed recipe (matched to Row4 except the two knobs above) ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export NUM_GPUS="${NUM_GPUS:-6}"
export FSDP_SHARD_SIZE="${FSDP_SHARD_SIZE:-6}"
export PAIR_BATCH_SIZE="${PAIR_BATCH_SIZE:-24}"
export GRAD_ACCUM_ITER="${GRAD_ACCUM_ITER:-5}"
export LR="${LR:-5e-05}"
export EMA_ENABLED="${EMA_ENABLED:-true}"
export LOAD_EMA_TO_REG="${LOAD_EMA_TO_REG:-false}"
export PAIR_MANIFEST_PATH="${PAIR_MANIFEST_PATH:-outputs/phase2/pair_future_frames/libero_pair_future_manifest_train.jsonl}"
export CV_FRAME_SET="${CV_FRAME_SET:-action+value+fproprio}"
export CV_NOISE_SHARED="${CV_NOISE_SHARED:-true}"
export CV_PAIR_MODE="${CV_PAIR_MODE:-matched}"
export MAX_ITER="${MAX_ITER:-10000}"
export CV_TOTAL_STEPS="${CV_TOTAL_STEPS:-${MAX_ITER}}"
export SAVE_ITER="${SAVE_ITER:-1000}"
export WANDB_MODE="${WANDB_MODE:-online}"
export JOB_NAME="${JOB_NAME:-phase3_row4b_scvc_${JOB_SUFFIX}}"

echo "[relaunch] JOB_NAME=${JOB_NAME} LAMBDA_CV=${LAMBDA_CV} warmup_end=${CV_WARMUP_END_FRACTION} seed=${SEED}"
bash cosmos_policy/experiments/robot/libero/launchers/scvc/run_scene_only_scvc_train_formal_6gpu.sh \
  trainer.seed="${SEED}" dataloader_train.sampler.seed="${SEED}" "$@"
