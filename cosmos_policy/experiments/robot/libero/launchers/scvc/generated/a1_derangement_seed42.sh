#!/usr/bin/env bash
set -euo pipefail

# Generated from phase3_run_registry.json for a1_derangement_seed42.
# Review GPU availability before executing.

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export NUM_GPUS=6
export FSDP_SHARD_SIZE=6
export PAIR_BATCH_SIZE=24
export GRAD_ACCUM_ITER=5
export LR=5e-05
export EMA_ENABLED=true
export LOAD_EMA_TO_REG=false
export PAIR_MANIFEST_PATH=outputs/phase2/pair_future_frames/libero_pair_future_manifest_train.jsonl
export JOB_NAME=phase3_a1_derangement_lambda010_seed42
export LAMBDA_CV=0.1
export CV_FRAME_SET=action+value+fproprio
export CV_NOISE_SHARED=true
export CV_PAIR_MODE=derangement
export CV_TOTAL_STEPS=10000
export MAX_ITER=10000
export SAVE_ITER=1000
export WANDB_MODE=online

bash cosmos_policy/experiments/robot/libero/launchers/scvc/run_scene_only_scvc_train_formal_6gpu.sh trainer.seed=42 dataloader_train.sampler.seed=42 "$@"
