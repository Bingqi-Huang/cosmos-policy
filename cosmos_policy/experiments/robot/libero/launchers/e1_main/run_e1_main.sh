#!/usr/bin/env bash
# E1-main dissociation study — orchestration driver (GPU box, e.g. the 2x RTX 5090).
#
# Produces the pre-registered LD13 readout on the FROZEN Row-1 scene-only checkpoint:
#   action side : rollout success under perturbed cameras (REUSE the existing frozen
#                 Row-1 camera eval JSONLs — NO rerun needed).
#   video side  : camera-conditioned excess-FVD of model-predicted futures vs GT replays.
#
# Stages B/C/D below are the NEW work. Stage A (model-future rollouts) needs the one-line
# eval wiring documented in handoff ("E1-main wiring") and the I3D checkpoint; both are
# first-GPU-run items, hence this is a driver to run/validate on the 5090, not on the
# training node. Read the inline NOTES before running.
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

export MUJOCO_GL="${MUJOCO_GL:-egl}"          # GPU EGL offscreen render (GT replays)
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

# ---- knobs ----------------------------------------------------------------
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
GPU_IDS="${GPU_IDS:-0,1}"                      # two 5090s by default
CKPT_PATH="${CKPT_PATH:-outputs/phase1/cosmos_policy/cosmos_v2_finetune/phase1_scene_only_stable_original_ema_b24_gacc5_lr5e5_seed42_15k/checkpoints/iter_000010000}"
CONFIG="${CONFIG:-cosmos_predict2_2b_480p_libero_scene_only__inference_only}"
DATASET_STATS="${DATASET_STATS:-nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json}"
T5_EMB="${T5_EMB:-nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl}"
CAMERA_TASKS_DIR="${CAMERA_TASKS_DIR:-outputs/phase0/libero_plus_camera_eval}"   # camera_task_names_<suite>.json
ACTION_JSONL_GLOB="${ACTION_JSONL_GLOB:-outputs/phase1/scene_only_stable_eval/iter_000010000/camera_full/*/shards/*/per_task.jsonl}"
TASK_CLASSIFICATION="${TASK_CLASSIFICATION:-$HOME/.cache/huggingface/hub/datasets--Sylvest--libero_plus_data_4suite/task_classification.json}"
NOMINAL_SR="${NOMINAL_SR:-0.932}"             # frozen Row-1 full-ID success rate (action nominal)
I3D_CKPT="${I3D_CKPT:-${E1_I3D_CKPT:-}}"      # TorchScript I3D for FVD (REQUIRED for stage C)
NUM_TRIALS="${NUM_TRIALS:-3}"
OUT="${OUT:-outputs/phase1/e1_main}"
mkdir -p "${OUT}"

echo "[E1-main] ckpt=${CKPT_PATH}"
[[ -d "${CKPT_PATH}" ]] || { echo "[ERROR] missing checkpoint ${CKPT_PATH}"; exit 1; }
[[ -f "${TASK_CLASSIFICATION}" ]] || echo "[WARN] task_classification.json not found at ${TASK_CLASSIFICATION} (needed for the report's difficulty levels)"

# ---- Stage A: model-predicted future rollouts (video side) ----------------
# NOTE (first-GPU-run wiring): run_libero_eval must save predicted future clips. Wire
# e1_main_fvd.save_model_future_clip(...) into run_libero_eval right after the future
# video save (~line 794), guarded by a --save_future_clips_dir flag, keyed by the current
# camera task name, and run with --ar_future_prediction True. See handoff "E1-main wiring".
# Until wired, Stage A is a no-op stub and you must supply MODEL_MANIFEST manually.
MODEL_MANIFEST="${MODEL_MANIFEST:-${OUT}/model_futures_manifest.jsonl}"
echo "[E1-main] Stage A: ensure ${MODEL_MANIFEST} exists (model-predicted future clips)."
if [[ ! -f "${MODEL_MANIFEST}" ]]; then
  echo "[E1-main] Stage A NOT wired yet — see handoff 'E1-main wiring'. Skipping."
fi

# ---- Stage B: GT-replay future clips at the benchmark eval cameras --------
for suite in ${SUITES}; do
  echo "[E1-main] Stage B: GT futures for ${suite}"
  uv run --extra cu128 --group libero --python 3.10 \
    python cosmos_policy/experiments/robot/libero/e1_main_render_gt_futures.py \
      --camera_tasks_file "${CAMERA_TASKS_DIR}/camera_task_names_${suite}.json" \
      --suite "${suite}" \
      --out_dir "${OUT}/gt_futures" \
      --gpu_device_id "${GPU_IDS%%,*}"
done
GT_MANIFEST="${OUT}/gt_futures/gt_futures_manifest_combined.jsonl"
cat "${OUT}"/gt_futures/gt_futures_manifest_*.jsonl > "${GT_MANIFEST}" 2>/dev/null || true

# ---- Stage C: per-cell excess-FVD ----------------------------------------
if [[ -n "${I3D_CKPT}" && -f "${MODEL_MANIFEST}" ]]; then
  echo "[E1-main] Stage C: excess-FVD"
  uv run --extra cu128 --group libero --python 3.10 \
    python cosmos_policy/experiments/robot/libero/e1_main_fvd.py \
      --model_manifest "${MODEL_MANIFEST}" \
      --gt_manifest "${GT_MANIFEST}" \
      --task_classification "${TASK_CLASSIFICATION}" \
      --i3d_ckpt "${I3D_CKPT}" \
      --out "${OUT}/excess_fvd.json"
else
  echo "[E1-main] Stage C skipped: need I3D_CKPT and ${MODEL_MANIFEST}."
fi

# ---- Stage D: dissociation criterion + report ----------------------------
echo "[E1-main] Stage D: dissociation report"
EXCESS_ARG=()
[[ -f "${OUT}/excess_fvd.json" ]] && EXCESS_ARG=(--excess_fvd_json "${OUT}/excess_fvd.json")
uv run --extra cu128 --group libero --python 3.10 \
  python cosmos_policy/experiments/robot/libero/e1_main_report.py \
    --jsonl_files ${ACTION_JSONL_GLOB} \
    --task_classification "${TASK_CLASSIFICATION}" \
    --nominal_success_rate "${NOMINAL_SR}" \
    "${EXCESS_ARG[@]}" \
    --out_dir "${OUT}/report"

echo "[E1-main] done -> ${OUT}/report/summary.md"
