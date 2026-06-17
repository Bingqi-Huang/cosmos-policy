#!/usr/bin/env bash
# E1-main dissociation study — orchestration driver (GPU box, e.g. the 2x RTX 5090).
#
# Produces the pre-registered LD13 readout on the FROZEN Row-1 scene-only checkpoint:
#   action side : rollout success under perturbed cameras (REUSE the existing frozen
#                 Row-1 camera eval JSONLs — NO rerun needed).
#   video side  : camera-conditioned excess-FVD of model-predicted futures vs GT replays.
#
# Method (researcher-decided 2026-06-17): the WAM emits a single predicted future-scene
# frame per query, so the LD13 video-side metric is realised as camera-conditioned
# excess-FID; GT is the rollout's OWN scenes under the same perturbed camera (state-matched,
# no separate demo render). Stages: A perturbed rollouts / A-nom nominal rollouts (Delta
# denominator) / C excess-FID / D dissociation report.
# Prereqs on the eval box: (1) LIBERO-Plus camera env + ~/.libero/config.yaml (stock libero
# has no camera-view handling); (2) cosmos-policy/assets/fid/inception_v3_imagenet.pth;
# (3) frozen Row-1 checkpoint + LIBERO success_only dataset + HF cache for base/LIBERO ckpts.
set -euo pipefail
# Resolve the cosmos-policy repo root from THIS script's location (robust to the caller's
# cwd; the project root is not a git repo, so git rev-parse there would mis-resolve paths).
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"

export MUJOCO_GL="${MUJOCO_GL:-egl}"          # GPU EGL offscreen render (GT replays)
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
# Avoid the Xet download path (it hangs on large gated .pt files on fresh boxes); the
# needed checkpoints resolve from the local HF cache. Matches bin/libero_local_env.sh.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
# Force offline HF resolution: the eval box may have NO internet, and otherwise hf_hub
# does a network HEAD even for cached files and hangs/retries on "Network is unreachable".
# All needed assets (base + LIBERO policy ckpts, tokenizer, dataset stats, t5) must be in
# the local HF cache. Set HF_HUB_OFFLINE=0 to allow downloads on an online box.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

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
# Inception weights for excess-FID (stage C). Default to the in-repo asset; rsync the
# 103MB file to cosmos-policy/assets/fid/ (untracked).
INCEPTION_CKPT="${INCEPTION_CKPT:-${E1_INCEPTION_CKPT:-assets/fid/inception_v3_imagenet.pth}}"
RUN_NOMINAL="${RUN_NOMINAL:-1}"               # also run a nominal-camera pass for the Delta denominator
NUM_TRIALS="${NUM_TRIALS:-3}"
LOAD_EMA_TO_REG="${LOAD_EMA_TO_REG:-True}"    # frozen Row-1 was EMA-enabled; eval with EMA weights
OUT="${OUT:-outputs/phase1/e1_main}"
mkdir -p "${OUT}"

echo "[E1-main] ckpt=${CKPT_PATH}"
[[ -d "${CKPT_PATH}" ]] || { echo "[ERROR] missing checkpoint ${CKPT_PATH}"; exit 1; }
[[ -f "${TASK_CLASSIFICATION}" ]] || echo "[WARN] task_classification.json not found at ${TASK_CLASSIFICATION} (needed for the report's difficulty levels)"

# Shared scene-only eval flags (mirror the frozen Row-1 camera eval). run_camera_task saves,
# per episode, the model's predicted future-scene frames AND the state-matched GT (the
# rollout's own scenes at query cadence under the same perturbed camera) -> per-shard
# e1_frames_manifest_shard*.jsonl. Forwarded to each shard by parse_known_args.
EVAL_FLAGS=(
  --ckpt_path "${CKPT_PATH}" --config "${CONFIG}"
  --dataset_stats_path "${DATASET_STATS}" --t5_text_embeddings_path "${T5_EMB}"
  --use_wrist_image False --use_third_person_image True --use_proprio True
  --normalize_proprio True --unnormalize_actions True --trained_with_image_aug True
  --chunk_size 16 --num_open_loop_steps 16 --flip_images True --use_jpeg_compression True
  --num_denoising_steps_action 5 --num_denoising_steps_future_state 1 --num_denoising_steps_value 1
  --ar_value_prediction True --ar_future_prediction True --deterministic True --randomize_seed False
  --load_ema_to_reg "${LOAD_EMA_TO_REG}" --save_rollout_videos False
)

run_rollouts() {  # $1=suite  $2=task_file  $3=frames_dir  $4=rollout_out_dir
  uv run --no-sync --extra cu128 --group libero --python 3.10 \
    python cosmos_policy/experiments/robot/libero/run_libero_camera_parallel.py \
      --task_suite_name "$1" --camera_tasks_file "$2" \
      --gpu_ids "${GPU_IDS}" --num_trials_per_task "${NUM_TRIALS}" \
      --output_dir "$4" --save_future_clips_dir "$3" "${EVAL_FLAGS[@]}"
}

# ---- Stage A: perturbed-camera rollouts (model futures + state-matched GT) -
MANIFEST="${OUT}/e1_frames_manifest.jsonl"
if [[ -s "${MANIFEST}" ]]; then   # -s: reuse only a NON-EMPTY manifest (an empty one is a failed run)
  echo "[E1-main] Stage A: reusing ${MANIFEST} ($(wc -l < "${MANIFEST}") rows)"
else
  for suite in ${SUITES}; do
    echo "[E1-main] Stage A: perturbed rollouts for ${suite}"
    run_rollouts "${suite}" "${CAMERA_TASKS_DIR}/camera_task_names_${suite}.json" \
      "${OUT}/frames_perturbed/${suite}" "${OUT}/rollouts/${suite}"
  done
  # Concatenate per-shard manifests. Use find (not a glob) so a missing match is an empty file,
  # not a literal-glob row; then ASSERT non-empty so a silent capture failure aborts loudly.
  find "${OUT}/frames_perturbed" -name 'e1_frames_manifest_shard*.jsonl' -exec cat {} + > "${MANIFEST}" 2>/dev/null || true
  if [[ ! -s "${MANIFEST}" ]]; then
    echo "[ERROR] Stage A produced an EMPTY frames manifest (${MANIFEST}). No model/GT frame"
    echo "        pairs were saved — check that rollouts ran and use_third_person_image=True."
    exit 1
  fi
  echo "[E1-main] Stage A done -> ${MANIFEST} ($(wc -l < "${MANIFEST}") rows)"
fi

# ---- Stage A-nom: nominal-camera rollouts (Delta denominator) --------------
NOM_MANIFEST=""
if [[ "${RUN_NOMINAL}" == "1" ]]; then
  NOM_MANIFEST="${OUT}/e1_frames_manifest_nominal.jsonl"
  if [[ ! -s "${NOM_MANIFEST}" ]]; then   # regenerate unless a NON-EMPTY nominal manifest exists
    for suite in ${SUITES}; do
      nom_file="${OUT}/nominal_tasks/camera_task_names_${suite}.json"
      mkdir -p "$(dirname "${nom_file}")"
      uv run --no-sync --extra cu128 --group libero --python 3.10 \
        python cosmos_policy/experiments/robot/libero/e1_main_make_nominal_tasks.py \
          --camera_tasks_file "${CAMERA_TASKS_DIR}/camera_task_names_${suite}.json" --out "${nom_file}"
      echo "[E1-main] Stage A-nom: nominal rollouts for ${suite}"
      run_rollouts "${suite}" "${nom_file}" "${OUT}/frames_nominal/${suite}" "${OUT}/rollouts_nominal/${suite}"
    done
    find "${OUT}/frames_nominal" -name 'e1_frames_manifest_shard*.jsonl' -exec cat {} + > "${NOM_MANIFEST}" 2>/dev/null || true
    if [[ ! -s "${NOM_MANIFEST}" ]]; then
      echo "[WARN] Stage A-nom produced an EMPTY nominal manifest — Delta will have no positive"
      echo "       oracle denominator; the report will mark fidelity verdicts provisional."
    fi
  fi
fi

# ---- Stage C: per-cell excess-FID -----------------------------------------
echo "[E1-main] Stage C: excess-FID"
FID_ARGS=(--manifest "${MANIFEST}" --task_classification "${TASK_CLASSIFICATION}" --out "${OUT}/excess_fid.json")
[[ -n "${INCEPTION_CKPT}" ]] && FID_ARGS+=(--inception_ckpt "${INCEPTION_CKPT}")
[[ -n "${NOM_MANIFEST}" && -f "${NOM_MANIFEST}" ]] && FID_ARGS+=(--nominal_manifest "${NOM_MANIFEST}")
uv run --no-sync --extra cu128 --group libero --python 3.10 \
  python cosmos_policy/experiments/robot/libero/e1_main_fid.py "${FID_ARGS[@]}"

# ---- Stage D: dissociation criterion + report ----------------------------
echo "[E1-main] Stage D: dissociation report"
EXCESS_ARG=()
[[ -f "${OUT}/excess_fid.json" ]] && EXCESS_ARG=(--excess_fvd_json "${OUT}/excess_fid.json")
uv run --no-sync --extra cu128 --group libero --python 3.10 \
  python cosmos_policy/experiments/robot/libero/e1_main_report.py \
    --jsonl_files ${ACTION_JSONL_GLOB} \
    --task_classification "${TASK_CLASSIFICATION}" \
    --nominal_success_rate "${NOMINAL_SR}" \
    "${EXCESS_ARG[@]}" \
    --out_dir "${OUT}/report"

echo "[E1-main] done -> ${OUT}/report/summary.md"
