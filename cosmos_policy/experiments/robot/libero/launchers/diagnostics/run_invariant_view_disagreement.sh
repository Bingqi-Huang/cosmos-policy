#!/usr/bin/env bash
# Held-out invariant-block cross-view disagreement: Row3 (lambda_cv=0 control) vs
# Row4 (SCVC lambda_cv=0.1), matched checkpoints. Early go/no-go for whether the
# current Row4 run is worth finishing to 10k.
#
# Two-sided read (the script prints a VERDICT):
#   PRIMARY = action-frame disagreement ratio row4/row3 (action is the load-bearing
#   frame for camera robustness; a falling aggregate driven by value/proprio while
#   action stays put helps little):
#     <= 0.80      => GO    (mechanism clearly engaging -> finish to 10k)
#     >= 0.95      => NO-GO  (essentially null -> stop & relaunch stronger)
#     0.80-0.95    => AMBIGUOUS (per-frame breakdown + tiny camera subset; maybe wait for iter_2000)
#   COVARIANT GUARD = held-out future-scene ratio must NOT collapse for Row4 vs Row3
#   (selective method leaves the dream per-view supervised; collapse = ghosting).
#
# Latent-space only (no rollouts). Pair with launchers/eval/run_scene_only_camera_screen120.sh
# for the success-level half of the read.
#
# Memory note: loads one 2B checkpoint at a time on a SINGLE GPU. The mainline
# training occupies GPUs 0-5; point GPU at a device with spare memory, run on the
# 5090 box, or run after this checkpoint when a GPU frees up. batch-size is kept at 2.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${SCRIPT_DIR}/../../../../../.." && pwd -P))"

PY="${PY:-.venv/bin/python}"
GPU="${GPU:-0}"
ITER="${ITER:-iter_000001000}"
CKPT_ROOT="${CKPT_ROOT:-outputs/phase3/cosmos_policy/cosmos_v2_finetune}"
ROW3_CKPT="${ROW3_CKPT:-${CKPT_ROOT}/phase3_row3_pair_fm_only_seed42/checkpoints/${ITER}}"
ROW4_CKPT="${ROW4_CKPT:-${CKPT_ROOT}/phase3_row4_scvc_lambda010_seed42/checkpoints/${ITER}}"
HOLDOUT="${HOLDOUT:-outputs/phase2/pair_future_frames/libero_pair_future_manifest_holdout_frozen.jsonl}"
OUT_DIR="${OUT_DIR:-outputs/phase3/diagnostics/invariant_view_disagreement/${ITER}}"
SIGMA_GRID="${SIGMA_GRID:-0.02,0.05,0.1,0.2,0.5,1.0,2.0,5.0}"
BATCH_SIZE="${BATCH_SIZE:-2}"
# Holdout has ~48k pairs; a few hundred suffice for the aggregate RMS. 0 = full pass.
MAX_BATCHES="${MAX_BATCHES:-150}"

for c in "$ROW3_CKPT" "$ROW4_CKPT"; do
  if [[ ! -e "$c" ]]; then
    echo "ERROR: checkpoint not found: $c" >&2
    echo "(Row4 ${ITER} may not be written yet; SAVE_ITER=1000, so wait for it.)" >&2
    exit 1
  fi
done

CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" \
  cosmos_policy/experiments/robot/libero/diagnostics/scvc/eval_invariant_view_disagreement.py \
  --checkpoints "$ROW3_CKPT" "$ROW4_CKPT" \
  --labels row3_fmonly row4_scvc \
  --holdout-manifest "$HOLDOUT" \
  --output-dir "$OUT_DIR" \
  --sigma-grid "$SIGMA_GRID" \
  --batch-size "$BATCH_SIZE" \
  --max-batches "$MAX_BATCHES" \
  --device "cuda:0"

echo "Done. Report: ${OUT_DIR}/invariant_view_disagreement.md"
