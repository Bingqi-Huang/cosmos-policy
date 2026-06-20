#!/usr/bin/env bash
set -uo pipefail
# -----------------------------------------------------------------------------
# Pilot orchestrator (2026-06-12).  Arm A (demo_only) is assumed ALREADY running
# (launched separately).  This script:
#   1. waits for Arm A to finish (final checkpoint present), with a divergence /
#      stall guard -- if Arm A diverges or dies without the final checkpoint it
#      aborts WITHOUT launching Arm B (task-book rule: triage, do not re-run).
#   2. launches Arm B (success_only) serially on the same node (blocking).
#   3. evaluates both arms at 6/9/12/15K on libero_10 (10 trials/task) via the
#      diagnostic checkpoint scanner.
#   4. writes an archived side-by-side pilot comparison. This is not the formal
#      single-checkpoint policy protocol.
#
# Run me in the background (harness-tracked) so completion / failure notifies.
# -----------------------------------------------------------------------------
cd /nvme02/bingqi/Project/Selective_Cross-View_Consistency_for_WAMs/cosmos-policy

LOGDIR=outputs/phase1/pilot_logs
mkdir -p "${LOGDIR}"
FINAL_ITER="${FINAL_ITER:-15000}"
FINAL_DIR_A="iter_$(printf '%09d' "${FINAL_ITER}")"

CKROOT=outputs/phase1/cosmos_policy/cosmos_v2_finetune
ARMA_CK="${CKROOT}/phase1_pilot_demo_only_6gpu_b30_15k/checkpoints"
ARMB_CK="${CKROOT}/phase1_pilot_success_only_mix_6gpu_b30_15k/checkpoints"
ARMA_LOG="${LOGDIR}/armA_demo_only_train.log"
ARMB_LOG="${LOGDIR}/armB_success_only_train.log"
EVAL_ITERS="6000 9000 12000 15000"

log() { echo "[orch $(date '+%H:%M:%S')] $*"; }

diverged() {  # $1 = logfile
  grep -qiE "Loss: nan|Loss: inf|Traceback \(most recent|CUDA out of memory|RuntimeError|AssertionError" "$1" 2>/dev/null
}

gpu_idle() {  # true if no compute apps occupying significant memory
  local used
  used=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
  [[ "${used:-0}" -lt 2000 ]]
}

wait_for_arm() {  # $1=ckpt_dir_root  $2=logfile  $3=arm_label
  local ck="$1" lg="$2" lbl="$3"
  log "waiting for ${lbl} final checkpoint ${ck}/${FINAL_DIR_A} ..."
  while true; do
    if [[ -d "${ck}/${FINAL_DIR_A}" ]]; then
      log "${lbl}: final checkpoint present."
      return 0
    fi
    if diverged "${lg}"; then
      log "ABORT: ${lbl} shows divergence/crash in ${lg}."
      grep -iE "Loss: nan|Traceback|CUDA out of memory|RuntimeError|AssertionError" "${lg}" | tail -5
      return 3
    fi
    # stall guard: log untouched for >20 min and no final ckpt -> dead
    if [[ -f "${lg}" ]]; then
      local age=$(( $(date +%s) - $(stat -c %Y "${lg}") ))
      if (( age > 1200 )); then
        log "ABORT: ${lbl} log idle ${age}s with no final checkpoint (stalled/dead)."
        return 4
      fi
    fi
    sleep 120
  done
}

run_eval() {  # $1=ckpt_root $2=label $3=out
  log "eval ${2}: scan ${EVAL_ITERS} on libero_10 (10 trials)"
  CKPT_ROOT="$1" ITERS="${EVAL_ITERS}" SUITES="libero_10" NUM_TRIALS=10 \
    OUT_ROOT="$3" FINAL_WINDOW=3 LABEL="$2" \
    bash cosmos_policy/experiments/robot/libero/launchers/eval/eval_checkpoint_scan.sh \
    > "${LOGDIR}/eval_$(basename "$3").log" 2>&1
}

# ---- 1. Arm A ----
wait_for_arm "${ARMA_CK}" "${ARMA_LOG}" "ArmA(demo_only)" || exit $?
log "waiting for GPUs to free after Arm A ..."
for i in $(seq 1 60); do gpu_idle && break; sleep 20; done

# ---- 2. Arm B ----
log "launching Arm B (success_only) serially ..."
bash cosmos_policy/experiments/robot/libero/launchers/archive/run_scene_only_pilot_train.sh success_only \
  > "${ARMB_LOG}" 2>&1 &
ARMB_PID=$!
log "Arm B pid=${ARMB_PID}"
wait_for_arm "${ARMB_CK}" "${ARMB_LOG}" "ArmB(success_only)" || exit $?
log "waiting for GPUs to free after Arm B ..."
for i in $(seq 1 60); do gpu_idle && break; sleep 20; done

# ---- 3. evals ----
run_eval "${ARMA_CK}" "ArmA demo-only" "outputs/phase1/pilot_eval/armA_demo_only"
run_eval "${ARMB_CK}" "ArmB success-only mixture" "outputs/phase1/pilot_eval/armB_success_only"

# ---- 4. comparison ----
log "building pilot comparison report ..."
python3 cosmos_policy/experiments/robot/libero/build_pilot_comparison.py \
  --armA outputs/phase1/pilot_eval/armA_demo_only \
  --armB outputs/phase1/pilot_eval/armB_success_only \
  --current_scan outputs/phase1/e2_row1_scene_only_eval/ckpt_scan_libero_10 \
  --iters 6000 9000 12000 15000 \
  --out outputs/phase1/pilot_eval/PILOT_COMPARISON.md \
  > "${LOGDIR}/comparison.log" 2>&1
log "DONE. comparison: outputs/phase1/pilot_eval/PILOT_COMPARISON.md"
