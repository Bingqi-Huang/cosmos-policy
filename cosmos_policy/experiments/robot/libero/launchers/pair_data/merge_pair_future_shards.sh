#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${REPO}/outputs/phase2/pair_future_frames}"
N_SHARDS="${N_SHARDS:-6}"

for KIND in \
  libero_pair_future_manifest_train \
  libero_pair_future_manifest_val \
  libero_wrong_pair_future_manifest_train \
  libero_wrong_pair_future_manifest_val; do
  OUT="${RESULTS_DIR}/${KIND}.jsonl"
  : > "${OUT}"
  TOTAL=0
  for SHARD in $(seq 0 $((N_SHARDS - 1))); do
    TAG=$(printf "shard%02d" "${SHARD}")
    SRC="${RESULTS_DIR}/${KIND}_${TAG}.jsonl"
    if [[ ! -f "${SRC}" ]]; then
      echo "[ERROR] missing shard file: ${SRC}" >&2
      exit 2
    fi
    N=$(wc -l < "${SRC}")
    cat "${SRC}" >> "${OUT}"
    TOTAL=$((TOTAL + N))
    echo "[merge] ${KIND} ${TAG}: ${N}"
  done
  echo "[merge] ${OUT}: ${TOTAL}"
done

echo "[done] merged pair future manifests in ${RESULTS_DIR}"
