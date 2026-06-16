#!/usr/bin/env bash
# Post-render pair-data gate pipeline (run AFTER the 6-shard render completes).
#
# Sequence (any failure aborts the whole gate, by design):
#   1. guard: all 6 per-shard train manifests exist (else render not done)
#   2. merge_pair_future_shards.sh        -> merged train/val (+ wrong) manifests
#   3. summarize_pair_manifest.py         -> scale/distribution stats (train)
#   4. validate_pair_future_data.py       -> camera-disjointness (standing rule 4)
#                                            + same-state hash recompute (matched pairs)
#   5. make_pair_visual_contact_sheet.py  -> visual pair spot-check (train)
#   6. freeze_pair_holdout_manifest.py    -> LOCKED DECISION 8 held-out freeze (val)
#   7. scale audit: effective_epochs = BUDGET * DEMO_FRACTION / distinct_pairs;
#                   WARN (non-fatal) if > EPOCH_LIMIT -> expand render before training.
#
# Safe to re-run: merge/summarize/validate/contact-sheet are idempotent; freeze
# refuses to overwrite an existing frozen manifest (LOCKED DECISION 8).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
PYTHON="${PYTHON:-${REPO}/.venv/bin/python}"
LDIR="${REPO}/cosmos_policy/experiments/robot/libero"
RESULTS_DIR="${RESULTS_DIR:-${REPO}/outputs/phase2/pair_future_frames}"
GATES_DIR="${GATES_DIR:-${RESULTS_DIR}/gates}"
N_SHARDS="${N_SHARDS:-6}"
STATE_HASH_ROWS="${STATE_HASH_ROWS:-256}"
CONTACT_SAMPLES="${CONTACT_SAMPLES:-48}"
# Scale-audit knobs (frozen budget = 7.2M sample presentations; Plan A pair fraction ~0.5).
BUDGET="${BUDGET:-7200000}"
DEMO_FRACTION="${DEMO_FRACTION:-0.5}"
EPOCH_LIMIT="${EPOCH_LIMIT:-10}"

TRAIN="${RESULTS_DIR}/libero_pair_future_manifest_train.jsonl"
VAL="${RESULTS_DIR}/libero_pair_future_manifest_val.jsonl"
mkdir -p "${GATES_DIR}"

echo "== [gate 1] guard: per-shard train manifests =="
missing=0
for s in $(seq 0 $((N_SHARDS - 1))); do
  f="${RESULTS_DIR}/libero_pair_future_manifest_train_shard$(printf '%02d' "$s").jsonl"
  if [[ ! -f "$f" ]]; then echo "  MISSING: $f"; missing=1; fi
done
if [[ "$missing" -ne 0 ]]; then
  echo "[ABORT] render not complete (per-shard manifests missing). Re-run after all shards finish." >&2
  exit 3
fi
echo "  all ${N_SHARDS} shard manifests present."

echo "== [gate 2] merge shards =="
bash "${LDIR}/launchers/pair_data/merge_pair_future_shards.sh"

echo "== [gate 3] summarize merged train manifest =="
"${PYTHON}" "${LDIR}/summarize_pair_manifest.py" \
  --manifest "${TRAIN}" --repo-root "${REPO}" \
  --output-json "${GATES_DIR}/summary_train.json" \
  --output-md "${GATES_DIR}/summary_train.md"

echo "== [gate 4] validate (camera-disjoint + same-state hash recompute) =="
"${PYTHON}" "${LDIR}/validate_pair_future_data.py" \
  --manifest "${TRAIN}" --repo-root "${REPO}" \
  --state-hash-check-rows "${STATE_HASH_ROWS}"

echo "== [gate 5] visual contact sheet (train) =="
"${PYTHON}" "${LDIR}/make_pair_visual_contact_sheet.py" \
  --manifest "${TRAIN}" --repo-root "${REPO}" \
  --output "${GATES_DIR}/contact_sheet_train.png" \
  --sample-count "${CONTACT_SAMPLES}"

echo "== [gate 6] freeze held-out (val) =="
"${PYTHON}" "${LDIR}/freeze_pair_holdout_manifest.py" \
  --val-manifest "${VAL}" \
  --frozen-out "${RESULTS_DIR}/libero_pair_future_manifest_holdout_frozen.jsonl"

echo "== [gate 7] paired-data scale audit =="
"${PYTHON}" - "${TRAIN}" "${BUDGET}" "${DEMO_FRACTION}" "${EPOCH_LIMIT}" "${GATES_DIR}/scale_audit.json" <<'PY'
import json, sys
train, budget, frac, limit, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
ids = set()
n = 0
with open(train) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        n += 1
        try:
            ids.add(json.loads(line)["pair_id"])
        except Exception:
            pass
distinct = len(ids) or n
pair_presentations = budget * frac
eff_epochs = pair_presentations / distinct if distinct else float("inf")
verdict = "OK" if eff_epochs <= limit else "EXPAND_RENDER"
rep = {
    "train_rows": n, "distinct_pairs": distinct,
    "budget_sample_presentations": budget, "demo_fraction": frac,
    "pair_presentations": pair_presentations,
    "effective_epochs": round(eff_epochs, 3), "epoch_limit": limit,
    "verdict": verdict,
}
print(json.dumps(rep, indent=2))
open(out, "w").write(json.dumps(rep, indent=2) + "\n")
if verdict != "OK":
    print(f"[WARN] effective_epochs {eff_epochs:.2f} > {limit}: expand rendering before training to avoid pair-distribution overfit.", file=sys.stderr)
PY

echo ""
echo "[done] pair-data gates passed. Artifacts in ${GATES_DIR}"
echo "       Next: two-branch memory smoke (bs10 x K_s=2) -> SCVC sanity ladder."
