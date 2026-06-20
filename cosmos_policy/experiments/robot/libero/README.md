# LIBERO Experiment Utilities

Core Python evaluators and report generators stay in this directory so module
imports remain stable. Shell launchers live under `launchers/`.

## Baseline Launchers

- `launchers/baseline/run_scene_only_stable_train_6gpu.sh`: current stable scene-only baseline training launcher. It keeps the original full-data recipe and applies variance controls: EMA, lower LR, lower local batch, larger effective batch through gradient accumulation.
- `launchers/baseline/run_scene_only_stability_eval.sh`: validation-only evaluator for selecting one checkpoint from a fixed LIBERO subset. It does not average checkpoints.
- `launchers/baseline/run_scene_only_train_formal_6gpu.sh`: legacy original-recipe training launcher kept for provenance.
- `launchers/baseline/run_scene_only_row1_eval_full.sh`: full standard plus camera evaluation launcher kept for provenance.

## Evaluation And Diagnostics

- `launchers/eval/run_scene_only_camera_screen120.sh`: quick camera-subset readout.
- `launchers/eval/eval_checkpoint_scan.sh`: archived diagnostic checkpoint scan. Cross-checkpoint summaries are diagnostic only.
- `generate_standard_report.py` and `generate_camera_report.py`: aggregate evaluator JSONL outputs.
- `select_stable_checkpoint.py`: selects one deployable checkpoint from validation reports.
- `prepare_libero_stability_subset.py`: writes the fixed validation subset manifest and task-index files.

## Pair Data

- `launchers/pair_data/launch_render_pair_future_6gpu.sh`: render same-state future-frame pairs.
- `launchers/pair_data/merge_pair_future_shards.sh`: merge rendered pair-data shard manifests.
- `render_libero_pair_future_frames.py`, `validate_pair_future_data.py`, `freeze_pair_holdout_manifest.py`, and related summarizers support pair-data construction and checks.

## Later Method Launchers

- `launchers/scvc/run_scene_only_scvc_train_formal_6gpu.sh`: cross-view consistency training launcher.
- `launchers/scvc/run_scvc_sanity_ladder.sh`: contract and sanity checks before launching those runs.
- `phase3_run_registry.json`: registry used by launcher generation for later method runs.

## Archive

- `launchers/archive/run_scene_only_pilot_train.sh` and `launchers/archive/run_pilot_orchestration.sh`: completed recipe-discrimination pilot launchers. Kept for reproducibility only.
- `phase1_pilot_run_registry.json`, `build_pilot_comparison.py`, and `aggregate_checkpoint_scan.py`: archived pilot diagnostics.

## Current Stable Baseline Commands

Manual training launch:

```bash
cd cosmos-policy
nohup bash cosmos_policy/experiments/robot/libero/launchers/baseline/run_scene_only_stable_train_6gpu.sh \
  > outputs/phase1/scene_only_stable_train_seed42.log 2>&1 &
```

After checkpoints exist, run validation checkpoint selection:

```bash
cd cosmos-policy
CKPT_ROOT=outputs/phase1/cosmos_policy/cosmos_v2_finetune/phase1_scene_only_stable_original_ema_b20_gacc6_lr5e5_seed42_15k/checkpoints \
ITERS="3000 6000 9000 12000 15000" \
OUT_ROOT=outputs/phase1/scene_only_stability_eval/seed42 \
bash cosmos_policy/experiments/robot/libero/launchers/baseline/run_scene_only_stability_eval.sh
```

The selector writes `checkpoint_selection.md`, `checkpoint_selection.csv`, and
`checkpoint_selection.json` under `OUT_ROOT`.
