# Local Patches

This repo is used as the working fork for the SCVC project.  The following
local changes are intentional and should be reviewed before any upstream rebase.

## NCCL Runtime Override (Blackwell fix, 2026-06-11)

- `pyproject.toml` + `uv.lock`
  - torch 2.7.0's wheel metadata hard-pins `nvidia-nccl-cu12==2.26.2`, which is broken on
    RTX PRO 6000 Blackwell (CUDA illegal memory access in P2P/SHM transports; corrupted
    NCCL object collectives). Forced to **2.30.7** via `[tool.uv] override-dependencies`
    (ABI-stable `libnccl.so.2` drop-in). Restores default fast transports: training goes
    5.9 → 3.4 s/iter (~1.73×), GPUs at full 600 W. Full record incl. failed attempts:
    `AGENT/training_speed_optimization.md`.
- `cosmos_policy/experiments/robot/libero/run_scene_only_train_formal_6gpu.sh`,
  `run_scene_only_scvc_train_formal_6gpu.sh`
  - Default to NCCL default transports; the 2.26.2-era conservative socket workaround is
    preserved behind `NCCL_CONSERVATIVE=1`.

## Blackwell / DCP Stability

- `cosmos_policy/_src/predict2/checkpointer/dcp.py`
  - Adds Gloo-backed DCP metadata collectives for checkpoint save/load.
  - Keeps training collectives on NCCL while avoiding NCCL object-collective
    failures observed on the 6x RTX PRO 6000 Blackwell host.

## Scene-Only / SCVC Pipeline

- `cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py`
  - Adds scene-only P2 config and SCVC scene-only paired-data config.
  - Pair dataset node wires `rollout_data_dir` (Plan A rollout mixture, 2026-06-11).
- `cosmos_policy/datasets/libero_pair_dataset.py`
  - Adds Phase-2 manifest-backed scene-only pair dataset returning `video_pair`.
  - Plan A rollout mixture (2026-06-11): embedded rollout-only `LIBERODataset`
    (`demonstration_sampling_prob=0.0`), 0.5:0.5 pair:rollout via index ranges;
    rollout samples pass through single-view with `pair_valid=0`.
  - Pair branches use photometric-only augmentation (B6b: spatial off,
    photometric independent across views).
- `cosmos_policy/datasets/dataset_utils.py`
  - Adds `photometric_only` switch to `apply_image_aug` / `preprocess_image`
    (skips random resized crop + rotation; keeps color jitter). Default
    behavior unchanged for all existing callers.
- `cosmos_policy/models/scvc_policy_video2world_model.py`
  - Adds paired SCVC training subclass with shared-noise, wrong-state,
    wrong-coordinate, and wrong-noise switches.
  - 2026-06-11 review fixes: CV loss normalized identically to the FM loss
    (same per-element coefficient, injected before `loss_scale`) so nominal
    `lambda_cv` equals the Lemma-1/Prop.-2 λ exactly; derangement via rejection
    sampling + conjugated-cycle fallback (no fixed points); VAE encode hoisted
    out of the K_s noise-draw loop; CV mask and shrinkage-ratio monitoring
    restricted to valid matched demo pairs on both comparison arms.
- `cosmos_policy/experiments/robot/libero/*`
  - Adds Row-1 eval launchers/reports, screen120 subset prep, pair future-frame
    renderer/merge/validation, and SCVC training launcher.
  - `render_libero_pair_future_frames.py` rejection-samples training cameras
    against the 1,599 LIBERO-Plus benchmark task poses (450 unique 5-tuples) —
    standing rule 4 train/eval camera disjointness by construction.
  - `validate_pair_future_data.py` audits manifest cameras against the
    benchmark poses and fails on any collision.
  - `freeze_pair_holdout_manifest.py` snapshots the val pair manifest with a
    SHA256 sidecar as the frozen A2 held-out set (LOCKED DECISION 8).
