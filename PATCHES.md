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
  - SCVC experiment declares `{"override /model": "scvc_policy_fsdp"}` (2026-06-11 audit):
    without it the config cannot compose — Hydra struct mode rejects `lambda_cv`/`cv_*`
    against the base `CosmosPolicyVideo2WorldConfig`-typed `/model` node.
- `cosmos_policy/config/defaults/model.py`
  - Registers the typed `scvc_policy_fsdp` model-group node
    (`SCVCPolicyVideo2WorldModel` + `SCVCPolicyVideo2WorldConfig`) used by the SCVC
    experiment above. Verified by a full `load_config` compose.
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
  - 2026-06-11 audit fixes: `generate_phase3_launchers.py` wires the registry seed
    into the launch line (`trainer.seed` + `dataloader_train.sampler.seed`; without
    this all multi-seed rows were identical runs); `validate_phase3_registry.py`
    requires an integer seed per run; `check_scvc_batch_contract.py` derangement
    check replicates the model's conjugated-cycle fallback instead of a strawman;
    `summarize_pair_manifest.py` reads `task_name`; `evaluate_scvc_shrinkage.py`
    skips fully-floor-excluded sigma bins in the aggregate (NaN poisoning).
- 2026-06-12 plan-risk hardening (researcher-approved, 7-point external review):
  - `scvc_policy_video2world_model.py`: `cv_frame_set` value `full` renamed to
    `invariant_plus_fscene` (= invariant block ∪ {future-scene}; the A2
    wrong-coordinates set — blank/conditioning frames are never in any CV set;
    passing `full` now raises with a migration hint). W&B key
    `scvc_cv_frame_set_full` → `scvc_cv_includes_fscene`. New init guard:
    refuses any loss/input mask flag on or `action_loss_multiplier≠1`
    (Prop.-2 FM-anchor + λ-bookkeeping precondition). New one-time first-step
    contract check: branch latent shapes + all `*_latent_idx` fields identical,
    shared (σ, n) bitwise equal post-split, invariant target frames
    (action/proprio/fproprio/value) bitwise-shared across branches for valid
    matched pairs.
  - `phase3_run_registry.json`: A2 rows renamed `a2_full_*` → `a2_fscene_*`,
    `cv_frame_set` → `invariant_plus_fscene` (nothing had run yet).
  - `validate_phase3_registry.py`: enum updated; new ERROR if an
    `E3_A2_wrong_coordinates` run does not use `invariant_plus_fscene`.
  - `evaluate_scvc_shrinkage.py`: checkpoint-load override uses the new enum.
  - `run_scvc_sanity_ladder.sh`: rung label updated.
  - `validate_pair_future_data.py`: new same-state pair guarantees —
    `--state-hash-check-rows` (default 256) recomputes state/action/robot-state
    hashes from the HDF5 source and compares to the manifest (end-to-end form of
    "assert value_0 == value_p"); `pair_type` semantics checked (matched ⇒ no
    differing `*_hash_b`; wrong_state ⇒ `state_hash_b` present, differing,
    `pair_confidence=0`). Functionally verified on synthetic HDF5 (clean pass;
    injected hash corruption and diluted wrong-state control both detected).
  - Verified: py_compile all touched files; registry validator pass (15 runs,
    no warnings); full `load_config` compose with
    `CV_FRAME_SET=invariant_plus_fscene` (state_t=7, chunk 25 intact); the
    renamed-enum ValueError fires on the real config class.
