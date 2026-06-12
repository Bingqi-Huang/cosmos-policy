# Variance-governance switches (Task 3, 2026-06-12)

Three knobs prepared for the researcher to combine **after** the recipe pilot
(Arm A / Arm B) reports. **Implement-only — do NOT launch a formal run with these
until researcher-approved.** All three are already first-class config fields, so
they are pure Hydra overrides on the existing scene-only launcher
(`run_scene_only_pilot_train.sh` / `run_scene_only_train_formal_6gpu.sh`); no code
change is required.

Baseline (current 30K run and both pilot arms): `optimizer.lr=1e-4`,
`dataloader_train.batch_size=30`, 6 GPUs, `trainer.grad_accum_iter=1`
→ **effective batch = 180**, `model.config.ema.enabled=False`.

## 1. EMA

Config: `EMAConfig` (`_src/imaginaire/config.py:209`) — fields `enabled` (default
`False`), `beta` (default `0.9999`). The base Predict2 model already carries the
`net_ema` shadow weights (visible as `net_ema.*` keys at load), so enabling is a
config flip:

```
model.config.ema.enabled=True
model.config.ema.beta=0.9999          # 0.999 for a shorter half-life if 15K steps is too short for 0.9999
```

Official policy finetune ships `ema.enabled=False`; **enabling EMA is a
deliberate deviation from the official recipe and must be recorded in the
registry.** Rationale: at effective batch 180 (1/10.7 of official's 1920) the
single-checkpoint signal oscillates 6%↔50%; EMA is the cheapest variance
damper that does not change the optimization.

**Eval-with-EMA toggle (OPEN — verify before use):** training will save both
`net` and `net_ema`. Which weights the eval loads is NOT yet wired/verified for
the scene-only eval path (`model_loader.py` + `checkpoint.load_ema_to_reg`
currently governs only the *base*-checkpoint load, per the libero10
investigation). Before reporting any EMA number, confirm the eval reads
`net_ema`. This is the one item in Task 3 that may need a small code touch; it is
left as a flagged TODO rather than guessed.

## 2. Gradient accumulation (effective batch)

Config: `trainer.grad_accum_iter` (`_src/imaginaire/config.py:398`, wired in
`trainer.py`). Already exercised by the dry-run launcher (`grad_accum_iter=4`).

```
trainer.grad_accum_iter=2     # effective batch 360
trainer.grad_accum_iter=4     # effective batch 720
```

**Critical accounting:** grad accumulation does **not** reduce compute. It trades
optimizer-step count for batch size at *equal sample throughput*. So a budget
expressed in optimizer steps is no longer comparable across `grad_accum_iter`
values — convert to **samples presented** (see table). To keep a row comparable
to the baseline you must hold *samples*, not *steps*, fixed.

## 3. Learning rate

Config: `optimizer.lr` (base value `1e-4`).

```
optimizer.lr=2e-5
optimizer.lr=5e-5
```

Linear-scaling reference: official `1920 @ 1e-4`; the LR matched to our effective
batch 180 under linear scaling would be `≈ 1e-4 × 180/1920 ≈ 9.4e-6`. The current
run uses `1e-4` at batch 180 (same LR as official, 1/10.7 the batch) — i.e. the
baseline LR is ~10.7× the linear-scaling value, a plausible contributor to
checkpoint oscillation. `2e-5`/`5e-5` bracket the linear-scaling regime for a
future sweep. If LR and batch are changed together, re-derive from samples.

## Sample-presentation conversion table

Effective batch `B = 30 × num_gpus(6) × grad_accum_iter`. Samples presented
`= B × optimizer_steps`. Per-optimizer-step wall-clock scales ~linearly with
`grad_accum_iter` (more micro-batches per step); **wall-clock is governed by
samples, not steps** — accumulation buys larger batch at ~equal wall-clock for
equal samples.

| grad_accum_iter | eff batch | steps for 2.70M samples* | steps for 15K-baseline samples (=2.70M) | samples at 15K steps | ~wall-clock at 15K steps** |
|---:|---:|---:|---:|---:|---:|
| 1 (baseline) | 180 | 15000 | 15000 | 2.70M | ~16 h |
| 2 | 360 | 7500 | 7500 | 5.40M | ~32 h |
| 4 | 720 | 3750 | 3750 | 10.8M | ~63 h |

\* "2.70M samples" = the Arm-A/B pilot budget (15000 × 180). The current 30K
single-arm run = 30000 × 180 = **5.40M** samples ( = official's 76.8M × 7.0%).
\*\* at the measured ~3.8 s per *baseline* optimizer step; an accum-k step costs
~k× that, so **equal-sample wall-clock is ~constant across k** (the table's
"15K steps" column processes k× more samples, hence k× the time).

**Researcher decision rule:** if a future run changes `grad_accum_iter`, set its
`trainer.max_iter` from a *samples* target (e.g. to match the 5.40M-sample 30K
baseline at eff-batch 720 → `max_iter = 5.40M / 720 = 7500`), and record both the
step count and the sample count in the registry.

## Smoke status

- **Compose smoke (CPU, no GPU) — recommended before any launch:**
  ```
  cd cosmos-policy && uv run --extra cu128 --group libero --python 3.10 \
    python -c "from cosmos_policy.config.config import ... ; <load_config with the three overrides>; print(ema, grad_accum_iter, lr)"
  ```
  (use the same `load_config` path the plan-risk hardening smokes used; do NOT set
  `JOB_NAME` in a compose smoke.) Validates the overrides resolve to the intended
  attrs. The three fields are existing dataclass attrs (`EMAConfig.enabled/beta`,
  `TrainerConfig.grad_accum_iter`, `optimizer.lr`), so failure risk is low.
- **GPU step-smoke (1–2 iters, finite loss, checkpoint save) — DEFERRED.** Needs
  GPUs, which are fully occupied by the pilot for ~32 h. Run in the post-pilot
  free window via `run_scene_only_train_step_dryrun.sh` with the chosen overrides
  added (it already sets `grad_accum_iter=4`, `max_iter=1`, `save_iter=999999`).
```
