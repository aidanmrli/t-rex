# TSMC Agnostic Plan Audit (2026-02-05)

## Scope
This audit compares the model-agnostic reproduction recipe in `docs/plans/TSMC_REPRODUCTION_PLAN.md` against the current repository implementation.

## What Is Implemented

### Stage 0: PRM800K download + schema exploration
Status: Implemented.

Evidence:
- PRM800K explore CLI args exist in `trex/prepare_math_datasets.py`.
- Explore stage writes schema/sample artifacts (`prm800k_schema.json`, `prm800k_samples.jsonl`).

Key refs:
- `trex/prepare_math_datasets.py:94`
- `trex/prepare_math_datasets.py:534`
- `trex/prepare_math_datasets.py:606`

### Stage 1: SFT generator with delimiter behavior
Status: Implemented.

Evidence:
- PRM800K formatter supports `--prm800k_use_pre_generated_steps`, correctness filtering, and `\n\n` normalization.
- SFT SLURM scripts train from base checkpoint with chat template off by default.
- Experiment log records completed PRM800K SFT jobs (`154122`, `154126`).

Key refs:
- `trex/prepare_math_datasets.py:353`
- `trex/prepare_math_datasets.py:441`
- `trex/scripts/tamia/run_sft_prm800k_h200.sh:51`
- `trex/scripts/tamia/run_sft_prm800k_h200.sh:63`
- `docs/EXPERIMENTS.md:49`

### Stage 2: Value-function learning from self-sampled trajectories
Status: Implemented (Mode-A baseline training path).

Evidence:
- Rollout collection from generator with multiple samples per prompt.
- Binary reward labels via verifier correctness.
- Value head training via `BCEWithLogitsLoss`.
- Delimiter-mode training filters to boundary-aligned states.

Key refs:
- `trex/training/value_trainer.py:98`
- `trex/training/value_trainer.py:131`
- `trex/training/value_trainer.py:194`
- `trex/training/value_trainer.py:213`
- `trex/training/train_value_head.py:24`

### Stage 3: TSMC inference loop
Status: Implemented (Mode-A base proposal + twist ratios).

Evidence:
- Twisted particle filter with ratio updates in prob/log-prob space.
- Delimiter-boundary step loop in TSMC path.
- Optional warmup and resampling controls.
- Optional final selection modes (`orm`, `majority_vote`, `twist_weight`).

Key refs:
- `trex/smc/tsmc_particle_filter.py:28`
- `trex/smc/tsmc_particle_filter.py:108`
- `trex/smc/tsmc_particle_filter.py:332`
- `trex/smc/tsmc_particle_filter.py:393`
- `trex/baselines/tsmc_config.py:41`

## Corrections Needed (High Priority First)

### 1) Inference script does not load a trained value head by default
Impact: High. Running the baseline script as-is can evaluate with a randomly initialized head.

Details:
- `run_tsmc_baseline.sh` does not pass `--value_head_path`.
- Baseline only loads trained head if `value_head_path` is set.

Refs:
- `trex/scripts/run_tsmc_baseline.sh:13`
- `trex/baselines/tsmc_baseline.py:139`

Fix:
- Require `--value_head_path` for TSMC runs intended as reproduction.
- Wire default to Stage-2 output path (or fail fast if missing).

### 2) First twist update is neutralized (no weighting at first scored step)
Impact: High. First resampling decision ignores twist signal.

Details:
- When `prev_value` metadata is missing, code sets `previous_values = current_values`, making delta/ratio = 0/1 on first update.

Refs:
- `trex/smc/tsmc_particle_filter.py:338`
- `trex/smc/tsmc_particle_filter.py:339`

Fix:
- Initialize `prev_value` at prompt state during `initialize()` (single prompt score copied across particles), or set first-step denominator to a shared prompt baseline.

### 3) Final selection default differs from agnostic recipe
Impact: Medium. Recipe says final vote among completed particles; default here is lineage twist-weight.

Refs:
- `docs/plans/TSMC_REPRODUCTION_PLAN.md:144`
- `trex/baselines/tsmc_config.py:63`
- `trex/scripts/run_tsmc_baseline.sh:20`

Fix:
- For strict reproduction mode, default `final_selection_mode=majority_vote`.

### 4) Default twist mode differs from recipe note
Impact: Medium. Recipe references sqrt-value weighting; implementation defaults to raw value.

Refs:
- `docs/plans/TSMC_REPRODUCTION_PLAN.md:132`
- `trex/baselines/tsmc_config.py:32`
- `trex/scripts/run_tsmc_baseline.sh:19`

Fix:
- Add a reproduction preset with `twist_mode=sqrt_value` and compare against `value` mode.

### 5) Resampling method default not aligned with recipe recommendation
Impact: Low-Medium. Recipe recommends stratified sampling; current default inherited from base config is systematic.

Refs:
- `docs/plans/TSMC_REPRODUCTION_PLAN.md:140`
- `trex/baselines/smc_config.py:46`

Fix:
- Set `resampling_method=stratified` in reproduction config/script.

### 6) Delimiter token supervision is boundary-aligned, but not exact-token guaranteed
Impact: Medium for faithful reproduction. Recipe warns about strict tokenizer-level delimiter semantics.

Details:
- Trainer uses `-1` (last token) for delimiter-ended states, not explicit second-newline token ID.
- No tokenizer sanity check that `\n\n` maps as expected.

Refs:
- `docs/plans/TSMC_REPRODUCTION_PLAN.md:150`
- `trex/training/value_trainer.py:136`
- `trex/models/twist_model.py:178`

Fix:
- Add tokenizer delimiter audit utility and compute explicit delimiter token index per state.

## What To Implement Next

1. Add a strict "reproduction config" for TSMC runs.
- Require `value_head_path`.
- Set `final_selection_mode=majority_vote`.
- Set `resampling_method=stratified`.
- Set `twist_mode=sqrt_value` (or run paired ablation value vs sqrt_value).

2. Fix first-step weighting.
- Initialize `prev_value` at prompt during `TSMCLLMParticleFilter.initialize()`.
- Add test asserting first scored step changes weights when particles diverge.

3. Implement delimiter-token alignment checks.
- Add utility to verify tokenizer behavior for `\n\n`.
- Store true boundary token index in trajectory triples instead of generic `-1`.
- Add tests for index correctness.

4. Add end-to-end smoke run and log in experiments doc.
- Train value head for a small subset.
- Run TSMC on a small eval slice with the strict reproduction config.
- Record command, metrics, and anomalies in `docs/EXPERIMENTS.md`.

## Verification Run Performed In This Audit
- Command:
  - `PYTHONPATH=/Users/aidanli/Documents/t-rex pytest trex/tests/test_models/test_value_head.py trex/tests/test_models/test_twist_model.py trex/tests/test_training/test_trajectory_buffer.py trex/tests/test_training/test_value_trainer.py trex/tests/test_smc/test_tsmc_particle_filter.py trex/tests/test_baselines/test_tsmc_baseline.py -q`
- Result: 19 passed.
