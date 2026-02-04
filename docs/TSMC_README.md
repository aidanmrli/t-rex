Temporary TSMC README
=====================

This document accompanies the TSMC implementation for code review. It summarizes
the design decisions locked in during planning, how the system works end-to-end,
and the key files and entry points.

Overview
--------
Twisted SMC (TSMC) replaces the external PRM with a learned value head (twist)
that estimates the likelihood of eventual correctness from partial reasoning
traces. We run standard SMC with the base model as the proposal (Mode A) and
apply twist ratios to update particle weights.

Core decisions (locked for baseline)
------------------------------------
1) Value head outputs raw logits only.
   - Mapping to psi or log_psi is done in the wrapper.
   - Avoids double activation and keeps BCEWithLogits consistent.

2) Default twist_space is log_prob.
   - TSMC baseline sets log_space=True explicitly.
   - TwistedSMCConfig default remains log_space=False for back-compat.

3) Per-token heads only for baseline runs.
   - Linear/MLP heads are used in the baseline.
   - Attention-pooled head is supported but is not used in the baseline path.

4) Step-based resampling is the default.
   - Uses fixed K-token boundaries for SMC iterations.
   - "## Step N:" boundaries are optional and supported.

5) Resampling defaults to systematic with every-step resampling.
   - Matches the existing SMC steering baseline for comparability.

6) Final selection uses ORM if available.
   - Falls back to best-weight particle if ORM is disabled or unavailable.

7) Log-space weight updates are used when log_prob is selected.
   - Uses log-sum-exp normalization for stability.

Algorithm summary (Mode A)
--------------------------
Proposal: q_t = p_0 (explicit sampling distribution after applying temperature/top-p/top-k)

Twist update:
  - Prob space: w_t = w_{t-1} * (psi_t / psi_{t-1})
  - Log space:  log w_t = log w_{t-1} + (log_psi_t - log_psi_{t-1})

Because q_t = p_0, this is the correct weight ratio for the twisted target
without requiring partition function estimation (Mode B is deferred).

Implementation structure
------------------------
Models:
  - trex/models/value_head.py
      Linear, MLP, AttentionPooled heads (all output raw logits).
  - trex/models/twist_model.py
      Wraps HF model + value head; maps logits to psi or log_psi.

SMC:
  - trex/smc/twisted_smc.py
      Core TSMC logic with log-space handling and per-particle metadata.
  - trex/smc/tsmc_particle_filter.py
      LLM-based TSMC filter that reuses LLMParticleFilter generation and
      replaces PRM scores with twist ratios.

Training:
  - trex/training/value_trainer.py
      Self-distillation training loop for the value head.
  - trex/training/trajectory_buffer.py
      Stores trajectories and expands to (partial_trace, reward) pairs.
  - trex/training/train_value_head.py
      CLI entry point for training.

Baselines:
  - trex/baselines/tsmc_config.py
      Config dataclass for TSMC runs.
  - trex/baselines/tsmc_baseline.py
      Baseline runner (LLM generation + twist scorer + optional ORM).
  - trex/scripts/train_value_head.sh
  - trex/scripts/run_tsmc_baseline.sh

Key correctness and stability points
------------------------------------
- Raw logits only in the value head; twist mapping in the wrapper.
- Log-space weight updates with logsumexp normalization (log_prob mode).
- Per-particle metadata stores prev_value to keep lineage correct after resampling.
- p_0 is the explicit sampling distribution after temperature/top_p/top_k are applied.
- Finished particles retain their previous twist values to avoid spurious updates.

Training details (self-distillation)
------------------------------------
1) Generate K trajectories per prompt with the base model (proposal p_0).
2) Verify final answer to get R in {0,1}.
3) Assign R to all intermediate steps (Monte Carlo targets).
4) Train value head with BCEWithLogits (optional class balance, label smoothing).

This avoids external labels and matches the twist-space contract.

Usage (examples)
----------------
Train a value head:
  python -m trex.training.train_value_head \
    --base_model Qwen/Qwen2.5-7B \
    --value_head_type mlp \
    --dataset trex/data/gsm8k_platinum_train.jsonl \
    --num_rollouts_per_prompt 8 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --max_steps 1000 \
    --update_frequency 100 \
    --output_dir ./results/value_head_training

Run TSMC baseline:
  python -m trex.baselines.tsmc_baseline \
    --dataset_path trex/data/gsm8k_platinum_test.jsonl \
    --generator_model_path Qwen/Qwen2.5-7B \
    --value_model_path Qwen/Qwen2.5-7B \
    --value_head_path ./results/value_head_training/value_head.pt \
    --value_head_type mlp \
    --twist_space log_prob \
    --twist_mode value \
    --n_particles 16 \
    --max_smc_iterations 20 \
    --temperature 0.8 \
    --output_dir ./results/tsmc_baseline

Testing
-------
New unit tests:
  - trex/tests/test_models/test_value_head.py
  - trex/tests/test_models/test_twist_model.py
  - trex/tests/test_smc/test_tsmc_particle_filter.py
  - trex/tests/test_training/test_trajectory_buffer.py

Notes / limitations
-------------------
- Mode B (twisted proposal with Z_t estimation) is not implemented.
- Shared-weight scoring with vLLM is not implemented (separate HF scorer is used).
- Attention-pooled head is sequence-level; not used in baseline runs.
- For unbiased weights under non-base proposal sampling, corrections are not applied.
