# Experiments for T-REX

**Last Updated:** 2026-02-05

**NOTE:** We should always update this document once we have planned an experiment and it works. Then, when the experiment has completed, we should immediately update this document with the results. Status entries are date-stamped; refresh them after each re-run.

---

## Current Experiment State (Summary) (2026-02-05)

**Solved so far:**
- Established BoN, GRPO-eval, and PPO-eval baselines with reproducible outputs.
- Confirmed large search headroom: BoN MATH-500 `pass@1=53.3%` to `pass@32=81.6%`.
- Found viable token-resampling regime: K=256 reaches `87.68%` on GSM8K Platinum.

**Outstanding problems / hypotheses:**
- Step-based SMC baseline is still invalid due to truncation/template-answer leakage.
- Hypothesis to test: finish-reason-aware step handling now allows full completions and valid `\boxed{...}` extraction.
- Hypothesis to test: high-K (>=256) or adaptive resampling preserves completion without leakage under stricter extraction.

**What to do now:**
- Re-run step-based SMC from a cleared checkpoint and verify leakage is fixed.
- Re-run token-resampling confirmation at K>=256 (and optionally adaptive ESS) with the same evaluation protocol.
- Compare valid SMC numbers directly against BoN/GRPO/PPO.
- Start evaluation runs for the new PRM800K SFT checkpoints (`job_154122`, `job_154126`) on GSM8K/MATH-500.

**Future work:**
- Move from baseline SMC to TSMC/T-REX comparisons once step-based SMC is valid.
- Add compute-efficiency comparisons (accuracy vs sample/token budget) across BoN, RL baselines, and SMC variants.

**Latest cluster submissions (2026-02-05):**
- Job `153914`: step-based SMC baseline re-run submitted with cleared checkpoint.
- Job `153915`: token sweep submitted with `K_VALUES="256 384 512"` and `TOTAL_TOKEN_BUDGET=2048`.
- Job `153916`: token sweep submitted with `K_VALUES="256 384 512"`, `TOTAL_TOKEN_BUDGET=2048`, `RESAMPLING_STRATEGY=ess_adaptive`.
- Submission hiccup: initial `Permission denied` on `run_smc_token_sweep_array.sh`; fixed by `chmod +x`.
- Job `154122`: PRM800K SFT (`run_sft_prm800k_h200.sh`) completed on `tg10502` (H200x8, `00:47:15`, `ExitCode=0:0`).
- Job `154126`: PRM800K SFT (`run_sft_prm800k_h100.sh`) completed on `tg11101` (H100x4, `01:28:23`, `ExitCode=0:0`).

**Outputs to check when jobs finish:**
- Job `153914` logs: `/scratch/l/liaidan/t-rex/slurm/smc_153914.out` and `/scratch/l/liaidan/t-rex/slurm/smc_153914.err`
- Job `153914` results: `/scratch/l/liaidan/t-rex/results/smc_baseline/summary.json` and `/scratch/l/liaidan/t-rex/results/smc_baseline/generations/generations.jsonl`
- Job `153915` logs: `/scratch/l/liaidan/t-rex/slurm/smc_token_sweep_153915_*.out` and `/scratch/l/liaidan/t-rex/slurm/smc_token_sweep_153915_*.err`
- Job `153915` results: `/scratch/l/liaidan/t-rex/results/smc_token_sweep/job_153915/k256/summary.json`, `/scratch/l/liaidan/t-rex/results/smc_token_sweep/job_153915/k384/summary.json`, `/scratch/l/liaidan/t-rex/results/smc_token_sweep/job_153915/k512/summary.json`
- Job `153916` logs: `/scratch/l/liaidan/t-rex/slurm/smc_token_sweep_153916_*.out` and `/scratch/l/liaidan/t-rex/slurm/smc_token_sweep_153916_*.err`
- Job `153916` results: `/scratch/l/liaidan/t-rex/results/smc_token_sweep/job_153916/k256/summary.json`, `/scratch/l/liaidan/t-rex/results/smc_token_sweep/job_153916/k384/summary.json`, `/scratch/l/liaidan/t-rex/results/smc_token_sweep/job_153916/k512/summary.json`

---

## PRM800K SFT Completion Check (2026-02-05)

**Command run:**
```bash
sbatch trex/scripts/tamia/run_sft_prm800k_h200.sh   # job 154122
sbatch trex/scripts/tamia/run_sft_prm800k_h100.sh   # job 154126
```

**Anything went wrong:**
- Both jobs completed successfully with `State=COMPLETED` and `ExitCode=0:0` (no OOM, no hard failure).
- Both logs emitted the standard PyTorch distributed warning: `destroy_process_group() was not called before program exit`.
- Job `154122` additionally emitted a post-training `TCPStore recvValue failed` warning + stack trace on rank 5 after completion; all ranks still exited successfully and checkpoint artifacts are complete.

### Results / Metrics

| Job | Script | Resources | Elapsed | Final `gpt_loss` | Train Progress | Output Artifact |
|-----|--------|-----------|---------|------------------|----------------|-----------------|
| `154122` | `trex/scripts/tamia/run_sft_prm800k_h200.sh` | `gpu:h200:8`, `cpu:64`, `mem:950G` | `00:47:15` | `0.306` | `11491/11491` steps (`1/1` epoch) | `/scratch/l/liaidan/t-rex/results/prm800k_sft/job_154122/ckpt` (15G) |
| `154126` | `trex/scripts/tamia/run_sft_prm800k_h100.sh` | `gpu:h100:4`, `cpu:48`, `mem:480G` | `01:28:23` | `0.256` | `22982/22982` steps (`1/1` epoch) | `/scratch/l/liaidan/t-rex/results/prm800k_sft/job_154126/ckpt` (15G) |

Common run config from logs:
- Dataset: `trex/data/prm800k_sft_train.jsonl` (`num_rows=91930`)
- Hyperparameters: `max_epochs=1`, `lr=5e-6`, `micro_train_batch_size=1`, `train_batch_size=128`, `max_len=2048`, `zero_stage=2`, `bf16`, gradient checkpointing on

### Interpretation (Experiment Context)

- The PRM800K SFT stage is now unblocked: both hardware variants produced complete checkpoints ready for downstream eval.
- Runtime scaled as expected with fewer GPUs (`154126` took ~1.87x longer and ran ~2x micro-steps), so loss numbers are not a strict apples-to-apples quality comparison.
- Immediate next decision point is model selection by evaluation quality and efficiency, not training completion status.
- Recommended follow-up: run the same GSM8K/MATH-500 eval harness on both checkpoints and choose a default SFT checkpoint for subsequent SMC/TSMC experiments.

---

## Results Folder Audit (2026-02-04)

**Command run:**
```bash
find results -name 'summary.json' -o -name 'sweep_results.json' | sort
```

**Anything went wrong:**
- No I/O or parse errors while reading result artifacts.
- The experiment issue is unchanged: step-based SMC outputs are still truncated and leak the template answer.

### Results / Metrics

| Artifact | Timestamp | Key Results |
|----------|-----------|-------------|
| `results/bon_baseline/Qwen2.5-7B/math_n32/summary.json` | 2026-01-27 22:43:12 | best_temp=0.8, pass@1=53.3%, pass@32=81.6% |
| `results/eval_grpo/Qwen2.5-7B/gsm8k_trained_n8/summary.json` | 2026-01-28 17:32:36 | best_temp=1.0, pass@1=38.6%, pass@16=98.3% |
| `results/eval_ppo/Qwen2.5-7B/gsm8k_trained_n1/summary.json` | 2026-01-28 17:34:51 | best_temp=1.0, pass@1=25.2%, pass@16=96.6% |
| `results/smc_baseline/summary.json` | 2026-02-02 19:55:17 | accuracy=0.08% (1/1209), avg ORM=0.9970, avg time=0.592s |

**Token-resampling sweep (`results/smc_token_sweep/job_152924`)**

| K | Accuracy | Correct | Avg Time/Problem | `extracted_answer == "answer"` | Missing `extracted_answer` |
|---|----------|---------|------------------|---------------------------------|----------------------------|
| 16 | 0.08% | 1/1209 | 0.209s | 1208/1209 | 0 |
| 32 | 0.08% | 1/1209 | 0.276s | 1208/1209 | 0 |
| 64 | 0.17% | 2/1209 | 0.420s | 1207/1209 | 0 |
| 128 | 17.54% | 212/1209 | 0.711s | 995/1209 | 1 |
| 256 | 87.68% | 1060/1209 | 1.222s | 102/1209 | 23 |

### Interpretation (Experiment Context)

- No new experiment outputs were added after 2026-02-02; current conclusions still hold.
- BoN, GRPO-eval, and PPO-eval baselines are complete and internally consistent with prior reported numbers.
- Step-based SMC baseline remains invalid for comparison due to template-answer leakage and premature truncation.
- Token-resampling remains promising only at larger chunks (K=256 in current sweep).

---

## Baseline Experiment Specifications

This section provides detailed technical specifications for reproducing the baseline experiments. These baselines establish the performance of standard methods before introducing T-REX's advanced inference techniques.

### Best-of-N Baseline (Brute-Force Rejection Sampling)

**Purpose:** Establish an upper bound on what can be achieved through pure sampling at inference time. This baseline answers: "Given a fixed compute budget (N samples), what accuracy can we achieve?"

**Script:** `trex/scripts/tamia/run_bon_baseline.sh`
**Implementation:** `trex/baselines/best_of_n_baseline.py`

#### Algorithm

```
Input: Dataset D, Model M, N samples, Temperatures T
Output: pass@k metrics for k ∈ {1, 2, 4, 8, ..., N}

Phase 1: Temperature Sweep (find optimal temperature)
  For each temperature t in T:
    For first K problems (sweep_size=100):
      Generate N samples at temperature t
      Verify each sample against ground truth
      Compute pass@N for this temperature
  Select t* = argmax_t pass@N

Phase 2: Full Evaluation (evaluate entire dataset)
  For each problem in D:
    Generate N samples at temperature t*
    Verify each sample against ground truth
    For each k in {1, 2, 4, 8, ..., N}:
      Compute pass@k using unbiased estimator
```

#### pass@k Metric (Unbiased Estimator)

The pass@k metric estimates the probability that at least one of k samples is correct, when we have n total samples with c correct:

```
pass@k = 1 - C(n-c, k) / C(n, k)
```

Where C(a,b) is the binomial coefficient. This is an unbiased estimator that avoids selection bias from "best-of" selection.

**Interpretation:**
- `pass@1`: Single-shot accuracy (what you'd get with greedy decoding at this temperature)
- `pass@N`: Upper bound with N samples (the "oracle" that picks the correct answer if any exists)
- The gap between pass@1 and pass@N quantifies the "search opportunity" that SMC/TSMC can exploit

**Greedy sampling guard (vLLM):**
vLLM requires `n=1` when `temperature=0`. The Best-of-N harness now enforces `effective_n = 1` for greedy runs,
so sweeps that include `temperature=0` are safe.

#### Verification Pipeline

The `MathVerifier` class (`trex/eval/math_verifier.py`) verifies answers through multiple backends:

1. **Answer Extraction:** Extract from `\boxed{}`, `####`, or last number
2. **HuggingFace math_verify:** Symbolic math comparison (handles LaTeX, fractions, etc.)
3. **SymPy:** Symbolic equality for algebraic expressions
4. **Numeric comparison:** Floating-point tolerance for decimal answers

#### Configuration for Reproducibility

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | `Qwen/Qwen2.5-7B` | Strong base model, widely available |
| Dataset | MATH-500 or GSM8K | Standard math reasoning benchmarks |
| N samples | 32 | High enough to see diminishing returns |
| Temperatures | [0.6, 0.8, 1.0, 1.2] | Range from conservative to exploratory |
| Sweep size | 100 | Sufficient for reliable temperature selection |
| Max tokens | 2048 | Long enough for multi-step reasoning |
| Chat template | False (base model) | Base models don't use chat templates |

#### Checkpointing

The script supports SLURM preemption through:
- Temperature sweep checkpoints (resume after each temperature)
- Evaluation chunk checkpoints (every 50 problems by default)
- Signal handlers for graceful shutdown (SIGUSR1, SIGTERM)

---

### GRPO Baseline (Group Relative Policy Optimization)

**Purpose:** Train the model to improve mathematical reasoning through reinforcement learning. GRPO is a simplified PPO variant that uses group normalization instead of a learned critic.

**Script:** `trex/scripts/tamia/run_grpo_baseline.sh`
**Reward Function:** `trex/baselines/grpo_reward_func.py`
**Framework:** OpenRLHF's `train_ppo_ray.py` with `--advantage_estimator group_norm`

#### Algorithm (GRPO)

GRPO differs from PPO by eliminating the critic network. Instead of GAE (Generalized Advantage Estimation), it uses group normalization:

```
Input: Policy π_θ, Dataset D, Group size G, KL coefficient β
Output: Trained policy π_θ'

For each training step:
  1. Sample batch of prompts from D

  2. For each prompt p:
     Generate G responses: {r_1, ..., r_G} ~ π_θ(·|p)
     Compute rewards: {R_1, ..., R_G} using reward_func

  3. Compute advantages via GROUP NORMALIZATION:
     μ = mean({R_1, ..., R_G})
     σ = std({R_1, ..., R_G})
     A_i = (R_i - μ) / (σ + ε)

     This replaces PPO's value function baseline V(s) with
     the empirical mean of the group.

  4. Update policy using PPO objective with KL penalty:
     L = E[min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t)] - β * KL(π_θ || π_ref)

     Where r_t = π_θ(a|s) / π_old(a|s)
```

**Key Insight:** Group normalization works because within a group of responses to the same prompt, the rewards are directly comparable. A response is "good" if it scores higher than its siblings.

#### Reward Function Design

The reward function (`grpo_reward_func.py`) computes:

```python
def reward_func(queries, prompts, labels):
    for query, prompt, label in zip(queries, prompts, labels):
        response = extract_response(query, prompt)
        correct = verifier.verify(response, label)
        has_boxed = extract_last_boxed(response) is not None

        # Reward shaping:
        if correct:
            reward = +1.0      # Correct answer
        elif not has_boxed:
            reward = -1.0      # Format penalty (no \boxed{})
        else:
            reward = 0.0       # Incorrect but properly formatted

    return {"rewards": tensor, "scores": tensor}
```

**Reward Shaping Rationale:**
- `+1.0` for correct: Strong positive signal
- `-1.0` for missing `\boxed{}`: Format penalty encourages structured output
- `0.0` for wrong but formatted: Neutral, lets group normalization work

#### Efficiency Tracking

The reward function includes an `EfficiencyTracker` singleton that logs:
- Cumulative samples and tokens processed
- Batch accuracy over time
- Time-to-threshold milestones (e.g., when 50%, 60%, 70% accuracy is reached)
- Samples/tokens needed to reach each threshold

This data is critical for comparing GRPO against T-REX methods.

#### Configuration for Reproducibility

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | `Qwen/Qwen2.5-7B` | Must match Best-of-N for fair comparison |
| Dataset | GSM8K train (7,473 problems) | Sufficient for RL training |
| Group size (N) | 8 | Balance between variance and compute |
| KL coefficient | 0.001 | Low penalty, allows exploration |
| KL estimator | k3 | More accurate KL estimate |
| Learning rate | 5e-7 | Conservative to avoid instability |
| Epochs | 1 | Single pass through data |
| Train batch size | 128 | Total samples per gradient update |
| Micro train batch | 4 | Per-GPU batch (memory constrained) |
| Rollout batch size | 128 | Samples per rollout phase |
| Micro rollout batch | 16 | Per-GPU rollout batch |
| Temperature | 1.0 | Standard sampling for RL |
| vLLM GPU utilization | 0.6 | Safe maximum with `--colocate_all_models` |

#### Memory Management

The script uses `--colocate_all_models` which shares GPU memory between:
- vLLM inference engine (for rollout generation)
- DeepSpeed training (for gradient updates)

This requires careful memory management:
- vLLM sleeps during training (`--vllm_enable_sleep`)
- DeepSpeed sleeps during generation (`--deepspeed_enable_sleep`)
- Conservative GPU utilization (0.6) prevents OOM on wake-up

#### Automatic Checkpointing and Requeue

The script handles SLURM's 24-hour time limit through:
1. Checkpoints saved every 50 steps to `$CKPT_DIR`
2. SIGUSR1 handler submits continuation job 120s before timeout
3. `--load_checkpoint` flag auto-detected from existing checkpoints
4. Completion marker prevents unnecessary requeues after training finishes

---

### GRPO Evaluation

**Purpose:** Evaluate the trained GRPO model using the same Best-of-N harness, enabling direct comparison.

**Script:** `trex/scripts/tamia/eval_grpo_baseline.sh`

#### Evaluation Protocol

```
1. Load latest checkpoint from GRPO training
2. Run Best-of-N evaluation with:
   - N = 16 samples per problem
   - Temperatures: [0.0 (greedy), 0.6, 1.0]
   - Dataset: GSM8K Platinum test set
3. Report pass@k for k ∈ {1, 2, 4, 8, 16}
```

**Why multiple temperatures?**
- Temperature 0.0 (greedy): Measures deterministic performance
- Temperature 0.6-1.0: Tests sampling diversity after training
- Comparing trained vs. base model at same temperature shows improvement

#### Expected Metrics to Compare

| Metric | Base Model | GRPO Trained | Interpretation |
|--------|------------|--------------|----------------|
| pass@1 (greedy) | X% | Y% | Single-shot improvement from training |
| pass@1 (T=0.6) | A% | B% | Sampling improvement |
| pass@16 | P% | Q% | Upper bound shift |
| Gap (pass@16 - pass@1) | P-X | Q-Y | Remaining search opportunity |

#### Results (2026-01-28, GSM8K Platinum Test)

**Best temperature:** 1.0 (N=16)

| Metric | Score |
|--------|-------|
| pass@1 | 38.6% |
| pass@2 | 61.1% |
| pass@4 | 82.7% |
| pass@8 | 94.7% |
| pass@16 | 98.3% |

---

## Script Optimization and Validation (2026-01-27)

### Changes Made

**GPU Utilization Optimizations:**

| Script | Setting | Before | After | Impact |
|--------|---------|--------|-------|--------|
| `run_grpo_baseline.sh` | `VLLM_GPU_UTIL` | 0.5 | 0.85 | +70% GPU memory usage during generation |
| `run_grpo_baseline.sh` | `MICRO_TRAIN_BATCH_SIZE` | 4 | 8 | +100% training throughput |
| `run_grpo_baseline.sh` | `MICRO_ROLLOUT_BATCH_SIZE` | 16 | 32 | +100% rollout throughput |
| `run_grpo_baseline.sh` | `--packing_samples` | Not used | Added | Reduces padding waste |
| `eval_grpo_baseline.sh` | `TP_SIZE` | 2 | 4 | Uses all 4 GPUs instead of 2 |
| `eval_grpo_baseline.sh` | `GPU_MEM_UTIL` | 0.90 | 0.95 | +5% GPU memory usage |

**Resource Configuration Fixes:**

| Script | Setting | Before | After | Reason |
|--------|---------|--------|-------|--------|
| All scripts | `--mem` | 512G | 480G | H100 nodes only have 500GB |
| All scripts | `--cpus-per-task` | 64 | 48 | H100 nodes have 48 CPUs |
| All scripts | CUDA module | Not loaded | `cuda/12.6` | Required for DeepSpeed |

**Offline Mode Configuration:**

Added for HPC clusters without internet on compute nodes:
- `export HF_HUB_OFFLINE=1` - Use cached models only
- `export WANDB_MODE=offline` - Log locally without network
- `export WANDB_API_KEY=...` - Added to all scripts

**Bug Fixes:**
- `run_bon_baseline.sh`: Removed accidental text insertion on lines 13-14

### Validation Status

**Status Note (last confirmed 2026-02-04):** Update this table after each new run.

| Script | Submitted | Started | Initialization | Training | Status |
|--------|-----------|---------|----------------|----------|--------|
| `run_bon_baseline.sh` | Yes | Yes | Success | **COMPLETE** | pass@32=81.6% on MATH-500 |
| `run_grpo_baseline.sh` | Yes | Yes | Success | **COMPLETE** | Model artifacts saved; metrics not present in `results/grpo_baseline/` |
| `eval_grpo_baseline.sh` | Yes | Yes | Success | **COMPLETE** | best_temp=1.0; pass@16=98.3% on GSM8K Platinum |
| `eval_ppo_baseline.sh` | Yes | Yes | Success | **COMPLETE** | best_temp=1.0; pass@16=96.6% on GSM8K Platinum |

### Final Configuration Notes

**Memory constraints with `--colocate_all_models`:**
- `VLLM_GPU_UTIL=0.6` is the safe maximum (0.85 causes OOM)
- `MICRO_TRAIN_BATCH_SIZE=4` and `MICRO_ROLLOUT_BATCH_SIZE=16` are conservative but stable

**Local model path required:**
- Scripts now use local cached model path instead of HuggingFace ID
- This avoids network calls even when `TRANSFORMERS_OFFLINE=1` is set

---

### SMC Steering Baseline

**Purpose:** Evaluate standard SMC steering with PRM guidance as an inference-time compute scaling baseline. This establishes what SMC can achieve before adding twisted targets or parallel tempering.

**Script:** `trex/scripts/tamia/run_smc_baseline.sh`
**Implementation:** `trex/baselines/smc_steering_baseline.py`, `trex/smc/llm_particle_filter.py`

#### Algorithm (SMC Steering)

```
Input: Problem prompt, Generator model M, Reward model R (PRM/ORM), N particles
Output: Best solution with highest ORM score

1. Initialize N particles with the problem prompt (with system prompt)
2. For each SMC iteration:
   a. Expand: Generate next reasoning step for each particle (stop at "## Step")
   b. Score: Get PRM score for latest step
   c. Update weights: w_t = w_{t-1} × PRM_score
   d. Resample: Duplicate high-weight particles, prune low-weight ones
3. Select best particle using ORM (final outcome score)
```

#### Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Generator Model | `Qwen/Qwen2.5-7B` | Match other baselines |
| Reward Model | `Qwen/Qwen2.5-Math-PRM-7B` | Process & outcome reward scoring |
| Dataset | GSM8K Platinum test (1,209 problems) | Standard eval set |
| N particles | 16 | Balance compute vs. diversity |
| Max SMC iterations | 20 | Per-particle generation cycles |
| Max reasoning steps | 15 | Cap on "## Step N:" patterns |
| ESS threshold | 0.5 | Trigger for resampling |
| Temperature | 0.7 | Moderate exploration |
| Resampling strategy | "every_step" | Standard SMC steering |

---

### SMC Token-Resampling Sweep (K)

**Purpose:** Find the best token chunk size K for resampling in token-interval SMC (no step template enforced).

**Scripts:**
- `trex/scripts/tamia/run_smc_token_sweep.sh` (single job, runs all K sequentially)
- `trex/scripts/tamia/run_smc_token_sweep_array.sh` (array job, one K per task)

**Key settings:**
- `K_VALUES`: space-separated list of K values (default: `32 64 128 256 512`)
- `TOTAL_TOKEN_BUDGET`: fixed per-particle budget to keep sweeps comparable (default: `2048`)
- `RESAMPLING_STRATEGY`: `every_step` or `ess_adaptive`

**Array usage example:**
```
K_VALUES="32 64 128 256 512" sbatch trex/scripts/tamia/run_smc_token_sweep_array.sh
```

**Recent runs (2026-02-02):**
```
sbatch trex/scripts/tamia/run_smc_token_sweep.sh h100 K_VALUES="64 128" TOTAL_TOKEN_BUDGET=512
sbatch trex/scripts/tamia/run_smc_token_sweep.sh
```

**Results (Job 152924, GSM8K Platinum Test, N=16):**

| K (tokens) | Accuracy | Correct | Avg Time/Problem |
|------------|----------|---------|------------------|
| 16 | 0.08% | 1/1209 | 0.209s |
| 32 | 0.08% | 1/1209 | 0.276s |
| 64 | 0.17% | 2/1209 | 0.420s |
| 128 | 17.54% | 212/1209 | 0.711s |
| 256 | 87.68% | 1060/1209 | 1.222s |

**Template-answer leakage (extracted_answer == "answer"):**
- K <= 64: 1207-1208/1209
- K = 128: 995/1209
- K = 256: 102/1209 (plus 23 missing extracted_answer)

---

## Experiment Log (Reverse Chronological)

### 2026-02-02: Experiment 4 - SMC Steering Baseline on GSM8K (Step-Based)

**Status:** RE-RUN REQUIRED (latest run is invalid)

**Configuration:**
- Generator: `Qwen/Qwen2.5-7B`
- Reward model: `Qwen/Qwen2.5-Math-PRM-7B`
- Dataset: GSM8K Platinum test (1,209 problems)
- N particles: 16
- Max SMC iterations: 20
- Max reasoning steps: 15
- Temperature: 0.7

**Latest run metrics (invalid):**

| Metric | Value |
|--------|-------|
| Reported Accuracy | 0.08% (1/1209) |
| Extracted Answer | "answer" for 1,208/1,209 problems |
| Average ORM Score | 0.9970 |
| Avg Time/Problem | 0.592s |

**Bug context:**
- Root cause observed in outputs: step-based generations still stop before final `\boxed{...}` completion.
- Existing fix already implemented in `trex/smc/llm_particle_filter.py`:
1. `include_stop_str_in_output=False`
2. `finish_reason`-aware continuation/termination
3. `inject_next_step_headers()` after scoring
4. EOS-based particle finishing

**Next run command path:**
1. `rm /scratch/l/liaidan/t-rex/results/smc_baseline/checkpoint.json`
2. `./trex/scripts/tamia/run_smc_baseline.sh h100`

**Output location:** `/scratch/l/liaidan/t-rex/results/smc_baseline/`

---

### 2026-02-02: SMC Token-Resampling Sweep (Job 152924)

**Status:** COMPLETE

**Configuration:**
- Dataset: GSM8K Platinum test
- N particles: 16
- Sweep: K in `{16, 32, 64, 128, 256}`

**Results:**

| K (tokens) | Accuracy | Correct | Avg Time/Problem |
|------------|----------|---------|------------------|
| 16 | 0.08% | 1/1209 | 0.209s |
| 32 | 0.08% | 1/1209 | 0.276s |
| 64 | 0.17% | 2/1209 | 0.420s |
| 128 | 17.54% | 212/1209 | 0.711s |
| 256 | 87.68% | 1060/1209 | 1.222s |

**Template-answer leakage (`extracted_answer == "answer"`):**
- K <= 64: 1207-1208/1209
- K = 128: 995/1209
- K = 256: 102/1209 (plus 23 missing `extracted_answer`)

**Output location:** `/scratch/l/liaidan/t-rex/results/smc_token_sweep/job_152924/`

---

### 2026-01-28: Experiment 3 - GRPO Evaluation on GSM8K (Platinum Test)

**Status:** COMPLETE

**Configuration:**
- Model: GRPO-trained `Qwen2.5-7B` (Experiment 1)
- Dataset: GSM8K Platinum test (1,209 problems)
- N samples: 16
- Temperatures: 0.0 (greedy), 0.6, 1.0
- Best temperature: 1.0

**Results:**

| Metric | Score |
|--------|-------|
| pass@1 | 38.6% |
| pass@2 | 61.1% |
| pass@4 | 82.7% |
| pass@8 | 94.7% |
| pass@16 | 98.3% |

**Output location:** `/scratch/l/liaidan/t-rex/results/eval_grpo/Qwen2.5-7B/gsm8k_trained_n8/`

---

### 2026-01-28: Supplemental Evaluation - PPO Baseline on GSM8K (Platinum Test)

**Status:** COMPLETE

**Configuration:**
- Model: PPO-trained `Qwen2.5-7B` (n=1)
- Dataset: GSM8K Platinum test (1,209 problems)
- N samples: 16
- Temperatures: 0.0 (greedy), 0.6, 1.0
- Best temperature: 1.0

**Results:**

| Metric | Score |
|--------|-------|
| pass@1 | 25.2% |
| pass@2 | 43.6% |
| pass@4 | 67.1% |
| pass@8 | 87.3% |
| pass@16 | 96.6% |

**Output location:** `/scratch/l/liaidan/t-rex/results/eval_ppo/Qwen2.5-7B/gsm8k_trained_n1/`

---

### 2026-01-27: Experiment 2 - Best-of-N Baseline on MATH-500

**Status:** COMPLETE

**Configuration:**
- Model: `Qwen/Qwen2.5-7B`
- Dataset: MATH-500 (500 problems)
- N samples: 32
- Best temperature: 0.8

**Results:**

| Metric | Score |
|--------|-------|
| pass@1 | 53.3% |
| pass@2 | 64.1% |
| pass@4 | 71.1% |
| pass@8 | 75.6% |
| pass@32 | 81.6% |

**Output location:** `/scratch/l/liaidan/t-rex/results/bon_baseline/Qwen2.5-7B/math_n32/`

---

### 2026-01 (Job 150200): Experiment 1 - GRPO Baseline Training on GSM8K

**Status:** COMPLETE

**Configuration:**
- Model: `Qwen/Qwen2.5-7B`
- Dataset: GSM8K train (7,473 problems)
- N samples: 8
- KL coefficient: 0.001
- Learning rate: 5e-7
- Epochs: 1
- Training steps: 59

**Training metrics (final):**

| Metric | Value |
|--------|-------|
| Reward | 0.938 |
| KL Divergence | 0.00711 |
| Policy Loss | 0.01458 |
| Response Length | 297.9 tokens |

**Output location:** `/scratch/l/liaidan/t-rex/results/grpo_baseline/Qwen2.5-7B/gsm8k_n8/`

---

## Interpretation & Implications (2026-02-04)

**What the new results say:**
- **GRPO evaluation:** best_temp=1.0 with pass@1=38.6% and pass@16=98.3% implies a very large remaining search gap (~59.7%). Training improves the model but still leaves substantial headroom for inference-time search.
- **PPO evaluation:** pass@1=25.2% and pass@16=96.6% trails GRPO across k, suggesting GRPO is the stronger RL baseline to beat for T-REX.
- **Token-resampling sweep:** a sharp jump from K=128 (17.5%) to K=256 (87.7%) indicates frequent resampling can prevent solutions from finishing. Larger chunk sizes (or adaptive K) are necessary for SMC to be viable.
- **Step-based SMC baseline:** still invalid due to premature truncation and template-answer leakage, so it cannot yet be compared to BoN or GRPO.

**Implications for next steps:**
- Prioritize stabilizing termination/stop handling for step-based SMC; the baseline is currently unusable for comparisons.
- Focus SMC development on token-resampling with K>=256 (or adaptive K) and validate whether the high K=256 accuracy holds under stricter answer extraction.
- Use GRPO as the primary RL baseline for comparisons; PPO appears weaker in both pass@1 and pass@16.

---

## Scientific Hypotheses and What These Baselines Test

### The "Narrow Passage" Problem

The core hypothesis of T-REX is that mathematical reasoning involves **narrow passages** in the solution space—correct reasoning paths that require specific sequences of steps. Standard autoregressive sampling struggles to find these because:

1. **Early errors propagate:** A wrong step at position 5 makes positions 6-100 useless
2. **Reward is sparse:** Binary correctness only at the end provides no intermediate signal
3. **Distribution mismatch:** The base model's prior p₀(x) doesn't concentrate on correct solutions

### What Best-of-N Reveals

The Best-of-N baseline directly measures the **search opportunity**:

```
Search Opportunity = pass@N - pass@1
```

**High search opportunity** (e.g., pass@32 = 81.6%, pass@1 = 53.3% → gap = 28.3%) indicates:
- The model "knows" how to solve more problems than greedy decoding extracts
- There's significant room for smarter search methods (SMC, TSMC, T-REX)
- The 28.3% of problems where sampling helps are likely "narrow passage" cases

**Low search opportunity** would indicate:
- The model either solves problems deterministically or genuinely can't
- Less benefit expected from inference-time compute scaling

### What GRPO Reveals

GRPO tests whether **training** can shift the base distribution toward correct solutions:

```
Training Effect = pass@k(trained) - pass@k(base)
```

This reveals:
1. **Direct improvement:** Does the model become better at greedy decoding?
2. **Coverage improvement:** Does pass@N also increase, or does training narrow the distribution?
3. **Diversity trade-off:** If pass@1 increases but pass@N decreases, training may be overfitting

### Key Comparisons for T-REX Development

| Comparison | What It Tests |
|------------|---------------|
| BoN pass@N vs pass@1 | Search opportunity in base model |
| GRPO pass@1 vs Base pass@1 | Training-time improvement |
| GRPO pass@N vs Base pass@N | Does training help search? |
| (Future) SMC pass@1 vs BoN pass@N | Does SMC approach BoN oracle? |
| (Future) TSMC vs SMC | Value of learned twist function |
| (Future) T-REX vs TSMC | Value of parallel tempering |

### Efficiency Metrics

The GRPO reward function tracks samples-to-threshold, enabling:

```
Sample Efficiency = samples_to_X%_accuracy
```

This metric will be critical for comparing:
- **Inference-time methods:** SMC needs K samples to reach accuracy A
- **Training methods:** GRPO needs M samples to reach the same accuracy A
- **T-REX:** Combines both—what's the optimal allocation of compute?

---

## Running the Experiments

### Prerequisites

1. **Download models** on the login node (compute nodes have no internet):
   ```bash
   source venv/bin/activate
   export HF_HOME="/scratch/l/liaidan/model_weights"
   python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B')"
   ```

2. **Verify scratch directories** exist:
   ```bash
   mkdir -p /scratch/l/liaidan/t-rex/results
   mkdir -p /scratch/l/liaidan/t-rex/slurm
   ```

### Best-of-N Baseline

```bash
# Submit the job
sbatch trex/scripts/tamia/run_bon_baseline.sh

# Monitor progress
tail -f /scratch/l/liaidan/t-rex/slurm/grpo-<JOBID>.err

# Check results
cat /scratch/l/liaidan/t-rex/results/bon_baseline/Qwen2.5-7B/math_n32/summary.json
```

### GRPO Training

```bash
# Submit the job
sbatch trex/scripts/tamia/run_grpo_baseline.sh

# Monitor training (look for iteration speed and loss)
tail -f /scratch/l/liaidan/t-rex/slurm/grpo-<JOBID>.err

# Check efficiency metrics
cat /scratch/l/liaidan/t-rex/results/grpo_baseline/Qwen2.5-7B/gsm8k_n8/efficiency_metrics.json

# List checkpoints
ls -la /scratch/l/liaidan/t-rex/results/grpo_baseline/Qwen2.5-7B/gsm8k_n8/checkpoints/
```

### GRPO Evaluation

```bash
# Run after GRPO training completes
sbatch trex/scripts/tamia/eval_grpo_baseline.sh

# Results
cat /scratch/l/liaidan/t-rex/results/eval_grpo/Qwen2.5-7B/gsm8k_trained_n8/summary.json
```
