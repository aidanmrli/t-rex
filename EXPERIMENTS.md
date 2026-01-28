# Experiments for T-REX

**Last Updated:** 2026-01-27

**NOTE:** We should always update this document once we have planned an experiment and it works. Then, when the experiment has completed, we should immediately update this document with the results.

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

| Script | Submitted | Started | Initialization | Training | Status |
|--------|-----------|---------|----------------|----------|--------|
| `run_bon_baseline.sh` | Yes | Yes | Success | **COMPLETE** | pass@32=81.6% on MATH-500 |
| `run_grpo_baseline.sh` | Yes | Yes | Success | **RUNNING** | ~2 it/s, 84% complete |
| `eval_grpo_baseline.sh` | N/A | N/A | N/A | N/A | Ready to run after GRPO |

### Final Configuration Notes

**Memory constraints with `--colocate_all_models`:**
- `VLLM_GPU_UTIL=0.6` is the safe maximum (0.85 causes OOM)
- `MICRO_TRAIN_BATCH_SIZE=4` and `MICRO_ROLLOUT_BATCH_SIZE=16` are conservative but stable

**Local model path required:**
- Scripts now use local cached model path instead of HuggingFace ID
- This avoids network calls even when `TRANSFORMERS_OFFLINE=1` is set

---

## Planned Experiments

### Experiment 1: GRPO Baseline on GSM8K

**Status:** RUNNING (Job 150200)

**Configuration:**
- Model: `Qwen/Qwen2.5-7B`
- Dataset: GSM8K train (7,473 problems)
- N samples: 8
- KL coefficient: 0.001
- Learning rate: 5e-7
- Epochs: 1

**Checkpoint Location:**
```
/scratch/l/liaidan/t-rex/results/grpo_baseline/Qwen2.5-7B/gsm8k_n8/checkpoints/
```

**To check training progress:**
```bash
tail -30 /scratch/l/liaidan/t-rex/slurm/grpo-150200.err
ls -la /scratch/l/liaidan/t-rex/results/grpo_baseline/Qwen2.5-7B/gsm8k_n8/checkpoints/
```

**When training completes, run evaluation:**
```bash
sbatch trex/scripts/tamia/eval_grpo_baseline.sh
```

---

## Completed Experiments

### Experiment 2: Best-of-N Baseline on MATH-500

**Status:** COMPLETE

**Configuration:**
- Model: `Qwen/Qwen2.5-7B`
- Dataset: MATH-500 (500 problems)
- N samples: 32
- Best temperature: 0.8 (selected from sweep)
- GPU memory utilization: 95%

**Results:**
| Metric | Score |
|--------|-------|
| pass@1 | 53.3% |
| pass@2 | 64.1% |
| pass@4 | 71.1% |
| pass@8 | 75.6% |
| pass@32 | 81.6% |

**Output Location:**
```
/scratch/l/liaidan/t-rex/results/bon_baseline/Qwen2.5-7B/math_n32/
```

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
