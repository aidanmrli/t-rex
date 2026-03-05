# Implementation Plan for T-REX

**Last Updated:** 2026-03-04

**NOTE:** We should always update this plan with our progress once we have implemented something and it works.

**Previous approach (archived):** The Feb 2026 approach (Twisted SMC + Block-Gibbs transport + learned value head/critic) is archived in `docs/archive/2026-feb/`. The project pivoted in March 2026 to the simpler multi-chain SMC with mixture proposals described below.

---

## Algorithm Summary

T-REX runs K independent SMC chains at different temperatures β_1=0 (hot, pure prior) to β_K=1 (cold, full posterior). Each chain uses a PRM for incremental reweighting. Cold chains absorb particles from adjacent hot chains via mixture proposals (Bernoulli coin flip with probability λ), with importance reweighting w = R(x)^{Δβ}. This is "zero-rejection" communication — no MH accept/reject step needed.

Key formula: `w_t^{k,(i)} = (R(x_{1:t}) / R(x_{1:t-1}))^{β_k}`

See `docs/HIGH_LEVEL_CONTEXT.md` for full mathematical specification.

---

## Current Repository Status

### Completed Components (from prior work, still reusable)

| Component | Location | Status | Notes |
|-----------|----------|--------|-------|
| **Best-of-N Baseline** | `trex/baselines/best_of_n_baseline.py` | Complete | Temperature sweep, checkpointing |
| **GRPO Training** | `trex/baselines/grpo_reward_func.py` | Complete | Efficiency tracking |
| **Math Verification** | `trex/eval/math_verifier.py` | Complete | HF math_verify, SymPy, timeout |
| **Answer Parsing** | `trex/eval/parser.py` | Complete | LaTeX, boxed format, numeric |
| **Math Grading** | `trex/eval/grader.py` | Complete | Symbolic equality, numeric comparison |
| **Datasets** | `trex/data/*.jsonl` | Complete | GSM8K, MATH, MATH-500 |
| **SLURM Scripts** | `trex/scripts/tamia/*.sh` | Complete | Auto-requeue, checkpointing |
| **SMC Core + Resampling** | `trex/smc/particle_filter.py`, `trex/smc/resampling.py` | Complete | Unit-tested, reusable |
| **LLM Particle Filter** | `trex/smc/llm_particle_filter.py` | Complete | Step-wise generation, PRM scoring |
| **Efficiency Tracking** | `trex/utils/efficiency_tracker.py` | Complete | Used by RL baselines |

### Code from old approach (may need cleanup/removal)

| Component | Location | Status | Notes |
|-----------|----------|--------|-------|
| **Twisted SMC Core** | `trex/smc/twisted_smc.py` | Obsolete | Old twist-based weighting with learned value head |
| **Tempering Primitives** | `trex/tempering/*.py` | Partially reusable | Temperature ladder is reusable; MH exchange logic is obsolete |
| **Value Head** | `trex/models/value_head.py` | Obsolete | No longer needed |
| **Twist Model** | `trex/models/twist_model.py` | Obsolete | No longer needed |
| **Value Trainer** | `trex/training/value_trainer.py` | Obsolete | No longer needed |
| **TSMC Baseline** | `trex/baselines/tsmc_baseline.py` | Obsolete | Old TSMC runner |

### Not Yet Implemented (new algorithm)

| Component | Priority | Complexity | Notes |
|-----------|----------|------------|-------|
| Single-chain SMC with PRM reweighting | High | Medium | Stage 3 of new spec |
| Multi-chain SMC with mixture proposals | High | High | Stage 4 — core novelty |
| Diagnostics (ESS, injection survival, log Z) | Medium | Low | Stage 5 |
| PRM800K SFT (may already have checkpoints) | Medium | Low | Stage 6 |
| Full experimental evaluation | Medium | Medium | Stage 7 |

---

## Stage 1: Base LLM Inference Wrapper

**Goal:** Build a module that takes a prompt and partial sequence and returns next-token samples and log-probabilities.

**What to implement:**
- `sample_next_token(model, x_{1:t-1}) → x_t`
- `log_prob(model, x_{1:t-1}, x_t) → float`
- Batched versions for N particles simultaneously

**Status:** Largely covered by existing `trex/smc/llm_particle_filter.py` and vLLM integration. May need refactoring to support the new multi-chain architecture.

**Success criteria:**
- Sampling produces valid tokens from model vocabulary
- Batched inference handles N particles efficiently on GPU without OOM (target N=64 or N=128)

---

## Stage 2: Process Reward Model (PRM) Integration

**Goal:** Integrate a PRM that scores partial sequences, returning values in (0, 1].

**What to implement:**
- `score_prm(prm_model, x_{1:t}) → float ∈ (0, 1]`
- Must handle partial sequences (not just complete solutions)
- Clamp outputs: if raw PRM ≤ 0, clamp to epsilon (e.g., 1e-8)
- Batched scoring for N particles

**Status:** Existing `trex/models/reward_model.py` wraps `Qwen/Qwen2.5-Math-PRM-7B`. Need to verify it meets the strict positivity requirement and supports the incremental weight computation pattern.

**Success criteria:**
- PRM scores in (0, 1] for all inputs
- Scoring N partial sequences is fast enough to not bottleneck the SMC loop

---

## Stage 3: Single-Chain SMC Sampler

**Goal:** Implement the Propagate → Reweight → Resample loop for a single chain with fixed temperature β.

**What to implement:**
- Incremental weight computation: `w_t^{(i)} = (R(x_{1:t}^{(i)}) / R(x_{1:t-1}^{(i)}))^β`
- All weight computations in **log space** for numerical stability
- Systematic resampling with ESS-based triggering (resample when ESS < N/2)
- Normalizing constant estimation: `log Z_hat = Σ_t [logsumexp(log_w) - log(N)]`
- Particle storage: full trajectories `x_{1:t}^{(i)}` and cached PRM scores
- EOS handling: freeze particles that generate EOS, set subsequent weights to 1.0

**Key implementation details:**
```python
log_w_local = beta_k * (log_R_curr - log_R_prev)
log_W = log_w - logsumexp(log_w)  # normalized log weights
W = exp(log_W)                     # for resampling
log_Z += logsumexp(log_w) - log(N) # accumulate normalizing constant
```

**Success criteria:**
- At β=0: all weights exactly 1.0, output indistinguishable from i.i.d. base model samples
- At β=1: particles concentrate on higher-reward trajectories vs β=0
- ESS never collapses to 1 for sustained periods
- On GSM8K, single-chain SMC at β=1 with N=64 should outperform Best-of-N with N=64

---

## Stage 4: Multi-Chain SMC with Mixture Proposals (T-REX)

**Goal:** Implement K chains with inter-chain communication via mixture proposals. This is the core novelty.

**What to implement:**
- K independent SMC chains with temperatures `β_1=0 < β_2 < ... < β_K=1`
- Mixture proposal mechanism: Bernoulli coin flip per particle (probability λ)
  - Local extension (prob 1-λ): extend from own chain's particle
  - Hot injection (prob λ): sample full trajectory from adjacent hot chain
- Hot injection weight: `w_{t,inject}^{(i)} = R(x_{1:t}^{(i)})^{β_{k+1} - β_k}`
- Trajectory replacement: when injecting, replace **entire** particle trajectory
- Cache management: after injection, update cached PRM scores (can reuse hot chain's cached value)
- Communication direction: unidirectional hot → cold (chain k feeds into chain k+1)

**Temperature schedule (starting defaults):**
- K=4, linear: `β = [0.0, 0.33, 0.67, 1.0]`
- Also try geometric: `β = [0.0, 0.1, 0.4, 1.0]`

**Success criteria:**
- Some injected particles survive resampling in the cold chain (injection rate × survival rate > 0)
- More diverse valid solutions in cold chain vs single-chain SMC
- Cold chain accuracy ≥ single-chain SMC at β=1 on GSM8K and MATH-500
- Accuracy monotonically improves as N increases

---

## Stage 5: Diagnostics

**Goal:** Implement monitoring tools for algorithm health.

**What to implement:**
- Per-chain ESS over time: `ESS_t^k` heatmaps
- Per-chain log Z estimate
- Injection survival rate per chain per step
- Particle genealogy tracking (ancestor indices through resampling)
- Reward distribution per chain (histograms of R values)

**Success criteria:**
- ESS never collapses to 1 for sustained periods
- Injection survival rate > 5% in adjacent colder chain
- Cold chain reward distribution shifts rightward over steps
- No NaN or Inf in weights or log Z

---

## Stage 6: SFT of Base Model

**Goal:** Fine-tune base model on step-by-step reasoning data.

**Status:** PRM800K SFT checkpoints may already exist from prior work:
- Job `154122` (H200x8): `/scratch/l/liaidan/t-rex/results/prm800k_sft/job_154122/ckpt`
- Job `154126` (H100x4): `/scratch/l/liaidan/t-rex/results/prm800k_sft/job_154126/ckpt`

**Models:** Qwen2.5-7B (primary), optionally Qwen2.5-14B, Qwen2.5-32B, LLaMA-3.1-8B.

**Success criteria:**
- SFT model produces step-by-step solutions in expected format
- Pass@1 accuracy on GSM8K improves over pre-SFT base
- Outputs parseable by PRM

---

## Stage 7: Experimental Evaluation

**Goal:** Run the full experimental comparison.

**Benchmarks:** GSM8K (primary), MATH-500 (secondary)

### Baselines

All baselines receive an equivalent computational budget for fair comparison.

1. **Best-of-N (BoN) Sampling** *(complete)*: Standard generation from the base LLM with no inference-time interventions. Sweep temperatures T ∈ {0.6, 0.8, 1.0, 1.2}. Defines the baseline difficulty and tests whether T-REX outperforms simply allocating equivalent compute to parallel random sampling.

2. **GRPO / PPO** *(complete)*: RL-trained models evaluated via BoN harness. Tests training-time vs inference-time compute scaling.

3. **Beam Search / Tree Search**: Deterministic heuristic search baseline. Provides a strong non-random search comparison to demonstrate the necessity of particle-based methods.

4. **Standard SMC with PRM**: Vanilla particle filtering (single chain, β=1) using the base model as proposal and the same PRM for reweighting. Critical scientific baseline — standard SMC can become trapped in local optima (locally-promising but fundamentally flawed strategies). Outperforming standard SMC at equal compute validates our core algorithmic claim.

5. **Twisted SMC (Guided Proposals)**: SMC with value-guided proposals (cf. Zhao et al., Feng et al.). Twisted SMC faces a chicken-and-egg problem: learning an accurate twist function requires valid solution trajectories, which are initially unknown and sparse. Isolates the necessity of replica exchange for uncovering sparse solutions.

6. **Swap-Based Parallel Tempering (PT / NRPT)**: Standard parallel tempering (including non-reversible PT) using Metropolis-Hastings swap acceptance criteria. In the high-dimensional discrete text space, standard PT suffers from vanishing swap acceptance rates because the typical sets of hot and cold chains have no overlap. Proves that performance gains come from our zero-rejection mixture injection, not merely from temperature scaling.

### Ablation Studies

Isolate contributions of T-REX's components: landscape smoothing, mode escape, and zero-rejection particle injection.

1. **Soft PRM Potential vs Binary Constraint**: Impact of smoothing the target landscape via intermediate step-level rewards vs enforcing binary correctness only at the terminal step.

2. **Mixture Proposal (Hot Injection) vs Standard SMC Proposal**: Replace the zero-rejection injection mechanism with standard base-model proposals (λ=0). Tests the value of cross-chain communication.

3. **Tempering On vs Off**: Whether maintaining multiple temperature chains is necessary to escape local reasoning modes.

4. **Online Twist Learning On vs Off**: Value of dynamically updating the twist function during inference.

5. **Zero-Rejection Injection vs Swap-Based Exchange**: Direct comparison of our communication rule against standard MH swap criteria. Highlights computational efficiency gained by bypassing high rejection rates.

6. **Auxiliary Masking/Infill Editor vs No Auxiliary Network**: Contribution of an auxiliary editing network in proposing valid local transitions in the discrete text space.

### Protocol

- Sweep N ∈ {16, 32, 64, 128, 256}
- Report: accuracy, sample efficiency (accuracy vs N), wall-clock time

### Success Criteria

- T-REX at N=64 matches or exceeds Best-of-N at N=256 on GSM8K (4× sample efficiency)
- T-REX outperforms standard SMC at equal N
- T-REX with communication (λ>0) outperforms without (λ=0)
- T-REX outperforms swap-based PT at equal compute (validates zero-rejection mechanism)

---

## Implementation Priority Order

### Sprint 1: Single-Chain SMC (Stage 3)
1. [ ] Refactor/verify PRM integration for strict positivity and caching
2. [ ] Implement single-chain SMC with log-space weights
3. [ ] Implement systematic resampling with ESS threshold
4. [ ] Validate sanity checks (β=0 and β=1)

### Sprint 2: Multi-Chain T-REX (Stage 4)
5. [ ] Implement multi-chain orchestration
6. [ ] Implement mixture proposal mechanism (Bernoulli + trajectory injection)
7. [ ] Implement hot injection reweighting
8. [ ] Validate inter-chain communication

### Sprint 3: Diagnostics + Evaluation (Stages 5, 7)
9. [ ] Implement diagnostic tools
10. [ ] Run single-chain SMC vs Best-of-N comparison
11. [ ] Run T-REX vs single-chain SMC comparison
12. [ ] Run λ=0 ablation (value of communication)

---

## Hyperparameter Summary

| Parameter | Default | Sweep Range | Notes |
|-----------|---------|-------------|-------|
| N (particles) | 64 | {16, 32, 64, 128, 256} | Primary compute knob |
| K (chains) | 4 | {2, 3, 4, 6} | More chains = more exploration |
| β schedule | Linear [0, 0.33, 0.67, 1] | Linear, geometric | Start with linear |
| λ (mix weight) | 0.1 | {0.05, 0.1, 0.2, 0.3} | Too high → cold chain flooded |
| ESS threshold | N/2 | {N/4, N/3, N/2} | Lower = less resampling |
| LLM temperature | 1.0 | {0.8, 1.0, 1.2} | Applied inside P_θ |

---

## GPU Allocation Strategy

**Total per step:** K × N LLM forward passes + K × N PRM evaluations.

Assuming 4-8 H100/H200 GPUs:
- Option A: All chains on all GPUs (batch dimension) — simplest, start here
- Option B: Chain partitioning (1-2 chains per GPU) — if memory constrained

---

## Testing Strategy

Existing test infrastructure from `docs/plans/UNIT_TESTING_PLAN.md` is still relevant for:
- Eval tests (parser, grader, math_verifier)
- Resampling tests
- Particle filter tests
- Temperature ladder tests

New tests needed:
- Single-chain SMC weight computation (log-space)
- Mixture proposal mechanism
- Hot injection reweighting
- ESS computation and threshold triggering
- End-to-end multi-chain smoke test
