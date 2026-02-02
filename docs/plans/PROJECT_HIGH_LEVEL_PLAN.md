# Implementation Plan for T-REX

**Last Updated:** 2026-02-02

**NOTE:** We should always update this plan with our progress once we have implemented something and it works.

We should have three abstract components:

1. Exploration through the usage of high temperatures and non-reversible parallel tempering to move between states
2. Twisted Sequential Monte Carlo (TSMC) to guide a model at a fixed temperature toward high-value regions efficiently while maintaining diversity of a set of particles
3. An abstract transport mechanism to help move particles between different temperature states. In high-dimensional spaces like language, the "Hot" distribution (Creative/Chaotic) and the "Cold" distribution (Rigorous/Narrow) might have almost zero overlap. If you try to swap them directly, the Cold model calculates an acceptance probability of almost zero.

---

## Current Repository Status

### ✅ Implemented Components

| Component | Location | Status | Notes |
|-----------|----------|--------|-------|
| **Best-of-N Baseline** | `trex/baselines/best_of_n_baseline.py` | Complete | Temperature sweep, checkpointing, WandB |
| **GRPO Training** | `trex/baselines/grpo_reward_func.py` | Complete | Efficiency tracking, threshold detection |
| **Math Verification** | `trex/eval/math_verifier.py` | Complete | HF math_verify, SymPy, timeout protection |
| **Answer Parsing** | `trex/eval/parser.py` | Complete | LaTeX, boxed format, numeric extraction |
| **Math Grading** | `trex/eval/grader.py` | Complete | Symbolic equality, numeric comparison |
| **Datasets** | `trex/data/*.jsonl` | Complete | GSM8K, MATH, MATH-500 |
| **SLURM Scripts** | `trex/scripts/*.sh` | Complete | Auto-requeue, checkpointing |
| **Standard SMC Steering** | `trex/baselines/smc_steering_baseline.py` | Complete | PRM/ORM guided, SLURM checkpointing |
| **SMC Core + Resampling** | `trex/smc/particle_filter.py`, `trex/smc/resampling.py` | Complete | Unit-tested |
| **LLM Particle Filter** | `trex/smc/llm_particle_filter.py` | Complete | Step-wise generation, PRM scoring |
| **Twisted SMC Core** | `trex/smc/twisted_smc.py` | Complete | Twist weights + tests; value head pending |
| **Tempering Primitives** | `trex/tempering/*.py` | Complete | Temperature ladder, swap pairs, replica exchange |
| **Efficiency Tracking** | `trex/utils/efficiency_tracker.py` | Complete | Used by RL baselines |

### 🔄 Ready for Testing

| Component | Location | Status | Notes |
|-----------|----------|--------|-------|
| **PPO Baseline** | `trex/baselines/ppo_reward_func.py` | Ready | Uses GAE, critic network |
| **SMC Steering Re-run** | `trex/baselines/smc_steering_baseline.py` | Ready | Post stop-string fix; needs re-eval |

### ❌ Not Yet Implemented

| Component | Priority | Complexity | Dependencies |
|-----------|----------|------------|--------------|
| Value Head + TSMC Baseline | High | Medium | Value head training, TSMC runner |
| Parallel Tempering (Full) | Medium | High | Multi-chain orchestration, exchange integration |
| Transport Mechanisms | Medium | High | Critic network, Block-Gibbs editor |
| Replica Exchange SMC | Low | High | Parallel tempering + SMC |

---

## Phase 1: Complete Baselines

### 1.1 PPO Baseline for Math Reasoning

**Goal:** Establish PPO baseline using existing OpenRLHF infrastructure.

**Implementation:**
- [x] Create `trex/baselines/ppo_reward_func.py` (reuses GRPO reward function)
- [x] Create `trex/scripts/tamia/run_ppo_baseline.sh` with PPO-specific hyperparameters
- [x] Key differences from GRPO:
  - Uses `--advantage_estimator gae` instead of `group_norm`
  - Requires critic network (value function)
  - Different KL penalty schedule

**Configuration:**
```bash
# PPO-specific args
--advantage_estimator gae
--critic_learning_rate 9e-6
--init_kl_coef 0.01  # Higher than GRPO
--kl_estimator k1
```

**Files Created:**
- `trex/baselines/ppo_reward_func.py`
- `trex/scripts/tamia/run_ppo_baseline.sh`

---

### 1.2 Standard SMC Steering (Rollout Roulette Baseline)

**Goal:** Implement the Puri et al. (2025) baseline for inference-time compute scaling.

**Reference:** Section 6.2 in HIGH_LEVEL_CONTEXT.md

**Core Algorithm:**
```
1. Initialize N particles with prompt
2. For each step t:
   a. Expansion: Generate next "step" (until \n or ## Step) for each particle
   b. Weighting: Score particles using Process Reward Model (PRM)
   c. Resampling: Multinomial/Systematic resampling proportional to weights
3. Final Selection: Best-of-N (ORM) or majority voting
```

**Implementation Status:** ✅ Complete

**Key Components Implemented:**
- **SMC Engine:** `LLMParticleFilter` extending base `ParticleFilter` to handle LLM generation and PRM scoring.
- **Reward Modeling:** `RewardModel` wrapper for `Qwen/Qwen2.5-Math-PRM-7B` supporting both PRM (step-wise) and ORM (final) scoring.
- **Step Detection:** Regex-based detection of `## Step N:` patterns.
- **Checkpointing:** Full SLURM-compatible checkpointing (state + particles).
- **Configuration:** `SMCSteeringConfig` for easy experimentation.

**Files Created:**
- `trex/baselines/smc_steering_baseline.py` - Main baseline runner
- `trex/baselines/smc_config.py` - Configuration classes
- `trex/smc/llm_particle_filter.py` - LLM-aware particle filter
- `trex/models/reward_model.py` - PRM/ORM wrapper
- `trex/models/prm_config.py` - PRM token configuration
- `trex/scripts/run_smc_baseline.sh` - Execution script
- `trex/smc/particle_filter.py` - Core logic
- `trex/smc/resampling.py` - Resampling logic

---

## Phase 2: Twisted Sequential Monte Carlo (TSMC)

### 2.1 Value Head Architecture

**Goal:** Add a learnable value head to predict expected future reward. This should probably use a Transformer but we can substitute it with different options.

**Reference:** Section 2.3 and 6.2 (Twisted SMC baseline) in HIGH_LEVEL_CONTEXT.md

**Architecture:**
```
Base LLM Hidden State [B, T, H]
    → Linear(H, H//4)
    → ReLU
    → Linear(H//4, 1)
    → Sigmoid
    → Value [B, T, 1]
```

**Implementation:**
- [ ] Create `trex/models/value_head.py` - Value head architecture
- [ ] Create `trex/models/twist_model.py` - Wrapper combining base LLM + value head
- [ ] Ensure compatibility with vLLM for efficient inference

**Value Head Options:**
We should be able to use all of these as the value head in a plug and play fashion.
1. **Simple Linear:** `Linear(Hidden_Dim, 1)` - Fastest, might underfit
2. **MLP:** `Linear(H, H//4) → ReLU → Linear(H//4, 1)` - Recommended starting point
3. **Attention-pooled:** Pool over sequence before prediction - For long contexts
4. **Transformers**

**Files to Create/Update:**
- `trex/models/__init__.py` (update exports)
- `trex/models/value_head.py`
- `trex/models/twist_model.py`

---

### 2.2 Value Head Training (Self-Distillation)

**Goal:** Train value head using Monte Carlo returns from rollouts.

**Reference:** Section 4 (Phase 3: Online Learning) in HIGH_LEVEL_CONTEXT.md

**Training Algorithm:**
```
1. Sample K trajectories per prompt using current policy
2. Verify final answers → Binary reward R ∈ {0, 1}
3. Assign final reward to ALL steps in trajectory (Monte Carlo)
4. Train: L = MSE(V(x_{1:t}), R)
5. Iterate: Use improved V to bias sampling, collect new data
```

**Implementation:**
- [ ] Create `trex/training/value_trainer.py` - Training loop for value head
- [ ] Create `trex/training/trajectory_buffer.py` - Store trajectories with rewards
- [ ] Integration with WandB for tracking value loss and accuracy correlation

**Training Configuration:**
```python
@dataclass
class ValueTrainingConfig:
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_rollouts_per_prompt: int = 8
    update_frequency: int = 100  # Steps between value updates
    ema_decay: float = 0.99      # For target value smoothing
```

**Files to Create:**
- `trex/training/__init__.py`
- `trex/training/value_trainer.py`
- `trex/training/trajectory_buffer.py`
- `trex/scripts/train_value_head.sh`

---

### 2.3 Twisted SMC Inference

**Goal:** Use learned value head to guide particle filtering.

**Reference:** Section 6.2 (Twisted SMC baseline) in HIGH_LEVEL_CONTEXT.md

**Key Difference from Standard SMC:**
- Standard SMC: Weight by PRM score `w_t = PRM(x_{1:t})`
- Twisted SMC: Weight by value ratio `w_t = ψ(x_{1:t}) / ψ(x_{1:t-1})`

**Implementation:**
- [x] Extend `trex/smc/particle_filter.py` to support twist-based weighting
- [x] Create `trex/smc/twisted_smc.py` - TSMC-specific logic
- [ ] Create `trex/models/value_head.py` and `trex/models/twist_model.py`
- [ ] Create `trex/training/value_trainer.py` and `trex/training/trajectory_buffer.py`
- [ ] Create `trex/baselines/tsmc_baseline.py` - Evaluation runner

**Weight Computation:**
```python
def compute_twisted_weights(values_t, values_t_minus_1):
    """
    Incremental importance weight for TSMC.

    w_t = ψ(x_{1:t}) / ψ(x_{1:t-1})

    Intuition: If value increased, this was a good step → high weight
    """
    return values_t / (values_t_minus_1 + 1e-8)
```

**Files Remaining:**
- `trex/models/value_head.py`
- `trex/models/twist_model.py`
- `trex/training/value_trainer.py`
- `trex/training/trajectory_buffer.py`
- `trex/baselines/tsmc_baseline.py`
- `trex/scripts/run_tsmc_baseline.sh`

---

## Phase 3: Non-Reversible Parallel Tempering

**Key Innovation:** We use a **non-reversible** parallel tempering scheme with a deterministic alternating swap schedule. This converts the O(K²) random walk of standard PT into O(K) ballistic flow, dramatically improving mixing efficiency.

**Reference:** Section 2.2 and 2.3 in HIGH_LEVEL_CONTEXT.md, Syed et al. (2022)

### 3.1 Temperature Ladder

**Goal:** Maintain K parallel chains at different temperatures with efficient non-reversible communication.

**Temperature Schedule:**
```
β_0 = 0.0  (Hot: Pure exploration, π ≈ p_0)
β_1 = 0.2
β_2 = 0.5
β_3 = 0.8
β_K = 1.0  (Cold: Strict posterior, π ∝ p_0 · φ)
```

**Non-Reversible Swap Schedule:**
```
S_odd  = {(1,2), (3,4), (5,6), ...}  # Odd timesteps
S_even = {(2,3), (4,5), (6,7), ...}  # Even timesteps
```
By alternating between these sets, particles flow ballistically through the ladder instead of diffusing randomly.

**Implementation:**
- [x] Create `trex/tempering/temperature_ladder.py` - Temperature schedule + swap pairs
- [x] Create `trex/tempering/exchange.py` - Acceptance ratios + swap primitive
- [ ] Create `trex/tempering/parallel_chains.py` - Multi-chain orchestration
- [ ] GPU parallelization: Each temperature on subset of GPUs

**Design:**
```python
@dataclass
class TemperingConfig:
    num_temperatures: int = 5
    beta_schedule: str = "linear"  # linear, geometric, adaptive
    min_beta: float = 0.0
    max_beta: float = 1.0
    particles_per_temp: int = 8
    non_reversible: bool = True    # Use deterministic alternating schedule
```

**Files Remaining:**
- `trex/tempering/parallel_chains.py`
- `trex/tempering/diagnostics.py`

---

### 3.2 Non-Reversible Replica Exchange

**Goal:** Enable efficient communication between temperature chains using ballistic flow.

**Reference:** Section 2.3 in HIGH_LEVEL_CONTEXT.md, Syed et al. (2022)

**The Problem with Standard PT:**
Standard parallel tempering chooses swap pairs at random, causing particles to perform a random walk through the temperature ladder. This requires $O(K^2)$ steps for a sample to propagate from prior to posterior.

**The Solution: Deterministic Alternating Schedule**

Instead of random swaps, we use a deterministic schedule that alternates between two swap sets:

```python
S_odd  = [(1,2), (3,4), (5,6), ...]  # Attempted at odd timesteps
S_even = [(2,3), (4,5), (6,7), ...]  # Attempted at even timesteps
```

This induces **ballistic flow**: successful particles propagate from prior to posterior in $O(K)$ steps.

**Exchange Algorithm:**
```
At timestep t:
1. Select swap set: S = S_odd if t % 2 == 1 else S_even
2. For each pair (k, k+1) in S (can be parallelized):
   a. Propose swap: x_k ↔ x_{k+1}
   b. Compute acceptance ratio:
      α = min(1, (φ(x_k)/φ(x_{k+1}))^{β_{k+1} - β_k})
   c. Accept with probability α
```

**Key Insight (from Section 3.2 in HIGH_LEVEL_CONTEXT.md):**
When using Block-Gibbs proposals from base model, the likelihood ratio cancels:
```
α = min(1, (φ(x')/φ(x))^{β_target})
```
This means we only need to evaluate the constraint φ, not compute sequence log-probs!

**Implementation:**
- [x] Create `trex/tempering/exchange.py` - Replica exchange logic
- [x] Implement alternating swap schedule (odd/even sets) in `get_swap_pairs()`
- [x] Implement acceptance ratio with constraint-only evaluation
- [ ] Track exchange statistics and flow direction for diagnostics

**Swap Schedule Implementation:**
```python
def get_swap_pairs(timestep: int, num_temperatures: int) -> List[Tuple[int, int]]:
    """
    Get swap pairs for non-reversible parallel tempering (0-indexed).

    Args:
        timestep: Current timestep (determines odd/even set)
        num_temperatures: Total number of temperature levels K

    Returns:
        List of (k, k+1) pairs to attempt swapping
    """
    if timestep % 2 == 1:  # Odd timestep
        # S_odd = {(0,1), (2,3), (4,5), ...}
        return [(k, k + 1) for k in range(0, num_temperatures, 2)]
    else:  # Even timestep
        # S_even = {(1,2), (3,4), (5,6), ...}
        return [(k, k + 1) for k in range(1, num_temperatures, 2)]
```

**Files Remaining:**
- `trex/tempering/diagnostics.py`

---

## Phase 4: Transport Mechanisms (Advanced)

### Overview

The transport mechanism bridges the gap between "hot" (creative but invalid) and "cold" (rigorous but narrow) distributions. We implement a plug-and-play system to compare different approaches.

**Reference:** IMPLEMENTATION_PLAN.md Section "Abstract Transport Mechanism"

### 4.1 Transport Interface

**Goal:** Define abstract interface for all transport mechanisms.

```python
class TransportMechanism(ABC):
    @abstractmethod
    def transport(
        self,
        x_source: str,           # Sample from source distribution
        beta_source: float,      # Source temperature
        beta_target: float,      # Target temperature
    ) -> Tuple[str, float]:      # (transported sample, acceptance prob)
        pass
```

**Files to Create:**
- `trex/transport/__init__.py`
- `trex/transport/base.py`

---

### 4.2 Level 1: Mathematical Tricks (Cheapest)

#### Standard Metropolis-Hastings
- Direct swap proposal
- Will have ~0% acceptance in high dimensions
- Useful as baseline/sanity check

#### Quantile Coupling
- Transport "luck" not text
- Map CDF values between distributions
- Known failure mode: tail mismatch

**Files to Create:**
- `trex/transport/metropolis_hastings.py`
- `trex/transport/quantile_coupling.py`

---

### 4.3 Level 2: Heuristic Bridges (Mid-Cost)

#### Likelihood Thresholding
- Mask tokens with P(t) < threshold under cold model
- Resample masked tokens
- Risk: Deletes novelty along with errors

**Implementation:**
- [ ] Create `trex/transport/likelihood_threshold.py`
- [ ] Implement adaptive threshold selection

---

### 4.4 Level 3: Verifier Bridges (High-Cost)

#### PRM-Guided Repair
- Use Process Reward Model to identify broken steps
- Mask lowest-scoring step
- Resample using cold model

**Implementation:**
- [ ] Create `trex/transport/prm_repair.py`
- [ ] Integrate with PRM (if available) or use verifier heuristics

---

### 4.5 Level 4: Learned Bridges (Research SOTA)

#### Learned Critic + Block-Gibbs Editor

**This is the core T-REX contribution.**

**Components:**
1. **Critic (C_ω):** Predicts mask M covering tokens causing rejection
2. **Editor:** Cold model resamples x_M conditioned on x_{-M}
3. **Twist:** Updates proposal to avoid these errors in future

**Implementation:**
- [ ] Create `trex/models/critic_head.py` - Token-level masking predictor
- [ ] Create `trex/transport/block_gibbs.py` - Conditional resampling
- [ ] Create `trex/transport/learned_transport.py` - Full learned transport
- [ ] Create `trex/training/critic_trainer.py` - Train critic on repair successes

**Critic Architecture:**
```
Hidden State [B, T, H] → Linear(H, 1) → Sigmoid → Mask Prob [B, T, 1]
```

**Training Signal:**
- Positive: Masks that led to successful repairs (acceptance)
- Negative: Masks that didn't help (rejection)

**Files to Create:**
- `trex/models/critic_head.py`
- `trex/transport/block_gibbs.py`
- `trex/transport/learned_transport.py`
- `trex/training/critic_trainer.py`

---

## Phase 5: Full T-REX Integration

### 5.1 Main T-REX Algorithm

**Goal:** Combine all components into the full algorithm.

**Reference:** Section 4 (T-REX Algorithm Lifecycle) in HIGH_LEVEL_CONTEXT.md

**Algorithm:**
```
Phase 1: Twisted SMC (Exploration)
  - Run K particles at each temperature
  - Use twist function to bias proposals
  - Resample within temperature rungs

Phase 2: Non-Reversible Transport (Communication)
  - At fixed intervals, attempt to promote hot samples
  - Edit: x' = Editor(x_hot)
  - Accept if φ(x')^β_target > U[0,1] · φ(x_current)^β_target

Phase 3: Online Learning (Self-Distillation)
  - Collect (x_{1:t}, R) pairs from successful transports
  - Update twist: L = ||V(x_{1:t}) - R||²
  - Update critic: Maximize likelihood of successful masks
```

**Implementation:**
- [ ] Create `trex/algorithms/trex.py` - Main algorithm orchestration
- [ ] Create `trex/algorithms/trex_config.py` - Full configuration
- [ ] Create `trex/scripts/run_trex.sh` - Training/inference script

**Files to Create:**
- `trex/algorithms/__init__.py`
- `trex/algorithms/trex.py`
- `trex/algorithms/trex_config.py`
- `trex/scripts/run_trex.sh`

---

## Implementation Priority Order

### Sprint 1: Baseline Completion (Week 1-2)
1. ✅ Best-of-N baseline - DONE
2. ✅ GRPO baseline - DONE
3. ✅ PPO baseline - DONE (ready for testing)
4. ✅ Standard SMC Steering baseline - DONE

### Sprint 2: TSMC Foundation (Week 3-4)
5. [ ] Value head architecture
6. [ ] Value head training
7. [ ] Twisted SMC inference
8. [ ] TSMC baseline evaluation

### Sprint 3: Parallel Tempering (Week 5-6)
9. [x] Temperature ladder + swap pairs
10. [ ] Parallel chains orchestration
11. [x] Replica exchange mechanism (core)
12. [ ] Diagnostics and visualization

### Sprint 4: Transport Mechanisms (Week 7-8)
13. [ ] Transport interface
14. [ ] Simple baselines (MH, threshold)
15. [ ] Learned critic + Block-Gibbs

### Sprint 5: Integration (Week 9-10)
16. [ ] Full T-REX algorithm
17. [ ] End-to-end training pipeline
18. [ ] Comprehensive evaluation

---

## GPU Allocation Strategy

Assuming 4-8 H100/H200 GPUs:

**Training:**
- Value head: 1-2 GPUs (gradient accumulation)
- Critic head: 1-2 GPUs
- Rollout generation: All GPUs via vLLM

**Inference (TSMC/Parallel Tempering):**
- Option A: All temperatures on all GPUs (batch dimension)
- Option B: Temperature partitioning (2 temps × 4 GPUs)
- Option C: Pipeline parallelism for large models

**Recommended:** Start with Option A for simplicity, move to B if memory constrained.

---

## Testing Strategy

Each component should have:
1. **Unit tests:** Isolated function testing
2. **Integration tests:** Component interaction
3. **Smoke tests:** Quick end-to-end on small data

**Test Files Structure:**
```
trex/tests/
├── test_eval/
│   ├── test_parser.py
│   ├── test_grader.py
│   └── test_math_verifier.py
├── test_baselines/
│   └── test_smc_config.py
├── test_models/
│   └── test_reward_model.py
├── test_smc/
│   ├── test_resampling.py
│   ├── test_particle_filter.py
│   ├── test_twisted_smc.py
│   └── test_llm_particle_filter.py
├── test_tempering/
│   ├── test_temperature_ladder.py
│   └── test_exchange.py
├── test_utils/
│   └── test_efficiency_tracker.py
└── test_transport/
    └── __init__.py
```

---

## Models to Evaluate

All baselines and T-REX methods should be evaluated on the following models:

| Model | Size | Notes |
|-------|------|-------|
| `Qwen/Qwen2.5-7B` | 7B | Primary model for development |
| `google/gemma-7b` | 7B | Secondary evaluation model |

**Note:** Additional models may be added based on compute availability and research needs.

---

## Experiment Tracking

All experiments should be logged to WandB with:
- Method name (bon, grpo, ppo, smc, tsmc, trex)
- Model name (qwen2.5-7b, gemma-7b)
- Dataset
- Key hyperparameters
- Efficiency metrics (samples to X% accuracy)

See `EXPERIMENTS.md` for experiment results and analysis.
