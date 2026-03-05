# T-REX: Implementation Specification

**Last Updated:** 2026-03-04

## Purpose

This document is a **precise mathematical and algorithmic specification** for implementing the T-REX (Twisted Replica Exchange) framework. It is intended to be consumed by downstream planning and coding agents. Every formula, index convention, and data structure is defined explicitly so that code can be written to match the math without ambiguity.

---

## 1. Notation and Definitions

| Symbol | Type | Definition |
|---|---|---|
| `x_0` | token sequence | The input prompt / problem statement. |
| `x_{1:T}` | token sequence | The generated solution, produced autoregressively. `x_t` is the token (or reasoning step) at position `t`. |
| `P_θ(x_t \| x_{1:t-1})` | probability | The base LLM's autoregressive next-token distribution, conditioned on all prior tokens. |
| `P_θ(x_{1:t})` | probability | Joint probability of the partial sequence under the base LLM: `∏_{s=1}^{t} P_θ(x_s \| x_{1:s-1})`. |
| `R(x_{1:t})` | float ∈ (0, 1] | Process Reward Model (PRM) score for the partial sequence `x_{1:t}`. Evaluates logical validity up to step `t`. Must be **strictly positive** — never exactly 0. |
| `K` | int ≥ 2 | Number of temperature chains (replicas). |
| `β_k` for `k ∈ {1, …, K}` | float ∈ [0, 1] | Inverse temperature for chain `k`. **Convention**: `β_1 = 0` (hottest, pure prior) and `β_K = 1` (coldest, full posterior). The schedule is monotonically increasing: `0 = β_1 < β_2 < … < β_K = 1`. |
| `N` | int | Number of particles per chain. |
| `T` | int | Maximum sequence length (number of generation steps). |
| `λ` | float ∈ (0, 1) | Mixing weight for hot-chain injection into cold chain proposals. Recommended default: `λ = 0.1`. |

---

## 2. Target Distribution

The goal is to sample from the posterior:

```
π(x_{1:T}) = (1 / Z_π) * P_θ(x_{1:T}) * R(x_{1:T})
```

where `Z_π = Σ_{x_{1:T}} P_θ(x_{1:T}) * R(x_{1:T})` is the intractable normalizing constant.

### Tempered targets

Each chain `k` has its own tempered target at step `t`:

```
π̃_t^k(x_{1:t}) = P_θ(x_{1:t}) * R(x_{1:t})^{β_k}
```

**Key consequences:**
- **Chain k=1 (β_1=0):** `π̃_t^1(x_{1:t}) = P_θ(x_{1:t})`. The hottest chain samples from the pure base LLM prior with **no reward influence**.
- **Chain k=K (β_K=1):** `π̃_t^K(x_{1:t}) = P_θ(x_{1:t}) * R(x_{1:t})`. The coldest chain samples from the full posterior.
- **Intermediate chains** interpolate between prior and posterior.

---

## 3. Core Algorithm: SMC per Chain

Each chain `k` runs an independent SMC sampler with `N` particles. At each generation step `t = 1, 2, …, T`, each chain performs the **Propagate → Reweight → Resample** cycle.

### 3.1 Propagate

For chain `k`, for each particle `i ∈ {1, …, N}`:

- Sample the next token: `x_t^{k,(i)} ~ P_θ(· | x_{1:t-1}^{k,(i)})`

This uses the **base LLM** as the proposal distribution. The proposal is the same regardless of chain temperature — temperature only affects the *reweighting*.

### 3.2 Reweight

Compute the **incremental importance weight** for particle `i` in chain `k`:

```
w_t^{k,(i)} = π̃_t^k(x_{1:t}^{k,(i)}) / [π̃_{t-1}^k(x_{1:t-1}^{k,(i)}) * P_θ(x_t^{k,(i)} | x_{1:t-1}^{k,(i)})]
```

Expanding using the tempered target definition:

```
w_t^{k,(i)} = [P_θ(x_{1:t}^{k,(i)}) * R(x_{1:t}^{k,(i)})^{β_k}] / [P_θ(x_{1:t-1}^{k,(i)}) * R(x_{1:t-1}^{k,(i)})^{β_k} * P_θ(x_t^{k,(i)} | x_{1:t-1}^{k,(i)})]
```

The `P_θ` terms cancel because `P_θ(x_{1:t}) = P_θ(x_{1:t-1}) * P_θ(x_t | x_{1:t-1})`:

```
w_t^{k,(i)} = [R(x_{1:t}^{k,(i)}) / R(x_{1:t-1}^{k,(i)})]^{β_k}
```

**This is the key formula for implementation.** The weight is the **ratio of consecutive PRM scores, raised to the inverse temperature**.

**Special cases:**
- **Hot chain (β_1=0):** `w_t^{1,(i)} = 1` for all particles. No reweighting — pure prior sampling.
- **Cold chain (β_K=1):** `w_t^{K,(i)} = R(x_{1:t}) / R(x_{1:t-1})`. Full reward ratio.

**At t=1 (first step):** Define `R(x_{1:0}) = 1` (empty sequence has reward 1), so `w_1^{k,(i)} = R(x_1^{k,(i)})^{β_k}`.

### 3.3 Resample

Normalize the weights across particles within chain `k`:

```
W_t^{k,(i)} = w_t^{k,(i)} / Σ_{j=1}^{N} w_t^{k,(j)}
```

Resample `N` particles from the categorical distribution `Categorical(W_t^{k,(1)}, …, W_t^{k,(N)})`. Particles with high normalized weight are duplicated; particles with low weight are eliminated.

**Resampling strategy:** Use **systematic resampling** (lower variance than multinomial). Optionally, resample only when the Effective Sample Size (ESS) drops below a threshold:

```
ESS_t^k = 1 / Σ_{i=1}^{N} (W_t^{k,(i)})^2
```

Resample when `ESS_t^k < N / 2`. Otherwise, carry forward unnormalized weights.

---

## 4. Inter-Chain Communication: Mixture Proposals

This is the core novelty of T-REX. Instead of Metropolis-Hastings swap moves (which have vanishing acceptance rates in high dimensions), cold chains **absorb** particles from adjacent hot chains through a mixture proposal.

### 4.1 The Mixture Proposal Distribution

At step `t`, for chain `k+1` (the colder chain), each of its `N` particles is proposed from a mixture:

```
q_t^{k+1}(x_{1:t}) = (1 - λ) * [Local Extension] + λ * [Hot Injection]
```

**For each particle `i` independently:**

1. **Draw a Bernoulli coin flip:** `z_i ~ Bernoulli(λ)`
2. **If `z_i = 0` (Local Extension, probability `1 - λ`):**
   - Extend from cold chain's own previous particle: `x_t^{k+1,(i)} ~ P_θ(· | x_{1:t-1}^{k+1,(i)})`
   - The particle history `x_{1:t-1}` comes from chain `k+1`'s own particle pool (after resampling at step `t-1`).
3. **If `z_i = 1` (Hot Injection, probability `λ`):**
   - Sample a **full trajectory** `x_{1:t}` uniformly from chain `k`'s current particle population: `x_{1:t} ~ η̂_t^k(x_{1:t})`
   - where `η̂_t^k(x_{1:t}) = (1/N) Σ_{j=1}^{N} δ_{x_{1:t}^{k,(j)}}(x_{1:t})` is the empirical measure of the hot chain's particles.
   - **This replaces the entire particle trajectory**, not just the current token. The cold chain receives a complete sequence from the hot chain.

### 4.2 Importance Weights for Mixture Proposals

The weight depends on which component generated the particle:

**Local Extension weight (same as standard SMC):**

```
w_{t,local}^{(i)} = [R(x_{1:t}^{(i)}) / R(x_{1:t-1}^{(i)})]^{β_{k+1}}
```

**Hot Injection weight:**

The injected particle was effectively drawn from `π̃_t^k ∝ P_θ · R^{β_k}` (the hot chain's target). It needs to be reweighted to the cold chain's target `π̃_t^{k+1} ∝ P_θ · R^{β_{k+1}}`:

```
w_{t,inject}^{(i)} = π̃_t^{k+1}(x_{1:t}^{(i)}) / π̃_t^k(x_{1:t}^{(i)})
                    = [P_θ(x_{1:t}) * R(x_{1:t})^{β_{k+1}}] / [P_θ(x_{1:t}) * R(x_{1:t})^{β_k}]
                    = R(x_{1:t}^{(i)})^{β_{k+1} - β_k}
```

**Key property:** Since `R(·) ∈ (0, 1]` and `β_{k+1} > β_k`, the exponent `β_{k+1} - β_k > 0`, so:
- If `R` is high (close to 1): the weight is close to 1 → particle survives.
- If `R` is low (close to 0): the weight is close to 0 → particle is washed out during resampling.

**This is "zero-rejection" communication.** Every injected particle enters the pool and gets reweighted. No Metropolis-Hastings accept/reject step is needed.

### 4.3 Communication Direction

Communication flows **unidirectionally from hot to cold**: chain `k` feeds into chain `k+1`.

- Chain 1 (hottest) has no injection source — it runs pure base LLM SMC.
- Chain 2 receives injections from chain 1.
- Chain 3 receives injections from chain 2.
- …
- Chain K (coldest) receives injections from chain K-1.

This means chains can run **asynchronously**: hot chains can run ahead of cold chains. A cold chain at step `t` only needs the hot chain's particles at step `t` (which the hot chain has already computed).

---

## 5. Full Algorithm Pseudocode

```
INPUTS:
  - x_0: prompt
  - P_θ: base LLM
  - R: Process Reward Model, R(x_{1:t}) → (0, 1]
  - K: number of chains
  - N: number of particles per chain
  - T: max generation steps
  - β_{1:K}: temperature schedule, β_1=0, β_K=1
  - λ: mixing weight (default 0.1)

INITIALIZE:
  For each chain k = 1, …, K:
    For each particle i = 1, …, N:
      x_{1:0}^{k,(i)} = x_0   # all particles start with the prompt
      R_prev^{k,(i)} = 1.0     # R of empty sequence

FOR t = 1 TO T:

  # --- PHASE 1: Propagate & Reweight each chain independently ---
  For chain k = 1 to K:
    For particle i = 1 to N:
      # Propagate: sample next token from base LLM
      x_t^{k,(i)} ~ P_θ(· | x_{1:t-1}^{k,(i)})
      
      # Compute PRM score for extended sequence
      R_curr^{k,(i)} = R(x_{1:t}^{k,(i)})
      
      # Compute incremental weight
      w^{k,(i)} = (R_curr^{k,(i)} / R_prev^{k,(i)})^{β_k}

  # --- PHASE 2: Inter-chain communication (hot → cold injection) ---
  For chain k = 2 to K:   # chain 1 has no source
    For particle i = 1 to N:
      z_i ~ Bernoulli(λ)
      If z_i = 1:  # Hot Injection
        j ~ Uniform({1, …, N})   # pick random particle from chain k-1
        # Replace particle i's ENTIRE trajectory with chain (k-1)'s particle j
        x_{1:t}^{k,(i)} = x_{1:t}^{k-1,(j)}
        R_curr^{k,(i)} = R(x_{1:t}^{k,(i)})  # may reuse cached value
        
        # Recompute weight for injected particle
        w^{k,(i)} = R_curr^{k,(i)}^{β_k - β_{k-1}}

  # --- PHASE 3: Resample within each chain ---
  For chain k = 1 to K:
    Normalize weights: W^{k,(i)} = w^{k,(i)} / Σ_j w^{k,(j)}
    Compute ESS_k = 1 / Σ_i (W^{k,(i)})^2
    
    If ESS_k < N / 2:
      ancestors = SystematicResample(W^{k,(1:N)})
      For i = 1 to N:
        x_{1:t}^{k,(i)} = x_{1:t}^{k,(ancestors[i])}
        R_prev^{k,(i)} = R_curr^{k,(ancestors[i])}
        w^{k,(i)} = 1.0   # reset weights after resampling
    Else:
      For i = 1 to N:
        R_prev^{k,(i)} = R_curr^{k,(i)}
        # carry forward unnormalized weights (accumulate)

OUTPUT:
  Return cold chain's (k=K) final particles {x_{1:T}^{K,(i)}}_{i=1}^{N}
  with normalized weights {W_T^{K,(i)}}
  
  Normalizing constant estimate:
    Z_hat = Π_{t=1}^{T} [(1/N) Σ_{i=1}^{N} w_t^{K,(i)}]
```

---

## 6. Implementation Stages and Success Criteria

### Stage 1: Base LLM Inference Wrapper

**Goal:** Build a module that takes a prompt and a partial sequence and returns next-token log-probabilities and samples.

**What to implement:**
- `sample_next_token(model, x_{1:t-1}) → x_t` — samples from `P_θ(· | x_{1:t-1})`
- `log_prob(model, x_{1:t-1}, x_t) → float` — returns `log P_θ(x_t | x_{1:t-1})`
- Batched versions of both for `N` particles simultaneously.

**Success criteria:**
- Sampling produces valid tokens from the model's vocabulary.
- Log-probs match the model's actual distribution (verify by comparing against HuggingFace `model.generate` or equivalent).
- Batched inference handles `N` particles efficiently on GPU without OOM for target `N` (e.g., N=64 or N=128).

### Stage 2: Process Reward Model (PRM) Integration

**Goal:** Integrate a PRM that scores partial sequences.

**What to implement:**
- `score_prm(prm_model, x_{1:t}) → float ∈ (0, 1]` — returns the PRM score for a partial sequence.
- Must handle **partial** sequences (not just complete solutions).
- Clamp outputs: if raw PRM output is ≤ 0, clamp to a small epsilon (e.g., `1e-8`) to maintain strict positivity.
- Batched scoring for `N` particles.

**Success criteria:**
- PRM scores are in `(0, 1]` for all inputs.
- Scores correlate with reasoning quality: correct partial solutions score higher than incorrect ones on a held-out set.
- Scoring `N` partial sequences is fast enough to not bottleneck the SMC loop (profile and report latency).

### Stage 3: Single-Chain SMC Sampler

**Goal:** Implement the basic SMC loop (Section 3) for a single chain with fixed temperature β.

**What to implement:**
- The Propagate → Reweight → Resample loop.
- Incremental weight computation: `w_t^{(i)} = (R(x_{1:t}^{(i)}) / R(x_{1:t-1}^{(i)}))^β`.
- Systematic resampling with ESS-based triggering.
- Normalizing constant estimation: `Z_hat = Π_t [(1/N) Σ_i w_t^{(i)}]` (accumulate as log sum to avoid numerical underflow).
- Particle storage: maintain full trajectories `x_{1:t}^{(i)}` and cached PRM scores.

**Success criteria:**
- **Sanity check at β=0:** All weights should be exactly 1.0. Output should be indistinguishable from i.i.d. base model samples. Verify by comparing token distribution statistics.
- **Sanity check at β=1:** Particles should concentrate on higher-reward trajectories compared to β=0.
- **ESS tracking:** Log ESS at each step. ESS should not collapse to 1 (if it does, N is too small or the PRM is too sharp).
- **Z_hat estimation:** For a problem with known difficulty, the log normalizing constant should be finite and reasonable.
- **Correctness test:** On GSM8K, single-chain SMC at β=1 with N=64 should outperform Best-of-N with N=64 (i.e., accuracy should be higher because SMC resamples mid-generation rather than only scoring complete solutions).

### Stage 4: Multi-Chain Parallel Tempering (without Twists)

**Goal:** Implement the full T-REX framework with K chains and inter-chain mixture proposals.

**What to implement:**
- K independent SMC chains with temperatures `β_1=0 < β_2 < … < β_K=1`.
- Mixture proposal mechanism (Section 4): Bernoulli coin flip, local extension vs. hot injection.
- Hot injection weight formula: `w_{t,inject}^{(i)} = R(x_{1:t}^{(i)})^{β_{k+1} - β_k}`.
- Trajectory replacement: when injecting, replace the **entire** particle trajectory, not just the current token.
- Cache management: after injection, update cached PRM scores.

**Temperature schedule design:**
- Start with `K=4` chains: `β = [0.0, 0.33, 0.67, 1.0]` (linear).
- Also try geometric spacing: `β = [0.0, 0.1, 0.4, 1.0]` (more chains near β=0 for exploration).
- The paper doesn't prescribe a specific schedule — this is a hyperparameter to tune.

**Success criteria:**
- **Communication verification:** Track injection events. Verify that some injected particles survive resampling in the cold chain (i.e., injection rate × survival rate > 0). If no injected particles ever survive, λ is too large or the temperature gap is too wide.
- **Diversity improvement:** Measure the number of distinct final answers across particles in the cold chain. T-REX should produce more diverse valid solutions than single-chain SMC.
- **Accuracy improvement:** On GSM8K and MATH500, the cold chain's accuracy (fraction of particles with correct answer) should be ≥ single-chain SMC at β=1.
- **Asymptotic correctness:** As N → ∞, the cold chain's empirical distribution should approach the true posterior. Test by increasing N and checking that accuracy monotonically improves.

### Stage 5: Normalizing Constant and Diagnostics

**Goal:** Implement diagnostic tools to monitor algorithm health.

**What to implement:**
- **Per-chain ESS over time:** `ESS_t^k` for each chain `k` at each step `t`. Plot as heatmaps.
- **Per-chain log Z estimate:** `log Z_hat^k = Σ_t log[(1/N) Σ_i w_t^{k,(i)}]`.
- **Injection survival rate:** For each chain `k > 1`, track `(# injected particles that survive resampling) / (# injected particles)` per step.
- **Particle genealogy:** Track ancestor indices through resampling to measure path degeneracy.
- **Reward distribution per chain:** Histogram of `R(x_{1:t}^{k,(i)})` values at each step, per chain.

**Success criteria:**
- ESS never collapses to 1 for sustained periods (indicates weight degeneracy).
- Injection survival rate is non-trivial (e.g., > 5% of injections survive in the adjacent colder chain).
- Cold chain's reward distribution shifts rightward over steps (particles improve).
- No NaN or Inf values in weights or log Z estimates.

### Stage 6: Supervised Fine-Tuning (SFT) of Base Model

**Goal:** Fine-tune the base model on step-by-step reasoning data so it can produce chain-of-thought solutions.

**What to implement:**
- Filter PRM800K dataset for high-quality step-by-step solutions (following Feng et al., 2025).
- Standard SFT with cross-entropy loss on the filtered dataset.
- Models: Qwen2.5-7B, Qwen2.5-14B, Qwen2.5-32B, LLaMA-3.1-8B.

**Success criteria:**
- After SFT, the base model can produce step-by-step solutions in the expected format.
- Pass@1 accuracy on GSM8K improves over the pre-SFT base model.
- The model's outputs are parseable by the PRM (i.e., the PRM can score intermediate steps).

### Stage 7: Experimental Evaluation

**Goal:** Run the full experimental comparison.

**Benchmarks:**
- **GSM8K**: Grade-school math (8.5K problems). Primary benchmark.
- **MATH500**: 500 harder math problems. Secondary benchmark.
- **Optional extensions:** AMC/AIME (competition math), HumanEval (code), GPQA Diamond (science).

**Baselines to implement:**
1. **Best-of-N rejection sampling:** Generate N complete solutions, pick the one with highest reward. Test at temperatures `{0.6, 0.8, 1.0, 1.2}`. Sweep on first 100 problems, then fix temperature.
2. **PPO / GRPO:** Fine-tune with KL penalty `β ∈ {0.01, 0.02, 0.05}` against the SFT model.
3. **Standard SMC with PRM** (Puri et al., 2025): Single chain, β=1, no tempering.
4. **Twisted SMC** (Feng et al., 2025): Uses learned twist/value function for reweighting only (not in proposal).
5. **Multi-temperature SMC without communication:** K chains at different β, but no mixture proposals (λ=0). Tests the value of inter-chain communication.

**Experimental protocol:**
- Use **stratified sampling** to reduce variance: partition benchmark by difficulty, sample proportionally.
- Report: accuracy (% problems solved), sample efficiency (accuracy vs. N), wall-clock time.
- For each method, sweep N ∈ {16, 32, 64, 128, 256}.

**Success criteria:**
- T-REX at N=64 should match or exceed Best-of-N at N=256 on GSM8K (4× sample efficiency).
- T-REX should outperform standard SMC (Puri et al.) at equal N on both GSM8K and MATH500.
- T-REX should produce more diverse correct solutions than GRPO (measured by distinct final answers among correct particles).
- Multi-temperature SMC with communication (T-REX) should outperform multi-temperature SMC without communication (baseline 5), validating the mixture proposal mechanism.

---

## 7. Key Implementation Details

### 7.1 Numerical Stability

All weight computations should be done in **log space**:

```python
log_w_local = beta_k * (log_R_curr - log_R_prev)
log_w_inject = (beta_k1 - beta_k) * log_R_curr
```

For resampling, convert to normalized probabilities:
```python
log_W = log_w - logsumexp(log_w)  # normalized log weights
W = exp(log_W)                     # normalized weights for resampling
```

For the normalizing constant estimate, accumulate in log space:
```python
log_Z += logsumexp(log_w) - log(N)  # at each step t
```

### 7.2 PRM Score Caching

Each particle stores its current PRM score `R(x_{1:t})`. After resampling, the surviving particle inherits the ancestor's cached score. After propagation, the PRM is called **once per particle per step** on the extended sequence. After hot injection, the injected particle carries the hot chain's cached PRM score (no recomputation needed since the trajectory is identical).

### 7.3 KV-Cache Management

Each particle has its own KV-cache for the base LLM. After resampling:
- Surviving particles: copy the ancestor's KV-cache.
- Eliminated particles: discard their KV-cache.
- After hot injection: the injected particle needs the hot chain's KV-cache for the injected trajectory. Either copy it or recompute from scratch.

This is the **primary memory bottleneck**. For N particles × K chains, we need `N × K` KV-caches. Strategies:
- Use KV-cache compression or quantization.
- Recompute KV-caches from sequences rather than storing all of them (trades compute for memory).
- Use PagedAttention (vLLM-style) for efficient KV-cache management.

### 7.4 Systematic Resampling

```python
def systematic_resample(weights, N):
    """
    Args:
        weights: normalized weights, shape (N,), sum to 1
        N: number of particles
    Returns:
        ancestors: array of ancestor indices, shape (N,)
    """
    positions = (np.arange(N) + np.random.uniform()) / N
    cumsum = np.cumsum(weights)
    ancestors = np.searchsorted(cumsum, positions)
    return ancestors
```

### 7.5 Effective Sample Size

```python
def compute_ess(log_weights):
    """
    Args:
        log_weights: unnormalized log weights, shape (N,)
    Returns:
        ESS: effective sample size, float in [1, N]
    """
    log_W = log_weights - logsumexp(log_weights)
    return np.exp(-logsumexp(2 * log_W))
```

### 7.6 Handling End-of-Sequence

When a particle generates an EOS token before step T:
- Freeze the particle: do not propagate further. Its PRM score remains fixed.
- Its incremental weight at all subsequent steps is 1.0 (since `R(x_{1:t}) / R(x_{1:t-1}) = 1` when the sequence is unchanged).
- It still participates in resampling and can be eliminated or duplicated.

### 7.7 Stratified Sampling for Evaluation

Partition the benchmark into difficulty buckets (e.g., by number of reasoning steps required or by historical solve rate). Sample problems proportionally from each bucket. Report per-bucket accuracy to understand where T-REX helps most (expected: hard problems with narrow passages).

---

## 8. Hyperparameter Summary

| Parameter | Default | Range to Sweep | Notes |
|---|---|---|---|
| N (particles) | 64 | {16, 32, 64, 128, 256} | Primary compute knob |
| K (chains) | 4 | {2, 3, 4, 6} | More chains = more exploration but more compute |
| β schedule | Linear [0, 0.33, 0.67, 1] | Linear, geometric, learned | Start with linear |
| λ (mix weight) | 0.1 | {0.05, 0.1, 0.2, 0.3} | Too high → cold chain is flooded with hot noise |
| ESS threshold | N/2 | {N/4, N/3, N/2} | Lower = less resampling, more weight variance |
| PRM granularity | Per reasoning step | Per token, per step, per sentence | Coarser = cheaper but less precise |
| Temperature for base LLM | 1.0 | {0.8, 1.0, 1.2} | Applied inside P_θ before SMC |

---

## 9. Data Flow Diagram

```
Step t:

Chain 1 (β=0, hot)     Chain 2 (β=0.33)      Chain 3 (β=0.67)      Chain 4 (β=1, cold)
─────────────────       ─────────────────      ─────────────────      ─────────────────
│ Propagate (LLM) │     │ Propagate (LLM) │    │ Propagate (LLM) │    │ Propagate (LLM) │
│ Reweight (β=0)  │     │ Reweight (β=.33)│    │ Reweight (β=.67)│    │ Reweight (β=1)  │
│ [weights = 1]   │     │                 │    │                 │    │                 │
└────────┬────────┘     └────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                       │                      │                      │
         │──── λ inject ────────>│                      │                      │
         │                       │──── λ inject ───────>│                      │
         │                       │                      │──── λ inject ───────>│
         │                       │                      │                      │
         │                       │ Reweight injected    │ Reweight injected    │ Reweight injected
         │                       │ w = R^{Δβ}           │ w = R^{Δβ}           │ w = R^{Δβ}
         │                       │                      │                      │
         v                       v                      v                      v
│ Resample (ESS)  │     │ Resample (ESS)  │    │ Resample (ESS)  │    │ Resample (ESS)  │
─────────────────       ─────────────────      ─────────────────      ─────────────────

                                                                      ↓ (at step T)
                                                                   OUTPUT: cold chain
                                                                   particles + weights
```

---

## 10. Expected Computational Cost

Per step `t`, per chain `k`:
- **LLM forward passes:** `N` (one per particle for propagation). This is the dominant cost.
- **PRM evaluations:** `N` (one per particle).
- **Resampling:** `O(N)` (negligible).
- **Injection:** `O(λN)` additional bookkeeping (negligible).

**Total per step:** `K × N` LLM forward passes + `K × N` PRM evaluations.
**Total for full generation:** `T × K × N` LLM forward passes.

**Comparison to Best-of-N:** Best-of-N with `M` samples requires `T × M` LLM forward passes (no PRM during generation). T-REX with equivalent total compute uses `M = K × N` effective forward passes per step but gets the benefit of mid-generation resampling and inter-chain communication.