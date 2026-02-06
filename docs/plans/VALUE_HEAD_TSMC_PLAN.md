# Implementation Plan: Value Head + Twisted SMC Baseline

**Status:** Active Implementation (Mode A baseline path implemented; experiments in progress)  
**Created:** 2026-01-31  
**Last Updated:** 2026-02-05  
**Goal:** Implement Value Head architecture and training to enable end-to-end Twisted SMC testing

---

## 1. Overview

### 1.1 Current Status Update (2026-02-05)

- Implemented: `trex/models/value_head.py` (Linear/MLP/Attention-pooled value heads).
- Implemented: `trex/models/twist_model.py` (raw-logit head output with explicit `prob` / `log_prob` mapping).
- Implemented: `trex/smc/tsmc_particle_filter.py` (LLM rollout + twist weighting + metadata-safe resampling).
- Implemented: `trex/training/value_trainer.py` and `trex/training/train_value_head.py` (self-distillation loop + CLI).
- Implemented: `trex/baselines/tsmc_baseline.py` + `trex/baselines/tsmc_config.py` (TSMC baseline runner/config).
- Implemented: baseline scripts for value-head training and TSMC inference.
- Default baseline behavior now uses `final_selection_mode="twist_weight"` (ORM optional, not default).
- Reproduction defaults keep chat templating off (`apply_chat_template=False`) and use delimiter boundaries (`\n\n`).

This plan implements the **Value Head** and **Twisted SMC (TSMC) Baseline** components of T-REX. The Value Head is a learnable **potential function** (unnormalized; may be zero) over partial reasoning traces. When integrated with Twisted SMC, it guides particle filtering toward high-value regions without requiring an external PRM.

### Key Differences from Standard SMC

| Aspect | Standard SMC (Implemented) | Twisted SMC (This Plan) |
|--------|---------------------------|------------------------|
| Scoring | External PRM (Qwen2.5-Math-PRM-7B) | Learned Value Head |
| Weight Update | `w_t = PRM_score` | **Mode A only (this plan):** base proposal `q_t = p_0`, twist on target: `w_t = ψ_t / ψ_{t-1}`. **Mode B (twisted proposal)** deferred to future work. |
| Training | None (inference only) | Self-distillation from rollouts |
| Compute Cost | High (separate PRM model) | Depends on proposal choice and model sharing |

---

## 2. Architecture Analysis

### 2.1 Integration Points (from codebase analysis)

**Existing Infrastructure:**
- [`trex/smc/twisted_smc.py`](trex/smc/twisted_smc.py:1) - Core TSMC logic with `compute_twisted_weights()` and `TwistedSMC` class
- [`trex/smc/llm_particle_filter.py`](trex/smc/llm_particle_filter.py:1) - LLM-aware particle filter (extends base ParticleFilter)
- [`trex/tests/test_smc/test_twisted_smc.py`](trex/tests/test_smc/test_twisted_smc.py:1) - Twisted SMC unit tests
- [`trex/tests/test_smc/test_llm_particle_filter.py`](trex/tests/test_smc/test_llm_particle_filter.py:1) - Stepwise generation tests
- [`openrlhf/models/model.py`](openrlhf/models/model.py:1) - Reference implementation of value head attachment
- [`trex/models/reward_model.py`](trex/models/reward_model.py:1) - PRM/ORM wrapper pattern to follow

**Key Observations:**
1. `TwistedSMC` already expects a `value_function: Callable[[List[str]], torch.Tensor]`
2. `LLMParticleFilter` already handles step-wise rollouts and PRM scoring; we can reuse its stepping logic
3. OpenRLHF's `get_llm_for_sequence_regression()` shows how to attach heads to base LLMs
4. Twist space must be explicit end-to-end. Head output, loss, and weight update must be consistent with the chosen space. Default: `twist_space="log_prob"`.

### 2.2 Mathematical Clarification: Twisted Proposal vs. Weights (MUST CHOOSE ONE)

**Optimal one-step twisted proposal (twist-induced proposal):**
$$q^{\psi}_t(s_t | s_{1:t-1}) = \frac{p_0(s_t | s_{1:t-1})\,\psi_t(s_{1:t})}{\int p_0(s'_t | s_{1:t-1})\,\psi_t(s_{1:t-1}, s'_t)\, ds'_t}$$

Define the normalizer (partition function):
$$Z_t(s_{1:t-1}) = \int p_0(s'_t | s_{1:t-1})\,\psi_t(s_{1:t-1}, s'_t)\, ds'_t$$

**Incremental importance weights (general):**
$$w_t(s_{1:t}) = \frac{\pi_t(s_{1:t})}{\pi_{t-1}(s_{1:t-1})\,q_t(s_t|s_{1:t-1})}$$

**Specialization:**
- **Mode A (base proposal)**: If we **do not** sample from the twisted proposal and keep $q_t = p_0$, then the p₀ terms cancel and the incremental weight reduces to the twist ratio:
  $$w_t = \frac{\psi_t(s_{1:t})}{\psi_{t-1}(s_{1:t-1})}$$
  In this mode, the **target** is twisted (unnormalized target $\tilde{\pi}_t \propto p_0\,\psi_t$), while the proposal stays base. This is the current `trex/smc/twisted_smc.py` behavior and is mathematically correct **only** under $q_t = p_0$.
- **Mode B (optimal twisted proposal)**: If we **do** sample from $q_t^{\psi}$, substitute $q_t = \frac{p_0\psi_t}{Z_t}$ into the standard SMC weight:
  $$w_t = \frac{p_0\,\psi_t}{p_0\,\psi_{t-1}} \cdot \frac{Z_t}{p_0\,\psi_t} = \frac{Z_t(s_{1:t-1})}{\psi_{t-1}(s_{1:t-1})}$$
  This requires computing/estimating $Z_t$, and **you must include this correction** or the method is biased.
  *Future work only in this plan; keep the math documented but do not implement in the baseline.*

**Potential definition:** $\psi$ is an **unnormalized potential** (nonnegative; may be zero). In `log_prob` space, we store $\log \psi$ (≤ 0, possibly −∞ for ψ=0).

**Choice for this plan:** **Mode A only (base proposal)**. Mode B is deferred to future work (see below). Keep configuration hooks and code structure to allow Mode B later, but do not implement or expose it in the baseline path.

### 2.3 Why Default to Base Proposal (Feng et al., 2024)

Feng et al. motivate using the base model proposal $q_t = p_0$ instead of the optimal twisted proposal $q_t^*$:
- **Intractability:** For multi-token steps, the normalization $Z_t$ requires summing over an exponentially large set of continuations.
- **Variance/degeneracy:** High-dimensional proposals can produce severe weight degeneracy even with approximations.

**Compromise (locally optimal twist for base proposal):**
When sampling from $p_0$, they derive a locally optimal twist that reduces weight variance:
$$\psi_t^{p}(x_{1:t}) \propto \sqrt{V^{p}(x_{1:t})}$$
Expose this as an optional variance-reduction mode: `twist_mode = "value"` vs. `"sqrt_value"`.
**Definition:** If `twist_space="prob"`, apply $\psi \leftarrow \sqrt{\psi}$; if `twist_space="log_prob"`, apply $\log\psi \leftarrow 0.5 \cdot \log\psi`.

**SIS weight for full sequence:**
$$w(s_{1:T}) = \frac{\tilde{\sigma}(s_{1:T})}{q(s_{1:T})}
= \frac{p_0(s_{1:T})\,\phi(s_{1:T})}{q(s_{1:T})}$$

**Mode B (future work): Estimating the partition function.** Implementing $q_t^{\psi}$ requires a *per-prefix* estimate of
$$Z_t(s_{1:t-1}) = \mathbb{E}_{s_t \sim p_0(\cdot | s_{1:t-1})} [\psi_t(s_{1:t})].$$
This is not obtained by averaging $\psi_t$ across different prefixes. A correct estimator requires **multiple** samples per prefix, e.g.
$$\hat{Z}_t^{(i)} \approx \frac{1}{M}\sum_{m=1}^{M}\psi_t(s_{1:t-1}^{(i)}, s_t^{(i,m)}).$$
If Mode B is later implemented, surface $M$ in config and log the estimator. Until then, keep the proposal base-only.

**Per-step local normalizer (conditional partition function):**
$$Z_t(s_{1:t-1}) = \mathbb{E}_{s_t \sim p_0(\cdot | s_{1:t-1})} \left[ \psi_t(s_{1:t}) \right]$$
For future Mode B, approximate **per prefix** using multiple samples from $p_0(\cdot|s_{1:t-1})$ (not across different prefixes).

**Future Mode B Note (when to compute $\hat{Z}_t$):**
If Mode B is implemented, compute $\hat{Z}_t$ **after generating** per-prefix candidate continuations for step $t$ and **before resampling**, using those pre-resampling samples. This keeps the estimator aligned with the proposal distribution used to generate the candidates.

**Plan requirement:** Mode A only in this plan. Keep the code structured so Mode B can be added later without rewiring the base proposal path.

**Prefix definition (state):** A “prefix” $s_{1:t}$ always means the **entire generated text so far** at the moment of a weight update / resampling decision (not just the last token or step header).

**Proposal consistency (Mode A default):** Define $p_0$ as the **explicit sampling distribution used in generation** after applying temperature/top-p/top-k. Any additional changes to sampling modify $p_0$; treat those as part of $q_t$ or add $p_0/q_t$ corrections.

**Step boundary policy (default):** Define update steps every **K tokens**. **K is fixed and user-configured.** Regex-based step protocols are optional and off by default; if enabled, emit a warning when no step markers are detected.

### 2.4 Issues / Gaps to Resolve Before Implementation

1. **Twist-space mismatch risk:** The plan mixes *logits*, *logψ*, and *ψ* in several places. If the value head applies `LogSigmoid` internally, then `BCEWithLogits` is **incorrect** (double activation). We must define a single source of truth: value head outputs **raw logits**; twist-space mapping happens in the wrapper/scorer.
2. **Step-boundary vs. token-boundary ambiguity:** TSMC updates are per-step, but the value head produces per-token values. We must define **which token index** represents step `t` (last token in the step segment) and ensure it is used consistently in weight updates.
3. **Proposal mismatch with sampling controls:** If generation uses top-p/top-k/temperature, then the proposal is **not** the base model distribution. Treat the actual sampling distribution as `q_t`, or record the correction `p_0/q_t` if we later need unbiased weights.
4. **Value head calibration + label imbalance:** Monte Carlo labels are sparse positives. Without class balancing or temperature scaling, the head can collapse to near-zero predictions and still minimize BCE. Add balancing or focal loss if necessary and track calibration metrics.
5. **Off-policy data drift:** If rollouts are ever collected under a twisted policy (future), the self-distillation targets become off-policy. The plan must document **when** to collect under p₀ vs. π^ψ, and whether to reweight or reset.
6. **Numerical stability for ψ≈0:** logψ can be −∞. Weight ratios must be computed in log-space with clipping/ε, or particles will die deterministically.
7. **Metadata-resampling consistency:** If we resample particles but not their associated `prev_logψ`, weights become incorrect. The current plan calls this out but needs an explicit data structure and tests.
8. **Shared-weight scoring vs. vLLM:** If we ever share weights between generator and scorer, we need a concrete integration path (vLLM does not expose hidden states by default). For now, keep a **separate** HF model and document memory cost.
9. **Attention-pooled head is sequence-level:** If we keep per-step updates, we must map the pooled value to step boundaries (or drop this head for the baseline).

### 2.5 Decisions (Locked In for Baseline)

1. **Value head output = raw logits.** No sigmoid/logsigmoid inside the head.
2. **Default twist_space = `log_prob`.** TSMC baseline will set `log_space=True`; `TwistedSMCConfig.log_space` default remains `False` for back-compat.
3. **Per-token heads only for baseline.** Attention-pooled head is allowed but **not used** in baseline runs.
4. **Step-resampling as default.** Use fixed K-token boundaries for SMC iterations; "## Step N:" markers are optional.
5. **Resampling defaults:** `resampling_strategy="every_step"`, `resampling_method="systematic"` (match SMC baseline for comparability).
6. **Final selection:** Default to twist lineage weight (`final_selection_mode="twist_weight"`); keep ORM and majority vote as optional modes.
7. **Log-space weight updates:** Use log-sum-exp normalization when `log_space=True`.

### 2.6 Value Head Architecture Options

Per IMPLEMENTATION_PLAN.md Section 2.1, we support multiple architectures:

```
┌─────────────────────────────────────────────────────────────────┐
│  Value Head Options (plug-and-play via config)                  │
├─────────────────────────────────────────────────────────────────┤
│  1. Simple Linear:                                              │
│     Hidden [B,T,H] → Linear(H, 1) → Logits [B,T,1]             │
│     - Fastest, might underfit complex reasoning                 │
│                                                                 │
│  2. MLP (Recommended):                                          │
│     Hidden [B,T,H] → Linear(H, H//4) → ReLU → Linear(H//4, 1)  │
│                    → Logits [B,T,1]                            │
│     - Good balance of capacity and efficiency                   │
│                                                                 │
│  3. Attention-Pooled:                                           │
│     Hidden [B,T,H] → AttentionPool(T, 1) → Linear(H, 1)         │
│                    → Logits [B,1]                              │
│     - For long contexts, aggregates sequence info               │
└─────────────────────────────────────────────────────────────────┘
```

**Note:** For step-wise weight updates, prefer per-token heads (Linear/MLP). The attention-pooled head is only valid if we define a sequence-level twist or map it to step boundaries.

---

## 3. Component Design

### 3.1 Module Structure

```
trex/
├── models/
│   ├── __init__.py              # Extend exports for ValueHead/TwistModel
│   ├── value_head.py            # NEW: Value head architectures
│   └── twist_model.py           # NEW: Base LLM + Value Head wrapper
├── training/
│   ├── __init__.py              # NEW: Training module exports
│   ├── value_trainer.py         # NEW: Self-distillation training loop
│   └── trajectory_buffer.py     # NEW: Rollout storage
├── smc/
│   ├── twisted_smc.py           # EXISTS: Twist weights + TwistedSMC base
│   ├── llm_particle_filter.py   # EXISTS: LLM rollouts + PRM scoring
│   └── tsmc_particle_filter.py  # OPTIONAL: TSMC with value head scoring (prefer extending llm_particle_filter.py)
└── baselines/
    └── tsmc_baseline.py         # NEW: Evaluation runner
```

### 3.2 Value Head Module (`trex/models/value_head.py`)

**Purpose:** Predict expected future reward from hidden states.

**Interface:**
```python
class ValueHead(nn.Module):
    """Base class for value heads."""
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
        Returns:
            logits: [batch_size, seq_len, 1] (raw, unnormalized)
        """
        pass

class LinearValueHead(ValueHead):
    """Simple linear projection."""
    def __init__(self, hidden_dim: int):
        self.linear = nn.Linear(hidden_dim, 1)
        
class MLPValueHead(ValueHead):
    """Two-layer MLP with ReLU."""
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: Optional[int] = None,
    ):
        intermediate_dim = intermediate_dim or hidden_dim // 4
        layers = [
            nn.Linear(hidden_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, 1),
        ]
        self.layers = nn.Sequential(*layers)
```

**Key Design Decisions:**
1. Output per-token values (not just sequence-level) to support step-by-step guidance
2. Twist space is explicit (see contract below). Default is `twist_space="log_prob"`.
3. No layer norm (unnecessary for single-task regression)
4. Initialize final layer to small values (logits near 0 → ψ≈0.5 in prob space; logψ≈log(0.5) in log-prob space) for stable training start
5. **Value head outputs raw logits only.** Convert to ψ/logψ in the wrapper/scorer to avoid double-activation.

**Twist-space contract (end-to-end):**
- **Value head output:** `z` (raw logits, no activation).
- `twist_space="prob"` ⇒ **ψ = σ(z)**, weights use ψ ratio, loss uses **BCEWithLogits(z, R)** or MSE on ψ (if explicitly requested).
- `twist_space="log_prob"` ⇒ **logψ = logσ(z)**, weights use exp(logψ_t−logψ_{t−1}); loss uses **BCEWithLogits(z, R)**, but values passed to TwistedSMC are **logψ**, not logits.
  - This allows ψ to be **zero** (logψ = −∞) and logψ to be negative. Use ε/clipping for numerical stability.
**TwistedSMC mapping:** `TwistedSMCConfig.log_space = (twist_space == "log_prob")`.

### 3.3 Twist Model Wrapper (`trex/models/twist_model.py`)

**Purpose:** Combine base LLM with value head for end-to-end inference.

**Interface:**
```python
class TwistModel(nn.Module):
    """
    Base LLM with attached value head for Twisted SMC.
    
    Wraps a causal LM and extracts hidden states at each position
    to compute value estimates.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        value_head_type: str = "mlp",
        twist_space: Literal["log_prob", "prob"] = "log_prob",
        freeze_base_model: bool = True,
        share_base_with_generator: bool = False,
    ):
        """
        Args:
            model_name_or_path: HuggingFace model ID or local path
            value_head_type: "linear", "mlp", or "attention_pooled"
            twist_space: "log_prob" (logσ output) or "prob" (σ output)
            freeze_base_model: Whether to freeze base LLM (recommended)
            share_base_with_generator: If True, use shared weights with generator (future optimization)
        """
        pass
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning logits and values.
        
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            values: [batch_size, seq_len, 1] (ψ or logψ depending on twist_space)
        """
        pass
    
    def get_values_for_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Convenience method for TSMC integration.
        
        Args:
            texts: List of partial reasoning traces
        Returns:
            values: [len(texts),] - Final position value for each text (prob or log-prob)
        """
        pass
```

**Implementation Notes:**
- Use `output_hidden_states=True` to extract final layer hidden states
- Extract value at the **last non-padding position** for each sequence
- If step boundaries are token-based (K tokens), choose the value at the **last token of the step** rather than the end of the full sequence
- Convert logits → ψ/logψ in the wrapper based on `twist_space`; keep logits for BCEWithLogits training
- Compatible with vLLM for efficient generation (value head runs separately)
- **Model sharing (future optimization):** default is **separate HF model** for scoring; expose a `share_base_with_generator` flag and a scorer interface to allow shared-weight implementations later.
- If `twist_space="log_prob"`, keep access to **raw logits** (pre-LogSigmoid) for BCEWithLogits during training; only convert to logψ for scoring/weights.

### 3.4 TSMC Particle Filter (extend `trex/smc/llm_particle_filter.py`; optional `trex/smc/tsmc_particle_filter.py`)

**Purpose:** Extend `TwistedSMC` with LLM generation and value head scoring.

**Implementation Note:** Prefer **extending** `trex/smc/llm_particle_filter.py` with a switchable scorer
interface (PRM vs. value head) to avoid duplicating rollout logic. Only introduce a new wrapper if
absolutely necessary.

**Correctness Note:** Store `ψ_{t-1}` in **per-particle metadata** so resampling carries the correct
value history with each particle. Otherwise, weight ratios are paired with the wrong particles and
the algorithm is invalid.

**Resampling Note (metadata sync):** When resampling, copy the full particle metadata (including
`prev_value` / `prev_log_value`) with the particle text. Do not keep previous values as a separate
global tensor unless it is permuted with the resampling indices.

**Config Note (make mode explicit):** Add config flags such as
`proposal_mode: Literal["base", "twisted"] = "base"` (keep `"twisted"` reserved for future),
`twist_mode: Literal["value", "sqrt_value"] = "value"`, and
`twist_space: Literal["log_prob", "prob"] = "log_prob"`
so logs/metrics clearly report which math is being used.

**Numerical Note (log-space):**
- Maintain weights in log-space: `log_w_t = log_w_{t-1} + (logψ_t - logψ_{t-1})`.
- Clamp `logψ` to `[-1e6, 0]` (or use `eps=1e-8` with `log(psi+eps)` in prob space).
- Normalize log-weights with `logsumexp` before ESS/resampling.

**Resampling Note (scheme + determinism):**
- Use **systematic** or **stratified** resampling to reduce variance.
- Seed and log resampling indices for reproducibility and debugging.

**Interface:**
```python
class TSMCLLMParticleFilter(TwistedSMC):
    """
    Twisted SMC with LLM generation and learned value function.
    
    Extends TwistedSMC to add:
    - LLM generation via vLLM
    - Value head scoring for weight updates
    - Self-contained (no external PRM needed)
    """
    
    def __init__(
        self,
        config: TSMCConfig,
        generator: "LLM",           # vLLM instance
        twist_scorer: "TwistScorer",    # Value head scorer (separate or shared)
    ):
        super().__init__(config)
        self.generator = generator
        self.twist_scorer = twist_scorer
        
        # Set value function for base TwistedSMC
        self.set_value_function(self._compute_values)
    
    def _compute_values(self, texts: List[str]) -> torch.Tensor:
        """Compute values using twist scorer."""
        return self.twist_scorer.score_texts(texts)
    
    def step(self) -> None:
        """
        Single TSMC iteration:
        1. Generate next step with LLM using the **base proposal** q_t = p_0
        2. Compute values with twist model
        3. Update weights using Mode A:
           - w_t = ψ_t / ψ_{t-1} (or exp(logψ_t - logψ_{t-1}) in log-prob space)
        4. Resample if ESS < threshold
        """
        pass
```

**Scorer Interface (future sharing):**
Define a small interface to decouple scoring from generation:
```
class TwistScorer(Protocol):
    def score_texts(self, texts: List[str]) -> torch.Tensor:
        """Return ψ or logψ for each text, per twist_space."""
```
This allows:
- `HFSeparateScorer` (default): separate HF model with value head
- `SharedWeightScorer` (future): shared base with generator + head
Both satisfy the same TSMC API, keeping integration unobtrusive.

### 3.5 Value Head Training (`trex/training/value_trainer.py`)

**Purpose:** Train value head via self-distillation from rollouts.

**Algorithm (from HIGH_LEVEL_CONTEXT.md Section 4, Phase 3):**
```
1. Data Collection (Rollouts) **[DEFAULT: p₀ only]**:
   For each prompt in training set:
     - Sample K trajectories using the **base proposal p₀** (temperature/top-p/top-k define the actual proposal)
     - Verify final answers → Binary reward R ∈ {0, 1}
   
2. Label Assignment (Monte Carlo):
   - Assign final reward R to ALL steps in trajectory:
     $$y_t = R \quad \forall t \in \{1,\dots,T\}$$
   - Dataset: D = {(x_{1:t}, R) for all steps t in all trajectories}
   - **Class balance:** maintain a target positive fraction (e.g., 30–50%) via downsampling negatives or class-weighted BCE.
   - **Sequence truncation:** cap to `max_seq_len` or `max_tokens_per_state` to keep scoring cost bounded.

3. Training:
   - If `twist_space="log_prob"`: use BCEWithLogits on logits vs. R, but convert to logψ for weight updates.
   - If `twist_space="prob"`: minimize MSE or BCE on ψ vs. R.
   - Only update value head (base model frozen)
   - Track calibration (ECE/Brier), not just loss.
   - Optional label smoothing (e.g., R ∈ {0.05, 0.95}) to stabilize early training.

4. Iteration:
   - (Optional) Use improved ψ to bias sampling
   - Collect new data with better guidance
   - If you deviate from base p₀, document that targets are under the twisted policy
```

**Interface:**
```python
@dataclass
class ValueTrainingConfig:
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_rollouts_per_prompt: int = 8
    update_frequency: int = 100  # Steps between value updates
    ema_decay: float = 0.99      # For target value smoothing
    max_steps: int = 1000

class ValueTrainer:
    """Trainer for value head via self-distillation."""
    
    def __init__(
        self,
        twist_model: TwistModel,
        config: ValueTrainingConfig,
    ):
        self.twist_model = twist_model
        self.config = config
        self.optimizer = AdamW(twist_model.value_head.parameters(), lr=config.learning_rate)
    
    def collect_rollouts(
        self,
        prompts: List[str],
        generator: "LLM",
        verifier: MathVerifier,
    ) -> List[Trajectory]:
        """Generate trajectories and assign rewards."""
        pass
    
    def train_step(self, trajectories: List[Trajectory]) -> float:
        """Single training step. Returns loss."""
        # Extract (state, reward) pairs from trajectories
        # Forward pass through twist model
        # Compute loss based on twist_space (BCEWithLogits or MSE/BCE)
        # Backprop and update
        pass
    
    def train(self, dataset: List[Dict], verifier: MathVerifier) -> None:
        """Full training loop with periodic rollout collection."""
        pass
```

**Implementation detail (batching):**
- Batch states by **length buckets** to minimize padding.
- Cache tokenizer outputs if multiple epochs over the same buffer.

### 3.6 Trajectory Buffer (`trex/training/trajectory_buffer.py`)

**Purpose:** Store and manage rollout data for training.
**Detail:** Store step boundaries (token indices or offsets) alongside text so we can map per-token values to step `t`.

**Interface:**
```python
@dataclass
class Trajectory:
    """Single reasoning trajectory with reward."""
    prompt: str
    steps: List[str]           # Reasoning steps
    full_text: str            # Complete generated text
    reward: float             # Binary: 1.0 if correct, 0.0 otherwise
    
    def get_state_reward_pairs(self) -> List[Tuple[str, float]]:
        """Return (partial_trace, reward) for each step."""
        pairs = []
        for i in range(len(self.steps)):
            partial = self.prompt + "".join(self.steps[:i+1])
            pairs.append((partial, self.reward))
        return pairs

    def get_state_token_indices(self) -> List[int]:
        """Return token indices for each step boundary (if using token-based steps)."""
        pass

class TrajectoryBuffer:
    """Buffer for storing and sampling trajectories."""
    
    def __init__(self, max_size: int = 10000):
        self.trajectories: Deque[Trajectory] = deque(maxlen=max_size)
    
    def add(self, trajectory: Trajectory) -> None:
        self.trajectories.append(trajectory)
    
    def sample(self, batch_size: int) -> List[Trajectory]:
        return random.sample(self.trajectories, min(batch_size, len(self.trajectories)))
    
    def get_all_state_reward_pairs(self) -> List[Tuple[str, float]]:
        """Flatten all trajectories into (state, reward) pairs."""
        pairs = []
        for traj in self.trajectories:
            pairs.extend(traj.get_state_reward_pairs())
        return pairs
```

### 3.7 TSMC Baseline Runner (`trex/baselines/tsmc_baseline.py`)

**Purpose:** Evaluation harness for Twisted SMC (similar to `smc_steering_baseline.py`).

**Interface:**
```python
@dataclass
class TSMCBaselineConfig:
    """Configuration for TSMC baseline evaluation."""
    # Model paths
    base_model_path: str = "Qwen/Qwen2.5-7B"
    value_head_path: Optional[str] = None  # If None, use untrained/random
    twist_space: Literal["log_prob", "prob"] = "log_prob"
    share_base_with_generator: bool = False  # Future optimization
    
    # TSMC parameters
    n_particles: int = 16
    max_steps: int = 10
    ess_threshold: float = 0.5
    
    # Generation parameters
    temperature: float = 0.8
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: int = 2048
    
    # Evaluation
    dataset: str = "gsm8k_test"
    output_dir: str = "./results/tsmc_baseline"

class TSMCBaselineRunner:
    """Runner for TSMC baseline evaluation."""
    
    def __init__(self, config: TSMCBaselineConfig):
        self.config = config
        self._load_models()
        self._setup_filter()
    
    def _load_models(self) -> None:
        """Load base LLM, value head, and verifier."""
        # Load vLLM generator
        # Load twist model (base + value head)
        # Load math verifier
        pass
    
    def _setup_filter(self) -> None:
        """Initialize TSMC particle filter."""
        self.filter = TSMCLLMParticleFilter(
            config=self.config.to_smc_config(),
            generator=self.generator,
            twist_scorer=self.twist_model,
        )
    
    def evaluate_problem(self, problem: Dict) -> Dict:
        """
        Evaluate single problem with TSMC.
        
        Returns:
            {
                "problem_id": str,
                "prompt": str,
                "ground_truth": str,
                "final_answer": str,
                "correct": bool,
                "n_particles": int,
                "n_steps": int,
                "particle_history": [...],
            }
        """
        pass
    
    def run_evaluation(self, dataset: List[Dict]) -> Dict:
        """Run full evaluation and return metrics."""
        pass
```

---

## 4. Training Pipeline

### 4.1 Self-Distillation Loop

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Self-Distillation Training                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Initialization: Load base LLM, attach random value head            │
│                     ↓                                                │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Iteration i                                                    │  │
│  │                                                                │  │
│  │  1. COLLECT ROLLOUTS (DEFAULT: p₀ only)                       │  │
│  │     For each training prompt:                                  │  │
│  │       - Generate K trajectories using base proposal p₀        │  │
│  │         (temperature/top-p/top-k define the actual proposal)  │  │
│  │       - Verify final answers                                   │  │
│  │       - Assign binary rewards R ∈ {0, 1}                       │  │
│  │                                                                │  │
│  │  2. BUILD TRAINING DATA                                        │  │
│  │     For each trajectory:                                       │  │
│  │       - Assign R to ALL intermediate states                    │  │
│  │       - Dataset: {(x_{1:t}, R)}                                │  │
│  │                                                                │  │
│  │  3. TRAIN VALUE HEAD                                           │  │
│  │     For N gradient steps:                                      │  │
│  │       - Sample batch from dataset                              │  │
│  │       - Forward: V(x_{1:t})                                    │  │
│  │       - Loss: BCEWithLogits or MSE/BCE (per twist_space)       │  │
│  │       - Update value head only (base model frozen)             │  │
│  │                                                                │  │
│  │  4. EVALUATE                                                   │  │
│  │     - Run TSMC on validation set                               │  │
│  │     - Track accuracy improvement                               │  │
│  │                     ↓                                          │  │
│  │  Converged? ──Yes──→ Save checkpoint & Exit                    │  │
│  │     │                                                          │  │
│  │     No                                                         │  │
│  │     ↓                                                          │  │
│  └────┘ (repeat)                                                  │  │
│                                                                    │  │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Training Script (`trex/scripts/train_value_head.sh`)

```bash
#!/bin/bash
#SBATCH --job-name=train-value-head
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=480G
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00

# ... (SLURM setup similar to other scripts)

python -m trex.training.train_value_head \
    --base_model Qwen/Qwen2.5-7B \
    --value_head_type mlp \
    --dataset gsm8k_train \
    --num_rollouts_per_prompt 8 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --max_iterations 100 \
    --output_dir $SCRATCH/t-rex/results/value_head_training
```

---

## 5. Evaluation Pipeline

### 5.1 TSMC Baseline Script (`trex/scripts/run_tsmc_baseline.sh`)

Similar structure to `run_smc_baseline.sh`, but:
- Loads trained value head instead of PRM
- Uses `TSMCLLMParticleFilter` instead of `LLMParticleFilter`
- Reports metrics: accuracy, avg particles surviving, value prediction correlation

### 5.2 Comparison Metrics

| Metric | Standard SMC | Twisted SMC |
|--------|-------------|-------------|
| Accuracy | Primary metric | Primary metric |
| Avg PRM calls/problem | N_steps × N_particles | 0 (value head is part of model) |
| Value prediction loss / correlation | N/A | Track correlation with actual outcomes |
| Particle diversity | ESS tracking | ESS tracking |
| Compute cost | High (separate PRM) | Low (amortized value head) |
| Calibration | N/A | ECE / Brier score |
| Weight degeneracy | N/A | log-weight variance, # unique ancestors |

---

## 6. Implementation Phases

### Phase 1: Value Head Core (Week 1)
- [x] Implement `trex/models/value_head.py` (Linear, MLP, Attention-pooled)
- [x] Implement `trex/models/twist_model.py` (wrapper)
- [x] Unit tests for value heads
- [x] Integration test: TwistModel forward pass

### Phase 2: TSMC Particle Filter (Week 1-2)
- [x] Add `trex/smc/tsmc_particle_filter.py` for twist-scorer-based TSMC rollout logic
- [x] Integrate with existing `TwistedSMC` math path and per-particle metadata tracking
- [x] Unit tests for TSMC loop and metadata consistency
- [ ] Smoke test on small dataset (cluster/integration run)

### Phase 3: Training Infrastructure (Week 2-3)
- [x] Implement `trex/training/trajectory_buffer.py`
- [x] Implement `trex/training/value_trainer.py`
- [x] Create `train_value_head.py` entry point
- [ ] Expand unit tests for value-training loop behaviors (currently partial)

### Phase 4: Baseline Runner (Week 3)
- [x] Implement `trex/baselines/tsmc_baseline.py`
- [x] Create `run_tsmc_baseline.sh` script
- [ ] Integration test: End-to-end evaluation
- [ ] SLURM testing on small dataset

### Phase 5: Experiments (Week 4)
- [ ] Train value head on GSM8K
- [ ] Run TSMC baseline evaluation
- [ ] Compare against Standard SMC baseline
- [ ] Document results in EXPERIMENTS.md

---

## 7. Architecture Diagram

```mermaid
flowchart TB
    subgraph Data["Training Data"]
        P[Math Problems]
        GT[Ground Truth]
    end
    
    subgraph Rollout["Rollout Collection"]
        G[vLLM Generator]
        TM[TwistModel<br/>Base LLM + Value Head]
        V[Math Verifier]
        TB[Trajectory Buffer]
        
        P --> G
        G -->|Generate K trajectories (base p₀)| TM
        TM -->|Verify answers| V
        V -->|Binary rewards| TB
    end
    
    subgraph Training["Value Head Training"]
        VT[Value Trainer]
        Loss[BCEWithLogits or MSE/BCE]
        
        TB -->|Sample trajectories| VT
        VT -->|Forward pass| TM
        TM -->|Predictions| Loss
        Loss -->|Backprop| TM
    end
    
    subgraph Inference["TSMC Inference"]
        TSMC[TSMC Particle Filter]
        PF[Particle Filter<br/>with Value Guidance]
        
        P -->|Initialize| TSMC
        TSMC -->|w_t = ψ_t/ψ_{t-1} (Mode A)| PF
        PF -->|Resample| G
        G -->|Generate| TM
        TM -->|Score| PF
    end
    
    TB -.->|Bootstrap| TSMC
```

---

## 8. Testing Strategy

### Unit Tests
- `test_value_head.py`: Test all three architectures, output range per `twist_space` (prob in [0,1], log_prob ≤ 0)
- `test_twist_model.py`: Test forward pass, hidden state extraction
- `test_trajectory_buffer.py`: Test storage, sampling, state-reward extraction
- `test_tsmc_particle_filter.py`: Test integration with TwistedSMC
- `test_twisted_smc_resampling_metadata.py`: Ensure resampling preserves per-particle prev values and weight ratios use the correct particle-aligned history
- `test_twisted_smc_weight_sanity.py`: ψ constant ⇒ all incremental weights = 1; ψ increasing ⇒ monotonic log-weights
- `test_twist_space_consistency.py`: logits → ψ/logψ mapping is consistent; BCEWithLogits matches target space

### Integration Tests
- Value head training for 10 steps on small dataset
- TSMC evaluation on 5 problems
- Checkpoint save/load

### SLURM Tests
- Train value head on GSM8K subset (100 problems)
- Run TSMC baseline on MATH-500
- Compare accuracy vs Standard SMC

---

## 9. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Freeze base model | Faster training, prevents catastrophic forgetting |
| Explicit twist space (`prob` / `log_prob`) | Avoids silent mismatch between head, loss, and weights |
| Value head outputs raw logits | Prevents double-activation; simplifies loss + scoring consistency |
| Per-token values | Enables step-by-step guidance |
| MLP default | Good capacity/efficiency tradeoff |
| Self-distillation (p₀ rollouts by default) | No human labels needed, bootstraps from verifier |
| Monte Carlo targets | Simple but effective (assign final reward to all steps) |

---

## 10. Files Implemented / Maintained

```
trex/models/value_head.py
trex/models/twist_model.py
trex/training/__init__.py
trex/training/trajectory_buffer.py
trex/training/value_trainer.py
trex/training/train_value_head.py
trex/smc/tsmc_particle_filter.py
trex/baselines/tsmc_baseline.py
trex/baselines/tsmc_config.py
trex/scripts/train_value_head.sh
trex/scripts/run_tsmc_baseline.sh
```

---

## 11. Open Questions

1. **Value head initialization**: Should we initialize logits near 0 (ψ≈0.5; logψ≈log(0.5)) or use learned initialization?
2. **Curriculum training**: Should we start with easier problems and progressively increase difficulty?
3. **Value head checkpointing**: Save only value head weights or full model?
4. **TSMC temperature**: Should we use temperature=1.0 or anneal during training?

---

**Next Step:** Run integration experiments with the current defaults (delimiter mode, no chat template, `twist_weight` final selection), then compare TSMC against the validated SMC baselines.
