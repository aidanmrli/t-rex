# Implementation Plan: Value Head + Twisted SMC Baseline

**Status:** Planning Phase  
**Created:** 2026-01-31  
**Goal:** Implement Value Head architecture and training to enable end-to-end Twisted SMC testing

---

## 1. Overview

This plan implements the **Value Head** and **Twisted SMC (TSMC) Baseline** components of T-REX. The Value Head is a learnable function that predicts the expected future reward (correctness) of a partial reasoning trace. When integrated with Twisted SMC, it guides particle filtering toward high-value regions without requiring an external PRM.

### Key Differences from Standard SMC

| Aspect | Standard SMC (Implemented) | Twisted SMC (This Plan) |
|--------|---------------------------|------------------------|
| Scoring | External PRM (Qwen2.5-Math-PRM-7B) | Learned Value Head |
| Weight Update | `w_t = PRM_score` | `w_t = V(x_t) / V(x_{t-1})` |
| Training | None (inference only) | Self-distillation from rollouts |
| Compute Cost | High (separate PRM model) | Low (single forward pass) |

---

## 2. Architecture Analysis

### 2.1 Integration Points (from codebase analysis)

**Existing Infrastructure:**
- [`trex/smc/twisted_smc.py`](trex/smc/twisted_smc.py:1) - Core TSMC logic with `compute_twisted_weights()` and `TwistedSMC` class
- [`trex/smc/llm_particle_filter.py`](trex/smc/llm_particle_filter.py:1) - LLM-aware particle filter (extends base ParticleFilter)
- [`openrlhf/models/model.py`](openrlhf/models/model.py:1) - Reference implementation of value head attachment
- [`trex/models/reward_model.py`](trex/models/reward_model.py:1) - PRM/ORM wrapper pattern to follow

**Key Observations:**
1. The `TwistedSMC` class already exists and expects a `value_function: Callable[[List[str]], torch.Tensor]`
2. OpenRLHF's `get_llm_for_sequence_regression()` shows how to attach heads to base LLMs
3. The Value Head must output values in [0, 1] (sigmoid) per HIGH_LEVEL_CONTEXT.md Section 2.4

### 2.2 Value Head Architecture Options

Per IMPLEMENTATION_PLAN.md Section 2.1, we support multiple architectures:

```
┌─────────────────────────────────────────────────────────────────┐
│  Value Head Options (plug-and-play via config)                  │
├─────────────────────────────────────────────────────────────────┤
│  1. Simple Linear:                                              │
│     Hidden [B,T,H] → Linear(H, 1) → Sigmoid → Value [B,T,1]    │
│     - Fastest, might underfit complex reasoning                 │
│                                                                 │
│  2. MLP (Recommended):                                          │
│     Hidden [B,T,H] → Linear(H, H//4) → ReLU → Linear(H//4, 1)  │
│                    → Sigmoid → Value [B,T,1]                    │
│     - Good balance of capacity and efficiency                   │
│                                                                 │
│  3. Attention-Pooled:                                           │
│     Hidden [B,T,H] → AttentionPool(T, 1) → Linear(H, 1)         │
│                    → Sigmoid → Value [B,1]                      │
│     - For long contexts, aggregates sequence info               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Design

### 3.1 Module Structure

```
trex/
├── models/
│   ├── __init__.py              # Add ValueHead, TwistModel exports
│   ├── value_head.py            # NEW: Value head architectures
│   └── twist_model.py           # NEW: Base LLM + Value Head wrapper
├── training/
│   ├── __init__.py              # NEW: Training module exports
│   ├── value_trainer.py         # NEW: Self-distillation training loop
│   └── trajectory_buffer.py     # NEW: Rollout storage
├── smc/
│   └── tsmc_particle_filter.py  # NEW: TSMC with value head scoring
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
            values: [batch_size, seq_len, 1] in [0, 1] via sigmoid
        """
        pass

class LinearValueHead(ValueHead):
    """Simple linear projection."""
    def __init__(self, hidden_dim: int):
        self.linear = nn.Linear(hidden_dim, 1)
        
class MLPValueHead(ValueHead):
    """Two-layer MLP with ReLU."""
    def __init__(self, hidden_dim: int, intermediate_dim: Optional[int] = None):
        intermediate_dim = intermediate_dim or hidden_dim // 4
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, 1),
            nn.Sigmoid(),
        )
```

**Key Design Decisions:**
1. Output per-token values (not just sequence-level) to support step-by-step guidance
2. Sigmoid output in [0, 1] per mathematical specification
3. No layer norm (unnecessary for single-task regression)
4. Initialize final layer to small values (near 0.5) for stable training start

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
        freeze_base_model: bool = True,
    ):
        """
        Args:
            model_name_or_path: HuggingFace model ID or local path
            value_head_type: "linear", "mlp", or "attention_pooled"
            freeze_base_model: Whether to freeze base LLM (recommended)
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
            values: [batch_size, seq_len, 1] in [0, 1]
        """
        pass
    
    def get_values_for_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Convenience method for TSMC integration.
        
        Args:
            texts: List of partial reasoning traces
        Returns:
            values: [len(texts),] - Final position value for each text
        """
        pass
```

**Implementation Notes:**
- Use `output_hidden_states=True` to extract final layer hidden states
- Extract value at the **last non-padding position** for each sequence
- Compatible with vLLM for efficient generation (value head runs separately)

### 3.4 TSMC Particle Filter (`trex/smc/tsmc_particle_filter.py`)

**Purpose:** Extend `TwistedSMC` with LLM generation and value head scoring.

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
        twist_model: TwistModel,    # Base LLM + Value Head
    ):
        super().__init__(config)
        self.generator = generator
        self.twist_model = twist_model
        
        # Set value function for base TwistedSMC
        self.set_value_function(self._compute_values)
    
    def _compute_values(self, texts: List[str]) -> torch.Tensor:
        """Compute values using twist model."""
        return self.twist_model.get_values_for_texts(texts)
    
    def step(self) -> None:
        """
        Single TSMC iteration:
        1. Generate next step with LLM
        2. Compute values with twist model
        3. Update weights: w_t = V_t / V_{t-1}
        4. Resample if ESS < threshold
        """
        pass
```

### 3.5 Value Head Training (`trex/training/value_trainer.py`)

**Purpose:** Train value head via self-distillation from rollouts.

**Algorithm (from HIGH_LEVEL_CONTEXT.md Section 4, Phase 3):**
```
1. Data Collection (Rollouts):
   For each prompt in training set:
     - Sample K trajectories using current policy
     - Verify final answers → Binary reward R ∈ {0, 1}
   
2. Label Assignment (Monte Carlo):
   - Assign final reward R to ALL steps in trajectory
   - Dataset: D = {(x_{1:t}, R) for all steps t in all trajectories}

3. Training:
   - Minimize MSE: L = ||V(x_{1:t}) - R||²
   - Only update value head (base model frozen)

4. Iteration:
   - Use improved V to bias sampling
   - Collect new data with better guidance
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
        # Compute MSE loss
        # Backprop and update
        pass
    
    def train(self, dataset: List[Dict], verifier: MathVerifier) -> None:
        """Full training loop with periodic rollout collection."""
        pass
```

### 3.6 Trajectory Buffer (`trex/training/trajectory_buffer.py`)

**Purpose:** Store and manage rollout data for training.

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
    
    # TSMC parameters
    n_particles: int = 16
    max_steps: int = 10
    ess_threshold: float = 0.5
    
    # Generation parameters
    temperature: float = 1.0
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
            twist_model=self.twist_model,
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
│  │  1. COLLECT ROLLOUTS                                          │  │
│  │     For each training prompt:                                  │  │
│  │       - Generate K trajectories using TSMC (or base LLM)      │  │
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
│  │       - Loss: MSE(V(x_{1:t}), R)                               │  │
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
| Value prediction MSE | N/A | Track correlation with actual outcomes |
| Particle diversity | ESS tracking | ESS tracking |
| Compute cost | High (separate PRM) | Low (amortized value head) |

---

## 6. Implementation Phases

### Phase 1: Value Head Core (Week 1)
- [ ] Implement `trex/models/value_head.py` (Linear, MLP, Attention-pooled)
- [ ] Implement `trex/models/twist_model.py` (wrapper)
- [ ] Unit tests for value heads
- [ ] Integration test: TwistModel forward pass

### Phase 2: TSMC Particle Filter (Week 1-2)
- [ ] Implement `trex/smc/tsmc_particle_filter.py`
- [ ] Integrate with existing `TwistedSMC` class
- [ ] Unit tests for TSMC loop
- [ ] Smoke test on small dataset

### Phase 3: Training Infrastructure (Week 2-3)
- [ ] Implement `trex/training/trajectory_buffer.py`
- [ ] Implement `trex/training/value_trainer.py`
- [ ] Create `train_value_head.py` entry point
- [ ] Unit tests for training loop

### Phase 4: Baseline Runner (Week 3)
- [ ] Implement `trex/baselines/tsmc_baseline.py`
- [ ] Create `run_tsmc_baseline.sh` script
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
        G -->|Generate K trajectories| TM
        TM -->|Verify answers| V
        V -->|Binary rewards| TB
    end
    
    subgraph Training["Value Head Training"]
        VT[Value Trainer]
        Loss[MSE Loss:<br/>||V(x) - R||²]
        
        TB -->|Sample trajectories| VT
        VT -->|Forward pass| TM
        TM -->|Predictions| Loss
        Loss -->|Backprop| TM
    end
    
    subgraph Inference["TSMC Inference"]
        TSMC[TSMC Particle Filter]
        PF[Particle Filter<br/>with Value Guidance]
        
        P -->|Initialize| TSMC
        TSMC -->|w_t = V_t/V_{t-1}| PF
        PF -->|Resample| G
        G -->|Generate| TM
        TM -->|Score| PF
    end
    
    TB -.->|Bootstrap| TSMC
```

---

## 8. Testing Strategy

### Unit Tests
- `test_value_head.py`: Test all three architectures, output range [0,1]
- `test_twist_model.py`: Test forward pass, hidden state extraction
- `test_trajectory_buffer.py`: Test storage, sampling, state-reward extraction
- `test_tsmc_particle_filter.py`: Test integration with TwistedSMC

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
| Sigmoid output | Matches mathematical spec (ψ ∈ [0,1]) |
| Per-token values | Enables step-by-step guidance |
| MLP default | Good capacity/efficiency tradeoff |
| Self-distillation | No human labels needed, bootstraps from verifier |
| Monte Carlo targets | Simple but effective (assign final reward to all steps) |

---

## 10. Files to Create

```
trex/models/value_head.py          # ~150 lines
trex/models/twist_model.py         # ~200 lines
trex/training/__init__.py          # ~10 lines
trex/training/trajectory_buffer.py # ~100 lines
trex/training/value_trainer.py     # ~250 lines
trex/smc/tsmc_particle_filter.py   # ~200 lines
trex/baselines/tsmc_baseline.py    # ~300 lines
trex/baselines/tsmc_config.py      # ~50 lines
trex/scripts/train_value_head.sh   # ~80 lines
trex/scripts/run_tsmc_baseline.sh  # ~80 lines

Total: ~1420 lines of new code
```

---

## 11. Open Questions

1. **Value head initialization**: Should we initialize to 0.5 (neutral) or use learned initialization?
2. **Curriculum training**: Should we start with easier problems and progressively increase difficulty?
3. **Value head checkpointing**: Save only value head weights or full model?
4. **TSMC temperature**: Should we use temperature=1.0 or anneal during training?

---

**Next Step:** Review this plan and proceed to implementation phase.