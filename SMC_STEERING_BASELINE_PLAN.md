# Detailed Implementation Plan: Standard SMC Steering Baseline

**Created:** 2026-01-29  
**Status:** ✅ Implementation Complete  
**Last Updated:** 2026-01-29 (Implementation complete)

> **Implementation Philosophy:** Test-Driven Development (TDD)
> - Write tests FIRST (Phase 1) ✅
> - Implement code to pass tests (Phase 2) ✅
> - Integration & smoke tests on GPU (Phases 3-4) - Ready for GPU testing
> - See Section 8 for detailed phased plan


---

## 1. Overview

The **Standard SMC Steering Baseline** implements **inference-time compute scaling** through particle filtering with Process Reward Model (PRM) guidance.

### 1.1 Key Components
- **Generator:** Qwen2.5-7B (or similar) for generating reasoning steps
- **PRM (Process Reward Model):** `Qwen/Qwen2.5-Math-PRM-7B` for scoring intermediate steps and guiding SMC resampling
- **ORM (Outcome Reward Model):** `Qwen/Qwen2.5-Math-PRM-7B` (same model, different input format) for final answer selection

### 1.2 SMC-LLM Integration Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  For each problem:                                                  │
│  1. Initialize N particles with prompt                              │
│  2. LOOP until all particles complete or max_steps:                 │
│     a. Generate next step for each particle (stop at "## Step")    │
│     b. Score each particle's latest step using PRM                  │
│     c. Update particle weights based on PRM scores                  │
│     d. Resample particles if ESS < threshold (SMC resampling)       │
│  3. Score all final particles using ORM                             │
│  4. Select highest ORM-scored particle as final answer              │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 Key Difference from Best-of-N

| Aspect | Best-of-N | SMC Steering |
|--------|-----------|--------------|
| Branching | None | After each `## Step N:` |
| Scoring | Final only (ORM) | After each step (PRM) + Final (ORM) |
| Compute allocation | Uniform | Adaptive (focus on promising particles) |
| Final selection | ORM on all N samples | ORM on surviving particles |

---

## 2. Reward Models

**Unified Model:** Both PRM and ORM use `Qwen/Qwen2.5-Math-PRM-7B`. The difference is solely in input formatting.

### 2.1 Process Reward Model (PRM)

The PRM scores **each intermediate step** during generation. Used for SMC weight updates and resampling decisions.

**Purpose:** Guide particle filtering by scoring partial solutions after each reasoning step.

**When Used:** After each `## Step N:` is generated, before resampling.

**Input Format (with `<reward_token>` after EACH step):**
```
## Step 1: [Concise description] 
[Brief explanation and calculations]
<reward_token> 
## Step 2: [Concise description] 
[Brief explanation and calculations] 
<reward_token> 
## Step 3: [Concise description] 
[Brief explanation and calculations] 
<reward_token>
```

**Output:** For each `<reward_token>` position, a probability score (0-1) indicating correctness likelihood.

**SMC Usage:** Extract the score at the **latest** `<reward_token>` position to weight the particle.

### 2.2 Outcome Reward Model (ORM)

The ORM scores **complete solutions** for final answer selection. After SMC finishes, **all surviving particles are scored by ORM and the highest-scoring one is selected for grading**.

**Purpose:** Select the best final answer from all surviving particles.

**When Used:** Once, after SMC loop terminates (all particles finished or max_steps reached).

**Input Format (single `<reward_token>` at END only):**
```
## Step 1: [Concise description] 
[Brief explanation and calculations] 
## Step 2: [Concise description] 
[Brief explanation and calculations] 
## Step 3: [Concise description] 
[Brief explanation and calculations] 
<reward_token>
```

**Output:** Single score for the complete solution.

**Final Selection:** `best_particle = argmax(ORM_scores)`

### 2.3 Reward Token Extraction

> **Design Note: Model-Agnostic Abstraction**
>
> Different PRMs use different token formats for step separation. The `RewardModel` class
> abstracts this via configurable token settings, allowing us to swap models without
> changing the SMC logic.

#### 2.3.1 Model-Specific Token Formats

| Model | Step Separator Token | Score Extraction Method |
|-------|---------------------|------------------------|
| **Qwen2.5-Math-PRM-7B** | `<extra_0>` | 2-class softmax, take index 1 |
| (Future: Other PRMs) | TBD | TBD |

#### 2.3.2 Qwen2.5-Math-PRM-7B Specifics (Default)

**From official HuggingFace documentation:**

- **Step separator token:** `<extra_0>` (NOT `<reward_token>`)
- **Output format:** 2-class logits at each `<extra_0>` position
- **Score extraction:** `softmax(logits, dim=-1)[:, 1]` (probability of positive class)

**Official extraction pattern:**
```python
import torch.nn.functional as F

def make_step_rewards(logits, token_masks):
    """
    Extract step rewards from Qwen2.5-Math-PRM-7B outputs.
    
    Args:
        logits: Model output logits, shape (batch, seq_len, 2)
        token_masks: Boolean mask for <extra_0> positions, shape (batch, seq_len)
    
    Returns:
        List of score lists, one per sample, scores in [0, 1]
    """
    probabilities = F.softmax(logits, dim=-1)  # (batch, seq_len, 2)
    probabilities = probabilities * token_masks.unsqueeze(-1)
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]  # (seq_len, 2)
        # Extract non-zero entries (where mask was True)
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1]  # Take positive class (index 1)
        all_scores_res.append(positive_probs.cpu().tolist())
    
    return all_scores_res

# Example output: [[1.0, 0.19, 0.98, 1.0]] — one score per step, in [0, 1]
```

**Input format for Qwen2.5-Math-PRM-7B:**
```python
# Steps are joined with <extra_0> separator
messages = [
    {"role": "system", "content": "Please reason step by step..."},
    {"role": "user", "content": problem},
    {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"},
]
```

---

## 3. Step Format Specification

### 3.1 Evaluation System Prompt

```
Solve the following math problem efficiently and clearly: 
- For simple problems (2 steps or fewer): Provide a concise solution with minimal explanation.  
- For complex problems (3 steps or more): Use this step-by-step format:  

## Step 1: [Concise description] 
[Brief explanation and calculations]  

## Step 2: [Concise description] 
[Brief explanation and calculations]  

Regardless of the approach, always conclude with:  

Therefore, the final answer is: $\boxed{answer}$. I hope it is correct. 

Where [answer] is just the final number or expression that solves the problem.
```

### 3.2 Step Detection

**Step delimiter:** `## Step` (regex: `r"## Step \d+:"`)

Each SMC iteration:
1. Generate until `## Step N:` pattern appears (or EOS/max tokens)
2. Score the new step using PRM
3. Resample particles based on PRM scores

---

## 4. Architecture Design

### 4.1 File Structure

```
trex/
├── baselines/
│   ├── smc_config.py              # SMC configuration (NEW)
│   ├── smc_steering_baseline.py   # Main baseline (NEW)
│   └── ...
├── smc/
│   ├── __init__.py                # Already exists
│   ├── resampling.py              # ✅ Already complete
│   ├── particle_filter.py         # ✅ Already complete (uses deepcopy)
│   ├── twisted_smc.py             # ✅ Already complete
│   └── llm_particle_filter.py     # LLM-aware particle filter (NEW)
├── models/
│   ├── __init__.py                # (NEW)
│   ├── prm_config.py              # Model-agnostic PRM configuration (NEW)
│   └── reward_model.py            # PRM/ORM wrapper using HF Transformers (NEW)
├── scripts/
│   └── run_smc_baseline.sh        # SLURM script (NEW)
└── tests/
    ├── conftest.py                # Shared test fixtures (NEW)
    └── test_smc/
        ├── test_llm_particle_filter.py  # (NEW)
        └── test_reward_model.py         # (NEW)
```

### 4.2 Component Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                  SMCSteeringBaseline                        │
│  (Orchestrator: loads data, runs loop, saves results)       │
│  + SLURM checkpointing support                              │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
┌─────────────────────────┐           ┌─────────────────────────┐
│   LLMParticleFilter     │           │    RewardModel          │
│   (Generation + SMC)    │           │    (PRM + ORM)          │
└─────────────────────────┘           └─────────────────────────┘
          │                                       │
          ▼                                       ▼
   ┌──────────────┐                    ┌──────────────────────┐
   │   vLLM LLM   │                    │ Qwen2.5-Math-PRM-7B  │
   │ (Generator)  │                    │   (Reward Model)     │
   └──────────────┘                    └──────────────────────┘
```

---

## 5. Core Components

### 5.1 `SMCSteeringConfig`

```python
@dataclass
class SMCSteeringConfig:
    """Configuration for SMC Steering Baseline."""
    
    # Generator model
    generator_model_path: str = "Qwen/Qwen2.5-7B"
    generator_tp_size: int = 1
    
    # Reward model (PRM/ORM)
    reward_model_path: str = "Qwen/Qwen2.5-Math-PRM-7B"
    reward_model_tp_size: int = 1
    
    # Dataset
    dataset_path: str = "trex/data/gsm8k_platinum_test.jsonl"
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    
    # SMC parameters
    n_particles: int = 16
    max_steps: int = 20
    step_pattern: str = r"## Step \d+:"  # Regex for step detection
    seed: Optional[int] = None  # Random seed for reproducibility
    
    # Resampling
    resampling_method: str = "systematic"
    ess_threshold: float = 0.5
    
    # Generation
    temperature: float = 0.7
    max_tokens_per_step: int = 512
    max_total_tokens: int = 2048
    
    # Final selection
    use_orm_for_final: bool = True  # Use ORM to select final answer
    
    # Checkpointing (SLURM)
    enable_checkpointing: bool = True
    checkpoint_file: str = "checkpoint.json"
    checkpoint_interval: int = 5  # Save every N problems
    checkpoint_time_interval: int = 600  # Save every N seconds (10 min default)
    
    # Output
    output_dir: str = "trex/results/smc_baseline"
    use_wandb: bool = False
    
    # System prompt
    system_prompt: str = """Solve the following math problem efficiently and clearly: 
- For simple problems (2 steps or fewer): Provide a concise solution with minimal explanation.  
- For complex problems (3 steps or more): Use this step-by-step format:  

## Step 1: [Concise description] 
[Brief explanation and calculations]  

## Step 2: [Concise description] 
[Brief explanation and calculations]  

Regardless of the approach, always conclude with:  

Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct. 

Where [answer] is just the final number or expression that solves the problem."""
```

### 5.2 `RewardModel` (PRM/ORM Wrapper)

> **Implementation Note:** This class is designed to be **model-agnostic**. Different PRMs
> use different token formats (see Section 2.3.1). The default configuration is for
> Qwen2.5-Math-PRM-7B, but the token settings can be overridden for other models.

```python
from dataclasses import dataclass
from typing import List, Union, Literal
import torch
import torch.nn.functional as F
import re

@dataclass
class PRMConfig:
    """
    Model-specific configuration for Process Reward Models.
    
    Different PRMs use different token formats for step separation and
    score extraction. This config abstracts those differences.
    """
    # Step separator token (inserted between/after reasoning steps)
    step_separator_token: str = "<extra_0>"  # Qwen2.5-Math-PRM-7B default
    
    # Score extraction method
    # - "binary_softmax": Model outputs 2-class logits, take softmax[:, 1]
    # - "single_logit": Model outputs single scalar logit, apply sigmoid
    extraction_method: Literal["binary_softmax", "single_logit"] = "binary_softmax"
    
    # Number of output classes (for binary_softmax, this is 2)
    num_classes: int = 2
    
    # Which class index represents "positive/correct" (for binary_softmax)
    positive_class_idx: int = 1


# Pre-defined configs for known models
QWEN_PRM_CONFIG = PRMConfig(
    step_separator_token="<extra_0>",
    extraction_method="binary_softmax",
    num_classes=2,
    positive_class_idx=1,
)


class RewardModel:
    """
    Model-agnostic wrapper for Process/Outcome Reward Models.
    
    Supports both PRM (score each step) and ORM (score final answer) modes.
    Default configuration is for Qwen2.5-Math-PRM-7B, but can be customized
    for other PRMs via PRMConfig.
    
    Attributes:
        prm_config: Model-specific token and extraction configuration
    """
    
    def __init__(
        self, 
        model_path: str, 
        tp_size: int = 1,
        prm_config: PRMConfig = QWEN_PRM_CONFIG,
    ):
        self.model_path = model_path
        self.prm_config = prm_config
        
        # Load model using HuggingFace Transformers (recommended approach)
        # vLLM's generate() API doesn't easily expose logits at arbitrary positions
        from transformers import AutoModel, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()
        
        # Get separator token ID
        self.separator_token_id = self.tokenizer.encode(
            self.prm_config.step_separator_token, 
            add_special_tokens=False
        )[0]
    
    def format_for_prm(self, steps: List[str]) -> str:
        """
        Format steps for PRM scoring by joining with separator token.
        
        Args:
            steps: List of reasoning step strings
            
        Returns:
            Formatted string with separator tokens between/after steps
        """
        separator = self.prm_config.step_separator_token
        return separator.join(steps) + separator
    
    def format_for_orm(self, text: str) -> str:
        """
        Format complete solution for ORM scoring.
        
        Args:
            text: Complete solution text
            
        Returns:
            Text with single separator token at end
        """
        separator = self.prm_config.step_separator_token
        return text.rstrip() + separator
    
    def _extract_scores_from_logits(
        self, 
        logits: torch.Tensor, 
        token_masks: torch.Tensor
    ) -> List[List[float]]:
        """
        Extract step reward scores from model logits.
        
        This follows the official Qwen2.5-Math-PRM-7B extraction pattern.
        
        Args:
            logits: Model output, shape (batch, seq_len, num_classes)
            token_masks: Boolean mask for separator positions, shape (batch, seq_len)
            
        Returns:
            List of score lists, one per sample, scores in [0, 1]
        """
        if self.prm_config.extraction_method == "binary_softmax":
            probabilities = F.softmax(logits, dim=-1)
            probabilities = probabilities * token_masks.unsqueeze(-1)
            
            all_scores = []
            for i in range(probabilities.size(0)):
                sample = probabilities[i]  # (seq_len, num_classes)
                # Extract non-zero entries (where mask was True)
                positive_probs = sample[sample != 0].view(-1, self.prm_config.num_classes)
                positive_probs = positive_probs[:, self.prm_config.positive_class_idx]
                all_scores.append(positive_probs.cpu().tolist())
            
            return all_scores
        
        elif self.prm_config.extraction_method == "single_logit":
            # For models that output single scalar logit per position
            scores = torch.sigmoid(logits.squeeze(-1))
            scores = scores * token_masks
            
            all_scores = []
            for i in range(scores.size(0)):
                sample_scores = scores[i][token_masks[i]].cpu().tolist()
                all_scores.append(sample_scores)
            
            return all_scores
        
        else:
            raise ValueError(f"Unknown extraction method: {self.prm_config.extraction_method}")
    
    def score_prm(self, texts: List[str]) -> List[List[float]]:
        """
        Score each step in formatted texts using PRM.
        
        Args:
            texts: List of formatted texts (with separator tokens already inserted)
            
        Returns:
            List of score lists, one per text, one score per step in [0, 1]
        """
        # Tokenize inputs
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
        ).to(self.model.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs[0]  # (batch, seq_len, num_classes)
        
        # Create mask for separator token positions
        token_masks = (inputs.input_ids == self.separator_token_id)
        
        return self._extract_scores_from_logits(logits, token_masks)
    
    def score_orm(self, texts: List[str]) -> List[float]:
        """
        Score complete solutions using ORM (takes last score only).
        
        Args:
            texts: List of complete solution texts (separator will be appended)
            
        Returns:
            List of single scores, one per text, in [0, 1]
        """
        formatted = [self.format_for_orm(t) for t in texts]
        all_scores = self.score_prm(formatted)
        # ORM uses only the final score
        return [scores[-1] if scores else 0.0 for scores in all_scores]
    
    def get_latest_step_scores(self, texts: List[str]) -> torch.Tensor:
        """
        Get PRM score for just the latest step in each text.
        Used for SMC weight updates.
        
        Args:
            texts: List of texts with separator tokens
            
        Returns:
            Tensor of scores, shape (n_texts,), values in [0, 1]
        """
```

> **Architecture Note:** The `RewardModel` class uses HuggingFace Transformers directly
> (via `AutoModel`) rather than vLLM's `generate()` API. This is because:
> 1. vLLM's API doesn't easily expose logits at arbitrary input positions
> 2. The official Qwen2.5-Math-PRM-7B examples use HuggingFace Transformers
> 3. Simpler implementation without monkey-patching vLLM internals
>
> **Trade-off:** This means the reward model loads separately from the generator,
> using additional GPU memory. For production, consider tensor parallelism or
> offloading strategies if memory is constrained.


### 5.3 `LLMParticleFilter`

> **Note:** `LLMParticleFilter` extends the existing `ParticleFilter` class from `trex/smc/particle_filter.py`. 
> The base `ParticleFilter` already supports text-based particles via the `Particle` dataclass.
> `LLMParticleFilter` adds LLM generation and PRM scoring capabilities.

```python
class LLMParticleFilter(ParticleFilter):
    """
    Particle Filter with LLM generation and PRM scoring.
    
    Inherits from ParticleFilter:
    - initialize(prompt) - Creates N particles with prompt
    - set_weights() / get_weights() - Weight management
    - normalize_weights() - Normalize to sum=1
    - effective_sample_size() - ESS computation
    - should_resample() - Check ESS threshold
    - resample() - Perform resampling (uses deepcopy for independence)
    - get_particle_texts() - Get all texts
    - get_best_particle() - Highest weight particle
    
    Adds:
    - expand_particles() - Generate next step with LLM
    - score_particles() - Score with PRM
    - step() - One SMC iteration
    - select_best_by_orm() - Select best particle using ORM scores
    - run() - Full SMC loop with ORM selection
    """
    
    STEP_PATTERN = re.compile(r"## Step \d+:")
    STEP_HEADER_PATTERN = re.compile(r"## Step (\d+):")  # For extracting step number
    
    def __init__(self, config: SMCSteeringConfig, generator: LLM, reward_model: RewardModel):
        smc_config = SMCConfig(
            n_particles=config.n_particles,
            ess_threshold=config.ess_threshold,
            resampling_method=config.resampling_method,
            seed=config.seed,  # Pass seed for reproducibility
        )
        super().__init__(smc_config)
        
        self.config = config
        self.generator = generator
        self.reward_model = reward_model
        self._smc_iteration = 0  # SMC loop iteration count (expand → score → resample cycles)
        # Note: Different from reasoning_step_count which tracks "## Step N:" patterns in particle text
    
    def _get_next_step_number(self, text: str) -> int:
        """
        Get the next step number to generate based on existing text.
        
        Counts existing "## Step N:" patterns and returns N+1.
        """
        matches = self.STEP_HEADER_PATTERN.findall(text)
        if not matches:
            return 1
        return max(int(m) for m in matches) + 1
    
    def _inject_step_header(self, text: str) -> str:
        """
        Inject the next step header if text ends mid-step.
        
        This ensures proper step numbering across expansion calls.
        """
        next_step = self._get_next_step_number(text)
        # Only inject if the text doesn't already end with a step header
        if not text.rstrip().endswith(":"):
            return text + f"## Step {next_step}:"
        return text
    
    def expand_particles(self) -> bool:
        """
        Generate next step for all particles until '## Step N:' or EOS.
        
        Strategy for correct step numbering:
        1. Stop generation when "## Step" is encountered
        2. INCLUDE the stop string in output (include_stop_str_in_output=True)
        3. Track step count per-particle and inject header if needed
        
        This ensures the LLM sees the complete "## Step N:" pattern in context
        for the next generation, maintaining correct step numbering.
        """
        texts = self.get_particle_texts()
        
        # Prepare prompts: inject step header if needed for particles mid-generation
        prompts = []
        for i, text in enumerate(texts):
            if self.particles[i].metadata.get("finished"):
                prompts.append(text)  # Don't modify finished particles
            elif self._smc_iteration > 0:
                # After first expansion, we may need to inject step header
                # Only inject if text doesn't end with step pattern
                if not self.STEP_PATTERN.search(text.rstrip()[-20:]):
                    prompts.append(self._inject_step_header(text))
                else:
                    prompts.append(text)
            else:
                prompts.append(text)
        
        # Generate until next step marker or end
        # INCLUDE stop string in output so LLM sees complete context
        sampling_params = SamplingParams(
            n=1,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens_per_step,
            stop=["## Step"],  # Stop when next step begins
            include_stop_str_in_output=True,  # ✅ Include "## Step" in output
        )
        
        outputs = self.generator.generate(prompts, sampling_params)
        
        still_generating = False
        for i, output in enumerate(outputs):
            if self.particles[i].metadata.get("finished"):
                continue  # Skip finished particles
            
            continuation = output.outputs[0].text
            self.particles[i].text = prompts[i] + continuation
            
            # Track per-particle reasoning step count ("## Step N:" patterns)
            reasoning_step_count = len(self.STEP_HEADER_PATTERN.findall(self.particles[i].text))
            self.particles[i].metadata["reasoning_step_count"] = reasoning_step_count
            
            # Check if particle is finished
            has_final_answer = "\\boxed{" in self.particles[i].text
            at_max_tokens = len(self.particles[i].text) > self.config.max_total_tokens
            at_max_steps = reasoning_step_count >= self.config.max_steps
            self.particles[i].metadata["finished"] = has_final_answer or at_max_tokens or at_max_steps
            
            if not self.particles[i].metadata.get("finished"):
                still_generating = True
        
        self._smc_iteration += 1
        return still_generating and self._smc_iteration < self.config.max_steps
    
    def score_particles(self) -> torch.Tensor:
        """Score all particles using PRM on latest step."""
        texts = self.get_particle_texts()
        return self.reward_model.get_latest_step_scores(texts)
    
    def select_best_by_orm(self) -> Particle:
        """
        Select the best particle using ORM (Outcome Reward Model) scores.
        
        This scores all particles using the ORM format (single reward token at end)
        and returns the particle with the highest score.
        
        Returns:
            Particle with highest ORM score
        """
        texts = self.get_particle_texts()
        orm_scores = self.reward_model.score_orm(texts)  # Returns List[float]
        orm_scores_tensor = torch.tensor(orm_scores)
        best_idx = int(torch.argmax(orm_scores_tensor))
        
        # Store ORM scores in metadata for logging
        for i, score in enumerate(orm_scores):
            self.particles[i].metadata["orm_score"] = score
        
        return self.particles[best_idx]
    
    def step(self) -> bool:
        """
        Single SMC step: expand, score with PRM, resample.
        
        Weight Update Strategy (Standard SMC Steering):
        - Uses MULTIPLICATIVE incremental weights: w_t = w_{t-1} × PRM(step_t)
        - This differs from Twisted SMC which uses value ratios (see below)
        
        Note: Twisted SMC uses:
            w_t(s_{1:t}) = [p_0(s_t|s_{1:t-1}) / q(s_t|s_{1:t-1})] × [ψ_t(s_{1:t}) / ψ_{t-1}(s_{1:t-1})]
        Where ψ is the twist function (value estimate). When q = p_0, the 
        likelihood ratio cancels, leaving only the twist ratio.
        """
        should_continue = self.expand_particles()
        
        # Get PRM scores for the latest step
        step_scores = self.score_particles()
        
        # Avoid all-zero (add small epsilon)
        step_scores = step_scores + 1e-8
        
        # MULTIPLICATIVE weight update: w_t = w_{t-1} × PRM(step_t)
        current_weights = self.get_weights()
        new_weights = current_weights * step_scores
        
        self.set_weights(new_weights)
        self.normalize_weights()
        
        if self.should_resample():
            self.resample()
        
        return should_continue
    
    def run(self) -> Particle:
        """Run full SMC loop, return ORM-selected best particle."""
        while self.step():
            pass
        
        # Final selection using ORM
        if self.config.use_orm_for_final:
            return self.select_best_by_orm()
        else:
            return self.get_best_particle()
```

> **⚠️ Known Future Concern: Degenerate Particles (Mode Collapse)**
> 
> After several resampling iterations, all particles may become exact duplicates (mode collapse).
> If all particles are identical, PRM scores them identically, and diversity is lost.
> 
> **Current Status:** Not addressed in this baseline. Monitor during experiments.
> 
> **Future Mitigations (if needed):**
> 1. Track particle diversity (number of unique particle texts)
> 2. Log/warn when diversity drops below threshold
> 3. Consider temperature annealing or diversity bonuses
> 4. Implement "rejuvenation" strategies from SMC literature

### 5.4 `SMCSteeringBaseline` with SLURM Checkpointing

```python
class SMCSteeringBaseline:
    """SMC Steering baseline with SLURM checkpointing support."""
    
    def __init__(self, config: SMCSteeringConfig):
        self.config = config
        self.verifier = MathVerifier()
        self.generator: Optional[LLM] = None
        self.reward_model: Optional[RewardModel] = None
        
        os.makedirs(config.output_dir, exist_ok=True)
        
        if config.enable_checkpointing:
            self.checkpoint_mgr = CheckpointManager(
                os.path.join(config.output_dir, config.checkpoint_file)
            )
            self._setup_signal_handlers()
        else:
            self.checkpoint_mgr = None
    
    def _setup_signal_handlers(self):
        """SLURM preemption handling."""
        def handler(signum, frame):
            print(f"Received signal {signum}, saving checkpoint...")
            if self.checkpoint_mgr:
                self.checkpoint_mgr.save()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGUSR1, handler)
    
    def run(self):
        """Main loop with checkpointing (problem-based AND time-based)."""
        # Resume from checkpoint if exists
        start_idx = 0
        results = []
        if self.checkpoint_mgr:
            start_idx = self.checkpoint_mgr.state.get("completed_idx", 0)
            results = self.checkpoint_mgr.state.get("results", [])
        
        # Load and process dataset
        dataset = self._load_dataset()
        
        # Time-based checkpointing
        last_checkpoint_time = time.time()
        
        for i in range(start_idx, len(dataset)):
            result = self.evaluate_single(dataset[i])
            results.append(result)
            
            # Check if we should checkpoint
            should_checkpoint = False
            current_time = time.time()
            
            # Problem-count based checkpoint
            if (i + 1) % self.config.checkpoint_interval == 0:
                should_checkpoint = True
            
            # Time-based checkpoint (every checkpoint_time_interval seconds)
            if current_time - last_checkpoint_time >= self.config.checkpoint_time_interval:
                should_checkpoint = True
            
            if self.checkpoint_mgr and should_checkpoint:
                self.checkpoint_mgr.state["completed_idx"] = i + 1
                self.checkpoint_mgr.state["results"] = results
                self.checkpoint_mgr.save()
                last_checkpoint_time = current_time
                print(f"Checkpoint saved at problem {i + 1}/{len(dataset)}")
        
        # Final save
        self._save_results(results)
```

---

## 6. Testing Strategy

> **TDD Approach:** All tests in Section 6.2 are written BEFORE implementation. 
> See Section 8 for the phased implementation order.

### 6.1 Testing Priorities

**Critical:** LLM-SMC integration must be seamless. The LLM must generate output in the correct `## Step N:` format.

| Priority | Component | Test Type | Purpose |
|----------|-----------|-----------|--------|
| P0 | Step format parsing | Unit | Ensure `## Step N:` detection works |
| P0 | PRM/ORM formatting | Unit | Correct `<reward_token>` insertion |
| P0 | **SLURM Checkpointing** | Unit | **Resume works correctly (CRITICAL for long runs)** |
| P0 | LLM output format | Integration | LLM follows system prompt format |
| P1 | SMC resampling | Unit | Correct particle duplication/culling |
| P1 | End-to-end flow | Integration | Full pipeline produces valid answers |

### 6.1.1 Mocking Strategy for Unit Tests

**Mock vLLM:** Unit tests should NOT require GPU or load actual models. We mock vLLM's `LLM` class.

**File:** `trex/tests/conftest.py` (shared across ALL tests)

```python
# trex/tests/conftest.py - Shared fixtures for mocking

import torch
import pytest
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass

@dataclass
class MockOutput:
    """Mock vLLM completion output."""
    text: str
    finish_reason: str = "stop"

@dataclass  
class MockRequestOutput:
    """Mock vLLM request output."""
    outputs: list
    prompt_logprobs: list = None  # For reward model logit extraction
    
    @classmethod
    def from_text(cls, text: str):
        return cls(outputs=[MockOutput(text=text)])

@pytest.fixture
def mock_llm():
    """Mock vLLM LLM class for unit tests."""
    llm = MagicMock()
    llm.generate = MagicMock(return_value=[
        MockRequestOutput.from_text("## Step 1: Calculate\n2+2=4")
    ])
    return llm

@pytest.fixture
def mock_reward_model():
    """Mock RewardModel for unit tests."""
    rm = MagicMock()
    rm.get_latest_step_scores = MagicMock(return_value=torch.tensor([0.8, 0.6, 0.9, 0.3]))
    rm.score_orm = MagicMock(return_value=[0.7, 0.5, 0.95, 0.2])
    rm.score_prm = MagicMock(return_value=[[0.8], [0.6], [0.9], [0.3]])
    return rm

@pytest.fixture
def mock_smc_config():
    """Mock SMCSteeringConfig for unit tests."""
    from trex.baselines.smc_config import SMCSteeringConfig
    return SMCSteeringConfig(
        n_particles=4,
        max_steps=5,
        temperature=0.7,
        seed=42,
    )
```

### 6.2 Unit Tests

**`test_reward_model.py`:**
```python
class TestRewardModelFormatting:
    """Test PRM/ORM input formatting - CRITICAL for correct scoring."""
    
    def test_prm_format_inserts_reward_tokens_after_each_step(self):
        """Each step gets a <reward_token> appended."""
        text = "## Step 1: Do X\nCalculation\n## Step 2: Do Y\nResult"
        formatted = reward_model.format_for_prm(text)
        assert formatted.count("<reward_token>") == 2
        # Verify token placement
        lines = formatted.split("\n")
        step_count = sum(1 for l in lines if l.strip() == "<reward_token>")
        assert step_count == 2
    
    def test_orm_format_single_token_at_end(self):
        """ORM format has exactly one <reward_token> at end."""
        text = "## Step 1: Do X\n## Step 2: Do Y\nFinal answer"
        formatted = reward_model.format_for_orm(text)
        assert formatted.count("<reward_token>") == 1
        assert formatted.rstrip().endswith("<reward_token>")
    
    def test_prm_format_handles_boxed_answer(self):
        """PRM correctly handles solutions with \\boxed{}."""
        text = "## Step 1: Calculate\n2+2=4\nTherefore, the final answer is: $\\boxed{4}$."
        formatted = reward_model.format_for_prm(text)
        assert "<reward_token>" in formatted
        assert "$\\boxed{4}$" in formatted

class TestRewardModelScoring:
    """Test actual model scoring (requires GPU)."""
    
    @pytest.mark.gpu
    def test_prm_returns_scores_per_step(self):
        """PRM returns one score per step."""
        reward_model = RewardModel("Qwen/Qwen2.5-Math-PRM-7B")
        text = "## Step 1: Add\n2+2=4\n## Step 2: Verify\n4 is correct"
        scores = reward_model.score_prm([text])
        assert len(scores[0]) == 2  # Two steps
        assert all(0 <= s <= 1 for s in scores[0])  # Valid probabilities
    
    @pytest.mark.gpu  
    def test_orm_returns_single_score(self):
        """ORM returns single score per text."""
        reward_model = RewardModel("Qwen/Qwen2.5-Math-PRM-7B")
        text = "## Step 1: Calculate\n2+2=4\nTherefore, the final answer is: $\\boxed{4}$."
        scores = reward_model.score_orm([text])
        assert len(scores) == 1
        assert 0 <= scores[0] <= 1
    
    @pytest.mark.gpu
    def test_correct_solution_scores_higher_than_incorrect(self):
        """Sanity check: correct solutions should score higher."""
        reward_model = RewardModel("Qwen/Qwen2.5-Math-PRM-7B")
        correct = "## Step 1: Add\n2+2=4\nTherefore, the final answer is: $\\boxed{4}$."
        incorrect = "## Step 1: Add\n2+2=5\nTherefore, the final answer is: $\\boxed{5}$."
        scores = reward_model.score_orm([correct, incorrect])
        assert scores[0] > scores[1], "Correct solution should score higher"
```

**`test_llm_particle_filter.py`:**
```python
class TestStepDetection:
    """Test step pattern detection - CRITICAL for SMC timing."""
    
    def test_detects_step_pattern(self):
        """Correctly identifies '## Step N:' pattern."""
        text = "## Step 1: Calculate\nDoing math\n## Step 2: Verify"
        steps = re.findall(r"## Step \d+:", text)
        assert len(steps) == 2
    
    def test_step_pattern_regex_strict(self):
        """Pattern requires exact format with colon."""
        pattern = re.compile(r"## Step \d+:")
        assert pattern.search("## Step 1: Begin")  # Valid
        assert pattern.search("## Step 10: Continue")  # Multi-digit
        assert not pattern.search("## Step1:")  # Missing space
        assert not pattern.search("## Step 1")  # Missing colon
        assert not pattern.search("# Step 1:")  # Single hash

class TestParticleExpansion:
    """Test particle generation (mocked LLM)."""
    
    def test_expansion_appends_to_particles(self, mock_llm):
        """Generated text is appended to particle."""
        mock_llm.generate.return_value = [MockOutput("## Step 1: Calculate\n2+2=4")]
        pf = LLMParticleFilter(config, mock_llm, mock_reward_model)
        pf.initialize("What is 2+2?")
        pf.expand_particles()
        assert "## Step 1" in pf.particles[0].text
    
    def test_finished_detection_on_boxed(self, mock_llm):
        """Particles with \\boxed{} are marked finished."""
        mock_llm.generate.return_value = [MockOutput("Therefore, $\\boxed{4}$")]
        pf = LLMParticleFilter(config, mock_llm, mock_reward_model)
        pf.initialize("prompt")
        pf.particles[0].text = "## Step 1: Calc\n"
        pf.expand_particles()
        assert pf.particles[0].metadata["finished"] == True

class TestSMCLoop:
    """Test full SMC loop (mocked)."""
    
    def test_resampling_triggered_on_low_ess(self):
        """Resampling occurs when ESS drops below threshold."""
        pf = LLMParticleFilter(config, mock_llm, mock_reward_model)
        pf.initialize("prompt")
        # Set highly skewed weights (low ESS)
        pf.set_weights(torch.tensor([0.99, 0.003, 0.003, 0.004]))
        pf.normalize_weights()
        assert pf.should_resample()  # ESS should be low
    
    def test_orm_selection_at_end(self):
        """Final particle is selected by highest ORM score."""
        mock_reward_model.score_orm.return_value = [0.3, 0.9, 0.5, 0.2]
        pf = LLMParticleFilter(config, mock_llm, mock_reward_model)
        pf.initialize("prompt")
        # Simulate finished particles
        for p in pf.particles:
            p.metadata["finished"] = True
        best = pf.select_best_by_orm()
        assert best == pf.particles[1]  # Index 1 had score 0.9
    
    def test_particle_independence_after_resampling(self):
        """
        CRITICAL: After resampling, particles must be independent.
        
        When a particle is duplicated during resampling, modifying the copy
        should NOT affect the original. This requires deepcopy, not shallow copy.
        """
        pf = LLMParticleFilter(config, mock_llm, mock_reward_model)
        pf.initialize("prompt")
        
        # Set weights so particle 0 will be duplicated
        pf.set_weights(torch.tensor([0.97, 0.01, 0.01, 0.01]))
        pf.normalize_weights()
        pf.resample()
        
        # After resampling, multiple particles may have same text
        # Find two particles with the same text
        texts_before = [p.text for p in pf.particles]
        
        # Modify the first particle's text and metadata
        pf.particles[0].text += " MODIFIED"
        pf.particles[0].metadata["test_key"] = "test_value"
        
        # Other particles with the same original text should NOT be affected
        for i in range(1, len(pf.particles)):
            assert "MODIFIED" not in pf.particles[i].text, \
                f"Particle {i} was affected by modification to particle 0 (shallow copy bug)"
            assert pf.particles[i].metadata.get("test_key") != "test_value", \
                f"Particle {i} metadata was affected (shallow copy bug)"

    def test_multiplicative_weight_update(self, mock_llm, mock_reward_model):
        """
        Verify weights are updated multiplicatively: w_t = w_{t-1} × PRM(step_t)
        """
        mock_reward_model.get_latest_step_scores.return_value = torch.tensor([0.8, 0.6, 0.9, 0.3])
        pf = LLMParticleFilter(config, mock_llm, mock_reward_model)
        pf.initialize("prompt")
        
        # Set initial weights (after normalization: [0.25, 0.25, 0.25, 0.25])
        initial_weights = pf.get_weights().clone()
        
        # Run one step (mocked expansion + scoring)
        mock_llm.generate.return_value = [MockOutput("step content")] * 4
        pf.step()
        
        # New weights should be proportional to initial * PRM scores
        # (normalized, so we check ratios)
        new_weights = pf.get_weights()
        expected_ratios = torch.tensor([0.8, 0.6, 0.9, 0.3])
        expected_ratios = expected_ratios / expected_ratios.sum()
        
        torch.testing.assert_close(new_weights, expected_ratios, rtol=1e-5, atol=1e-5)
```

### 6.3 Integration Tests (LLM Format Compliance)

**`test_smc_integration.py`:**
```python
@pytest.mark.integration
@pytest.mark.gpu
class TestLLMFormatCompliance:
    """Verify LLM generates text in correct format - CRITICAL."""
    
    def test_llm_follows_step_format(self):
        """LLM output must follow '## Step N:' format for SMC to work."""
        llm = LLM(model="Qwen/Qwen2.5-7B", ...)
        prompt = apply_chat_template(SYSTEM_PROMPT, "What is 15 + 27?")
        output = llm.generate([prompt], SamplingParams(max_tokens=500))
        text = output[0].outputs[0].text
        
        # Must contain at least one step
        assert re.search(r"## Step \d+:", text), f"No step found in: {text[:200]}"
        
        # Must end with boxed answer
        assert "\\boxed{" in text, f"No boxed answer in: {text[-200:]}"
    
    def test_llm_step_numbers_sequential(self):
        """Steps must be numbered sequentially: 1, 2, 3, ..."""
        output_text = "## Step 1: First\n...\n## Step 2: Second\n..."
        steps = re.findall(r"## Step (\d+):", output_text)
        step_nums = [int(s) for s in steps]
        assert step_nums == list(range(1, len(step_nums) + 1))

@pytest.mark.integration
@pytest.mark.gpu
class TestSMCIntegration:
    """Full integration tests with real models."""
    
    def test_single_problem_end_to_end(self):
        """Run SMC on one problem, verify all components work together."""
        config = SMCSteeringConfig(
            n_particles=4, 
            max_steps=5,
            generator_model_path="Qwen/Qwen2.5-7B",
            reward_model_path="Qwen/Qwen2.5-Math-PRM-7B",
        )
        baseline = SMCSteeringBaseline(config)
        
        result = baseline.evaluate_single(
            prompt="What is 2 + 2?",
            ground_truth="4"
        )
        
        assert "final_answer" in result
        assert "orm_score" in result
        assert "n_steps" in result
        assert result["n_steps"] <= 5
    
    def test_step_numbering_across_expansions(self):
        """
        CRITICAL: Verify steps are numbered sequentially across multiple expansions.
        
        ⚠️ This tests a potential issue: When we stop generation at "## Step",
        the next generation call needs to continue with the correct step number.
        If the LLM doesn't naturally continue the sequence, steps may be misnumbered.
        
        If this test fails, we may need to:
        1. Include the stop string in output and re-add it to the prompt
        2. Add explicit step number tracking in expand_particles()
        3. Use a different stop pattern
        """
        config = SMCSteeringConfig(
            n_particles=2,
            max_steps=5,
            generator_model_path="Qwen/Qwen2.5-7B",
        )
        generator = LLM(model=config.generator_model_path, ...)
        pf = LLMParticleFilter(config, generator, mock_reward_model)
        
        prompt = apply_chat_template(SYSTEM_PROMPT, "What is 15 + 27?")
        pf.initialize(prompt)
        
        # Run multiple expansion steps
        for _ in range(3):
            if not pf.expand_particles():
                break
        
        # Check that all particles have sequential step numbers
        for particle in pf.particles:
            steps = re.findall(r"## Step (\d+):", particle.text)
            step_nums = [int(s) for s in steps]
            if len(step_nums) > 1:
                # Steps should be sequential: 1, 2, 3, ...
                expected = list(range(1, len(step_nums) + 1))
                assert step_nums == expected, \
                    f"Non-sequential steps: {step_nums}, expected {expected}\n" \
                    f"Particle text: {particle.text[:500]}"
    
    def test_prm_scores_available_at_each_step(self):
        """PRM scores are computed after each step generation."""
        # Run with logging enabled, verify PRM called after each step
        ...
    
    def test_orm_selects_from_all_surviving_particles(self):
        """ORM scores all final particles and selects best."""
        # Verify ORM is called once at end with all particles
        ...
```

### 6.4 Smoke Tests

```python
@pytest.mark.smoke
@pytest.mark.gpu
def test_smc_baseline_runs_on_small_dataset():
    """Run on 5 problems without crashing."""
    config = SMCSteeringConfig(
        dataset_path="trex/data/gsm8k_test_small.jsonl",  # 5 problems
        n_particles=4,
        max_steps=10,
    )
    baseline = SMCSteeringBaseline(config)
    summary = baseline.run()
    
    assert summary["total_problems"] == 5
    assert 0 <= summary["accuracy"] <= 1
    assert "avg_orm_score" in summary

@pytest.mark.smoke
def test_checkpointing_save_and_resume():
    """Verify checkpoint can be saved and resumed."""
    config = SMCSteeringConfig(enable_checkpointing=True)
    baseline = SMCSteeringBaseline(config)
    
    # Simulate partial progress
    baseline.checkpoint_mgr.state["completed_idx"] = 5
    baseline.checkpoint_mgr.save()
    
    # Create new instance, should resume
    baseline2 = SMCSteeringBaseline(config)
    assert baseline2.checkpoint_mgr.state["completed_idx"] == 5
```

---

## 7. SLURM Checkpointing

### 7.1 Checkpoint State Structure

> **Design Decision:** We only checkpoint **between problems**, not mid-problem.
> This simplifies the checkpoint format and avoids storing large particle states.

```python
checkpoint_state = {
    "completed_idx": 0,              # Number of problems completed
    "results": [],                    # Per-problem results (compact)
    "config_hash": "...",             # Hash of config for validation
    "finished": False,
}
```

> **Note:** We do NOT checkpoint mid-problem because:
> 1. Storing N particles with full text is expensive
> 2. SLURM signals give 120s warning - enough to finish most problems
> 3. Simpler resume logic (always restart current problem)

### 7.2 Signal Handling

```python
# Handle SLURM preemption signals
signal.signal(signal.SIGTERM, graceful_shutdown)  # SLURM kill
signal.signal(signal.SIGUSR1, graceful_shutdown)  # SLURM warning
```

### 7.3 SLURM Script Template

```bash
#!/bin/bash
#SBATCH --job-name=smc_baseline
#SBATCH --output=logs/smc_%j.out
#SBATCH --error=logs/smc_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --requeue                    # Enable requeue on preemption
#SBATCH --signal=SIGUSR1@120         # Send signal 120s before timeout

# Automatic resubmission on preemption
if [[ -f "${OUTPUT_DIR}/checkpoint.json" ]]; then
    echo "Resuming from checkpoint..."
fi

python -m trex.baselines.smc_steering_baseline \
    --generator_model_path Qwen/Qwen2.5-7B \
    --reward_model_path Qwen/Qwen2.5-Math-PRM-7B \
    --n_particles 16 \
    --enable_checkpointing \
    --output_dir ${OUTPUT_DIR}
```

---

## 8. Implementation Order (TDD Approach)

> **Philosophy:** Write tests FIRST, then implement code to pass them. This ensures clear specifications and catches regressions early.

---

### Phase 1: Unit Tests for New Components (Day 1, Morning)

Write all unit tests BEFORE any implementation. Tests use mocks for external dependencies (vLLM, GPU models).

#### 1.1 Create Test Files

| Priority | Test File | Location |
|----------|-----------|----------|
| P0 | `test_smc_config.py` | `trex/tests/test_baselines/` |
| P0 | `test_reward_model.py` | `trex/tests/test_models/` |
| P0 | `test_llm_particle_filter.py` | `trex/tests/test_smc/` |
| P0 | `test_checkpoint_manager.py` | `trex/tests/test_baselines/` |
| P1 | `test_smc_steering_baseline.py` | `trex/tests/test_baselines/` |

#### 1.2 Test Specifications

**`test_smc_config.py`** (5 tests)
```python
class TestSMCSteeringConfig:
    def test_default_values_are_sensible(self): ...
    def test_custom_n_particles(self): ...
    def test_invalid_resampling_method_raises(self): ...
    def test_config_from_dict(self): ...
    def test_config_to_dict(self): ...
```

**`test_reward_model.py`** (12 tests, NO GPU required - mocked)
```python
class TestRewardModelFormatting:
    # PRM formatting tests
    def test_format_for_prm_inserts_reward_token_after_each_step(self): ...
    def test_format_for_prm_handles_single_step(self): ...
    def test_format_for_prm_handles_empty_input(self): ...
    def test_format_for_prm_preserves_boxed_answer(self): ...
    
    # ORM formatting tests
    def test_format_for_orm_single_token_at_end(self): ...
    def test_format_for_orm_no_intermediate_tokens(self): ...
    
    # Edge cases
    def test_format_handles_malformed_steps(self): ...
    def test_format_handles_unicode(self): ...

class TestRewardModelScoring:
    # Mocked scoring tests
    def test_score_prm_returns_list_per_text(self, mock_llm): ...
    def test_score_orm_returns_single_score(self, mock_llm): ...
    def test_get_latest_step_scores_extracts_last(self, mock_llm): ...
    def test_batch_scoring_multiple_texts(self, mock_llm): ...
```

**`test_llm_particle_filter.py`** (17 tests, vLLM mocked)
```python
class TestStepDetection:
    def test_step_pattern_matches_valid_format(self): ...
    def test_step_pattern_requires_colon(self): ...
    def test_step_pattern_requires_space_after_hash(self): ...
    def test_step_pattern_multi_digit_numbers(self): ...
    def test_count_steps_in_text(self): ...

class TestParticleExpansion:
    def test_expand_appends_generated_text(self, mock_llm): ...
    def test_expand_stops_at_step_delimiter(self, mock_llm): ...
    def test_expand_marks_finished_on_boxed(self, mock_llm): ...
    def test_expand_marks_finished_on_max_tokens(self, mock_llm): ...
    def test_expand_returns_false_when_all_finished(self, mock_llm): ...

class TestSMCWeighting:
    def test_score_particles_calls_prm(self, mock_reward_model): ...
    def test_weights_updated_from_prm_scores(self, mock_reward_model): ...
    def test_epsilon_added_to_avoid_zero_weights(self): ...
    def test_multiplicative_weight_update(self, mock_llm, mock_reward_model): ...  # NEW

class TestORMSelection:
    def test_select_best_by_orm_returns_highest(self, mock_reward_model): ...
    def test_select_best_by_orm_handles_ties(self, mock_reward_model): ...

class TestParticleIndependence:
    # NEW: Critical test for deepcopy fix
    def test_particle_independence_after_resampling(self): ...
```

**`test_checkpoint_manager.py`** (10 tests)
```python
class TestCheckpointManager:
    def test_save_creates_file(self, tmp_path): ...
    def test_load_restores_state(self, tmp_path): ...
    def test_load_nonexistent_returns_default(self, tmp_path): ...
    def test_atomic_save_prevents_corruption(self, tmp_path): ...
    def test_checkpoint_interval_respected(self): ...

class TestSignalHandling:
    def test_sigterm_triggers_save(self, mock_signal): ...
    def test_sigusr1_triggers_save(self, mock_signal): ...
    def test_graceful_shutdown_saves_state(self): ...

class TestResumeLogic:
    def test_resume_skips_completed_problems(self): ...
    def test_resume_preserves_partial_results(self): ...
```

#### 1.3 Checklist

- [ ] Create `trex/tests/conftest.py` (shared fixtures)
- [ ] Create `trex/tests/test_baselines/__init__.py`
- [ ] Create `trex/tests/test_models/__init__.py`
- [ ] Write `trex/tests/test_baselines/test_smc_config.py` (5 tests)
- [ ] Write `trex/tests/test_models/test_reward_model.py` (12 tests)
- [ ] Write `trex/tests/test_smc/test_llm_particle_filter.py` (17 tests)
- [ ] Write `trex/tests/test_baselines/test_checkpoint_manager.py` (10 tests)
- [ ] Run tests: ALL SHOULD FAIL (no implementation yet)

---

### Phase 2: Implement Core Components (Day 1, Afternoon)

Implement code to make Phase 1 tests pass.

#### 2.1 Implementation Order (Dependency-Sorted)

```
1. SMCSteeringConfig (no dependencies)
   ↓
2. CheckpointManager (no dependencies)
   ↓
3. RewardModel (needs tokenizer mock setup)
   ↓
4. LLMParticleFilter (extends ParticleFilter, uses RewardModel)
   ↓
5. SMCSteeringBaseline (orchestrates all)
```

#### 2.2 File Specifications

**`trex/baselines/smc_config.py`**
- `SMCSteeringConfig` dataclass (includes `seed` and `checkpoint_time_interval`)
- `CheckpointManager` class
- Validation methods

**`trex/models/__init__.py`**
- Export `RewardModel`
- Export `HFRewardModelBackend` (alternative backend)

**`trex/models/reward_model.py`**
- `RewardModel` class with:
  - `format_for_prm(text: str) -> str`
  - `format_for_orm(text: str) -> str`
  - `_extract_reward_scores(texts: List[str], mode: str) -> Union[List[List[float]], List[float]]`
  - `score_prm(texts: List[str]) -> List[List[float]]`
  - `score_orm(texts: List[str]) -> List[float]`
  - `get_latest_step_scores(texts: List[str]) -> torch.Tensor`

**`trex/models/vllm_logit_sampler.py`**
- `patch_vllm_for_logits()` function
- `HFRewardModelBackend` class (alternative backend using HuggingFace)

**`trex/smc/llm_particle_filter.py`**
- `LLMParticleFilter(ParticleFilter)` class with:
  - `STEP_PATTERN: re.Pattern`
  - `STEP_HEADER_PATTERN: re.Pattern`
  - `_get_next_step_number(text: str) -> int`
  - `_inject_step_header(text: str) -> str`
  - `expand_particles() -> bool`
  - `score_particles() -> torch.Tensor`
  - `select_best_by_orm() -> Particle`
  - `step() -> bool`
  - `run() -> Particle`

**`trex/baselines/smc_steering_baseline.py`**
- `SMCSteeringBaseline` class with:
  - SLURM signal handling
  - Time-based AND problem-count based checkpointing
  - Main evaluation loop

#### 2.3 Checklist

- [ ] Create `trex/models/__init__.py`
- [ ] Implement `trex/baselines/smc_config.py`
- [ ] Implement `trex/models/vllm_logit_sampler.py` (start with `HFRewardModelBackend`)
- [ ] Implement `trex/models/reward_model.py`
- [ ] Implement `trex/smc/llm_particle_filter.py`
- [ ] Implement `trex/baselines/smc_steering_baseline.py`
- [ ] Update `trex/smc/__init__.py` to export `LLMParticleFilter`
- [x] ~~**FIX: Update `trex/smc/particle_filter.py` to use `deepcopy`**~~ ✅ DONE
  - `from copy import deepcopy` already in place
  - Particle independence is now guaranteed after resampling
- [ ] Run unit tests: ALL SHOULD PASS

---

### Phase 3: Integration Tests (Day 2, Morning)

Tests requiring actual GPU/models. These verify real-world behavior.

#### 3.1 GPU-Based Test Specifications

**`test_reward_model_gpu.py`** (requires GPU)
```python
@pytest.mark.gpu
class TestRewardModelGPU:
    def test_prm_returns_valid_probabilities(self): ...
    def test_orm_returns_valid_probabilities(self): ...
    def test_correct_solution_scores_higher(self): ...
    def test_multi_step_scoring_correct_count(self): ...
```

**`test_llm_format_compliance.py`** (requires GPU)
```python
@pytest.mark.gpu
class TestLLMFormatCompliance:
    def test_qwen_follows_step_format(self): ...
    def test_step_numbers_are_sequential(self): ...
    def test_ends_with_boxed_answer(self): ...
```

**`test_smc_integration.py`** (requires GPU)
```python
@pytest.mark.gpu
class TestSMCIntegration:
    def test_single_problem_end_to_end(self): ...
    def test_resampling_occurs_on_low_ess(self): ...
    def test_orm_selection_at_completion(self): ...
```

#### 3.2 Checklist

- [ ] Write `trex/tests/test_models/test_reward_model_gpu.py`
- [ ] Write `trex/tests/test_smc/test_llm_format_compliance.py`
- [ ] Write `trex/tests/test_smc/test_smc_integration.py`
- [ ] Run integration tests on GPU node

---

### Phase 4: Smoke Tests & SLURM Validation (Day 2, Afternoon)

#### 4.1 Smoke Test Specifications

```python
@pytest.mark.smoke
@pytest.mark.gpu
def test_smc_baseline_5_problems():
    """Run on 5 problems without crashing."""
    ...

@pytest.mark.smoke
def test_checkpoint_save_resume_cycle():
    """Verify checkpoint can be saved and resumed."""
    ...

@pytest.mark.smoke
def test_slurm_script_syntax():
    """Verify SLURM script has valid syntax."""
    ...
```

#### 4.2 Checklist

- [ ] Create `trex/data/gsm8k_test_5.jsonl` (5-problem test set)
- [ ] Create `trex/scripts/run_smc_baseline.sh` (SLURM script)
- [ ] Write smoke tests in `trex/tests/test_baselines/test_smc_smoke.py`
- [ ] Run smoke tests on SLURM cluster
- [ ] Verify checkpoint resume works after manual interruption

---

### Phase 5: Full Evaluation (Day 3)

- [ ] Run on GSM8K-500 or MATH-500
- [ ] Compare with Best-of-N baseline (same compute budget)
- [ ] Log to WandB with proper tags
- [ ] Update EXPERIMENTS.md with results

---

## 9. References

- **IMPLEMENTATION_PLAN.md**: Section 1.2 (Standard SMC Steering)
- **HIGH_LEVEL_CONTEXT.md**: Section 6.2 (SMC baselines)
- **Qwen2.5-Math-PRM-7B**: [HuggingFace Model Card](https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B)
- **Existing Code:**
  - `trex/smc/particle_filter.py` - Base ParticleFilter class
  - `trex/smc/resampling.py` - Resampling algorithms
  - `trex/baselines/best_of_n_baseline.py` - Reference for checkpointing pattern
