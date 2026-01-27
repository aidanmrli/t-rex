# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

T-REX (Twisted Replica Exchange for Bootstrapping Reasoning) is a research project extending [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) for advanced RLHF techniques focused on mathematical reasoning. The project implements probabilistic inference methods to solve the "narrow passage" problem in constrained language generation.

**Key concepts:**
- **Parallel Tempering:** Multiple chains at different temperatures (β=0 hot/exploration → β=1 cold/exploitation)
- **Twisted SMC:** Value function-guided particle filtering for efficient search
- **Non-Reversible Transport:** Block-Gibbs editing to bridge creative and rigorous distributions

## Repository Structure

```
t-rex/
├── openrlhf/           # Core RLHF framework (forked from OpenRLHF)
│   ├── cli/            # Entry points: train_sft, train_rm, train_ppo_ray, etc.
│   ├── trainer/        # SFTTrainer, RMTrainer, PPOTrainer implementations
│   └── models/         # Actor, Critic, Reward model wrappers
├── trex/               # T-REX research code
│   ├── baselines/      # Best-of-N, GRPO reward functions
│   ├── eval/           # MathVerifier, answer parsing, grading
│   ├── data/           # GSM8K, MATH, MATH-500 datasets (JSONL)
│   ├── scripts/        # SLURM job scripts with auto-requeue
│   └── utils/          # Efficiency tracking utilities
├── HIGH_LEVEL_CONTEXT.md   # Mathematical specification (read for algorithm details)
├── IMPLEMENTATION_PLAN.md  # Development roadmap with status
└── EXPERIMENTS.md          # Experiment tracking
```

## Development Commands

### Environment Setup
```bash
module load python/3.12.4 scipy-stack arrow/21.0.0 gcc opencv/4.13.0 rust
virtualenv --no-download venv
source venv/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch deepspeed
pip install -r requirements_tamia.txt --no-index
pip install --no-index vllm
pip install ~/flash_attn-2.8.3+cu126torch2.9-cp312-cp312-linux_x86_64.whl --no-deps
```

- Model weights should be saved in `/scratch/l/liaidan/model_weights`.
- The scratch directory for this project is in `/scratch/l/liaidan/t-rex/`. Any experimental results should be in `/scratch/l/liaidan/t-rex/results`. Any sbatch out and err logs should be in `/scratch/l/liaidan/t-rex/slurm`.

### Pre-commit Hooks
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files  # Manual run
```

### Running Tests
```bash
pytest                           # All tests
pytest -m unit                   # Unit tests only
pytest -m integration            # Integration tests only
pytest tests/test_eval/          # Specific module
```

### Training Commands

**GRPO Training:**
```bash
python -m openrlhf.cli.train_ppo_ray \
    --pretrain Qwen/Qwen2.5-7B \
    --remote_rm_url trex/baselines/grpo_reward_func.py \
    --prompt_data trex/data/gsm8k_train.jsonl \
    --advantage_estimator group_norm \
    --n_samples_per_prompt 8 \
    --colocate_all_models \
    --vllm_enable_sleep
```

**Best-of-N Evaluation:**
```bash
python -m trex.baselines.best_of_n_baseline \
    --model Qwen/Qwen2.5-7B \
    --dataset trex/data/gsm8k_test.jsonl
```

**SLURM Job Submission:**
```bash
sbatch trex/scripts/run_grpo_baseline.sh
sbatch trex/scripts/run_bon_baseline.sh
```

For SLURM, tasks are assigned to complete nodes. The nodes have 48 cores each and 512GB of available memory. Use one of the following Slurm options:

- For a task on a node with an H100 GPU: `--gpus=h100:4`
- For a task on a node with an H200 GPU: `--gpus=h200:8`
- For tasks with multiple nodes, use `--gpus-per-nodes=h100:4` or `--gpus-per-nodes=h200:8`.

NOTE: We should always check that we are maximizing our GPU usage before submitting a long job. The Tamia time limit is 24 hours. Beyond that, we should use checkpointing. 

Checkpointing should be built into every script that we use for training or evaluation.

Make sure that we export these variables in our SLURM scripts:
```bash
export HF_HOME="$SCRATCH_WEIGHTS"
export HF_DATASETS_CACHE="$SCRATCH_WEIGHTS"
```

## Key Components

### Math Verification (`trex/eval/`)
```python
from trex.eval import MathVerifier

verifier = MathVerifier()
is_correct = verifier.verify(model_answer="\\boxed{42}", ground_truth="42")
```

Uses HuggingFace `math_verify` with SymPy fallback. Handles LaTeX, boxed format, numeric comparison.

### Reward Function Interface (`trex/baselines/grpo_reward_func.py`)
```python
def reward_func(queries, prompts, labels):
    # queries: prompts + responses
    # labels: ground truth answers
    return {
        "rewards": tensor,      # For advantage calculation
        "scores": tensor,       # For dynamic filtering (0-1)
        "extra_logs": {...}     # WandB logging
    }
```

### Checkpoint Management
SLURM scripts support automatic requeue with checkpoint resumption. Key environment variables:
- `TREX_SUBMISSION_COUNT`: Tracks requeue count
- `TREX_EFFICIENCY_PATH`: Path for efficiency metrics JSON

## Architecture Notes

### OpenRLHF Integration
- Uses Ray for distributed scheduling (Actor, Critic, Reward, Reference on separate GPUs)
- vLLM for accelerated generation (`--vllm_num_engines`, `--vllm_tensor_parallel_size`)
- DeepSpeed ZeRO-3 for memory-efficient training

### Advantage Estimators
- `group_norm`: GRPO (group normalization without critic)
- `gae`: PPO (requires critic network)
- `reinforce`: REINFORCE++
- `rloo`: RLOO variant

### Temperature in Parallel Tempering
Target distribution: `π_k(x) ∝ p_0(x) · φ(x)^β_k`
- β ≈ 0: Pure base model (hot, diverse)
- β = 1: Strict posterior (cold, valid)

## Implementation Status

**Completed:**
- Best-of-N baseline with temperature sweep
- GRPO training integration
- Math verification (multi-backend)
- Dataset preparation

**In Progress (see IMPLEMENTATION_PLAN.md):**
- PPO math baseline
- Standard SMC steering (Rollout Roulette)
- Value head architecture
- Twisted SMC inference
- Parallel tempering chains