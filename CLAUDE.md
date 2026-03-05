# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

T-REX (Twisted Replica Exchange) is a research project implementing probabilistic inference methods for mathematical reasoning. The core algorithm runs K parallel SMC chains at different temperatures with inter-chain communication via mixture proposals, avoiding the vanishing acceptance rates of traditional Metropolis-Hastings swaps.

**Key concepts:**
- **Multi-chain SMC:** K independent particle filter chains at temperatures β_1=0 (hot/prior) to β_K=1 (cold/posterior)
- **Mixture Proposals:** Cold chains absorb particles from adjacent hot chains via Bernoulli coin flip (probability λ), with importance reweighting R(x)^{Δβ} — "zero-rejection" communication
- **PRM-guided Reweighting:** Incremental weights w_t = (R(x_{1:t}) / R(x_{1:t-1}))^β using Process Reward Model scores

## Context Management
- See `docs/HIGH_LEVEL_CONTEXT.md` for mathematical specification and algorithm details
- See `docs/plans/PROJECT_HIGH_LEVEL_PLAN.md` for details of the development roadmap with status. We will make more detailed plans for each item in this plan in separate Markdown files before any implementation is done.
- See `docs/EXPERIMENTS.md` to keep track of all experiments that have been done. This should contain all details about what experiments have been ran, what hypothesis we are testing with each experiment, and what the results are.
- Old docs from the Feb 2026 approach (Twisted SMC + Block-Gibbs transport) are archived in `docs/archive/2026-feb/`.

Update these markdown contexts after carrying out actions.

## Repository Structure

```
t-rex/
├── openrlhf/           # Core RLHF framework (forked from OpenRLHF)
│   ├── cli/            # Entry points: train_sft, train_rm, train_ppo_ray, etc.
│   ├── trainer/        # SFTTrainer, RMTrainer, PPOTrainer implementations
│   └── models/         # Actor, Critic, Reward model wrappers
├── trex/               # T-REX research code
│   ├── baselines/      # Best-of-N, GRPO, SMC steering baselines
│   ├── eval/           # MathVerifier, answer parsing, grading
│   ├── data/           # GSM8K, MATH, MATH-500 datasets (JSONL)
│   ├── smc/            # SMC core: particle filter, resampling
│   ├── scripts/        # SLURM job scripts with auto-requeue
│   └── utils/          # Efficiency tracking utilities
├── docs/
│   ├── HIGH_LEVEL_CONTEXT.md   # Mathematical specification (read for algorithm details)
│   ├── EXPERIMENTS.md          # Experiment tracking
│   ├── plans/                  # Detailed implementation plans
│   └── archive/2026-feb/       # Archived docs from old approach
└── CLAUDE.md
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

In order to run anything in an interactive session (note that interactive sessions do not have GPU access):
```bash
module load python/3.12.4 scipy-stack arrow/21.0.0 gcc opencv/4.13.0 rust
source venv/bin/activate
```
- Model weights should be saved in `/scratch/l/liaidan/model_weights`.
- The scratch directory for this project is in `/scratch/l/liaidan/t-rex/`. Any experimental results should be in `/scratch/l/liaidan/t-rex/results`. Any sbatch out and err logs should be in `/scratch/l/liaidan/t-rex/slurm`.

### Running Tests

```bash
# Run all tests
pytest trex/tests/ -v

# Run only fast unit tests (exclude slow/integration)
pytest trex/tests/ -v -m "not slow and not integration"

# Run specific module
pytest trex/tests/test_eval/ -v

# Run with coverage
pytest trex/tests/ --cov=trex --cov-report=html

# Run tests matching pattern
pytest trex/tests/ -v -k "parser"
```

**SLURM Job Submission:**

Use the sbatch scripts in the `trex/scripts/tamia` folder. These should contain training and evaluation scripts.

For example:
```bash
sbatch trex/scripts/tamia/run_grpo_baseline.sh
sbatch trex/scripts/tamia/run_bon_baseline.sh
```

For SLURM, tasks are assigned to complete nodes. Use one of the following configurations:

| GPU Type | GPUs/Node | Memory | CPUs | SBATCH Options |
|----------|-----------|--------|------|----------------|
| H100     | 4         | 500GB  | 48   | `--gres=gpu:h100:4 --mem=480G --cpus-per-task=48` |
| H200     | 8         | 1TB    | 64   | `--gres=gpu:h200:8 --mem=950G --cpus-per-task=64` |

**IMPORTANT: Compute nodes have NO internet access.** You must:
1. Download models to the cache on the login node first
2. Use `HF_HUB_OFFLINE=1` in SLURM scripts
3. Use `WANDB_MODE=offline` for WandB logging

NOTE: We should always check that we are maximizing our GPU usage before submitting a long job. The Tamia time limit is 24 hours. Beyond that, we should use checkpointing.

Checkpointing should be built into every script that we use for training or evaluation.

Make sure that we export these variables in our SLURM scripts:
```bash
export HF_HOME="$SCRATCH_WEIGHTS"
export HF_DATASETS_CACHE="$SCRATCH_WEIGHTS"
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
```

### Pre-downloading Models

Before running SLURM jobs, download models on the login node:
```bash
source venv/bin/activate
export HF_HOME="/scratch/l/liaidan/model_weights"
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B')"
```

### Memory Constraints with `--colocate_all_models`

When using `--colocate_all_models` (recommended for 4-GPU setup), memory is shared between vLLM and DeepSpeed:
- `VLLM_GPU_UTIL=0.6` is the safe maximum (0.85+ causes OOM on wake-up)
- Use conservative batch sizes: `MICRO_TRAIN_BATCH_SIZE=4`, `MICRO_ROLLOUT_BATCH_SIZE=16`

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
