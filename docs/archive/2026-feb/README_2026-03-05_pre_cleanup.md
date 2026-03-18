# T-REX: Twisted Replica Exchange for Bootstrapping Reasoning

(NOTE: This repo is VERY much a work in progress.)

T-REX is a research project that extends [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) with probabilistic inference methods for improving mathematical reasoning in large language models. It addresses the "narrow passage" problem in constrained language generation — where correct reasoning paths require specific sequences of steps that standard autoregressive sampling struggles to find.

The core idea: instead of relying solely on training-time optimization (RLHF/GRPO) or brute-force sampling (Best-of-N), T-REX combines **parallel tempering**, **twisted Sequential Monte Carlo (SMC)**, and **non-reversible transport** to efficiently search the space of reasoning traces at inference time.

## Key Components

- **Parallel Tempering:** Maintains multiple chains at different temperatures (hot chains explore freely, cold chains enforce correctness), with non-reversible swap schedules for efficient propagation.
- **Twisted SMC:** A learned value function (twist) biases particle filtering toward high-reward reasoning trajectories, converting expensive search into efficient guided generation.
- **Non-Reversible Transport:** A Block-Gibbs editor that repairs "hot" exploratory samples into valid solutions, bridging the gap between creative exploration and rigorous correctness.

## Built on OpenRLHF

This repository is a fork of [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), an open-source RLHF framework built on Ray, vLLM, and DeepSpeed. We reuse OpenRLHF's infrastructure for:

- **SFT / Reward Model / PPO / GRPO training** — the full `openrlhf/` directory provides battle-tested distributed training pipelines
- **Ray-based distributed scheduling** — placing Actor, Critic, Reward, and Reference models across GPUs
- **vLLM integration** — accelerated generation with Hybrid Engine support (`--colocate_all_models`, `--vllm_enable_sleep`)
- **Reinforced fine-tuning with custom reward functions** — GRPO/PPO with pluggable `reward_func` (see `trex/baselines/grpo_reward_func.py`)

T-REX adds the `trex/` directory on top of this foundation, implementing the novel probabilistic inference components (SMC, tempering, twist learning) while leveraging OpenRLHF for all standard RLHF workflows.

## Repository Structure

```
t-rex/
├── openrlhf/                  # Core RLHF framework (forked from OpenRLHF)
│   ├── cli/                   # Entry points: train_sft, train_rm, train_ppo_ray
│   ├── trainer/               # SFTTrainer, RMTrainer, PPOTrainer
│   └── models/                # Actor, Critic, Reward model wrappers
├── trex/                      # T-REX research code
│   ├── baselines/             # Best-of-N, GRPO, SMC steering, TSMC baselines
│   ├── eval/                  # MathVerifier, answer parsing, grading
│   ├── smc/                   # Particle filter, resampling, twisted SMC
│   ├── tempering/             # Temperature ladder, replica exchange
│   ├── models/                # Value head, twist model, reward model wrapper
│   ├── training/              # Value trainer, trajectory buffer
│   ├── data/                  # GSM8K, MATH, MATH-500 datasets (JSONL)
│   ├── scripts/tamia/         # SLURM job scripts for the Tamia cluster
│   └── tests/                 # Test suite
├── docs/
│   ├── HIGH_LEVEL_CONTEXT.md  # Mathematical specification and algorithm details
│   └── EXPERIMENTS.md         # Experiment tracking and results
└── CLAUDE.md                  # Development guide for Claude Code
```

## Current Status

**Implemented and validated:**
- Best-of-N baseline with temperature sweeps and pass@k metrics
- GRPO and PPO training and evaluation pipelines
- SMC steering baseline with PRM-guided particle filtering
- Twisted SMC (Mode-A, base-proposal path) end-to-end
- Value head self-distillation training loop
- Math verification pipeline (HuggingFace math_verify + SymPy fallback)

**Baseline results (Qwen2.5-7B):**

| Method | Dataset | Key Metric |
|--------|---------|------------|
| Best-of-N (N=32) | MATH-500 | pass@1=53.3%, pass@32=81.6% |
| GRPO (trained, N=16) | GSM8K Platinum | pass@1=38.6%, pass@16=98.3% |
| PPO (trained, N=16) | GSM8K Platinum | pass@1=25.2%, pass@16=96.6% |
| SMC token-resample (K=256) | GSM8K Platinum | 87.68% accuracy |

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for full experiment details and results.

## Setup

### Tamia Cluster (DRAC)

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

Pre-download models on the login node (compute nodes have no internet):
```bash
export HF_HOME="/scratch/l/liaidan/model_weights"
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B')"
```

### Mila Cluster

```bash
source .venv/bin/activate
module load cuda/12.6.0
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
pip install -e .
pip install vllm flash-attn
```

## Running Experiments

All SLURM scripts are in `trex/scripts/tamia/`. Examples:

```bash
# Best-of-N baseline
sbatch trex/scripts/tamia/run_bon_baseline.sh

# GRPO training
sbatch trex/scripts/tamia/run_grpo_baseline.sh

# GRPO evaluation
sbatch trex/scripts/tamia/eval_grpo_baseline.sh

# SMC steering baseline
sbatch trex/scripts/tamia/run_smc_baseline.sh

# SMC token-resampling sweep
sbatch trex/scripts/tamia/run_smc_token_sweep_array.sh

# PRM800K SFT
sbatch trex/scripts/tamia/run_sft_prm800k_h200.sh
```

Scripts support automatic checkpointing and SLURM requeue for the 24-hour time limit.

## Running Tests

```bash
# All tests
pytest trex/tests/ -v

# Fast unit tests only
pytest trex/tests/ -v -m "not slow and not integration"

# Specific module
pytest trex/tests/test_eval/ -v
```

## Architecture

T-REX builds on top of OpenRLHF's distributed infrastructure:
- **Ray** for distributed scheduling across GPUs
- **vLLM** for fast inference-time generation
- **DeepSpeed ZeRO-3** for memory-efficient training

The T-REX-specific components (SMC, tempering, twist learning) are in the `trex/` directory and are designed to be modular — each can be used independently or composed together.
