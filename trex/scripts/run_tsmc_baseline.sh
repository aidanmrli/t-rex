#!/bin/bash
#SBATCH --job-name=tsmc-baseline
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=480G
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00

set -euo pipefail

export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

python -m trex.baselines.tsmc_baseline \
    --dataset_path trex/data/gsm8k_platinum_test.jsonl \
    --generator_model_path Qwen/Qwen2.5-7B \
    --value_model_path Qwen/Qwen2.5-7B \
    --value_head_type mlp \
    --twist_space log_prob \
    --twist_mode value \
    --n_particles 16 \
    --max_smc_iterations 20 \
    --temperature 0.8 \
    --output_dir ./results/tsmc_baseline
