#!/bin/bash
#SBATCH --job-name=tsmc-baseline
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=480G
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00

set -euo pipefail

export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

VALUE_HEAD_PATH="${VALUE_HEAD_PATH:-}"
if [[ -z "${VALUE_HEAD_PATH}" ]]; then
    echo "ERROR: VALUE_HEAD_PATH is required (path to trained value_head.pt)."
    echo "Example: VALUE_HEAD_PATH=./results/value_head_training_prm800k/value_head.pt sbatch trex/scripts/run_tsmc_baseline.sh"
    exit 1
fi

TWIST_MODE="${TWIST_MODE:-sqrt_value}"
FINAL_SELECTION_MODE="${FINAL_SELECTION_MODE:-majority_vote}"
RESAMPLING_METHOD="${RESAMPLING_METHOD:-stratified}"

python -m trex.baselines.tsmc_baseline \
    --dataset_path trex/data/gsm8k_platinum_test.jsonl \
    --generator_model_path Qwen/Qwen2.5-7B \
    --value_model_path Qwen/Qwen2.5-7B \
    --value_head_path "$VALUE_HEAD_PATH" \
    --value_head_type mlp \
    --twist_space log_prob \
    --twist_mode "$TWIST_MODE" \
    --final_selection_mode "$FINAL_SELECTION_MODE" \
    --step_boundary_mode delimiter \
    --step_delimiter "\n\n" \
    --n_particles 16 \
    --max_smc_iterations 20 \
    --resampling_method "$RESAMPLING_METHOD" \
    --warmup_steps 0 \
    --warmup_tokens 0 \
    --temperature 0.8 \
    --output_dir ./results/tsmc_baseline
