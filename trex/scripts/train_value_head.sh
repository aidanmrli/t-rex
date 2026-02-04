#!/bin/bash
#SBATCH --job-name=train-value-head
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=480G
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00

set -euo pipefail

export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

BASE_MODEL="Qwen/Qwen2.5-7B"
DATASET="trex/data/gsm8k_platinum_train.jsonl"
OUTPUT_DIR="./results/value_head_training"

python -m trex.training.train_value_head \
    --base_model "$BASE_MODEL" \
    --value_head_type mlp \
    --dataset "$DATASET" \
    --num_rollouts_per_prompt 8 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --max_steps 1000 \
    --update_frequency 100 \
    --output_dir "$OUTPUT_DIR"
