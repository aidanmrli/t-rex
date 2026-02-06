#!/bin/bash
#SBATCH --job-name=train-value-head
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=480G
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00

set -euo pipefail

export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

# Stage-1 SFT generator checkpoint and PRM800K-formatted dataset by default.
BASE_MODEL="${BASE_MODEL:-/scratch/l/liaidan/t-rex/results/prm800k_sft/job_154126/ckpt}"
DATASET="${DATASET:-trex/data/prm800k_sft_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/value_head_training_prm800k}"

python -m trex.training.train_value_head \
    --base_model "$BASE_MODEL" \
    --value_head_type mlp \
    --dataset "$DATASET" \
    --num_rollouts_per_prompt 8 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --max_steps 1000 \
    --update_frequency 100 \
    --no_chat_template \
    --output_dir "$OUTPUT_DIR"
