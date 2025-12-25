#!/bin/bash

# Configuration
DATASET=${1:-"gsm8k"}
MODEL=${2:-"Qwen/Qwen2.5-Math-7B-Instruct"}
N_SAMPLES=${3:-8}
TP_SIZE=${4:-1}

# Set path to the dataset based on input
if [ "$DATASET" == "gsm8k" ]; then
    DATASET_PATH="trex/data/gsm8k_platinum_test.jsonl"
elif [ "$DATASET" == "math" ]; then
    DATASET_PATH="trex/data/math_test.jsonl"
else
    echo "Unknown dataset: $DATASET. Using default gsm8k path."
    DATASET_PATH="trex/data/gsm8k_platinum_test.jsonl"
fi

# Run the baseline script
# We use python -m to ensure the package structure is respected
python -m trex.baselines.best_of_n_baseline \
    --model_path "$MODEL" \
    --dataset "$DATASET" \
    --dataset_path "$DATASET_PATH" \
    --n_samples "$N_SAMPLES" \
    --tp_size "$TP_SIZE" \
    --sweep_size 100 \
    --use_wandb
