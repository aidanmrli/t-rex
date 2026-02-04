#!/bin/bash

#SBATCH --job-name=eval-ppo-baseline
#SBATCH --account=aip-apsarath
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH -o /scratch/l/liaidan/t-rex/slurm/eval-ppo-%j.out
#SBATCH -e /scratch/l/liaidan/t-rex/slurm/eval-ppo-%j.err

# =============================================================================
# T-REX PPO Evaluation Script
# Evaluates a trained PPO model using the Best-of-N harness.
# =============================================================================

# 1. Load the required modules
module load python/3.12.4 scipy-stack arrow/21.0.0 gcc opencv/4.13.0 rust cuda/12.6

# 2. Load your environment
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
SEARCH_DIR="${REPO_ROOT}"
while [[ "${SEARCH_DIR}" != "/" && ! -d "${SEARCH_DIR}/.git" ]]; do
    SEARCH_DIR="$(dirname "${SEARCH_DIR}")"
done
if [[ -d "${SEARCH_DIR}/.git" ]]; then
    REPO_ROOT="${SEARCH_DIR}"
fi
if [[ ! -f "${REPO_ROOT}/venv/bin/activate" ]]; then
    echo "ERROR: Could not find virtualenv at ${REPO_ROOT}/venv/bin/activate"
    exit 1
fi
cd "${REPO_ROOT}"
source "${REPO_ROOT}/venv/bin/activate"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Scratch and weights setup
SCRATCH_DIR="/scratch/l/liaidan/t-rex"
SCRATCH_WEIGHTS="/scratch/l/liaidan/model_weights"

export HF_HOME="$SCRATCH_WEIGHTS"
export HF_DATASETS_CACHE="$SCRATCH_WEIGHTS"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY="8ac57bf9aa5138a9e30d747070d1ebc22b581efc"
export WANDB_MODE=offline
mkdir -p "$SCRATCH_WEIGHTS"

# 4. Set up experimental results on scratch
mkdir -p "$SCRATCH_DIR/results"
mkdir -p "$SCRATCH_DIR/slurm"

if [ -d "trex/results" ] && [ ! -L "trex/results" ]; then
    # If it's a real directory, symlink logic handles it (assumed handled by training script)
    :
elif [ ! -e "trex/results" ]; then
    ln -s "$SCRATCH_DIR/results" trex/results
fi

# =============================================================================
# CONFIGURATION
# =============================================================================

# 1. Identify Valid Checkpoint
# -----------------------------------------------------------------------------
# Base model used for training (used to construct path)
BASE_MODEL="Qwen/Qwen2.5-7B"
MODEL_NAME=$(basename "$BASE_MODEL")
DATASET="gsm8k"
TRAIN_N_SAMPLES=1  # PPO uses N_SAMPLES=1 (no group sampling)

# Construct Path to PPO Output
OUTPUT_BASE_DIR="/scratch/l/liaidan/t-rex/results/ppo_baseline/${MODEL_NAME}/${DATASET}_n${TRAIN_N_SAMPLES}"
CKPT_BASE_DIR="${OUTPUT_BASE_DIR}/checkpoints"

# Prefer final model if training completed, otherwise use latest checkpoint
if [ -f "${OUTPUT_BASE_DIR}/.training_complete" ]; then
    # Training completed - use the final model in the base directory
    MODEL_PATH="$OUTPUT_BASE_DIR"
    echo "Training completed. Using final model: $MODEL_PATH"
elif [ -d "$CKPT_BASE_DIR" ]; then
    # Training incomplete - find latest checkpoint
    LATEST_CKPT=$(ls -td "$CKPT_BASE_DIR"/*/ 2>/dev/null | grep -v "_actor" | head -n 1)

    if [ -z "$LATEST_CKPT" ]; then
        echo "ERROR: No checkpoints found in $CKPT_BASE_DIR"
        echo "Please check your training run."
        exit 1
    fi

    # Remove trailing slash
    MODEL_PATH=${LATEST_CKPT%/}
    echo "Using latest checkpoint: $MODEL_PATH"
else
    echo "ERROR: Neither final model nor checkpoint directory found at: $OUTPUT_BASE_DIR"
    exit 1
fi

# 2. Evaluation Settings
# -----------------------------------------------------------------------------
TEST_DATASET="gsm8k"
TEST_DATASET_PATH="/project/6100862/liaidan/t-rex/trex/data/gsm8k_platinum_test.jsonl"

# Eval N samples (Best-of-N during eval)
# Set N=1 for simple pass@1, or N=16 to see pass@1...pass@16 capabilities
EVAL_N_SAMPLES=16

# Temperatures to test
# 0.0 for Greedy (deterministic)
# 0.6 for Sampling (standard exploration)
TEMPS="0.0 0.6 1.0"

# Output Directory
OUTPUT_DIR="/scratch/l/liaidan/t-rex/results/eval_ppo/${MODEL_NAME}/${TEST_DATASET}_trained_n${TRAIN_N_SAMPLES}"

# vLLM Settings (optimized to use all 4 GPUs)
TP_SIZE=4                       # Increased from 2 to use all GPUs
GPU_MEM_UTIL=0.95               # Increased from 0.90 for better utilization

# =============================================================================
# EXECUTION
# =============================================================================

echo "============================================="
echo "Starting PPO Evaluation"
echo "Model: $MODEL_PATH"
echo "Dataset: $TEST_DATASET"
echo "Eval N: $EVAL_N_SAMPLES"
echo "Temps: $TEMPS"
echo "Output: $OUTPUT_DIR"
echo "============================================="

python -m trex.baselines.best_of_n_baseline \
    --model_path "$MODEL_PATH" \
    --dataset "$TEST_DATASET" \
    --dataset_path "$TEST_DATASET_PATH" \
    --n_samples "$EVAL_N_SAMPLES" \
    --temperatures $TEMPS \
    --tp_size "$TP_SIZE" \
    --gpu_memory_utilization "$GPU_MEM_UTIL" \
    --output_dir "$OUTPUT_DIR" \
    --sweep_size 100 \
    --eval_chunk_size 50 \
    --use_wandb
