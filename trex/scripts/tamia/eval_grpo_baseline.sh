#!/bin/bash

#SBATCH --job-name=eval-grpo-baseline
#SBATCH --account=def-bengioy                       
#SBATCH --cpus-per-task=64                                
#SBATCH --gres=gpu:h100:4                                     
#SBATCH --mem=512G                                        
#SBATCH --time=24:00:00                                   
#SBATCH -o /scratch/l/liaidan/t-rex/slurm/eval-grpo-%j.out
#SBATCH -e /scratch/l/liaidan/t-rex/slurm/eval-grpo-%j.err

# =============================================================================
# T-REX GRPO Evaluation Script
# Evaluates a trained GRPO model using the Best-of-N harness.
# =============================================================================

# 1. Load the required modules
module load python/3.12.4 scipy-stack arrow/21.0.0 gcc opencv/4.13.0 rust

# 2. Load your environment
source venv/bin/activate

# Scratch and weights setup
SCRATCH_DIR="/scratch/l/liaidan/t-rex"
SCRATCH_WEIGHTS="/scratch/l/liaidan/model_weights"

export HF_HOME="$SCRATCH_WEIGHTS"
export HF_DATASETS_CACHE="$SCRATCH_WEIGHTS"
export TOKENIZERS_PARALLELISM=false
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
TRAIN_N_SAMPLES=8

# Construct Path to GRPO Checkpoints
CKPT_BASE_DIR="/scratch/l/liaidan/t-rex/results/grpo_baseline/${MODEL_NAME}/${DATASET}_n${TRAIN_N_SAMPLES}/checkpoints"

# Find latest checkpoint (globally or step-based)
if [ -d "$CKPT_BASE_DIR" ]; then
    # Look for directories like 'global_step_*' or similar, sort by modification time (latest first)
    # Using 'ls -td' to sort by time, head -n 1 to get latest
    LATEST_CKPT=$(ls -td "$CKPT_BASE_DIR"/*/ | head -n 1)
    
    if [ -z "$LATEST_CKPT" ]; then
        echo "ERROR: No checkpoints found in $CKPT_BASE_DIR"
        echo "Please check your training run."
        exit 1
    fi
    
    # Remove trailing slash
    MODEL_PATH=${LATEST_CKPT%/}
    echo "Using latest checkpoint: $MODEL_PATH"
else
    echo "ERROR: Checkpoint directory not found: $CKPT_BASE_DIR"
    exit 1
fi

# 2. Evaluation Settings
# -----------------------------------------------------------------------------
TEST_DATASET="gsm8k"
TEST_DATASET_PATH="/scratch/l/liaidan/t-rex/data/gsm8k_platinum_test.jsonl" # Consistent with training script

# Eval N samples (Best-of-N during eval)
# Set N=1 for simple pass@1, or N=16 to see pass@1...pass@16 capabilities
EVAL_N_SAMPLES=16

# Temperatures to test
# 0.0 for Greedy (deterministic)
# 0.6 for Sampling (standard exploration)
TEMPS="0.0 0.6 1.0"

# Output Directory
OUTPUT_DIR="/scratch/l/liaidan/t-rex/results/eval_grpo/${MODEL_NAME}/${TEST_DATASET}_trained_n${TRAIN_N_SAMPLES}"

# vLLM Settings
TP_SIZE=2
GPU_MEM_UTIL=0.90

# =============================================================================
# EXECUTION
# =============================================================================

echo "============================================="
echo "Starting GRPO Evaluation"
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

