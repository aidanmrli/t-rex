#!/bin/bash

#SBATCH --partition=short-unkillable                           
#SBATCH --cpus-per-task=64                                
#SBATCH --gres=gpu:h100:4                                     
#SBATCH --mem=512G                                        
#SBATCH --time=3:00:00                                   
#SBATCH -o /network/scratch/l/lia/t-rex/slurm/slurm-%j.out  # Write the log on scratch

# 1. Load the required modules
module load cuda/12.6.0

# 2. Load your environment
source .venv/bin/activate

# Scratch and weights setup
SCRATCH_DIR="/network/scratch/l/lia/t-rex"
SCRATCH_WEIGHTS="/network/scratch/l/lia/model_weights"

# 3. Set environment variables for model weights and datasets
export HF_HOME="$SCRATCH_WEIGHTS"
export HF_DATASETS_CACHE="$SCRATCH_WEIGHTS"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$SCRATCH_WEIGHTS"

# 4. Set up experimental results on scratch and symlink back
mkdir -p "$SCRATCH_DIR/results"

# If trex/results is a directory but not a symlink, move its content to scratch and symlink
if [ -d "trex/results" ] && [ ! -L "trex/results" ]; then
    echo "Moving existing results to scratch..."
    cp -r trex/results/* "$SCRATCH_DIR/results/"
    rm -rf trex/results
    ln -s "$SCRATCH_DIR/results" trex/results
elif [ ! -e "trex/results" ]; then
    ln -s "$SCRATCH_DIR/results" trex/results
fi

# NOTE: The model should be saved in $SLURM_TMPDIR

# =============================================================================
# EXPERIMENT CONFIGURATION - Edit these values directly
# =============================================================================

# Model
MODEL="Qwen/Qwen2.5-Math-7B-Instruct"
# Extracts the model name from the path (e.g., Qwen2.5-Math-7B-Instruct)
MODEL_NAME=$(basename "$MODEL")

# Dataset: "gsm8k" or "math"
# DATASET="gsm8k"
# DATASET_PATH="trex/data/gsm8k_platinum_test.jsonl"
DATASET="math"
DATASET_PATH="trex/data/math_test.jsonl"

# Sampling
N_SAMPLES=32                # Number of samples per prompt (Best-of-N)
SWEEP_SIZE=100             # Number of problems for temperature sweep

# vLLM / GPU settings
TP_SIZE=4                  # Tensor parallelism (must match #SBATCH --gres=gpu:X)
MAX_NUM_SEQS=512           # Max sequences in KV cache
GPU_MEM_UTIL=0.95          # GPU memory utilization (0.0-1.0)

# Chat Template: Set to false for base models, true for instruct models
APPLY_CHAT_TEMPLATE=true

# Output
OUTPUT_DIR="trex/results/bon_baseline/${MODEL_NAME}/${DATASET}_n${N_SAMPLES}"

# =============================================================================

# Additional arguments
EXTRA_ARGS=""
if [ "$APPLY_CHAT_TEMPLATE" = false ]; then
    EXTRA_ARGS="$EXTRA_ARGS --no_chat_template"
fi

# Run the baseline script
# We use python -m to ensure the package structure is respected
python -m trex.baselines.best_of_n_baseline \
    --model_path "$MODEL" \
    --dataset "$DATASET" \
    --dataset_path "$DATASET_PATH" \
    --n_samples "$N_SAMPLES" \
    --tp_size "$TP_SIZE" \
    --max_num_seqs "$MAX_NUM_SEQS" \
    --gpu_memory_utilization "$GPU_MEM_UTIL" \
    --output_dir "$OUTPUT_DIR" \
    --sweep_size "$SWEEP_SIZE" \
    --use_wandb \
    $EXTRA_ARGS

# Then, after the job is finished, copy whatever you want to save on $SCRATCH
# cp $SLURM_TMPDIR/<to_save> $SCRATCH/t-rex