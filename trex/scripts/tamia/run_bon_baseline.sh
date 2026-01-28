#!/bin/bash

#SBATCH --job-name=bon-baseline
#SBATCH --account=aip-apsarath                          
#SBATCH --cpus-per-task=64                                
#SBATCH --gres=gpu:h100:4                                     
#SBATCH --mem=512G                                        
#SBATCH --time=24:00:00                                   
#SBATCH -o /scratch/l/liaidan/t-rex/slurm/grpo-%j.out
#SBATCH -e /scratch/l/liaidan/t-rex/slurm/grpo-%j.err
#SBATCH --requeue                                         # Requeue job on preemption
#SBATCH --signal=SIGUSR1@120                              # Send SIGUSR1 120s before timeout
- Model weights should be saved in `/scratch/l/liaidan/model_weights`.
- The scratch directory for this project is in `/scratch/l/liaidan/t-rex/`. Any experimental results should be in `/scratch/l/liaidan/t-rex/results`. Any sbatch out and err logs should be in `/scratch/l/liaidan/t-rex/slurm`.

# 1. Load the required modules
module load python/3.12.4 scipy-stack arrow/21.0.0 gcc opencv/4.13.0 rust

# 2. Load your environment
source venv/bin/activate

# Scratch and weights setup
SCRATCH_DIR="/scratch/l/liaidan/t-rex"
SCRATCH_WEIGHTS="/scratch/l/liaidan/model_weights"

# 3. Set environment variables for model weights and datasets
export HF_HOME="$SCRATCH_WEIGHTS"
export HF_DATASETS_CACHE="$SCRATCH_WEIGHTS"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$SCRATCH_WEIGHTS"

# 4. Set up experimental results on scratch and symlink back
mkdir -p "$SCRATCH_DIR/results"
mkdir -p "$SCRATCH_DIR/slurm"  # Ensure slurm log directory exists

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

# =============================================================================
# MODEL OPTIONS:
# -----------------------------------------------------------------------------
# Model Name              | Path                              | Chat Template
# -----------------------------------------------------------------------------
# Qwen2.5-7B (Base)       | Qwen/Qwen2.5-7B                   | false
# Qwen3-8B (Base)         | Qwen/Qwen3-8B-Base                | false
# Qwen2.5-7B-Instruct     | Qwen/Qwen2.5-7B-Instruct          | true
# Qwen2.5-Math-7B-Instruct| Qwen/Qwen2.5-Math-7B-Instruct     | true
# =============================================================================
# NOTE: Base models do NOT have chat templates. Set APPLY_CHAT_TEMPLATE=false.
#       Instruct models require chat templates. Set APPLY_CHAT_TEMPLATE=true.
# =============================================================================

MODEL="Qwen/Qwen2.5-7B"
# Extracts the model name from the path (e.g., Qwen2.5-7B)
MODEL_NAME=$(basename "$MODEL")

# =============================================================================
# DATASET OPTIONS:
# -----------------------------------------------------------------------------
# Dataset Name    | Path                                   | Description
# -----------------------------------------------------------------------------
# gsm8k           | trex/data/gsm8k_test.jsonl             | GSM8K test set
# gsm8k           | trex/data/gsm8k_platinum_test.jsonl    | GSM8K Platinum (cleaned)
# math            | trex/data/math_test.jsonl              | Full MATH test (5000 problems)
# math            | trex/data/math500_test.jsonl           | MATH-500 (curated 500 subset)
# =============================================================================
# DATASET="gsm8k"
# DATASET_PATH="trex/data/gsm8k_test.jsonl"
# DATASET="math"
# DATASET_PATH="trex/data/math500_test.jsonl"
DATASET="math"
DATASET_PATH="trex/data/math500_test.jsonl"

# Sampling
N_SAMPLES=32                # Number of samples per prompt (Best-of-N)
SWEEP_SIZE=100             # Number of problems for temperature sweep

# vLLM / GPU settings
TP_SIZE=4                  # Tensor parallelism (must match #SBATCH --gres=gpu:X)
MAX_NUM_SEQS=512           # Max sequences in KV cache
GPU_MEM_UTIL=0.95          # GPU memory utilization (0.0-1.0)

# Chat Template: Set to false for base models, true for instruct models
# NOTE: Base models do NOT have chat templates or thinking modes.
APPLY_CHAT_TEMPLATE=false

# Reasoning / Thinking Mode (for Qwen3, DeepSeek-R1, etc.)
# NOTE: Only applicable for models trained with reasoning tokens (like <think>)
ENABLE_THINKING=false       # Enable thinking in chat template
ENABLE_REASONING=false      # Enable reasoning in vLLM engine
REASONING_PARSER=""         # Parser for reasoning tokens (e.g., "deepseek_r1" for Qwen3/DeepSeek)

# Checkpointing: Saves progress for preemptible clusters
ENABLE_CHECKPOINTING=true
EVAL_CHUNK_SIZE=50         # Save checkpoint every N problems during full eval

# Output
OUTPUT_DIR="/scratch/l/liaidan/t-rex/results/bon_baseline/${MODEL_NAME}/${DATASET}_n${N_SAMPLES}"

# =============================================================================

# Additional arguments
EXTRA_ARGS=""
if [ "$APPLY_CHAT_TEMPLATE" = false ]; then
    EXTRA_ARGS="$EXTRA_ARGS --no_chat_template"
fi
if [ "$ENABLE_CHECKPOINTING" = false ]; then
    EXTRA_ARGS="$EXTRA_ARGS --no_checkpointing"
fi
if [ "$ENABLE_THINKING" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --enable_thinking"
fi
if [ "$ENABLE_REASONING" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --enable_reasoning"
fi
if [ -n "$REASONING_PARSER" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --reasoning_parser $REASONING_PARSER"
fi

# Log job info
echo "============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Output Dir: $OUTPUT_DIR"
echo "Checkpointing: $ENABLE_CHECKPOINTING"
echo "============================================="

# Check if this is a requeued job (checkpoint should exist)
if [ -f "$OUTPUT_DIR/checkpoint.json" ]; then
    echo "Found existing checkpoint - will resume from previous state"
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
    --eval_chunk_size "$EVAL_CHUNK_SIZE" \
    --use_wandb \
    $EXTRA_ARGS

EXIT_CODE=$?

echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "============================================="

# Then, after the job is finished, copy whatever you want to save on $SCRATCH
# cp $SLURM_TMPDIR/<to_save> $SCRATCH/t-rex
