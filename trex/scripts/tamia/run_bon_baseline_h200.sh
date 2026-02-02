#!/bin/bash
#SBATCH --job-name=bon-baseline
#SBATCH --account=aip-apsarath
#SBATCH --output=/scratch/l/liaidan/t-rex/slurm/bon_%j.out
#SBATCH --error=/scratch/l/liaidan/t-rex/slurm/bon_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h200:8
#SBATCH --mem=950G
#SBATCH --cpus-per-task=64
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@120

# =============================================================================
# Best-of-N Baseline - H200 Configuration (8 GPUs)
# =============================================================================

set -e

module load python/3.12.4 scipy-stack arrow/21.0.0 gcc opencv/4.13.0 rust cuda/12.6
source venv/bin/activate

# Offline mode
export SCRATCH_WEIGHTS="/scratch/l/liaidan/model_weights"
export HF_HOME="$SCRATCH_WEIGHTS"
export HF_DATASETS_CACHE="$SCRATCH_WEIGHTS"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline

mkdir -p /scratch/l/liaidan/t-rex/slurm
mkdir -p /scratch/l/liaidan/t-rex/results

# H200: 8 GPUs - full tensor parallelism
TP_SIZE=8
GPU_MEM_UTIL=0.95
MAX_NUM_SEQS=1024  # More sequences with more memory

# Model
MODEL_ID="Qwen2.5-7B"
MODEL="${SCRATCH_WEIGHTS}/${MODEL_ID}"

# Dataset
DATASET="math"
DATASET_PATH="trex/data/math500_test.jsonl"

# Sampling
N_SAMPLES=32
SWEEP_SIZE=100

# Output
OUTPUT_DIR="/scratch/l/liaidan/t-rex/results/bon_baseline/${MODEL_ID}/${DATASET}_n${N_SAMPLES}"

echo "=============================================="
echo "Best-of-N Baseline (H200 - 8 GPUs)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "TP Size: $TP_SIZE"
echo "GPU Mem Util: $GPU_MEM_UTIL"
echo "=============================================="

if [[ -f "$OUTPUT_DIR/checkpoint.json" ]]; then
    echo "Found checkpoint - will resume"
fi

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
    --eval_chunk_size 50 \
    --use_wandb \
    --no_chat_template

echo "=============================================="
echo "Best-of-N completed!"
echo "Results: $OUTPUT_DIR"
echo "=============================================="
nvidia-smi
