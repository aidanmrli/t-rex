#!/bin/bash
#SBATCH --job-name=bon-baseline
#SBATCH --account=aip-apsarath
#SBATCH --output=/scratch/l/liaidan/t-rex/slurm/bon_%j.out
#SBATCH --error=/scratch/l/liaidan/t-rex/slurm/bon_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=480G
#SBATCH --cpus-per-task=48
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@120

# =============================================================================
# Best-of-N Baseline - H100 Configuration (4 GPUs)
# =============================================================================

set -e

module load python/3.12.4 scipy-stack arrow/21.0.0 gcc opencv/4.13.0 rust cuda/12.6
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

# H100: 4 GPUs
TP_SIZE=4
GPU_MEM_UTIL=0.95
MAX_NUM_SEQS=512

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
echo "Best-of-N Baseline (H100 - 4 GPUs)"
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
