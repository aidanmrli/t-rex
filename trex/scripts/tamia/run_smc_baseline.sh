#!/bin/bash
#SBATCH --job-name=smc_baseline
#SBATCH --account=aip-apsarath
#SBATCH --output=/scratch/l/liaidan/t-rex/slurm/smc_%j.out
#SBATCH --error=/scratch/l/liaidan/t-rex/slurm/smc_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=480G
#SBATCH --cpus-per-task=48
#SBATCH --requeue                    # Enable requeue on preemption
#SBATCH --signal=SIGUSR1@120         # Send signal 120s before timeout

# =============================================================================
# SMC Steering Baseline SLURM Script
# =============================================================================
#
# Usage:
#   sbatch scripts/run_smc_baseline.sh [--n_particles 16] [--dataset_path ...]
#
# Supports automatic resumption from checkpoints on preemption.
# =============================================================================

set -e

# =============================================================================
# CRITICAL: Offline mode for compute nodes (no internet access)
# =============================================================================
export SCRATCH_WEIGHTS="/scratch/l/liaidan/model_weights"
export HF_HOME="$SCRATCH_WEIGHTS"
export HF_DATASETS_CACHE="$SCRATCH_WEIGHTS"
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

# Create required directories
mkdir -p /scratch/l/liaidan/t-rex/slurm
mkdir -p /scratch/l/liaidan/t-rex/results

# =============================================================================
# MODEL OPTIONS:
# -----------------------------------------------------------------------------
# Model Name                                                  | Path                             
# -----------------------------------------------------------------------------
# Qwen2.5-7B (Base)                                           | Qwen2.5-7B
# Qwen2.5-Math-PRM-7B                                         | Qwen2.5-Math-PRM-7B              
# gemma-3-4b-pt (pre-trained, no instruction tuning)          | gemma-3-4b-pt                        | false
# gemma-3-12b-pt (pre-trained, no instruction tuning)         | gemma-3-12b-pt                        | false
# =============================================================================
GENERATOR_MODEL_ID="Qwen2.5-7B"
REWARD_MODEL_ID="Qwen2.5-Math-PRM-7B"

# Resolve local model path from HuggingFace cache
# This avoids network calls when using TRANSFORMERS_OFFLINE=1
MODEL_CACHE_DIR="$SCRATCH_WEIGHTS/$MODEL_ID"

# =============================================================================
# DATASET OPTIONS:
# -----------------------------------------------------------------------------
# Dataset   | Train Path                      | Eval Path
# -----------------------------------------------------------------------------
# gsm8k     | trex/data/gsm8k_train.jsonl     | trex/data/gsm8k_platinum_test.jsonl
# math      | trex/data/math_train.jsonl      | trex/data/math500_test.jsonl
# =============================================================================
DATASET="gsm8k"
DATASET_PATH="trex/data/gsm8k_platinum_test.jsonl"
EVAL_DATA="trex/data/gsm8k_platinum_test.jsonl"

# Default parameters (can be overridden via environment variables)
export GENERATOR_MODEL_PATH="${SCRATCH_WEIGHTS}/${GENERATOR_MODEL_ID}"
export REWARD_MODEL_PATH="${SCRATCH_WEIGHTS}/${REWARD_MODEL_ID}"
export N_PARTICLES="${N_PARTICLES:-16}"
export MAX_SMC_ITERATIONS="${MAX_SMC_ITERATIONS:-20}"
export MAX_REASONING_STEPS="${MAX_REASONING_STEPS:-15}"
export ESS_THRESHOLD="${ESS_THRESHOLD:-0.5}"
export TEMPERATURE="${TEMPERATURE:-0.7}"
export OUTPUT_DIR="${OUTPUT_DIR:-/scratch/l/liaidan/t-rex/results/smc_baseline}"
export SEED="${SEED:-42}"

# vLLM settings
export GENERATOR_TP_SIZE="${GENERATOR_TP_SIZE:-1}"
export REWARD_MODEL_TP_SIZE="${REWARD_MODEL_TP_SIZE:-1}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

# Checkpointing settings
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5}"
export CHECKPOINT_TIME_INTERVAL="${CHECKPOINT_TIME_INTERVAL:-600}"

# Print configuration
echo "=============================================="
echo "SMC Steering Baseline"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${SLURM_GPUS_ON_NODE:-${CUDA_VISIBLE_DEVICES}}"
echo "HF_HOME: ${HF_HOME}"
echo "HF_HUB_OFFLINE: ${HF_HUB_OFFLINE}"
echo ""
echo "Configuration:"
echo "  Generator Model: ${GENERATOR_MODEL_PATH}"
echo "  Reward Model: ${REWARD_MODEL_PATH}"
echo "  Dataset: ${DATASET_PATH}"
echo "  N Particles: ${N_PARTICLES}"
echo "  Max SMC Iterations: ${MAX_SMC_ITERATIONS}"
echo "  Max Reasoning Steps: ${MAX_REASONING_STEPS}"
echo "  ESS Threshold: ${ESS_THRESHOLD}"
echo "  Temperature: ${TEMPERATURE}"
echo "  Output Dir: ${OUTPUT_DIR}"
echo "  Seed: ${SEED}"
echo "=============================================="

# Check for existing checkpoint (DISABLED - forcing fresh start)
# if [[ -f "${OUTPUT_DIR}/checkpoint.json" ]]; then
#     echo ""
#     echo "Found existing checkpoint - will resume from saved state"
#     echo ""
# fi

# Create output directory
mkdir -p "${OUTPUT_DIR}/generations"

# Remove existing checkpoint to force fresh start
if [[ -f "${OUTPUT_DIR}/checkpoint.json" ]]; then
    echo "Removing existing checkpoint to start fresh..."
    rm -f "${OUTPUT_DIR}/checkpoint.json"
fi

# Run the baseline
python -m trex.baselines.smc_steering_baseline \
    --generator_model_path "${GENERATOR_MODEL_PATH}" \
    --reward_model_path "${REWARD_MODEL_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --n_particles ${N_PARTICLES} \
    --max_smc_iterations ${MAX_SMC_ITERATIONS} \
    --max_reasoning_steps ${MAX_REASONING_STEPS} \
    --ess_threshold ${ESS_THRESHOLD} \
    --temperature ${TEMPERATURE} \
    --seed ${SEED} \
    --generator_tp_size ${GENERATOR_TP_SIZE} \
    --reward_model_tp_size ${REWARD_MODEL_TP_SIZE} \
    --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
    --output_dir "${OUTPUT_DIR}" \
    # --enable_checkpointing \  # DISABLED - forcing fresh start
    # --checkpoint_interval ${CHECKPOINT_INTERVAL} \
    # --checkpoint_time_interval ${CHECKPOINT_TIME_INTERVAL} \
    "$@"

echo ""
echo "=============================================="
echo "SMC Baseline completed successfully!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=============================================="
