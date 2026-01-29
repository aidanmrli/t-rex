#!/bin/bash
#SBATCH --job-name=smc_baseline
#SBATCH --output=logs/smc_%j.out
#SBATCH --error=logs/smc_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
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

# Create logs directory
mkdir -p logs

# Default parameters (can be overridden via environment variables)
export GENERATOR_MODEL_PATH="${GENERATOR_MODEL_PATH:-Qwen/Qwen2.5-7B}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-Qwen/Qwen2.5-Math-PRM-7B}"
export DATASET_PATH="${DATASET_PATH:-trex/data/gsm8k_platinum_test.jsonl}"
export N_PARTICLES="${N_PARTICLES:-16}"
export MAX_STEPS="${MAX_STEPS:-20}"
export ESS_THRESHOLD="${ESS_THRESHOLD:-0.5}"
export TEMPERATURE="${TEMPERATURE:-0.7}"
export OUTPUT_DIR="${OUTPUT_DIR:-trex/results/smc_baseline}"
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
echo ""
echo "Configuration:"
echo "  Generator Model: ${GENERATOR_MODEL_PATH}"
echo "  Reward Model: ${REWARD_MODEL_PATH}"
echo "  Dataset: ${DATASET_PATH}"
echo "  N Particles: ${N_PARTICLES}"
echo "  Max Steps: ${MAX_STEPS}"
echo "  ESS Threshold: ${ESS_THRESHOLD}"
echo "  Temperature: ${TEMPERATURE}"
echo "  Output Dir: ${OUTPUT_DIR}"
echo "  Seed: ${SEED}"
echo "=============================================="

# Check for existing checkpoint
if [[ -f "${OUTPUT_DIR}/checkpoint.json" ]]; then
    echo ""
    echo "Found existing checkpoint - will resume from saved state"
    echo ""
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}/generations"

# Run the baseline
python -m trex.baselines.smc_steering_baseline \
    --generator_model_path "${GENERATOR_MODEL_PATH}" \
    --reward_model_path "${REWARD_MODEL_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --n_particles ${N_PARTICLES} \
    --max_steps ${MAX_STEPS} \
    --ess_threshold ${ESS_THRESHOLD} \
    --temperature ${TEMPERATURE} \
    --seed ${SEED} \
    --generator_tp_size ${GENERATOR_TP_SIZE} \
    --reward_model_tp_size ${REWARD_MODEL_TP_SIZE} \
    --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
    --output_dir "${OUTPUT_DIR}" \
    --enable_checkpointing \
    --checkpoint_interval ${CHECKPOINT_INTERVAL} \
    --checkpoint_time_interval ${CHECKPOINT_TIME_INTERVAL} \
    "$@"

echo ""
echo "=============================================="
echo "SMC Baseline completed successfully!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=============================================="
