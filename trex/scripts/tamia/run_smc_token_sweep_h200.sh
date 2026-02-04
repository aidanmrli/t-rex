#!/bin/bash
#SBATCH --job-name=smc_token_sweep
#SBATCH --account=aip-apsarath
#SBATCH --output=/scratch/l/liaidan/t-rex/slurm/smc_token_sweep_%j.out
#SBATCH --error=/scratch/l/liaidan/t-rex/slurm/smc_token_sweep_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h200:8
#SBATCH --mem=950G
#SBATCH --cpus-per-task=64
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@120

# =============================================================================
# SMC Token-Resampling Sweep - H200 Configuration (8 GPUs)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Load modules and environment
module load python/3.12.4 scipy-stack arrow/21.0.0 gcc opencv/4.13.0 rust cuda/12.6
source "${REPO_ROOT}/venv/bin/activate"
cd "${REPO_ROOT}"

# Offline mode for compute nodes
export SCRATCH_WEIGHTS="/scratch/l/liaidan/model_weights"
export HF_HOME="$SCRATCH_WEIGHTS"
export HF_DATASETS_CACHE="$SCRATCH_WEIGHTS"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

mkdir -p /scratch/l/liaidan/t-rex/slurm
mkdir -p /scratch/l/liaidan/t-rex/results

# H200: 8 GPUs - TP=4 per model
GENERATOR_TP_SIZE=4
REWARD_MODEL_TP_SIZE=4
GPU_MEMORY_UTILIZATION=0.92
N_PARTICLES=32

# Models
GENERATOR_MODEL_PATH="${SCRATCH_WEIGHTS}/Qwen2.5-7B"
REWARD_MODEL_PATH="${SCRATCH_WEIGHTS}/Qwen2.5-Math-PRM-7B"

# Dataset
DATASET_PATH="${REPO_ROOT}/trex/data/gsm8k_platinum_test.jsonl"

# Sweep parameters
K_VALUES_DEFAULT="16 32 64 128 256"
K_VALUES="${K_VALUES:-$K_VALUES_DEFAULT}"
TOTAL_TOKEN_BUDGET="${TOTAL_TOKEN_BUDGET:-2048}"
RESAMPLING_STRATEGY="${RESAMPLING_STRATEGY:-every_step}"

# Shared SMC parameters
ESS_THRESHOLD=0.5
TEMPERATURE=0.7
SEED=42
USE_TOKEN_PROMPTS=1

# Output
OUTPUT_DIR_BASE="/scratch/l/liaidan/t-rex/results/smc_token_sweep/job_${SLURM_JOB_ID}"

echo "=============================================="
echo "SMC Token-Resampling Sweep (H200 - 8 GPUs)"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Generator TP: ${GENERATOR_TP_SIZE}, Reward TP: ${REWARD_MODEL_TP_SIZE}"
echo "N Particles: ${N_PARTICLES}"
echo "GPU Memory Util: ${GPU_MEMORY_UTILIZATION}"
echo "K Values: ${K_VALUES}"
echo "Total Token Budget: ${TOTAL_TOKEN_BUDGET}"
echo "Resampling Strategy: ${RESAMPLING_STRATEGY}"
echo "=============================================="

mkdir -p "${OUTPUT_DIR_BASE}"

for K in ${K_VALUES}; do
    MAX_SMC_ITERATIONS=$(( (TOTAL_TOKEN_BUDGET + K - 1) / K ))
    if [[ -n "${MAX_SMC_ITERATIONS_OVERRIDE}" ]]; then
        MAX_SMC_ITERATIONS="${MAX_SMC_ITERATIONS_OVERRIDE}"
    fi

    OUTPUT_DIR="${OUTPUT_DIR_BASE}/k${K}"
    mkdir -p "${OUTPUT_DIR}/generations"

    echo "----------------------------------------------"
    echo "Running K=${K} (max_smc_iterations=${MAX_SMC_ITERATIONS})"
    echo "Output: ${OUTPUT_DIR}"
    echo "----------------------------------------------"

    python -m trex.baselines.smc_steering_baseline \
        --generator_model_path "${GENERATOR_MODEL_PATH}" \
        --reward_model_path "${REWARD_MODEL_PATH}" \
        --dataset_path "${DATASET_PATH}" \
        --n_particles ${N_PARTICLES} \
        --max_smc_iterations ${MAX_SMC_ITERATIONS} \
        --ess_threshold ${ESS_THRESHOLD} \
        --temperature ${TEMPERATURE} \
        --seed ${SEED} \
        --generator_tp_size ${GENERATOR_TP_SIZE} \
        --reward_model_tp_size ${REWARD_MODEL_TP_SIZE} \
        --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
        --output_dir "${OUTPUT_DIR}" \
        --resampling_unit token \
        --resample_every_tokens ${K} \
        --resampling_strategy "${RESAMPLING_STRATEGY}" \
        --use_token_prompts \
        --enable_checkpointing \
        --checkpoint_interval 5 \
        --checkpoint_time_interval 600 \
        --log_level INFO \
        --log_file "${OUTPUT_DIR}/run.log" \
        "$@"
done

echo "=============================================="
echo "SMC Token Sweep completed!"
echo "Results: ${OUTPUT_DIR_BASE}"
echo "=============================================="
nvidia-smi
