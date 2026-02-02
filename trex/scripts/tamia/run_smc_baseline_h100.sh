#!/bin/bash
#SBATCH --job-name=smc_baseline
#SBATCH --account=aip-apsarath
#SBATCH --output=/scratch/l/liaidan/t-rex/slurm/smc_%j.out
#SBATCH --error=/scratch/l/liaidan/t-rex/slurm/smc_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=480G
#SBATCH --cpus-per-task=48
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@120

# =============================================================================
# SMC Steering Baseline - H100 Configuration (4 GPUs)
# =============================================================================

set -e

# Load modules and environment
module load python/3.12.4 scipy-stack arrow/21.0.0 gcc opencv/4.13.0 rust cuda/12.6
source venv/bin/activate

# Offline mode for compute nodes
export SCRATCH_WEIGHTS="/scratch/l/liaidan/model_weights"
export HF_HOME="$SCRATCH_WEIGHTS"
export HF_DATASETS_CACHE="$SCRATCH_WEIGHTS"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

mkdir -p /scratch/l/liaidan/t-rex/slurm
mkdir -p /scratch/l/liaidan/t-rex/results

# H100: 4 GPUs - TP=2 per model
GENERATOR_TP_SIZE=2
REWARD_MODEL_TP_SIZE=2
GPU_MEMORY_UTILIZATION=0.90
N_PARTICLES=16

# Models
GENERATOR_MODEL_PATH="${SCRATCH_WEIGHTS}/Qwen2.5-7B"
REWARD_MODEL_PATH="${SCRATCH_WEIGHTS}/Qwen2.5-Math-PRM-7B"

# Dataset
DATASET_PATH="trex/data/gsm8k_platinum_test.jsonl"

# SMC parameters
MAX_SMC_ITERATIONS=20
MAX_REASONING_STEPS=15
ESS_THRESHOLD=0.5
TEMPERATURE=0.7
SEED=42

# Output
OUTPUT_DIR="/scratch/l/liaidan/t-rex/results/smc_baseline"

echo "=============================================="
echo "SMC Steering Baseline (H100 - 4 GPUs)"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Generator TP: ${GENERATOR_TP_SIZE}, Reward TP: ${REWARD_MODEL_TP_SIZE}"
echo "N Particles: ${N_PARTICLES}"
echo "GPU Memory Util: ${GPU_MEMORY_UTILIZATION}"
echo "=============================================="

if [[ -f "${OUTPUT_DIR}/checkpoint.json" ]]; then
    echo "Found existing checkpoint - will resume"
fi

mkdir -p "${OUTPUT_DIR}/generations"

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
    --enable_checkpointing \
    --checkpoint_interval 5 \
    --checkpoint_time_interval 600 \
    "$@"

echo "=============================================="
echo "SMC Baseline completed!"
echo "Results: ${OUTPUT_DIR}"
echo "=============================================="
nvidia-smi
