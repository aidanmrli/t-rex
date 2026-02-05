#!/bin/bash
#SBATCH --job-name=prm800k_sft
#SBATCH --account=aip-apsarath
#SBATCH --output=/scratch/l/liaidan/t-rex/slurm/prm800k_sft_%j.out
#SBATCH --error=/scratch/l/liaidan/t-rex/slurm/prm800k_sft_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h200:8
#SBATCH --mem=950G
#SBATCH --cpus-per-task=64
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@120

# =============================================================================
# PRM800K SFT - H200 Configuration (8 GPUs)
# =============================================================================

set -e

# Load modules and environment
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

# Offline mode for compute nodes
export SCRATCH_WEIGHTS="/scratch/l/liaidan/model_weights"
export HF_HOME="$SCRATCH_WEIGHTS"
export HF_DATASETS_CACHE="$SCRATCH_WEIGHTS"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

mkdir -p /scratch/l/liaidan/t-rex/slurm
mkdir -p /scratch/l/liaidan/t-rex/results

NUM_GPUS="${SLURM_GPUS_ON_NODE:-8}"

# Paths
BASE_MODEL_PATH="${SCRATCH_WEIGHTS}/Qwen2.5-7B"
DATASET_PATH="${REPO_ROOT}/trex/data/prm800k_sft_train.jsonl"
OUTPUT_DIR="/scratch/l/liaidan/t-rex/results/prm800k_sft/job_${SLURM_JOB_ID}"

# Training hyperparameters (override via env)
MAX_EPOCHS="${MAX_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
MAX_LEN="${MAX_LEN:-2048}"
MICRO_BSZ="${MICRO_BSZ:-1}"
TRAIN_BSZ="${TRAIN_BSZ:-128}"
ZERO_STAGE="${ZERO_STAGE:-2}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-1}"

echo "=============================================="
echo "PRM800K SFT (H200 - 8 GPUs)"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Base Model: ${BASE_MODEL_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Epochs: ${MAX_EPOCHS}, LR: ${LEARNING_RATE}"
echo "Micro BSZ: ${MICRO_BSZ}, Train BSZ: ${TRAIN_BSZ}"
echo "Max Len: ${MAX_LEN}, ZeRO Stage: ${ZERO_STAGE}"
echo "=============================================="

mkdir -p "${OUTPUT_DIR}"

EXTRA_ARGS=()
if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
    EXTRA_ARGS+=(--gradient_checkpointing)
fi
if [[ "${APPLY_CHAT_TEMPLATE}" == "1" ]]; then
    EXTRA_ARGS+=(--apply_chat_template)
fi

deepspeed --num_gpus "${NUM_GPUS}" openrlhf/cli/train_sft.py \
    --pretrain "${BASE_MODEL_PATH}" \
    --dataset "${DATASET_PATH}" \
    --input_key input \
    --output_key output \
    --max_len "${MAX_LEN}" \
    --max_epochs "${MAX_EPOCHS}" \
    --learning_rate "${LEARNING_RATE}" \
    --micro_train_batch_size "${MICRO_BSZ}" \
    --train_batch_size "${TRAIN_BSZ}" \
    --zero_stage "${ZERO_STAGE}" \
    --bf16 \
    --save_path "${OUTPUT_DIR}/ckpt" \
    --ckpt_path "${OUTPUT_DIR}/checkpoints_sft" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "=============================================="
echo "PRM800K SFT completed!"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================="
nvidia-smi
