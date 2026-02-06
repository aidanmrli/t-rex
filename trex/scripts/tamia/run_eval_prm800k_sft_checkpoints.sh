#!/bin/bash
#SBATCH --job-name=eval_prm800k_sft
#SBATCH --account=aip-apsarath
#SBATCH --output=/scratch/l/liaidan/t-rex/slurm/eval_prm800k_sft_%j.out
#SBATCH --error=/scratch/l/liaidan/t-rex/slurm/eval_prm800k_sft_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=480G
#SBATCH --cpus-per-task=48
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@120

# =============================================================================
# PRM800K SFT Checkpoint Evaluation
# Evaluates multiple Stage-1 SFT checkpoints on GSM8K + MATH-500 with
# the same Best-of-N harness for checkpoint selection.
#
# Usage:
#   sbatch trex/scripts/tamia/run_eval_prm800k_sft_checkpoints.sh
#   bash trex/scripts/tamia/run_eval_prm800k_sft_checkpoints.sh
#
# Optional convenience:
#   bash trex/scripts/tamia/run_eval_prm800k_sft_checkpoints.sh --submit
# =============================================================================

set -euo pipefail

if [[ "${1:-}" == "--submit" ]]; then
    sbatch "$0" "${@:2}"
    exit $?
fi

# Load modules when available (cluster/login shells).
if command -v module >/dev/null 2>&1; then
    module load python/3.12.4 scipy-stack arrow/21.0.0 gcc opencv/4.13.0 rust cuda/12.6
fi

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

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/l/liaidan/t-rex}"
SCRATCH_WEIGHTS="${SCRATCH_WEIGHTS:-/scratch/l/liaidan/model_weights}"
PRM800K_SFT_ROOT="${PRM800K_SFT_ROOT:-${SCRATCH_ROOT}/results/prm800k_sft}"

export HF_HOME="${SCRATCH_WEIGHTS}"
export HF_DATASETS_CACHE="${SCRATCH_WEIGHTS}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline

mkdir -p "${SCRATCH_ROOT}/results"
mkdir -p "${SCRATCH_ROOT}/slurm"

# Checkpoints and datasets.
CHECKPOINT_IDS="${CHECKPOINT_IDS:-154122 154126}"
GSM8K_DATASET_PATH="${GSM8K_DATASET_PATH:-${REPO_ROOT}/trex/data/gsm8k_platinum_test.jsonl}"
MATH_DATASET_PATH="${MATH_DATASET_PATH:-${REPO_ROOT}/trex/data/math500_test.jsonl}"

# Best-of-N settings.
N_SAMPLES_GSM8K="${N_SAMPLES_GSM8K:-16}"
N_SAMPLES_MATH="${N_SAMPLES_MATH:-32}"
TEMPERATURES="${TEMPERATURES:-0.6 0.8 1.0 1.2}"
SWEEP_SIZE="${SWEEP_SIZE:-100}"
EVAL_CHUNK_SIZE="${EVAL_CHUNK_SIZE:-50}"
TP_SIZE="${TP_SIZE:-4}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
USE_WANDB="${USE_WANDB:-0}"

RUN_TAG="${RUN_TAG:-${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH_ROOT}/results/eval_prm800k_sft/${RUN_TAG}}"

EXTRA_ARGS=("--no_chat_template")
if [[ "${USE_WANDB}" == "1" ]]; then
    EXTRA_ARGS+=("--use_wandb")
fi

run_eval() {
    local checkpoint_id="$1"
    local dataset="$2"
    local dataset_path="$3"
    local n_samples="$4"

    local model_path="${PRM800K_SFT_ROOT}/job_${checkpoint_id}/ckpt"
    if [[ ! -d "${model_path}" ]]; then
        echo "ERROR: Missing model checkpoint directory: ${model_path}"
        return 1
    fi
    if [[ ! -f "${dataset_path}" ]]; then
        echo "ERROR: Missing dataset file: ${dataset_path}"
        return 1
    fi

    local output_dir="${OUTPUT_ROOT}/job_${checkpoint_id}/${dataset}_n${n_samples}"
    mkdir -p "${output_dir}"

    echo "----------------------------------------------"
    echo "Checkpoint: job_${checkpoint_id}"
    echo "Dataset: ${dataset}"
    echo "Model: ${model_path}"
    echo "Output: ${output_dir}"
    echo "----------------------------------------------"

    python -m trex.baselines.best_of_n_baseline \
        --model_path "${model_path}" \
        --dataset "${dataset}" \
        --dataset_path "${dataset_path}" \
        --n_samples "${n_samples}" \
        --temperatures ${TEMPERATURES} \
        --tp_size "${TP_SIZE}" \
        --max_num_seqs "${MAX_NUM_SEQS}" \
        --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
        --output_dir "${output_dir}" \
        --sweep_size "${SWEEP_SIZE}" \
        --eval_chunk_size "${EVAL_CHUNK_SIZE}" \
        "${EXTRA_ARGS[@]}"
}

echo "=============================================="
echo "PRM800K SFT checkpoint evaluation"
echo "=============================================="
echo "Run tag: ${RUN_TAG}"
echo "Checkpoint IDs: ${CHECKPOINT_IDS}"
echo "Temperatures: ${TEMPERATURES}"
echo "GSM8K N: ${N_SAMPLES_GSM8K}, MATH N: ${N_SAMPLES_MATH}"
echo "Output root: ${OUTPUT_ROOT}"
echo "=============================================="

for ckpt in ${CHECKPOINT_IDS}; do
    run_eval "${ckpt}" "gsm8k" "${GSM8K_DATASET_PATH}" "${N_SAMPLES_GSM8K}"
    run_eval "${ckpt}" "math" "${MATH_DATASET_PATH}" "${N_SAMPLES_MATH}"
done

echo "=============================================="
echo "Evaluation complete. Summary files:"
find "${OUTPUT_ROOT}" -name "summary.json" | sort
echo "=============================================="

if command -v jq >/dev/null 2>&1; then
    for ckpt in ${CHECKPOINT_IDS}; do
        for pair in "gsm8k:${N_SAMPLES_GSM8K}" "math:${N_SAMPLES_MATH}"; do
            dataset="${pair%%:*}"
            n_samples="${pair##*:}"
            summary_path="${OUTPUT_ROOT}/job_${ckpt}/${dataset}_n${n_samples}/summary.json"
            if [[ -f "${summary_path}" ]]; then
                echo "=== job_${ckpt} / ${dataset} ==="
                jq '{best_temp, final_metrics}' "${summary_path}"
            fi
        done
    done
fi
