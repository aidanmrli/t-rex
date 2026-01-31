#!/bin/bash

#SBATCH --job-name=train-ppo-baseline
#SBATCH --account=aip-apsarath
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH -o /scratch/l/liaidan/t-rex/slurm/ppo-%j.out
#SBATCH -e /scratch/l/liaidan/t-rex/slurm/ppo-%j.err
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@120

# =============================================================================
# T-REX PPO Baseline Training Script
# Uses OpenRLHF's PPO with GAE (critic-based advantage estimation)
#
# Key differences from GRPO:
# - Uses GAE instead of group_norm for advantage estimation
# - Requires a critic network (adds memory overhead)
# - Single sample per prompt (no group sampling)
# - Higher KL penalty to prevent policy collapse
#
# AUTOMATIC REQUEUE: This script automatically resubmits itself on SLURM
# timeout, allowing training to span multiple job submissions. Training
# resumes from the latest checkpoint.
# =============================================================================

# -----------------------------------------------------------------------------
# AUTOMATIC REQUEUE CONFIGURATION
# -----------------------------------------------------------------------------
# Maximum number of consecutive job submissions (safety limit)
MAX_SUBMISSIONS=${MAX_SUBMISSIONS:-20}

# Track submission count across requeues (inherited from parent job)
export TREX_SUBMISSION_COUNT=$((${TREX_SUBMISSION_COUNT:-0} + 1))

# Absolute path to this script for self-resubmission
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

# Check if we've exceeded max submissions (prevents infinite loops on real errors)
if [ "$TREX_SUBMISSION_COUNT" -gt "$MAX_SUBMISSIONS" ]; then
    echo "============================================="
    echo "ERROR: Exceeded maximum submissions ($MAX_SUBMISSIONS)"
    echo "This may indicate a persistent error. Check logs and restart manually."
    echo "To restart: export TREX_SUBMISSION_COUNT=0 && sbatch $SCRIPT_PATH"
    echo "============================================="
    exit 1
fi

echo "============================================="
echo "Job Submission: $TREX_SUBMISSION_COUNT / $MAX_SUBMISSIONS"
echo "============================================="

# -----------------------------------------------------------------------------
# SIGNAL HANDLING FOR GRACEFUL REQUEUE
# -----------------------------------------------------------------------------
REQUEUE_SUBMITTED=false

submit_continuation_job() {
    if [ "$REQUEUE_SUBMITTED" = true ]; then
        echo "Continuation job already submitted"
        return
    fi

    # Check completion marker (set after this section is defined)
    if [ -n "$COMPLETION_MARKER" ] && [ -f "$COMPLETION_MARKER" ]; then
        echo "Training complete - no requeue needed"
        return
    fi

    if [ "$TREX_SUBMISSION_COUNT" -ge "$MAX_SUBMISSIONS" ]; then
        echo "Max submissions reached - no requeue"
        return
    fi

    echo "Submitting continuation job..."
    sbatch --export=ALL,TREX_SUBMISSION_COUNT=$TREX_SUBMISSION_COUNT "$SCRIPT_PATH"
    REQUEUE_SUBMITTED=true
    echo "Continuation job submitted. Current job will continue until killed."
}

# Trap SIGUSR1 (sent 120s before timeout) - submit new job but keep running
# to maximize checkpoint opportunities
handle_sigusr1() {
    echo ""
    echo "============================================="
    echo "SIGUSR1 received - $(date)"
    echo "Job will be killed in ~120 seconds"
    echo "Submitting continuation job now..."
    echo "============================================="
    submit_continuation_job
}
trap handle_sigusr1 SIGUSR1

# Also trap EXIT for cases where script is killed before reaching completion handling
handle_exit() {
    local trap_exit_code=$?
    # Use PYTHON_EXIT_CODE if set (script got past the python command), otherwise use trap's $?
    # This handles cases where the script itself is killed before reaching completion handling
    local actual_exit_code=${PYTHON_EXIT_CODE:-$trap_exit_code}

    if [ "$actual_exit_code" -ne 0 ] && [ "$REQUEUE_SUBMITTED" = false ]; then
        # Check if checkpoints exist before requeuing
        if [ -n "$CKPT_DIR" ] && [ -d "$CKPT_DIR" ] && [ "$(ls -A "$CKPT_DIR" 2>/dev/null)" ]; then
            echo "EXIT trap: Non-zero exit ($actual_exit_code) with checkpoints - submitting continuation job"
            submit_continuation_job
        fi
    fi
}
trap handle_exit EXIT

# -----------------------------------------------------------------------------
# ENVIRONMENT SETUP
# -----------------------------------------------------------------------------

# 1. Load the required modules
module load python/3.12.4 scipy-stack arrow/21.0.0 gcc opencv/4.13.0 rust cuda/12.6

# 2. Load your environment
source venv/bin/activate

# Scratch and weights setup
SCRATCH_DIR="/scratch/l/liaidan/t-rex"
SCRATCH_WEIGHTS="/scratch/l/liaidan/model_weights"

# 3. Set environment variables
export HF_HOME="$SCRATCH_WEIGHTS"
export HF_DATASETS_CACHE="$SCRATCH_WEIGHTS"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export WANDB_MODE=offline
mkdir -p "$SCRATCH_WEIGHTS"

# 4. Set up results directory
mkdir -p "$SCRATCH_DIR/results"
mkdir -p "$SCRATCH_DIR/slurm"

if [ -d "trex/results" ] && [ ! -L "trex/results" ]; then
    echo "Moving existing results to scratch..."
    cp -r trex/results/* "$SCRATCH_DIR/results/"
    rm -rf trex/results
    ln -s "$SCRATCH_DIR/results" trex/results
elif [ ! -e "trex/results" ]; then
    ln -s "$SCRATCH_DIR/results" trex/results
fi

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

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
MODEL_ID="Qwen2.5-7B"
MODEL_NAME=$(basename "$MODEL_ID")

# Resolve local model path from HuggingFace cache
# This avoids network calls when using TRANSFORMERS_OFFLINE=1
MODEL_CACHE_DIR="$SCRATCH_WEIGHTS/Qwen2.5-7B"
if [ -d "$MODEL_CACHE_DIR/snapshots" ]; then
    SNAPSHOT=$(ls "$MODEL_CACHE_DIR/snapshots" | head -1)
    MODEL="$MODEL_CACHE_DIR/snapshots/$SNAPSHOT"
    echo "Using cached model: $MODEL"
else
    MODEL="$MODEL_ID"
    echo "Warning: Model not cached at $MODEL_CACHE_DIR, using HuggingFace ID: $MODEL"
fi

# =============================================================================
# DATASET OPTIONS:
# -----------------------------------------------------------------------------
# Dataset   | Train Path                      | Eval Path
# -----------------------------------------------------------------------------
# gsm8k     | trex/data/gsm8k_train.jsonl     | trex/data/gsm8k_platinum_test.jsonl
# math      | trex/data/math_train.jsonl      | trex/data/math500_test.jsonl
# =============================================================================
DATASET="gsm8k"
TRAIN_DATA="trex/data/gsm8k_train.jsonl"
EVAL_DATA="trex/data/gsm8k_platinum_test.jsonl"

# PPO Hyperparameters (differ from GRPO)
N_SAMPLES=1                     # PPO doesn't need group sampling
KL_COEF=0.01                    # Higher KL penalty prevents policy collapse
KL_ESTIMATOR="k1"               # k1 sufficient for non-KL-loss mode
MAX_EPOCHS=1                    # Number of training epochs
LEARNING_RATE=5e-7              # Actor learning rate
CRITIC_LEARNING_RATE=9e-6       # Critic learning rate (PPO-specific)
GAE_LAMBDA=0.95                 # GAE lambda parameter
VALUE_CLIP=0.5                  # PPO value function clipping

# Batch sizes (slightly reduced for critic memory overhead)
TRAIN_BATCH_SIZE=128
MICRO_TRAIN_BATCH_SIZE=4        # Conservative for memory sharing
ROLLOUT_BATCH_SIZE=128
MICRO_ROLLOUT_BATCH_SIZE=16     # Conservative for memory sharing

# Generation settings
TEMPERATURE=1.0
PROMPT_MAX_LEN=1024
GENERATE_MAX_LEN=2048

# vLLM settings (4x H100) - reduced GPU util for critic memory overhead
VLLM_NUM_ENGINES=2
VLLM_TP_SIZE=2
VLLM_GPU_UTIL=0.5               # Reduced from 0.6 to account for critic network

# Checkpointing (~30 min intervals)
SAVE_STEPS=50

# Chat Template: Set to false for base models, true for instruct models
APPLY_CHAT_TEMPLATE=false

# WandB
USE_WANDB=true
WANDB_API_KEY="8ac57bf9aa5138a9e30d747070d1ebc22b581efc"
WANDB_PROJECT="t-rex"

# Output
OUTPUT_DIR="/scratch/l/liaidan/t-rex/results/ppo_baseline/${MODEL_NAME}/${DATASET}_n${N_SAMPLES}"
CKPT_DIR="${OUTPUT_DIR}/checkpoints"
EFFICIENCY_PATH="${OUTPUT_DIR}/efficiency_metrics.json"
COMPLETION_MARKER="${OUTPUT_DIR}/.training_complete"

# =============================================================================
# TRAINING COMPLETION CHECK
# =============================================================================
# Skip if training already completed (prevents wasted job submissions)
if [ -f "$COMPLETION_MARKER" ]; then
    echo "============================================="
    echo "TRAINING ALREADY COMPLETE"
    echo "Completion marker: $COMPLETION_MARKER"
    echo "To restart training, remove the marker and checkpoints:"
    echo "  rm -f $COMPLETION_MARKER"
    echo "  rm -rf $CKPT_DIR/*"
    echo "============================================="
    exit 0
fi

# =============================================================================
# AUTOMATIC CHECKPOINT RESUMPTION
# =============================================================================
# Automatically detect if a checkpoint exists and resume from it.
# This enables training across multiple job submissions on time-limited clusters.

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CKPT_DIR"

# Check if checkpoint exists (look for any checkpoint directory)
LOAD_CHECKPOINT=false
if [ -d "$CKPT_DIR" ] && [ "$(ls -A $CKPT_DIR 2>/dev/null)" ]; then
    echo "============================================="
    echo "CHECKPOINT DETECTED - RESUMING TRAINING"
    echo "Checkpoint dir: $CKPT_DIR"
    echo "Submission: $TREX_SUBMISSION_COUNT / $MAX_SUBMISSIONS"
    echo "============================================="
    LOAD_CHECKPOINT=true
else
    echo "============================================="
    echo "NO CHECKPOINT FOUND - STARTING FRESH"
    echo "============================================="
fi

# =============================================================================

# Build extra arguments
EXTRA_ARGS=""
if [ "$APPLY_CHAT_TEMPLATE" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --apply_chat_template"
fi
if [ "$LOAD_CHECKPOINT" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --load_checkpoint"
fi
if [ "$USE_WANDB" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --use_wandb $WANDB_API_KEY --wandb_project $WANDB_PROJECT"
fi

# Set efficiency tracking environment variables
export TREX_METHOD="ppo"
export TREX_MODEL="$MODEL_NAME"
export TREX_DATASET="$DATASET"
export TREX_EFFICIENCY_PATH="$EFFICIENCY_PATH"

# Log job info
echo "============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Advantage Estimator: GAE (PPO)"
echo "KL Coef: $KL_COEF"
echo "Critic LR: $CRITIC_LEARNING_RATE"
echo "GAE Lambda: $GAE_LAMBDA"
echo "Output Dir: $OUTPUT_DIR"
echo "Resume from checkpoint: $LOAD_CHECKPOINT"
echo "============================================="

# Run PPO training using OpenRLHF
python -m openrlhf.cli.train_ppo_ray \
    --pretrain "$MODEL" \
    --remote_rm_url trex/baselines/ppo_reward_func.py \
    --prompt_data "$TRAIN_DATA" \
    --input_key prompt \
    --label_key label \
    --advantage_estimator gae \
    --kl_estimator "$KL_ESTIMATOR" \
    --init_kl_coef "$KL_COEF" \
    --n_samples_per_prompt "$N_SAMPLES" \
    --critic_learning_rate "$CRITIC_LEARNING_RATE" \
    --lambd "$GAE_LAMBDA" \
    --value_clip "$VALUE_CLIP" \
    --normalize_reward \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --micro_train_batch_size "$MICRO_TRAIN_BATCH_SIZE" \
    --rollout_batch_size "$ROLLOUT_BATCH_SIZE" \
    --micro_rollout_batch_size "$MICRO_ROLLOUT_BATCH_SIZE" \
    --max_epochs "$MAX_EPOCHS" \
    --actor_learning_rate "$LEARNING_RATE" \
    --temperature "$TEMPERATURE" \
    --prompt_max_len "$PROMPT_MAX_LEN" \
    --generate_max_len "$GENERATE_MAX_LEN" \
    --vllm_num_engines "$VLLM_NUM_ENGINES" \
    --vllm_tensor_parallel_size "$VLLM_TP_SIZE" \
    --vllm_gpu_memory_utilization "$VLLM_GPU_UTIL" \
    --colocate_all_models \
    --vllm_enable_sleep \
    --deepspeed_enable_sleep \
    --zero_stage 3 \
    --bf16 \
    --packing_samples \
    --gradient_checkpointing \
    --save_steps "$SAVE_STEPS" \
    --save_path "$OUTPUT_DIR" \
    --ckpt_path "$CKPT_DIR" \
    --save_hf_ckpt \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 4 \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node 4 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 4 \
    $EXTRA_ARGS

# Capture Python exit code in a variable the EXIT trap can access
PYTHON_EXIT_CODE=$?

echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: $PYTHON_EXIT_CODE"
echo "============================================="

# =============================================================================
# TRAINING COMPLETION HANDLING
# =============================================================================
if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "============================================="
    echo "TRAINING COMPLETED SUCCESSFULLY"
    echo "============================================="
    # Create completion marker to prevent unnecessary requeues
    touch "$COMPLETION_MARKER"
    echo "Created completion marker: $COMPLETION_MARKER"
else
    echo "============================================="
    echo "TRAINING INTERRUPTED (exit code: $PYTHON_EXIT_CODE)"
    echo "============================================="
    # Explicitly handle requeue here - don't rely solely on EXIT trap
    # (EXIT trap is backup for when script is killed before reaching this point)
    if [ "$REQUEUE_SUBMITTED" = false ]; then
        if [ -d "$CKPT_DIR" ] && [ "$(ls -A "$CKPT_DIR" 2>/dev/null)" ]; then
            echo "Checkpoints exist - submitting continuation job"
            submit_continuation_job
        else
            echo "No checkpoints found - not requeuing (training may have failed before first checkpoint)"
        fi
    else
        echo "Continuation job already submitted via SIGUSR1"
    fi
fi
