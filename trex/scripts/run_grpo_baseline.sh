#!/bin/bash

#SBATCH --partition=short-unkillable                           
#SBATCH --cpus-per-task=64                                
#SBATCH --gres=gpu:h100:4                                     
#SBATCH --mem=512G                                        
#SBATCH --time=6:00:00                                   
#SBATCH -o /network/scratch/l/lia/t-rex/slurm/grpo-%j.out
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@120

# =============================================================================
# T-REX GRPO Baseline Training Script
# Uses OpenRLHF's native GRPO support via train_ppo_ray.py
# =============================================================================

# 1. Load the required modules
module load cuda/12.6.0

# 2. Load your environment
source .venv/bin/activate

# Scratch and weights setup
SCRATCH_DIR="/network/scratch/l/lia/t-rex"
SCRATCH_WEIGHTS="/network/scratch/l/lia/model_weights"

# 3. Set environment variables
export HF_HOME="$SCRATCH_WEIGHTS"
export HF_DATASETS_CACHE="$SCRATCH_WEIGHTS"
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
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
# Model Name              | Path                              | Chat Template
# -----------------------------------------------------------------------------
# Qwen2.5-7B (Base)       | Qwen/Qwen2.5-7B                   | false
# Qwen3-8B (Base)         | Qwen/Qwen3-8B-Base                | false
# Qwen2.5-7B-Instruct     | Qwen/Qwen2.5-7B-Instruct          | true
# Qwen2.5-Math-7B-Instruct| Qwen/Qwen2.5-Math-7B-Instruct     | true
# =============================================================================
MODEL="Qwen/Qwen2.5-7B"
MODEL_NAME=$(basename "$MODEL")

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

# GRPO Hyperparameters
N_SAMPLES=8                     # Group size (G) for GRPO
KL_COEF=0.001                   # KL penalty coefficient
KL_ESTIMATOR="k3"               # k1, k2, or k3
MAX_EPOCHS=1                    # Number of training epochs
LEARNING_RATE=5e-7              # Actor learning rate

# Batch sizes
TRAIN_BATCH_SIZE=128
MICRO_TRAIN_BATCH_SIZE=4
ROLLOUT_BATCH_SIZE=128
MICRO_ROLLOUT_BATCH_SIZE=16

# Generation settings
TEMPERATURE=1.0
PROMPT_MAX_LEN=1024
GENERATE_MAX_LEN=2048

# vLLM settings (4x H100)
VLLM_NUM_ENGINES=2
VLLM_TP_SIZE=2
VLLM_GPU_UTIL=0.5

# Checkpointing (~30 min intervals)
SAVE_STEPS=50

# Chat Template: Set to false for base models, true for instruct models
APPLY_CHAT_TEMPLATE=false

# WandB
USE_WANDB=true
WANDB_PROJECT="t-rex"

# Output
OUTPUT_DIR="trex/results/grpo_baseline/${MODEL_NAME}/${DATASET}_n${N_SAMPLES}"
CKPT_DIR="${OUTPUT_DIR}/checkpoints"
EFFICIENCY_PATH="${OUTPUT_DIR}/efficiency_metrics.json"

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
export TREX_METHOD="grpo"
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
echo "Group Size (N): $N_SAMPLES"
echo "KL Coef: $KL_COEF"
echo "Output Dir: $OUTPUT_DIR"
echo "Resume from checkpoint: $LOAD_CHECKPOINT"
echo "============================================="

# Run GRPO training using OpenRLHF
python -m openrlhf.cli.train_ppo_ray \
    --pretrain "$MODEL" \
    --remote_rm_url trex/baselines/grpo_reward_func.py \
    --prompt_data "$TRAIN_DATA" \
    --input_key prompt \
    --label_key label \
    --advantage_estimator group_norm \
    --use_kl_loss \
    --kl_estimator "$KL_ESTIMATOR" \
    --init_kl_coef "$KL_COEF" \
    --n_samples_per_prompt "$N_SAMPLES" \
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
    $EXTRA_ARGS

EXIT_CODE=$?

echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "============================================="
