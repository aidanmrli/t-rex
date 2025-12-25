#!/bin/bash

#SBATCH --partition=unkillable                           # Ask for unkillable job
#SBATCH --cpus-per-task=2                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=10G                                        # Ask for 10 GB of RAM
#SBATCH --time=3:00:00                                   # The job will run for 3 hours
#SBATCH -o /network/scratch/l/lia/t-rex/slurm/slurm-%j.out  # Write the log on scratch

# 1. Load the required modules
module load cuda/12.6.0

# 2. Load your environment
source .venv/bin/activate

# Scratch and weights setup
SCRATCH_DIR="/network/scratch/l/lia/t-rex"
SCRATCH_WEIGHTS="/network/scratch/l/lia/model_weights"

# 3. Set environment variables for model weights and datasets
export HF_HOME="$SCRATCH_WEIGHTS"
export HF_DATASETS_CACHE="$SCRATCH_WEIGHTS"
mkdir -p "$SCRATCH_WEIGHTS"

# 4. Set up experimental results on scratch and symlink back
mkdir -p "$SCRATCH_DIR/results"

# If trex/results is a directory but not a symlink, move its content to scratch and symlink
if [ -d "trex/results" ] && [ ! -L "trex/results" ]; then
    echo "Moving existing results to scratch..."
    cp -r trex/results/* "$SCRATCH_DIR/results/"
    rm -rf trex/results
    ln -s "$SCRATCH_DIR/results" trex/results
elif [ ! -e "trex/results" ]; then
    ln -s "$SCRATCH_DIR/results" trex/results
fi

# NOTE: The model should be saved in $SLURM_TMPDIR

# Configuration
DATASET=${1:-"gsm8k"}
MODEL=${2:-"Qwen/Qwen2.5-Math-7B-Instruct"}
N_SAMPLES=${3:-8}
TP_SIZE=${4:-1}

# Set output directory
OUTPUT_DIR="trex/results/bon_baseline/${DATASET}_n${N_SAMPLES}"

# Set path to the dataset based on input
if [ "$DATASET" == "gsm8k" ]; then
    DATASET_PATH="trex/data/gsm8k_platinum_test.jsonl"
elif [ "$DATASET" == "math" ]; then
    DATASET_PATH="trex/data/math_test.jsonl"
else
    echo "Unknown dataset: $DATASET. Using default gsm8k path."
    DATASET_PATH="trex/data/gsm8k_platinum_test.jsonl"
fi

# Run the baseline script
# We use python -m to ensure the package structure is respected
python -m trex.baselines.best_of_n_baseline \
    --model_path "$MODEL" \
    --dataset "$DATASET" \
    --dataset_path "$DATASET_PATH" \
    --n_samples "$N_SAMPLES" \
    --tp_size "$TP_SIZE" \
    --output_dir "$OUTPUT_DIR" \
    --sweep_size 100 \
    --use_wandb

# Then, after the job is finished, copy whatever you want to save on $SCRATCH
# cp $SLURM_TMPDIR/<to_save> $SCRATCH/t-rex