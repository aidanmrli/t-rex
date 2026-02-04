#!/bin/bash
# =============================================================================
# SMC Token-Resampling Sweep (Array) - Smart Launcher
# =============================================================================
# Submits an array job so each K runs as its own SLURM job.
#
# Usage:
#   K_VALUES="32 64 128 256 512" ./trex/scripts/tamia/run_smc_token_sweep_array.sh
#   ./trex/scripts/tamia/run_smc_token_sweep_array.sh h100
#   ./trex/scripts/tamia/run_smc_token_sweep_array.sh h200
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

K_VALUES_DEFAULT="32 64 128 256 512"
K_VALUES="${K_VALUES:-$K_VALUES_DEFAULT}"
NUM_K=$(echo "${K_VALUES}" | awk '{print NF}')
if [[ "${NUM_K}" -lt 1 ]]; then
    echo "K_VALUES must contain at least one integer; got '${K_VALUES}'"
    exit 1
fi
ARRAY_SPEC="0-$((NUM_K - 1))"

# Check if user specified a GPU type
if [[ "$1" == "h100" ]]; then
    shift
    echo "Forcing H100 submission..."
    sbatch --array="${ARRAY_SPEC}" "${SCRIPT_DIR}/run_smc_token_sweep_array_h100.sh" "$@"
    exit $?
elif [[ "$1" == "h200" ]]; then
    shift
    echo "Forcing H200 submission..."
    sbatch --array="${ARRAY_SPEC}" "${SCRIPT_DIR}/run_smc_token_sweep_array_h200.sh" "$@"
    exit $?
fi

# Auto-detect best queue by checking pending jobs and available nodes
echo "Checking queue availability..."

# Count idle nodes for each GPU type
H100_IDLE=$(sinfo -N -h -t idle -o "%G" 2>/dev/null | grep -c "h100" || true)
H200_IDLE=$(sinfo -N -h -t idle -o "%G" 2>/dev/null | grep -c "h200" || true)
H100_IDLE=${H100_IDLE:-0}
H200_IDLE=${H200_IDLE:-0}

echo "H100: ${H100_IDLE} idle nodes"
echo "H200: ${H200_IDLE} idle nodes"

# Decision logic: prefer H200 when available (more GPUs), fall back to H100
if [[ $H200_IDLE -gt 0 ]]; then
    echo "Submitting to H200 (idle nodes available)..."
    sbatch --array="${ARRAY_SPEC}" "${SCRIPT_DIR}/run_smc_token_sweep_array_h200.sh" "$@"
elif [[ $H100_IDLE -gt 0 ]]; then
    echo "Submitting to H100 (idle nodes available)..."
    sbatch --array="${ARRAY_SPEC}" "${SCRIPT_DIR}/run_smc_token_sweep_array_h100.sh" "$@"
else
    # No idle nodes - default to H100 (more nodes available)
    echo "No idle nodes - submitting to H100 (larger pool)..."
    sbatch --array="${ARRAY_SPEC}" "${SCRIPT_DIR}/run_smc_token_sweep_array_h100.sh" "$@"
fi
