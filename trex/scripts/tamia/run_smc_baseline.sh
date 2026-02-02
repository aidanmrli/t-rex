#!/bin/bash
# =============================================================================
# SMC Steering Baseline - Smart Launcher
# =============================================================================
# Checks queue availability and submits to H100 or H200, whichever is faster.
#
# Usage:
#   ./trex/scripts/tamia/run_smc_baseline.sh          # Auto-select best queue
#   ./trex/scripts/tamia/run_smc_baseline.sh h100     # Force H100
#   ./trex/scripts/tamia/run_smc_baseline.sh h200     # Force H200
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if user specified a GPU type
if [[ "$1" == "h100" ]]; then
    echo "Forcing H100 submission..."
    sbatch "${SCRIPT_DIR}/run_smc_baseline_h100.sh"
    exit $?
elif [[ "$1" == "h200" ]]; then
    echo "Forcing H200 submission..."
    sbatch "${SCRIPT_DIR}/run_smc_baseline_h200.sh"
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
    sbatch "${SCRIPT_DIR}/run_smc_baseline_h200.sh"
elif [[ $H100_IDLE -gt 0 ]]; then
    echo "Submitting to H100 (idle nodes available)..."
    sbatch "${SCRIPT_DIR}/run_smc_baseline_h100.sh"
else
    # No idle nodes - default to H100 (more nodes available)
    echo "No idle nodes - submitting to H100 (larger pool)..."
    sbatch "${SCRIPT_DIR}/run_smc_baseline_h100.sh"
fi
