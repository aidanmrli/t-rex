#!/bin/bash
# =============================================================================
# Best-of-N Baseline - Smart Launcher
# =============================================================================
# Checks queue availability and submits to H100 or H200, whichever is faster.
#
# Usage:
#   ./trex/scripts/tamia/run_bon_baseline.sh          # Auto-select best queue
#   ./trex/scripts/tamia/run_bon_baseline.sh h100     # Force H100
#   ./trex/scripts/tamia/run_bon_baseline.sh h200     # Force H200
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$1" == "h100" ]]; then
    echo "Forcing H100 submission..."
    sbatch "${SCRIPT_DIR}/run_bon_baseline_h100.sh"
    exit $?
elif [[ "$1" == "h200" ]]; then
    echo "Forcing H200 submission..."
    sbatch "${SCRIPT_DIR}/run_bon_baseline_h200.sh"
    exit $?
fi

echo "Checking queue availability..."

H100_IDLE=$(sinfo -N -h -t idle -o "%G" 2>/dev/null | grep -c "h100" || true)
H200_IDLE=$(sinfo -N -h -t idle -o "%G" 2>/dev/null | grep -c "h200" || true)
H100_IDLE=${H100_IDLE:-0}
H200_IDLE=${H200_IDLE:-0}

echo "H100: ${H100_IDLE} idle nodes"
echo "H200: ${H200_IDLE} idle nodes"

if [[ $H200_IDLE -gt 0 ]]; then
    echo "Submitting to H200 (idle nodes available)..."
    sbatch "${SCRIPT_DIR}/run_bon_baseline_h200.sh"
elif [[ $H100_IDLE -gt 0 ]]; then
    echo "Submitting to H100 (idle nodes available)..."
    sbatch "${SCRIPT_DIR}/run_bon_baseline_h100.sh"
else
    echo "No idle nodes - submitting to H100 (larger pool)..."
    sbatch "${SCRIPT_DIR}/run_bon_baseline_h100.sh"
fi
