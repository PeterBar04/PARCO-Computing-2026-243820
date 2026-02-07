#!/bin/bash

# ==============================================================================
# Script to Generate Weak Scaling Plots
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/.."
PYTHON_SCRIPT="$SCRIPT_DIR/plot_weak_scaling.py"
RESULTS="${1:-$REPO_ROOT/results/weak_scaling_results.csv}" # Default input file

# 1. Validation
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ Error: plot_weak_scaling.py not found."
    exit 1
fi

if [ ! -f "$RESULTS" ]; then
    echo "❌ Error: Results file not found at $RESULTS"
    exit 1
fi

echo "========================================"
echo "Generating Weak Scaling Plots"
echo "Input: $RESULTS"
echo "========================================"

# 2. Environment Setup
module load Miniforge3/24.11.3-0 2>/dev/null || echo "-> Module 'Miniforge3' load skipped (not found or local)"

# 3. Activate Virtual Environment
# Checks for .venv (hidden) or venv (standard)
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
    echo "-> Activated .venv"
elif [ -f "$REPO_ROOT/venv/bin/activate" ]; then
    source "$REPO_ROOT/venv/bin/activate"
    echo "-> Activated venv"
else
    echo "❌ Error: Virtual environment not found in $REPO_ROOT"
    echo "   Please run the scaling script first to generate it."
    exit 1
fi

# 3. Dependencies Check (Critical Fixes)
# forcing numpy<2.0.0 prevents binary incompatibility errors with pandas/matplotlib
echo "-> Checking libraries..."
pip install --upgrade pip --quiet
pip install "pandas==1.5.3" "numpy<2.0.0" matplotlib --quiet

# 4. Run Script
python3 "$PYTHON_SCRIPT" "$RESULTS"

if [ $? -eq 0 ]; then
    echo "✅ Success! Plots saved in Deliverable2/plots/"
else
    echo "❌ Error running python script."
    exit 1
fi
