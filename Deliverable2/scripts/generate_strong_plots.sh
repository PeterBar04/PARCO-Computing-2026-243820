#!/bin/bash

# ==============================================================================
# Script to Generate Strong Scaling Plot independently
# Usage: ./generate_plot.sh [optional_path_to_csv]
# ==============================================================================

# 1. Setup Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/.."
PYTHON_SCRIPT="$SCRIPT_DIR/plot_strong_scaling.py"

if [ -n "$1" ]; then
    RESULTS="$1"
else
    RESULTS="$REPO_ROOT/results/strong_scaling_results.csv"
fi

# 2. Validation
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python plotting script not found at $PYTHON_SCRIPT"
    exit 1
fi

if [ ! -f "$RESULTS" ]; then
    echo "Error: Results file not found at $RESULTS"
    exit 1
fi

# 3. Execution environment
echo "========================================"
echo "Plotting Strong Scaling Data"
echo "CSV File: $RESULTS"
echo "Script:   $PYTHON_SCRIPT"
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

# 5. Run the Python Script
python3 "$PYTHON_SCRIPT" "$RESULTS"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Success! Plot generated."
else
    echo ""
    echo "❌ Error: Python script failed."
    exit 1
fi