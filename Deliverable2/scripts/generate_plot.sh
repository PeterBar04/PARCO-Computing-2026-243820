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

module load python-3.10.14_gcc91 2>/dev/null || echo "  -> Module load skipped (local?)"

# 4. Activate & Fix Dependencies
# Find the environment
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
elif [ -f "$REPO_ROOT/venv/bin/activate" ]; then
    source "$REPO_ROOT/venv/bin/activate"
else
    echo "Error: Could not find virtual environment (.venv or venv)"
    exit 1
fi

echo "  -> Environment activated."
echo "  -> Fixing dependencies (Numpy incompatibility)..."

# CRITICAL FIX: Force reinstall compatible versions
# 1. Upgrade pip to handle wheels better
# 2. Install Pandas 1.5.3 (compatible with old GCC)
# 3. Install Numpy < 2.0 (compatible with Pandas 1.5.3)
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