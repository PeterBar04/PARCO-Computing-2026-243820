#!/bin/bash

# --- SETUP (Same as before) ---
module purge
module load gcc91
module load mpich-3.2.1--gcc-9.1.0

SCRIPT_DIR=$(dirname "$0")
REPO_ROOT="$SCRIPT_DIR/.."
SRC_DIR="$REPO_ROOT/src"
DATA_DIR="$REPO_ROOT/data/weak_scaling" # Points to new folder
OUT_FILE="$SRC_DIR/distributed_spmv.out"
RESULTS="$REPO_ROOT/results/weak_scaling_results.csv"

VENV_DIR="$REPO_ROOT/.venv"


# 2. Define the "Check File" 
# If this file exists, we assume all matrices are generated
CHECK_FILE="$DATA_DIR/weak_128.mtx"

# --- GENERATION STEP (Conditional) ---
if [ -f "$CHECK_FILE" ]; then
    echo "Check file '$CHECK_FILE' found. Skipping Python generation."
else
    echo "Matrix files missing. Running Python generator..."
    
    # --- PYTHON ENVIRONMENT SETUP & GENERATION ---

    # Check if the virtual environment exists. If not, create it.
    if [ ! -d "$VENV_DIR" ]; then
        echo "Virtual environment not found. Creating one..."
        
        # Load the cluster python module
        module load python-3.10.14_gcc91
        
        # 1. Create the virtual environment in the project folder
        python3 -m venv "$VENV_DIR"
        
        # 2. Activate it (This temporarily switches 'python3' to use the one inside .venv)
        source "$VENV_DIR/bin/activate"
        
        # 3. Install dependencies inside this isolated environment
        echo "Installing Scipy and Numpy..."
        pip install --upgrade pip --quiet
        pip install numpy scipy --quiet
        
    else
        echo "Virtual environment found. Activating..."
        module load python-3.10.14_gcc91
        source "$VENV_DIR/bin/activate"
    fi

    # Run the script
    python3 "$SCRIPT_DIR/generate_weak_matrices.py"
    
    deactivate
    # Unload python to keep environment clean (optional)
    module unload python-3.10.14_gcc91 
fi


# --- COMPILE C++ ---
echo "Compiling C++ Code..."
# --- COMPILE ---
mpicxx -std=c++17 -O3 $SRC_DIR/distributed_spmv.cpp "$SRC_DIR/matrixManager.cpp" "$SRC_DIR/processManager.cpp" -o "$OUT_FILE"

if [ $? -ne 0 ]; then
    echo "C++ Compilation failed!"
    exit 1
fi

# --- RUN WEAK SCALING (TEST MODE) ---
mkdir -p "$REPO_ROOT/results"
echo "Matrix,NumProcess,ExecutionTime_P90,CommunicationTime_P90" > "$RESULTS"

# --- RUN WEAK SCALING ---
# We manually define the pairs: (Procs MatrixFile)
declare -A BENCHMARKS
BENCHMARKS[1]="weak_1.mtx"
BENCHMARKS[2]="weak_2.mtx"
BENCHMARKS[4]="weak_4.mtx"
BENCHMARKS[8]="weak_8.mtx"
BENCHMARKS[16]="weak_16.mtx"
BENCHMARKS[32]="weak_32.mtx"
BENCHMARKS[64]="weak_64.mtx"
BENCHMARKS[128]="weak_128.mtx"

# Iterate through the keys (1, 2, 4...)
for NP in 1 2 4 8 16 32 64 128; do
    MATRIX_FILE="$DATA_DIR/${BENCHMARKS[$NP]}"
    MATRIX_NAME=$(basename "$MATRIX_FILE")
    
    echo "=================================================="
    echo "Weak Scaling: NP=$NP on $MATRIX_NAME"
    echo "=================================================="

    if [ -f "$MATRIX_FILE" ]; then
        # Run exactly like before
        OUTPUT_LINE=$(mpirun -np $NP $OUT_FILE "$MATRIX_FILE" 100 | grep "EXEC_TIME")
        
        TOTAL_TIME=$(echo "$OUTPUT_LINE" | awk '{print $2}')
        COMM_TIME=$(echo "$OUTPUT_LINE" | awk '{print $4}')
        
        if [ ! -z "$TOTAL_TIME" ]; then
            echo "$MATRIX_NAME,$NP,$TOTAL_TIME,$COMM_TIME" >> "$RESULTS"
            echo "  -> Done. Time: ${TOTAL_TIME}ms"
        else
            echo "  -> Failed (Crash or no output)."
        fi
    else
        echo "  -> Error: File $MATRIX_FILE not found."
    fi
done