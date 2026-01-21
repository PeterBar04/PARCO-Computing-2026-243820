#!/bin/bash

# Modules for python and MPI
module load GCC/12.3.0
module load gompi/2023a

# --- Verification ---
echo "Using the following compiler versions:"
# Check if the commands are found
if ! command -v mpicxx &> /dev/null; then
    echo "Error: mpicxx could not be found. Check 'module avail gompi'"
    exit 1
fi

gcc --version
mpicxx --version


# Define paths
SCRIPT_DIR=$(dirname "$0")
REPO_ROOT="$SCRIPT_DIR/.."
SRC_DIR="$REPO_ROOT/src"
MATRIX_DIR="$REPO_ROOT/data"

OUT_FILE="$SRC_DIR/distributed_spmv.out"

# 1. Compile ONCE
echo "Compiling..."
mpicxx -o3 $SRC_DIR/distributed_spmv.cpp "$SRC_DIR/matrixManager.cpp" "$SRC_DIR/processManager.cpp" -o "$OUT_FILE"

# 2. Check if compilation succeeded
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

# 3. Run for each matrix
NP=4 #NP = number of processes
MATRICES=($MATRIX_DIR/*.mtx)

for MATRIX_FILE in "${MATRICES[@]}"; do
    echo "Processing $MATRIX_FILE with $NP processes..."
    
    mpirun -np $NP $OUT_FILE "$MATRIX_FILE"


done
