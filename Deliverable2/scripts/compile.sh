#!/bin/bash

# Modules for python and MPI
module load gcc91
module load mpich-3.2.1--gcc-9.1.0

g++() {
    g++-9.1.0 "$@"
}

g++ --version

gcc() {
    gcc-9.1.0 "$@"
}
gcc --version

# Define paths
SCRIPT_DIR=$(dirname "$0")
REPO_ROOT="$SCRIPT_DIR/.."
SRC_DIR="$REPO_ROOT/src"
MATRIX_DIR="$REPO_ROOT/data"

OUT_FILE="$SRC_DIR/distributed_spmv.out"

# 1. Compile ONCE
echo "Compiling..."
mpicxx $SRC_DIR/distributed_spmv.cpp "$SRC_DIR/matrixManager.cpp" "$SRC_DIR/processManager.cpp" -o "$OUT_FILE"

# 2. Check if compilation succeeded
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

# 3. Run for each matrix
NP=5 #NP = number of processes
MATRICES=($MATRIX_DIR/*.mtx)

for MATRIX_FILE in "${MATRICES[@]}"; do
    echo "Processing $MATRIX_FILE with $NP processes..."
    
    mpirun -np $NP $OUT_FILE "$MATRIX_FILE"


done
