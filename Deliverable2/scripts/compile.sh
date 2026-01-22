#!/bin/bash

# --- 1. LOAD OLD CLUSTER MODULES ---
# We load the compiler first, then the MPI library built for it.
module purge             # Clear any default modules to avoid conflicts
module load gcc91        # Load GCC 9.1.0
module load mpich-3.2.1--gcc-9.1.0  # Load MPICH built with GCC 9.1

# --- Verification ---
echo "Using the following compiler versions:"

# Safety Check: If the specific MPI module above failed, try the generic one
if ! command -v mpicxx &> /dev/null; then
    echo "Specific MPI module not found. Trying generic mpich-3.2..."
    module load mpich-3.2
fi

# Final check
if ! command -v mpicxx &> /dev/null; then
    echo "Error: mpicxx could not be found. Please check 'module avail' names."
    exit 1
fi

gcc --version
mpicxx --version  # Note: MPICH often uses -version or --version

# Define paths
SCRIPT_DIR=$(dirname "$0")
REPO_ROOT="$SCRIPT_DIR/.."
SRC_DIR="$REPO_ROOT/src"
MATRIX_DIR="$REPO_ROOT/data"

OUT_FILE="$SRC_DIR/distributed_spmv.out"
RESULTS="$REPO_ROOT/results/results.csv"

# Verify directory exists
mkdir -p "$REPO_ROOT/results"

# Write Header (Overwrites old file)
echo "Matrix,NumProcess,Iteration,ExecutionTime,CommunicationTime" > "$RESULTS"


# --- 2. COMPILE ---
# GCC 9.1.0 supports C++17, so we keep the flag.
echo "Compiling..."
mpicxx -std=c++17 -O3 $SRC_DIR/distributed_spmv.cpp "$SRC_DIR/matrixManager.cpp" "$SRC_DIR/processManager.cpp" -o "$OUT_FILE"

# Check if compilation succeeded
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi


# --- 3. RUN LOOP ---
NUM_RUNS=10
MAX_PROCESS=128
MATRICES=($MATRIX_DIR/*.mtx)

for MATRIX_FILE in "${MATRICES[@]}"; do
    # Extract the filename (e.g., "twotone.mtx")
    MATRIX_NAME=$(basename "$MATRIX_FILE")

    echo "=================================================="
    echo "Processing Matrix: $MATRIX_NAME"
    echo "=================================================="
    
    # Loop for Process Counts: 1, 2, 4, ... 128
    NP=1
    while [ $NP -le $MAX_PROCESS ]; do
        echo "  -> Running with NP = $NP ($NUM_RUNS iterations)"
        
        for (( i=1; i<=NUM_RUNS; i++ )); do
            echo -n "."
            
            # Run MPI and filter for the "EXEC_TIME" line
            # Get the full line first
            OUTPUT_LINE=$(mpirun -np $NP $OUT_FILE "$MATRIX_FILE" | grep "EXEC_TIME")
            
            # Extract Total Time (2nd word)
            EXEC_TIME=$(echo "$OUTPUT_LINE" | awk '{print $2}')

            # Extract Comm Time (4th word)
            COMM_TIME=$(echo "$OUTPUT_LINE" | awk '{print $4}')

            # Save to CSV if the run was successful
            if [ ! -z "$EXEC_TIME" ]; then
                echo "$MATRIX_NAME,$NP,$i,$EXEC_TIME,$COMM_TIME" >> "$RESULTS"
            else
                echo "Warning: Run failed for NP=$NP Iter=$i"
            fi
        done
        
        echo "" 
        NP=$((NP * 2))
    done
done

echo "Done! Results saved in $RESULTS"