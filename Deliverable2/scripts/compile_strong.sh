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
MATRIX_DIR="$REPO_ROOT/data/strong_scaling"

OUT_FILE="$SRC_DIR/distributed_spmv.out"
RESULTS="$REPO_ROOT/results/strong_scaling_results.csv"

# Verify directory exists
mkdir -p "$REPO_ROOT/results"

# Write Header 
echo "Matrix,NumProcess,ExecutionTime_P90,CommunicationTime_P90,Min_NNZ,Max_NNZ,Avg_NNZ,Imbalance_Ratio,Min_Comm,Max_Comm,Avg_Comm,GFLOPS,Efficiency,Speedup" > "$RESULTS"

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
        echo "=================================================="
        echo "Strong Scaling: NP=$NP on $MATRIX_NAME"
        echo "=================================================="
        
        if [ -f "$MATRIX_FILE" ]; then
            # Run the program and capture ALL output to a variable
            # We use '2>&1' to ensure we capture both stdout and stderr just in case
            FULL_OUTPUT=$(mpirun -np $NP $OUT_FILE "$MATRIX_FILE" 100 2>&1)

            # 1. PARSE STANDARD TIMES (Existing logic)
            # We use 'grep' on the captured variable $FULL_OUTPUT
            TIME_LINE=$(echo "$FULL_OUTPUT" | grep "EXEC_TIME")
            EXEC_TIME=$(echo "$TIME_LINE" | awk '{print $2}')
            COMM_TIME=$(echo "$TIME_LINE" | awk '{print $4}')

            # Sum them up for the "Real" Total Time
            TOTAL_TIME=$(awk -v e="$EXEC_TIME" -v c="$COMM_TIME" 'BEGIN {print e + c}')
            
            # 2. PARSE BONUS METRICS (New logic)
            BONUS_LINE=$(echo "$FULL_OUTPUT" | grep "BONUS_DATA")
            
            # Extract the 6 values (Cols 2 through 7 because Col 1 is the tag "BONUS_DATA")
            MIN_NNZ=$(echo "$BONUS_LINE" | awk '{print $2}')
            MAX_NNZ=$(echo "$BONUS_LINE" | awk '{print $3}')
            AVG_NNZ=$(echo "$BONUS_LINE" | awk '{print $4}')
            IMBALANCE_RATIO=$(echo "$BONUS_LINE" | awk '{print $5}')
            MIN_COMM=$(echo "$BONUS_LINE" | awk '{print $6}')
            MAX_COMM=$(echo "$BONUS_LINE" | awk '{print $7}')
            AVG_COMM=$(echo "$BONUS_LINE" | awk '{print $8}')

            # --- STRONG SCALING MATH ---
            if [ "$NP" -eq 1 ]; then
                T_BASE=$TOTAL_TIME
            fi

            # 2.5. Calculate metrics using awk
            METRICS=$(awk -v t1="$T_BASE" -v tn="$TOTAL_TIME" -v np="$NP" 'BEGIN {
                if (tn > 0 && t1 > 0) {
                    speedup = t1 / tn;       # Formula: Old_Time / New_Time
                    eff = speedup / np;      # Formula: Speedup / N
                    printf "%.4f %.4f", eff, speedup;
                } else {
                    print "0.0000 0.0000"
                }
            }')

            EFF=$(echo "$METRICS" | awk '{print $1}')
            SPD=$(echo "$METRICS" | awk '{print $2}')

             # 3. CALCULATE GFLOPS
            # Formula: (2 * Avg_NNZ * NP) / (Total_Time_ms * 1,000,000)
            GFLOPS=$(awk -v nnz="$AVG_NNZ" -v np="$NP" -v time="$TOTAL_TIME" 'BEGIN {
                if (time > 0) printf "%.4f", (2 * nnz * np) / (time * 1000000);
                else print "0"
            }')
                
            # 4. SAVE TO CSV
            if [ ! -z "$EXEC_TIME" ]; then
                echo "$MATRIX_NAME,$NP,$EXEC_TIME,$COMM_TIME,$MIN_NNZ,$MAX_NNZ,$AVG_NNZ,$IMBALANCE_RATIO,$MIN_COMM,$MAX_COMM,$AVG_COMM,$GFLOPS,$EFF,$SPD" >> "$RESULTS"
                echo "  -> Done. Time: ${TOTAL_TIME}ms"
            else
                echo "  -> Failed (Crash or no output)."
            fi
            
            NP=$((NP * 2))
        else
            echo "  -> Error: File $MATRIX_FILE not found."
        fi
    done
done

echo "Done! Results saved in $RESULTS"
