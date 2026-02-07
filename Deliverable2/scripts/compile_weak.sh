#!/bin/bash

# --- SETUP (Same as before) ---
module purge
module load GCC/12.3.0
module load gompi/2023a

SCRIPT_DIR=$(dirname "$0")
REPO_ROOT="$SCRIPT_DIR/.."
SRC_DIR="$REPO_ROOT/src"
DATA_DIR="$REPO_ROOT/data/weak_scaling" 
OUT_FILE="$SRC_DIR/distributed_spmv.out"
RESULTS="$REPO_ROOT/results/weak_scaling_results.csv"

VENV_DIR="$REPO_ROOT/.venv"

# 2. Define the "Check File" 
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
        
        # --- CHANGE 1: Load the available Miniforge module ---
        module load Miniforge3/24.11.3-0
        
        # 1. Create the virtual environment in the project folder
        python3 -m venv "$VENV_DIR"
        
        # 2. Activate it
        source "$VENV_DIR/bin/activate"
        
        # 3. Install dependencies inside this isolated environment
        echo "Installing Scipy and Numpy..."
        pip install --upgrade pip --quiet
        pip install numpy scipy --quiet
        pip install pandas matplotlib --quiet
        
    else
        echo "Virtual environment found. Activating..."
        # --- CHANGE 2: Load the available Miniforge module ---
        module load Miniforge3/24.11.3-0
        source "$VENV_DIR/bin/activate"
    fi

    # Run the script
    python3 "$SCRIPT_DIR/generate_weak_matrices.py"
    
    deactivate
    # --- CHANGE 3: Unload the correct module ---
    module unload Miniforge3/24.11.3-0 
fi


# --- COMPILE C++ ---
echo "Compiling C++ Code..."
# ... (Rest of your script remains unchanged)
mpicxx -std=c++17 -O3 $SRC_DIR/distributed_spmv.cpp "$SRC_DIR/matrixManager.cpp" "$SRC_DIR/processManager.cpp" -o "$OUT_FILE"

if [ $? -ne 0 ]; then
    echo "C++ Compilation failed!"
    exit 1
fi

# --- RUN WEAK SCALING (TEST MODE) ---
mkdir -p "$REPO_ROOT/results"
echo "Matrix,NumProcess,ExecutionTime_P90,CommunicationTime_P90,Min_NNZ,Max_NNZ,Avg_NNZ,Imbalance_Ratio,Min_Comm,Max_Comm,Avg_Comm,GFLOPS,Efficiency,Speedup" > "$RESULTS"

# --- RUN WEAK SCALING ---
declare -A BENCHMARKS
BENCHMARKS[1]="weak_1.mtx"
BENCHMARKS[2]="weak_2.mtx"
BENCHMARKS[4]="weak_4.mtx"
BENCHMARKS[8]="weak_8.mtx"
BENCHMARKS[16]="weak_16.mtx"
BENCHMARKS[32]="weak_32.mtx"
BENCHMARKS[64]="weak_64.mtx"
BENCHMARKS[128]="weak_128.mtx"

T_BASE=0

for NP in 1 2 4 8 16 32 64 128; do
    MATRIX_FILE="$DATA_DIR/${BENCHMARKS[$NP]}"
    MATRIX_NAME=$(basename "$MATRIX_FILE")
    
    echo "=================================================="
    echo "Weak Scaling: NP=$NP on $MATRIX_NAME"
    echo "=================================================="

    if [ -f "$MATRIX_FILE" ]; then
       FULL_OUTPUT=$(mpirun --oversubscribe -np $NP $OUT_FILE "$MATRIX_FILE" 100 2>&1)
           
        TIME_LINE=$(echo "$FULL_OUTPUT" | grep "EXEC_TIME")
        EXEC_TIME=$(echo "$TIME_LINE" | awk '{print $2}')
        COMM_TIME=$(echo "$TIME_LINE" | awk '{print $4}')

        TOTAL_TIME=$(awk -v e="$EXEC_TIME" -v c="$COMM_TIME" 'BEGIN {print e + c}')
        
        BONUS_LINE=$(echo "$FULL_OUTPUT" | grep "BONUS_DATA")
        
        MIN_NNZ=$(echo "$BONUS_LINE" | awk '{print $2}')
        MAX_NNZ=$(echo "$BONUS_LINE" | awk '{print $3}')
        AVG_NNZ=$(echo "$BONUS_LINE" | awk '{print $4}')
        IMBALANCE_RATIO=$(echo "$BONUS_LINE" | awk '{print $5}')
        MIN_COMM=$(echo "$BONUS_LINE" | awk '{print $6}')
        MAX_COMM=$(echo "$BONUS_LINE" | awk '{print $7}')
        AVG_COMM=$(echo "$BONUS_LINE" | awk '{print $8}')

        if [ "$NP" -eq 1 ]; then
            T_BASE=$TOTAL_TIME
        fi

        METRICS=$(awk -v t1="$T_BASE" -v tn="$TOTAL_TIME" -v np="$NP" 'BEGIN {
            if (tn > 0 && t1 > 0) {
                spd = np * (t1 / tn);    
                eff = spd / np;          
                printf "%.4f %.4f", eff, spd;
            } else {
                print "0.0000 0.0000"
            }
        }')

        EFFICIENCY=$(echo "$METRICS" | awk '{print $1}')
        SPEEDUP=$(echo "$METRICS" | awk '{print $2}')

        GFLOPS=$(awk -v nnz="$AVG_NNZ" -v np="$NP" -v time="$TOTAL_TIME" 'BEGIN {
            if (time > 0) printf "%.4f", (2 * nnz * np) / (time * 1000000);
            else print "0"
        }')
        
        if [ ! -z "$EXEC_TIME" ]; then
            echo "$MATRIX_NAME,$NP,$EXEC_TIME,$COMM_TIME,$MIN_NNZ,$MAX_NNZ,$AVG_NNZ,$IMBALANCE_RATIO,$MIN_COMM,$MAX_COMM,$AVG_COMM,$GFLOPS,$EFFICIENCY,$SPEEDUP" >> "$RESULTS"
            echo "  -> Done. Time: ${TOTAL_TIME}ms"
        else
            echo "  -> Failed (Crash or no output)."
        fi
    else
        echo "  -> Error: File $MATRIX_FILE not found."
    fi
done