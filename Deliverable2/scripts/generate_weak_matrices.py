import scipy.sparse as sp
import scipy.io as sio
import numpy as np
import os

# --- CONFIGURATION ---
OUTPUT_DIR = "../data/weak_scaling"
ROWS_PER_PROC = 10000   # Must match what you used before!
COLS_PER_PROC = 10000   
NNZ_PER_ROW = 10

# Create output directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Full list of process counts
proc_counts = [1, 2, 4, 8, 16, 32, 64, 128]

print(f"Generating matrices in '{OUTPUT_DIR}'...")

for p in proc_counts:
    filename = f"weak_{p}.mtx"
    filepath = os.path.join(OUTPUT_DIR, filename)

    # --- THE FIX: CHECK IF FILE EXISTS ---
    if os.path.exists(filepath):
        print(f"  -> NP={p}: File '{filename}' already exists. Skipping.")
        continue  # Jump to next iteration
    
    # If not found, generate it
    total_rows = ROWS_PER_PROC * p
    total_cols = COLS_PER_PROC * p
    total_nnz = total_rows * NNZ_PER_ROW
    
    print(f"  -> NP={p}: Generating {total_rows}x{total_cols}...", end="")
    
    # --- OPTIMIZED GENERATION ---
    # Instead of sp.random (slow/heavy), we build the COO arrays directly.
    
    # 1. ROW INDICES: repeat each row index 10 times (0,0,0... 1,1,1...)
    # This ensures exact load balancing (exactly 10 items per row)
    rows = np.repeat(np.arange(total_rows, dtype=np.int32), NNZ_PER_ROW)
    
    # 2. COL INDICES: random columns for each entry
    cols = np.random.randint(0, total_cols, size=total_nnz, dtype=np.int32)
    
    # 3. DATA: random values between 0 and 1
    data = np.random.rand(total_nnz)

    # 4. Create Matrix directly (Instant, low memory)
    matrix = sp.coo_matrix((data, (rows, cols)), shape=(total_rows, total_cols))
    
    sio.mmwrite(filepath, matrix)
    print(" Done.")

print("All matrices check/generation complete.")