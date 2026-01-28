import matplotlib
matplotlib.use('Agg') # Force headless mode
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import sys
import os

def plot_strong_scaling(csv_file):
    # --- 1. Read Data ---
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found.")
        return
    
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    
    # --- 2. Setup Data ---
    matrices = df['Matrix'].unique()
    processes = sorted(df['NumProcess'].unique())
    indices = np.arange(len(processes))
    
    total_width = 0.8
    bar_width = total_width / len(matrices)
    cmap = plt.get_cmap('tab10')

    # --- 3. Plotting ---
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Calculate safe minimum for log scale (ignoring zeros)
    all_values = df['ExecutionTime_P90'].tolist() + df['CommunicationTime_P90'].tolist()
    non_zeros = [v for v in all_values if v > 0]
    min_val = min(non_zeros) * 0.5 if non_zeros else 0.1

    print(f"DEBUG: Setting Log Scale Base: {min_val}")

    for i, matrix in enumerate(matrices):
        subset = df[df['Matrix'] == matrix]
        exec_times = []
        comm_times = []
        
        for p in processes:
            row = subset[subset['NumProcess'] == p]
            if not row.empty:
                e = row.iloc[0]['ExecutionTime_P90']
                c = row.iloc[0]['CommunicationTime_P90']
                exec_times.append(max(e, 0)) 
                comm_times.append(max(c, 0))
            else:
                exec_times.append(0)
                comm_times.append(0)
        
        # Positions
        x_positions = indices - (total_width / 2) + (i * bar_width) + (bar_width / 2)
        
        # BAR 1: Execution (Solid)
        ax.bar(x_positions, exec_times, width=bar_width, 
               color=cmap(i), edgecolor='black', linewidth=0.5,
               label=matrix)

        # BAR 2: Communication (Hatched), stacked on top
        ax.bar(x_positions, comm_times, width=bar_width, 
               bottom=exec_times,
               color=cmap(i), edgecolor='black', linewidth=0.5, 
               hatch='///', alpha=0.6)

    # --- 4. Force Log Scale ---
    ax.set_yscale('log')
    ax.set_ylim(bottom=min_val)
    
    # Labels
    ax.set_xlabel('Number of Processes (NP)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (ms) - Log Scale', fontsize=12, fontweight='bold')
    ax.set_title('Strong Scaling: Execution vs Communication Time (Log Scale)', fontsize=14, fontweight='bold')
    
    ax.set_xticks(indices)
    ax.set_xticklabels(processes)
    
    # Grid for log scale (Major and Minor lines)
    ax.grid(True, which="both", axis="y", linestyle='--', alpha=0.5)

    # Legend
    legend_elements = [Patch(facecolor=cmap(i), edgecolor='black', label=m) for i, m in enumerate(matrices)]
    legend_elements.append(Patch(facecolor='white', edgecolor='black', label='Solid: Computation'))
    legend_elements.append(Patch(facecolor='white', edgecolor='black', hatch='///', label='Hatched: Communication'))
    
    ax.legend(handles=legend_elements, title="Matrices & Metrics", loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()

    # --- 5. Save ---
    # Determine where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Resolve the path to Clean Up the ".."
    # This turns ".../scripts/../plots" into ".../plots"
    output_dir = os.path.abspath(os.path.join(script_dir, "..", "plots"))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")
    
    # UPDATED FILENAME HERE
    output_filename = os.path.join(output_dir, "strong_scaling_time_log.png")
    
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved to: {output_filename}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_strong_scaling.py <csv_file>")
    else:
        plot_strong_scaling(sys.argv[1])
