import matplotlib
matplotlib.use('Agg') # Force headless mode
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def setup_plot_style():
    plt.rc('font', size=11)
    plt.rc('axes', titlesize=12, labelsize=11)
    plt.rc('xtick', labelsize=9)
    plt.rc('ytick', labelsize=9)
    plt.rc('legend', fontsize=9)

def plot_weak_scaling(csv_file):
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found.")
        return
    
    try:
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    df = df.sort_values(by='NumProcess')
    
    # Convert to Milliseconds
    df['ExecutionTime_P90'] = df['ExecutionTime_P90'] * 1000
    df['CommunicationTime_P90'] = df['CommunicationTime_P90'] * 1000
    df['TotalTime'] = df['ExecutionTime_P90'] + df['CommunicationTime_P90']
    
    setup_plot_style()
    
    # Create Figure with 3 Columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    processes = df['NumProcess'].astype(str).tolist()
    indices = np.arange(len(processes))

    # =======================================================
    # PLOT 1 (LEFT): Total Time (Log Scale)
    # =======================================================
    ax = axes[0]
    total_times = df['TotalTime'].values
    
    ax.plot(indices, total_times, marker='o', linewidth=2, color='#d62728', label='Measured')
    ax.axhline(y=total_times[0], color='gray', linestyle='--', label='Ideal (Constant)')
    
    ax.set_title('Total Time (Log scale)', fontweight='bold')
    ax.set_xlabel('Processes', fontweight='bold')
    ax.set_ylabel('Time (ms)', fontweight='bold')
    ax.set_xticks(indices)
    ax.set_xticklabels(processes)
    ax.set_yscale('log')
    ax.grid(True, which="both", axis="y", linestyle='--', alpha=0.5)
    ax.legend()

    # =======================================================
    # PLOT 2 (CENTER): Efficiency (WITH NUMBERS RESTORED)
    # =======================================================
    ax = axes[1]
    efficiency = df['Efficiency'].values
    
    ax.plot(indices, efficiency, marker='s', linewidth=2, color='#1f77b4', label='Efficiency')
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Ideal (1.0)')

    # --- RESTORED LABELS HERE ---
    for i, eff in enumerate(efficiency):
        # Shift text slightly up (+0.03) so it doesn't overlap the dot
        ax.text(indices[i], eff + 0.03, f"{eff:.2f}", 
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1f77b4')

    ax.set_title('Efficiency', fontweight='bold')
    ax.set_xlabel('Processes', fontweight='bold')
    ax.set_ylabel('Eff (0-1)', fontweight='bold')
    ax.set_xticks(indices)
    ax.set_xticklabels(processes)
    ax.set_ylim(0, 1.15) # Increased upper limit to fit the text
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    # =======================================================
    # PLOT 3 (RIGHT): Breakdown (Filtered NP <= 64)
    # =======================================================
    ax = axes[2]
    
    # FILTER: Exclude NP > 64 to avoid squashing the graph
    limit_p = 64
    df_sub = df[df['NumProcess'] <= limit_p]
    
    sub_indices = np.arange(len(df_sub))
    sub_exec = df_sub['ExecutionTime_P90'].values
    sub_comm = df_sub['CommunicationTime_P90'].values
    sub_total = sub_exec + sub_comm
    sub_procs = df_sub['NumProcess'].astype(str).tolist()
    
    bar_width = 0.6
    
    # Draw Stacked Bars
    p1 = ax.bar(sub_indices, sub_exec, width=bar_width, 
           label='Compute', color='#2ca02c', edgecolor='black', alpha=0.85) # Green
    p2 = ax.bar(sub_indices, sub_comm, width=bar_width, bottom=sub_exec,
           label='Comm', color='#d62728', edgecolor='black', alpha=0.85)   # Red
    
    # Add Percentage Labels
    for i in range(len(df_sub)):
        pct_exec = (sub_exec[i] / sub_total[i]) * 100
        pct_comm = (sub_comm[i] / sub_total[i]) * 100
        
        # Label Compute
        if pct_exec > 10:
            ax.text(sub_indices[i], sub_exec[i]/2, f"{int(pct_exec)}%", 
                    ha='center', va='center', color='white', fontsize=8, fontweight='bold')
            
        # Label Comm
        if pct_comm > 10:
            ax.text(sub_indices[i], sub_exec[i] + sub_comm[i]/2, f"{int(pct_comm)}%", 
                    ha='center', va='center', color='white', fontsize=8, fontweight='bold')

    ax.set_title(f'Time Breakdown (<= {limit_p}P)', fontweight='bold')
    ax.set_xlabel('Processes', fontweight='bold')
    ax.set_ylabel('Time (ms)', fontweight='bold')
    ax.set_xticks(sub_indices)
    ax.set_xticklabels(sub_procs)
    
    ax.grid(True, axis="y", linestyle='--', alpha=0.5)
    ax.legend(loc='upper left')

    # --- SAVE ---
    plt.tight_layout()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(os.path.join(script_dir, "..", "plots"))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    output_filename = os.path.join(output_dir, "weak_scaling_combined.png")
    plt.savefig(output_filename, dpi=300)
    print(f"Combined plot saved to: {output_filename}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_weak_scaling.py <csv_file>")
    else:
        plot_weak_scaling(sys.argv[1])
