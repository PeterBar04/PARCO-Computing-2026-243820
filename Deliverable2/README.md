# Optimization of Sparse Matrix Vector Multiplication (SpMV) using MPI

## Overview
This project implements a parallel **Sparse Matrix-Vector Multiplication (SpMV)** kernel using **MPI** on a distributed memory system. It evaluates performance through:
* **Strong Scaling:** Fixed problem size, increasing processors.
* **Weak Scaling:** Fixed workload per processor, increasing problem size.

## How to build/run
1. **Cluster Execution (Recommended):** The project is designed to run on the UniTn HPC cluster. We provide an automated script that handles environment setup, matrix generation, compilation, and execution.
2. **Analysis:** The execution generates two `.csv` files in the `results/` directory containing detailed performance metrics (P90 execution time, GFLOPS, Efficiency, Speedup).

## Compiler Version & Flags
* **Compiler:** `mpicxx` (wrapping GCC 12.3.0)
* **MPI Library:** `gompi/2023a` (OpenMPI)
* **Flags:** `-std=c++17 -O3

## How to Compile and Run (Local)
The project is primarily intended for cluster execution, but for local testing and compilation, follow these steps from the root directory of the repository:
1.  **Navigate to scripts:**
    ```bash
    cd repo/Deliverable2/scripts
    ```
2.  **Make scripts executable:**
    ```bash
    chmod +x *.sh
    ```
3.  **Run Benchmarks:**
    ```bash
    ./run_strong_scaling.sh
    ./run_weak_scaling.sh
    ```
4.  **Generate Plots:**
    ```bash
    ./generate_strong_plot.sh
    ./generate_weak_plots.sh
    ```

---

## How to Compile and Run (Cluster)
The compilation, execution, and data generation are handled automatically by the PBS job script.

1.  Navigate to the scripts directory:
    ```bash
    cd repo/Deliverable2/scripts
    ```
2.  Submit the job to the queue:
    ```bash
    qsub distributed_spmv.pbs
    ```
3.  Monitor progress:
    ```bash
    qstat -u <your_username>
    ``` 

**What this script does:**
1.  Loads the required modules (`GCC/12.3.0`, `gompi/2023a`, `python-3.10.14`).
2.  Generates weak scaling matrices (if missing).
3.  Compiles the C++ source code using `mpicxx`.
4.  Executes the benchmark for $NP = \{1, 2, 4, 8, 16, 32, 64, 128\}$.
5.  Generates performance plots.

## Input / Output

## Input Data

### 1. Strong Scaling (Real Matrices)
**Note:** These matrices are too large for the repository. You must download them manually and place them in `data/strong_scaling/`.

| Matrix Name | Download Link |
| :--- | :--- |
| **atmosmodj** | [Link](https://sparse.tamu.edu/Bourchtein/atmosmodj) |
| **ML_Laplace**| [Link](https://sparse.tamu.edu/Janna/ML_Laplace) |
| **venkat25**  | [Link](https://sparse.tamu.edu/Simon/venkat25) |
| **torso1** | [Link](https://sparse.tamu.edu/Norris/torso1) |
| **twotone** | [Link](https://sparse.tamu.edu/ATandT/twotone) |

### 2. Weak Scaling (Synthetic Matrices)
Input matrices are **generated automatically** to ensure constant workload per processor.
* **Generator:** `scripts/generate_weak_matrices.py`
* **Location:** `data/weak_scaling/`

---

### OUTPUT
Results are stored in the `results/` directory as CSV files:
* `strong_scaling_results.csv`
* `weak_scaling_results.csv`

**Metrics Reported:**
* `ExecutionTime_P90`: Computation time (90th percentile).
* `CommunicationTime_P90`: Ghost value exchange time.
* `Avg_NNZ`: Load balance metric.
* `GFLOPS`: Effective throughput.
* `Avg_Comm`: Communication volume. 
* `Efficiency` & `Speedup`.

---

## Changing Parameters
To modify the test scenario (e.g., change iterations or process counts), edit the variables inside `run_weak_scaling.sh` or `run_strong_scaling.sh`.

* **BENCHMARKS**: An associative array mapping the number of processes (NP) to specific matrix files (only for weak scaling).
* **NP Loop**: The `for NP in ...` loop controls which process counts are tested (currently 1 to 128).

## Cluster Notes
Below are the resource allocation directives used in the script. Users may need to adjust the queue name (-q) and resource limits (walltime, select) based on their specific cluster environment.
```bash
#PBS -q shortCPUQ: Specifies the job queue (e.g., a queue dedicated to shorter CPU-bound tasks).
#PBS -l walltime=2:00:00: Sets the maximum wall-clock time for the job (3 hours).
#PBS -l select=2:ncpus=64:mpiprocs=64:mem=64gb: Requests 2 node with 64 CPUs available for the job execution

**Required Modules:**
```bash
module load GCC/12.3.0
module load gompi/2023a

module load python-3.10.14_gcc91
