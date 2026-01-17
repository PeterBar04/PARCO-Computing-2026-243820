# Optimization of Sparse Matrix Vector Multiplication (SpMV) using OpenMP

## How to build/run
1. Submit the job to the UniTn cluster (see How to Compile and Run (Cluster) below). This will compile the C++ code, execute the benchmarks, and generate the necessary .csv files in the results/ directory.
2. To obtain the plots, run the MATLAB script to generate the final plots, which are saved in the plots/ directory.

## Compiler Version & Flags
Compiler version: GNU g++ 9.1.0
Flags:
- std=c++17
- Wall
- g
- o3
- fopenmp (only used in the parallel version of the code)

## How to Compile and Run (Local)
The project is primarily intended for cluster execution, but for local testing and compilation, follow these steps from the root directory of the repository:
1. Navigate to the scripts directory: cd repo/scripts
2. Ensure the script is executable: chmod +x compile.sh
3. Run the compilation and execution script: ./compile.sh
4. Generate plots: start MATLAB, navigate to the scripts/ directory, and execute the plotting function: plot_results


## How to Compile and Run (Cluster)
The compilation, execution, and data generation are handled automatically by the PBS job script.
1. Navigate to the scripts directory: cd repo/scripts
2. Submit the job to the queue system: qsub job.pbs

You can monitor the progress of your job using the cluster's status command (e.g., qstat).

3. Once the job finishes, start MATLAB, navigate to the scripts/ directory, and execute the plotting function: plot_results

## Input / Output
INPUT
The matrices required for running the benchmarks are too large to be included in this repository.

Please download the necessary files from the following sources:
- atmosmodj: https://sparse.tamu.edu/Bourchtein/atmosmodj
- ML_Laplace: https://sparse.tamu.edu/Janna/ML_Laplace
- PR02R: https://sparse.tamu.edu/Fluorem/PR02R
- torso1: https://sparse.tamu.edu/Norris/torso1
- twotone: https://sparse.tamu.edu/ATandT/twotone

After downloading, ensure all .mtx files are placed inside the data/matrix/ subdirectory of this project.

OUTPUT
The execution of the compiled program generates two CSV files in the results/ directory, containing performance and cache metrics for each benchmark run.

Both output files share the same set of descriptive columns, which specify the context of the run: Matrix,Mode,Threads,Schedule,Chunk_size.

time_results.csv
This file contains the key metrics related to the execution time and parallel efficiency of the runs. The values reported are based on the 90th percentile (p90) of the measured run times:
p90_run_time,p90_speed_up,p90_efficiency

cache_results.csv: 
This file details the cache usage and miss statistics, obtained via hardware counters, providing insight into the memory hierarchy performance. The metrics are reported as average values across the measurement runs:
avg_l1_load,avg_l1_miss,l1_miss_rate,avg_ll_load,avg_ll_miss,ll_miss_rate

## Changing Parameters
The primary execution parameters controlling the benchmark runs are defined within the compile.sh script. To modify the test scenario, adjust the following variables:
- NUM_RUNS: The number of internal iterations the C++ code performs to measure timing for a single run configuration.
- CHUNKS: The size of the chunk used by OpenMP scheduling 
- THREADS: The specific number of OpenMP threads to be tested.
- SCHEDULE: The type of scheduling used by OpenMP
- REPEAT: The number of times the executable must be run for a specific configuration to calculate the average cache statistics.

The type of matrixes used can also be changed inside the folder data/matrix/
The job.pbs script is configured to automatically iterate through all .mtx files found in this directory. To change the set of matrices used, simply add or remove .mtx files from the data/matrix/ folder.

## Cluster Notes
Below are the resource allocation directives used in the script. Users may need to adjust the queue name (-q) and resource limits (walltime, select) based on their specific cluster environment.
- #PBS -q short_cpuQ: Specifies the job queue (e.g., a queue dedicated to shorter CPU-bound tasks).
- #PBS -l walltime=3:00:00: Sets the maximum wall-clock time for the job (3 hours).
- #PBS -l select=1:ncpus=64: Requests 1 node with 64 CPUs available for the job execution

In the job.pbs you must include the following modules:
- module load gcc91
- module load perf  
