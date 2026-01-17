#!/bin/bash

module load perf  

REPO_ROOT=$(dirname "$0")/..

TIME_OUTFILE="$REPO_ROOT/results/time_results.csv"
CACHE_OUTFILE="$REPO_ROOT/results/cache_results.csv"

NUM_RUNS=10

SCHEDULES=("static" "dynamic" "guided")
CHUNKS=(10 100 1000)
THREADS=(1 2 4 8 16 32 64)

MATRIX_DIR="$REPO_ROOT/data/matrix"
MATRICES=($(ls $MATRIX_DIR/*.mtx)) #array of all files .mtx

echo "Matrix,Mode,Threads,Schedule,Chunk_size,p90_run_time,p90_speed_up,p90_efficiency" > "$TIME_OUTFILE"

echo "Matrix,Mode,Threads,Schedule,Chunk_size,avg_l1_load,avg_l1_miss,l1_miss_rate,avg_ll_load,avg_ll_miss,ll_miss_rate" > "$CACHE_OUTFILE"

calculate_cache() {
        total_l1_load=0.0
        total_l1_miss=0.0
        total_ll_load=0.0
        total_ll_miss=0.0 
        
        REPEAT=5     
        
        for ((i=1; i<=REPEAT; i++)); do
          #in the code, num_runs=1, is the bash that executes many times the program
          result=$( (perf stat -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses $REPO_ROOT/src/deliverable1 $s $c 1) 2>&1 )
          

          l1_cache_load=$(echo "$result" | grep -E "L1-dcache-loads(:| |$)" | awk '{print $1}' | tr -d ',')
          l1_cache_miss=$(echo "$result" | grep -E "L1-dcache-load-misses(:| |$)" | awk '{print $1}' | tr -d ',')
          ll_cache_load=$(echo "$result" | grep -E "LLC-loads(:| |$)" | awk '{print $1}' | tr -d ',')
          ll_cache_miss=$(echo "$result" | grep -E "LLC-load-misses(:| |$)" | awk '{print $1}' | tr -d ',')
          
          # If one of the variable is empty, put 0
          l1_cache_load=${l1_cache_load:-0}
          l1_cache_miss=${l1_cache_miss:-0}
          ll_cache_load=${ll_cache_load:-0}
          ll_cache_miss=${ll_cache_miss:-0}
  

          #total data
          total_l1_load=$(echo "$total_l1_load + $l1_cache_load" | bc -l)
          total_l1_miss=$(echo "$total_l1_miss + $l1_cache_miss" | bc -l)
          total_ll_load=$(echo "$total_ll_load + $ll_cache_load" | bc -l)
          total_ll_miss=$(echo "$total_ll_miss + $ll_cache_miss" | bc -l)
            
        done
        
        #average data
        avg_l1_load=$(printf "%.2f" "$(echo "$total_l1_load / $REPEAT" | bc -l)")
        avg_l1_miss=$(printf "%.2f" "$(echo "$total_l1_miss / $REPEAT" | bc -l)")
        avg_ll_load=$(printf "%.2f" "$(echo "$total_ll_load / $REPEAT" | bc -l)")
        avg_ll_miss=$(printf "%.2f" "$(echo "$total_ll_miss / $REPEAT" | bc -l)")
        
        
        avg_l1_miss=${avg_l1_miss:-0}
        avg_l1_load=${avg_l1_load:-0}
        avg_ll_miss=${avg_ll_miss:-0}
        avg_ll_load=${avg_ll_load:-0}
        
    
         # calculate miss rate
        if (( $(echo "$avg_l1_load > 0" | bc -l) )); then
            l1_miss_rate=$(printf "%.2f" "$(echo "($avg_l1_miss / $avg_l1_load)*100" | bc -l)")
        else
            l1_miss_rate=0
        fi
        
        if (( $(echo "$avg_ll_load > 0" | bc -l) )); then
            ll_miss_rate=$(printf "%.2f" "$(echo "($avg_ll_miss / $avg_ll_load)*100" | bc -l)")
        else
            ll_miss_rate=0
        fi
        
        
        #save on csv
        echo "$avg_l1_load, $avg_l1_miss, $l1_miss_rate, $avg_ll_load, $avg_ll_miss, $ll_miss_rate" >> "$CACHE_OUTFILE"
}


for MATRIX_FILE in "${MATRICES[@]}"; do

echo "Processing matrix: $MATRIX_FILE"

#-------------------------------------------
#SEQUENTIAL CODE

g++-9.1.0 -Wall -g -O3 -std=c++17 $REPO_ROOT/src/deliverable1.cpp -o $REPO_ROOT/src/deliverable1
result=$($REPO_ROOT/src/deliverable1 "$MATRIX_FILE" none 0 $NUM_RUNS) #no schedule and chuck size
time_val_serial=$(echo "$result" | grep "P90" | awk '{printf "%f", $2}')


echo "$(basename "$MATRIX_FILE"), Sequential, NaN, NaN, NaN, $time_val_serial, NaN, NaN">> "$TIME_OUTFILE"
echo -n "$(basename "$MATRIX_FILE"), Sequential, NaN, NaN, NaN, ">> "$CACHE_OUTFILE"
calculate_cache

#------------------------------------------


#------------------------------------------
#PARALLEL CODE
g++-9.1.0 -Wall -g -O3 -fopenmp -std=c++17 $REPO_ROOT/src/deliverable1.cpp -o $REPO_ROOT/src/deliverable1


for t in "${THREADS[@]}"; do
  export OMP_NUM_THREADS=$t
  
  for s in "${SCHEDULES[@]}"; do
    for c in "${CHUNKS[@]}"; do
      
    result=$($REPO_ROOT/src/deliverable1 "$MATRIX_FILE" $s $c $NUM_RUNS)
    time_val_parallel=$(echo "$result" | grep "P90" | awk '{printf "%f", $2}')
    p90_speed_up=$(printf "%.2f" "$(echo "$time_val_serial / $time_val_parallel" | bc -l)")
    p90_efficiency=$(printf "%.2f" "$(echo "($p90_speed_up / $t)*100" | bc -l)") #speed_up/num_threads
    
    
    time_val_parallel=${time_val_parallel:-0} #if time_val is empty put equal to 0
    p90_speed_up=${p90_speed_up:-0}
    p90_efficiency=${p90_efficiency:-0}
 
    echo "$(basename "$MATRIX_FILE"), Parallel, $t, $s, $c, $time_val_parallel, $p90_speed_up, $p90_efficiency" >> "$TIME_OUTFILE"

    done
  done
done


for t in "${THREADS[@]}"; do
  export OMP_NUM_THREADS=$t
  
  for s in "${SCHEDULES[@]}"; do
    for c in "${CHUNKS[@]}"; do
        
        echo -n "$(basename "$MATRIX_FILE"), Parallel, $t, $s, $c, " >> "$CACHE_OUTFILE"
        calculate_cache 
                
    done
  done
done
done