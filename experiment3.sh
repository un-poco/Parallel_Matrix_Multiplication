#!/bin/bash

gcc -Wall -std=c99 -fopenmp -o experiment3 experiment3.c matrixOps.c matrixMultiplication.c strassen.c -O2

# Define the combinations of m, n, p as array of arrays
declare -a combinations
combinations=(
    # "512 512 512"
    # "2048 2048 2048"
    # "4096 4096 4096"
    "8192 8192 8192"
)

# Define block sizes
# block_size_values=(16 32 64)
block_size_values=(4)
# block_size_values=(4 8 16 32 64)

# Number of runs
runs=5

# Loop over each combination of m, n, p
for combination in "${combinations[@]}"; do
    read -r m n p <<< "$combination"

    for block_size in "${block_size_values[@]}"; do
        # Initialize total time variables
        total_time_parallel=0
        total_time_strassen=0

        for i in $(seq 1 $runs); do
            # Run the command and capture the output
            output=$(./experiment3 $m $n $p $block_size)

            # Extract times and add to total
            time_parallel=$(echo "$output" | grep "Parallel matrix multiplication took" | awk '{print $5}')
            total_time_parallel=$(echo "$total_time_parallel + $time_parallel" | bc)

            time_strassen=$(echo "$output" | grep "Parallel strassen matrix multiplication took" | awk '{print $6}')
            total_time_strassen=$(echo "$total_time_strassen + $time_strassen" | bc)
        done

        # Calculate average
        avg_time_parallel=$(echo "scale=6; $total_time_parallel / $runs" | bc)
        avg_time_strassen=$(echo "scale=6; $total_time_strassen / $runs" | bc)

        # Print average times
        echo "For m=$m, n=$n, p=$p"
        echo "Average time for parallel matrix multiplication: $avg_time_parallel seconds"
        echo "Average time for parallel strassen matrix multiplication: $avg_time_strassen seconds"
        echo ""
    done
done