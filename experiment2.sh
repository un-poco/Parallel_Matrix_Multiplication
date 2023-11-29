#!/bin/bash

gcc -Wall -std=c99 -fopenmp -o matrixMultiplication main.c matrixOps.c matrixMultiplication.c strassen.c -O2

# Define the combinations of m, n, p as array of arrays
declare -a combinations
combinations=(
    # Small matrices
    "100 100 100"
    "200 50 100"
    "200 100 50"
    "100 200 50"
    "100 50 200"
    "50 200 100"
    "50 100 200"
)

# Define block sizes
# block_size_values=(16 32 64)
# block_size_values=(4)
block_size_values=(4 8 16 32 64)

# Number of runs
runs=40

# Loop over each combination of m, n, p
for combination in "${combinations[@]}"; do
    read -r m n p <<< "$combination"

    for block_size in "${block_size_values[@]}"; do
        # Initialize total time variables
        total_time_block=0

        for i in $(seq 1 $runs); do
            # Run the command and capture the output
            output=$(./matrixMultiplication $m $n $p $block_size)
            time_block=$(echo "$output" | grep "Parallel block matrix multiplication took" | awk '{print $6}')
            total_time_block=$(echo "$total_time_block + $time_block" | bc)
        done

        # Calculate average
      
        avg_time_block=$(echo "scale=6; $total_time_block / $runs" | bc)
    
        # Print average times
        echo "For m=$m, n=$n, p=$p, block_size=$block_size:"
        echo "Average time for parallel block matrix multiplication: $avg_time_block seconds"
        echo ""
    done
done