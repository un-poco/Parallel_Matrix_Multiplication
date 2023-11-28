#!/bin/bash

gcc -Wall -std=c99 -fopenmp -o matrixMultiplication main.c matrixOps.c matrixMultiplication.c strassen.c

# Define the combinations of m, n, p as array of arrays
declare -a combinations
combinations=(
    # Small matrices
    "100 100 100"
    "200 50 200"
    "50 200 50"
    # Big matrices
    "2000 2000 2000"
    "1000 4000 1000"
    "4000 2000 1000"
    # Special Matrices (for deep learning)
    "1 2048 1000"
    "512 768 3072"
    # Add more combinations as needed
)

# Define block sizes
block_size_values=(16 32 64)

# Number of runs
runs=5

# Loop over each combination of m, n, p
for combination in "${combinations[@]}"; do
    read -r m n p <<< "$combination"

    for block_size in "${block_size_values[@]}"; do
        # Initialize total time variables
        total_time_parallel=0
        total_time_block=0
        # total_time_transpose=0

        for i in $(seq 1 $runs); do
            # Run the command and capture the output
            output=$(./matrixMultiplication $m $n $p $block_size)

            # Extract times and add to total
            time_parallel=$(echo "$output" | grep "Parallel matrix multiplication took" | awk '{print $5}')
            time_block=$(echo "$output" | grep "Parallel block matrix multiplication took" | awk '{print $6}')
            # time_transpose=$(echo "$output" | grep "Parallel transpose matrix multiplication took" | awk '{print $6}')
            
            total_time_parallel=$(echo "$total_time_parallel + $time_parallel" | bc)
            total_time_block=$(echo "$total_time_block + $time_block" | bc)
            # total_time_transpose=$(echo "$total_time_transpose + $time_transpose" | bc)
        done

        # Calculate average
        avg_time_parallel=$(echo "scale=6; $total_time_parallel / $runs" | bc)
        avg_time_block=$(echo "scale=6; $total_time_block / $runs" | bc)
        # avg_time_transpose=$(echo "scale=6; $total_time_transpose / $runs" | bc)

        # Print average times
        echo "For m=$m, n=$n, p=$p, block_size=$block_size:"
        echo "Average time for parallel matrix multiplication: $avg_time_parallel seconds"
        echo "Average time for parallel block matrix multiplication: $avg_time_block seconds"
        # echo "Average time for parallel transpose matrix multiplication: $avg_time_transpose seconds"
        echo ""
    done
done
