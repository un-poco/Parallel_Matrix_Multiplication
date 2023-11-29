#!/bin/bash

gcc -Wall -std=c99 -fopenmp -o matrixMultiplication experiment1.c matrixOps.c matrixMultiplication.c strassen.c -O2

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
block_size_values=(4)
# block_size_values=(4 8 16 32 64)

# Number of runs
runs=80

# Loop over each combination of m, n, p
for combination in "${combinations[@]}"; do
    read -r m n p <<< "$combination"

    for block_size in "${block_size_values[@]}"; do
        # Initialize total time variables
        total_time_serial=0
        total_time_parallel=0
        total_time_LoopUnrolling=0
        total_time_transpose=0

        for i in $(seq 1 $runs); do
            # Run the command and capture the output
            output=$(./matrixMultiplication $m $n $p $block_size)

            time_serial=$(echo "$output" | grep "Serial matrix multiplication took" | awk '{print $5}')
            total_time_serial=$(echo "$total_time_serial + $time_serial" | bc)

            # Extract times and add to total
            time_parallel=$(echo "$output" | grep "Parallel matrix multiplication took" | awk '{print $5}')
            total_time_parallel=$(echo "$total_time_parallel + $time_parallel" | bc)

            time_block_LoopUnrolling=$(echo "$output" | grep "Parallel block matrix multiplication LoopUnrolling took" | awk '{print $7}')
            total_time_LoopUnrolling=$(echo "$total_time_LoopUnrolling + $time_block_LoopUnrolling" | bc)

            time_transpose=$(echo "$output" | grep "Parallel transpose matrix multiplication took" | awk '{print $6}')
            total_time_transpose=$(echo "$total_time_transpose + $time_transpose" | bc)

            # time_strassen=$(echo "$output" | grep "Parallel strassen matrix multiplication took" | awk '{print $6}')
            # total_time_strassen=$(echo "$total_time_strassen + $time_strassen" | bc)
        done

        # Calculate average
        avg_time_serial=$(echo "scale=6; $total_time_serial / $runs" | bc)
        avg_time_parallel=$(echo "scale=6; $total_time_parallel / $runs" | bc)
        # avg_time_block=$(echo "scale=6; $total_time_block / $runs" | bc)
        avg_time_LoopUnrolling=$(echo "scale=6; $total_time_LoopUnrolling / $runs" | bc)
        avg_time_transpose=$(echo "scale=6; $total_time_transpose / $runs" | bc)
        # avg_time_strassen=$(echo "scale=6; $total_time_strassen / $runs" | bc)

        # Print average times
        echo "For m=$m, n=$n, p=$p"
        echo "Average time for serial matrix multiplication: $avg_time_serial seconds"
        echo "Average time for parallel matrix multiplication: $avg_time_parallel seconds"
        # echo "Average time for parallel block matrix multiplication: $avg_time_block seconds"
        echo "Average time for parallel block matrix multiplication LoopUnrolling: $avg_time_LoopUnrolling seconds"
        echo "Average time for parallel transpose matrix multiplication: $avg_time_transpose seconds"
        # echo "Average time for parallel strassen matrix multiplication: $avg_time_strassen seconds"
        echo ""
    done
done