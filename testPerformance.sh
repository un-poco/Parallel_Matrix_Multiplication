#!/bin/bash

gcc -Wall -std=c99 -fopenmp -o matrixMultiplication main.c matrixOps.c matrixMultiplication.c strassen.c

# Define the combinations of m, n, p as array of arrays
declare -a small_combinations
small_combinations=(
    # Small matrices
    "100 100 100"
    "200 50 200"
    "50 200 50"
)

declare -a big_combinations
big_combinations=(
    # Big matrices
    "2000 2000 2000"
    "1000 2000 4000"
    "1000 4000 2000"
    "2000 1000 4000"
    "2000 4000 1000"
    "4000 1000 2000"
    "4000 2000 1000"
    # Special Matrices (for deep learning)
    "1 2048 1000"
    "512 768 3072"
    # Add more combinations as needed
)

# Define block sizes
block_size_values=(16 32 64)
# block_size_values=(4 8 16 32)

# Number of runs
runs=5

# Loop over each combination of small size m, n, p. Run multiple times for more precise results.
for combination in "${small_combinations[@]}"; do
    read -r m n p <<< "$combination"

    for block_size in "${block_size_values[@]}"; do
        # Initialize total time variables
        total_time_parallel=0
        total_time_block=0
        total_time_strassen=0
        total_time_transpose=0

        for i in $(seq 1 $runs); do
            # Run the command and capture the output
            output=$(./matrixMultiplication $m $n $p $block_size)

            # Extract times and add to total
            time_parallel=$(echo "$output" | grep "Parallel matrix multiplication took" | awk '{print $5}')
            total_time_parallel=$(echo "$total_time_parallel + $time_parallel" | bc)
            
            time_block=$(echo "$output" | grep "Parallel block matrix multiplication took" | awk '{print $6}')
            total_time_block=$(echo "$total_time_block + $time_block" | bc)
            
            time_transpose=$(echo "$output" | grep "Parallel transpose matrix multiplication took" | awk '{print $6}')
            total_time_transpose=$(echo "$total_time_transpose + $time_transpose" | bc)

            # time_strassen=$(echo "$output" | grep "Parallel strassen matrix multiplication took" | awk '{print $6}')
            # total_time_strassen=$(echo "$total_time_strassen + $time_strassen" | bc)
        done

        # Calculate average
        avg_time_parallel=$(echo "scale=6; $total_time_parallel / $runs" | bc)
        avg_time_block=$(echo "scale=6; $total_time_block / $runs" | bc)
        avg_time_transpose=$(echo "scale=6; $total_time_transpose / $runs" | bc)
        # avg_time_strassen=$(echo "scale=6; $total_time_strassen / $runs" | bc)

        # Print average times
        echo "For m=$m, n=$n, p=$p, block_size=$block_size:"
        echo "Average time for parallel matrix multiplication: $avg_time_parallel seconds"
        echo "Average time for parallel block matrix multiplication: $avg_time_block seconds"
        echo "Average time for parallel transpose matrix multiplication: $avg_time_transpose seconds"
        # echo "Average time for parallel strassen matrix multiplication: $avg_time_strassen seconds"
    done
done

# Loop over each combination of big size m, n, p
for combination in "${big_combinations[@]}"; do
    read -r m n p <<< "$combination"

    for block_size in "${block_size_values[@]}"; do
        # Initialize total time variables
        total_time_parallel=0
        total_time_block=0
        total_time_strassen=0
        total_time_transpose=0

        # Run the command and capture the output
        output=$(./matrixMultiplication $m $n $p $block_size)

        # Extract times and add to total
        time_serial=$(echo "$output" | grep "Updated serial matrix multiplication took" | awk '{print $5}')

        time_parallel=$(echo "$output" | grep "Parallel matrix multiplication took" | awk '{print $5}')
        
        time_block=$(echo "$output" | grep "Parallel block matrix multiplication took" | awk '{print $6}')
        
        time_transpose=$(echo "$output" | grep "Parallel transpose matrix multiplication took" | awk '{print $6}')

        # time_strassen=$(echo "$output" | grep "Parallel strassen matrix multiplication took" | awk '{print $6}')
        # total_time_strassen=$(echo "$total_time_strassen + $time_strassen" | bc)

        # Print average times
        echo "For m=$m, n=$n, p=$p, block_size=$block_size:"
        echo "Time for serial matrix multiplication: $time_serial seconds"
        echo "Time for parallel matrix multiplication: $time_parallel seconds"
        echo "Time for parallel block matrix multiplication: $time_block seconds"
        echo "Time for parallel transpose matrix multiplication: $time_transpose seconds"
        # echo "Average time for parallel strassen matrix multiplication: $avg_time_strassen seconds"
    done
done
