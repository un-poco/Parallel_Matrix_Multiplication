#!/bin/bash

# gcc -Wall -std=c99 -fopenmp -O2 -o mm matrixMultiplication.c strassen.c
gcc -Wall -std=c99 -fopenmp -O3 -ftree-vectorize -o mm matrixMultiplication.c strassen.c
# gcc -Wall -std=c99 -O3 -ftree-vectorize -march=native -fopenmp -fopt-info-vec -o mm matrixMultiplication.c strassen.c

# Initialize total time variables
total_time_parallel=0
total_time_block=0
total_time_transpose=0
total_time_strassen=0

# Number of runs
runs=5

for i in $(seq 1 $runs); do
    # Run the command and capture the output
    output=$(./mm 512 512 512)
    # output=$(./mm 4096 4096 4096)

    # Extract times and add to total
    time_parallel=$(echo "$output" | grep "Parallel matrix multiplication took" | awk '{print $5}')
    total_time_parallel=$(echo "$total_time_parallel + $time_parallel" | bc)

    time_block=$(echo "$output" | grep "Parallel block matrix multiplication took" | awk '{print $6}')
    total_time_block=$(echo "$total_time_block + $time_block" | bc)
    
    time_transpose=$(echo "$output" | grep "Parallel transpose matrix multiplication took" | awk '{print $6}')
    total_time_transpose=$(echo "$total_time_transpose + $time_transpose" | bc)

    time_strassen=$(echo "$output" | grep "Parallel strassen matrix multiplication took" | awk '{print $6}')
    total_time_strassen=$(echo "$total_time_strassen + $time_strassen" | bc)
    
done

# Calculate average
avg_time_parallel=$(echo "scale=6; $total_time_parallel / $runs" | bc)
avg_time_block=$(echo "scale=6; $total_time_block / $runs" | bc)
avg_time_transpose=$(echo "scale=6; $total_time_transpose / $runs" | bc)
avg_time_strassen=$(echo "scale=6; $total_time_strassen / $runs" | bc)

# Print average times
echo "Average time for parallel matrix multiplication: $avg_time_parallel seconds"
echo "Average time for parallel block matrix multiplication: $avg_time_block seconds"
echo "Average time for parallel transpose matrix multiplication: $avg_time_transpose seconds"
echo "Average time for parallel strassen matrix multiplication: $avg_time_strassen seconds"
