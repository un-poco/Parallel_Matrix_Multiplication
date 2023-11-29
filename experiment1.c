#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "matrixOps.h"
#include "matrixMultiplication.h"
#include "strassen.h"

int main(int argc, char *argv[]) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <m> <n> <p> <block size>\n", argv[0]);
        return 1;
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int p = atoi(argv[3]);
    int block_size = atoi(argv[4]);

    if (m <= 0 || n <= 0 || p <= 0) {
        fprintf(stderr, "Matrix dimensions must be positive integers\n");
        return 1;
    }

    if (block_size <= 1) {
        fprintf(stderr, "Block size must be a positive integer");
    }

    int** a = allocateMatrix(m, n); // Matrix A of size m x n
    int** b = allocateMatrix(n, p); // Matrix B of size n x p
    int** c = allocateMatrix(m, p); // Result matrix C of size m x p

    // Initialize matrices
    initializeMatrix(a, m, n);
    initializeMatrix(b, n, p);
    setMatrixZeros(c, m, p);

    // Print initial matrices
    // printf("Matrix A:\n");
    // printMatrix(a, m, n);

    // printf("\nMatrix B:\n");
    // printMatrix(b, n, p);

    double start, end;
   
    // Perform updated version of serial matrix multiplication
    start = omp_get_wtime();
    matrixMultiplySerial2(a, b, c, m, n, p);
    end = omp_get_wtime();
    printf("Serial matrix multiplication took %f seconds.\n", end - start);
    
    // Parallel matrix multiplication
    setMatrixZeros(c, m, p); // reset all elements in result matrix to zeros
    start = omp_get_wtime();
    matrixMultiplyParallel(a, b, c, m, n, p);
    end = omp_get_wtime();
    printf("Parallel matrix multiplication took %f seconds.\n", end - start);

    // // Block matrix multiplication
    // setMatrixZeros(c, m, p); // reset all elements in result matrix to zeros
    // start = omp_get_wtime();
    // matrixMultiplyParallelBlock(a, b, c, m, n, p, block_size);
    // end = omp_get_wtime();
    // printf("Parallel block matrix multiplication took %f seconds.\n", end - start);

    // Block matrix loop Unrolling multiplication
    setMatrixZeros(c, m, p); // reset all elements in result matrix to zeros
    start = omp_get_wtime();
    matrixMultiplyParallelBlockLoopUnrolling(a, b, c, m, n, p, block_size);
    end = omp_get_wtime();
    printf("Parallel block matrix multiplication LoopUnrolling took %f seconds.\n", end - start);
    
    // Transpose
    setMatrixZeros(c, m, p);
    start = omp_get_wtime();
    matrixMultiplyParallelTranspose(a, b, c, m, n, p);
    end = omp_get_wtime();
    printf("Parallel transpose matrix multiplication took %f seconds.\n", end - start);

    // Strassen matrix multiplication(only m=n=p=2^k)
    if(n==m && m==p && (m & (m - 1)) == 0){
        start = omp_get_wtime();
        omp_set_num_threads(8);
        #pragma omp parallel
        {
        #pragma omp single
            {
                strassen(n, a, b);
            }
        }
        end = omp_get_wtime();
        printf("Parallel strassen matrix multiplication took %f seconds.\n", end - start);
    }

    // Free the allocated memory
    for (int i = 0; i < m; i++) free(a[i]);
    for (int i = 0; i < n; i++) free(b[i]);
    for (int i = 0; i < m; i++) free(c[i]);
    free(a);
    free(b);
    free(c);

    return 0;
}