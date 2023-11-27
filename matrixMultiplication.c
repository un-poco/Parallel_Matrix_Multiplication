#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "strassen.h"


#define BLOCK_SIZE 64

// Serial matrix multiplication
void matrixMultiplySerial(int **a, int **b, int **c, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < n; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

// Serial matrix multiplication
void matrixMultiplySerial2(int **a, int **b, int **c, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < p; j++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

// Parallel matrix multiplication using OpenMP
void matrixMultiplyParallel(int **a, int **b, int **c, int m, int n, int p) {
    #pragma omp parallel for collapse(2) 
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < p; j++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

// Block Matrix Multiplication Algorithm
void matrixMultiplyParallelBlock(int **a, int **b, int **c, int m, int n, int p) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i += BLOCK_SIZE) {
        for (int j = 0; j < p; j += BLOCK_SIZE) {
            for (int ii = i; ii < i + BLOCK_SIZE && ii < m; ++ii) {
                for (int k = 0; k < n; ++k) {
                    for (int jj = j; jj < j + BLOCK_SIZE && jj < p; ++jj) {
                        c[ii][jj] += a[ii][k] * b[k][jj];
                    }
                }
            }
        }
    }
}

int** transposeMatrix(int **matrix, int rows, int cols) {
    int **transposed = (int **)malloc(cols * sizeof(int *));
    #pragma omp parallel for
    for (int i = 0; i < cols; ++i) {
        transposed[i] = (int *)malloc(rows * sizeof(int));
        for (int j = 0; j < rows; ++j) {
            transposed[i][j] = matrix[j][i];
        }
    }
    return transposed;
}


void matrixMultiplyParallelTranspose(int **a, int **b, int **c, int m, int n, int p) {
    int **bTransposed = transposeMatrix(b, n, p);
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++)  {
            for (int k = 0; k < n; k++) {
                c[i][j] += a[i][k] * bTransposed[j][k]; // Use transposed B
            }
        }
    }

    // Free the transposed matrix after use
    for (int i = 0; i < p; ++i) {
        free(bTransposed[i]);
    }
    free(bTransposed);
}


// Function to allocate memory for a matrix
int** allocateMatrix(int rows, int cols) {
    int** matrix = (int**)malloc(rows * sizeof(int*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (int*)malloc(cols * sizeof(int));
    }
    return matrix;
}

// Function to initialize a matrix with random values
void initializeMatrix(int **matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = rand() % 10;  // Initialize with random numbers between 0 and 9
        }
    }
}

// Function to set a matrix with all zeros
void setMatrixZeros(int **matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = 0;
        }
    }
}

// Function to print a matrix
void printMatrix(int **matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <m> <n> <p>\n", argv[0]);
        return 1;
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int p = atoi(argv[3]);

    if (m <= 0 || n <= 0 || p <= 0) {
        fprintf(stderr, "Matrix dimensions must be positive integers\n");
        return 1;
    }

    int** a = allocateMatrix(m, n); // Matrix A of size m x n
    int** b = allocateMatrix(n, p); // Matrix B of size n x p
    int** c = allocateMatrix(m, p); // Result matrix C of size m x p

    // Initialize matrices
    initializeMatrix(a, m, n);
    initializeMatrix(b, n, p);
    setMatrixZeros(c, m, p);


    double start, end;
    /*
    // Perform serial matrix multiplication
    start = omp_get_wtime();
    matrixMultiplySerial(a, b, c, m, n, p);
    end = omp_get_wtime();
    printf("Serial matrix multiplication took %f seconds.\n", end - start);
    // printf("\nResult of Serial Multiplication:\n");
    // printMatrix(c, m, p);

    // Perform updated version of serial matrix multiplication
    start = omp_get_wtime();
    matrixMultiplySerial2(a, b, c, m, n, p);
    end = omp_get_wtime();
    printf("Updated serial matrix multiplication took %f seconds.\n", end - start);
    */

    // parallel matrix multiplication
    setMatrixZeros(c, m, p); // reset all elements in result matrix to zeros
    start = omp_get_wtime();
    matrixMultiplyParallel(a, b, c, m, n, p);
    end = omp_get_wtime();
    printf("Parallel matrix multiplication took %f seconds.\n", end - start);

    // block matrix multiplication
    setMatrixZeros(c, m, p); // reset all elements in result matrix to zeros
    start = omp_get_wtime();
    matrixMultiplyParallelBlock(a, b, c, m, n, p);
    end = omp_get_wtime();
    printf("Parallel block matrix multiplication took %f seconds.\n", end - start);

    // transpose(not helpful)
    start = omp_get_wtime();
    matrixMultiplyParallelTranspose(a, b, c, m, n, p);
    end = omp_get_wtime();
    printf("Parallel transpose matrix multiplication took %f seconds.\n", end - start);

    // strassen matrix multiplication(only m=n=p=2^k)
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
    

    // Free the allocated memory
    for (int i = 0; i < m; i++) free(a[i]);
    for (int i = 0; i < n; i++) free(b[i]);
    for (int i = 0; i < m; i++) free(c[i]);
    free(a);
    free(b);
    free(c);

    return 0;
}