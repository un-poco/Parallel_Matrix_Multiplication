#include <omp.h>
#include <stdlib.h>
#include "matrixMultiplication.h"

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
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < p; j++) {
                c[i][j] += a[i][k] * b[k][j];
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

// Block Matrix Multiplication Algorithm
void matrixMultiplyParallelBlock(int **a, int **b, int **c, int m, int n, int p, int block_size) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i += block_size) {
        for (int j = 0; j < p; j += block_size) {
            for (int ii = i; ii < i + block_size && ii < m; ++ii) {
                for (int k = 0; k < n; ++k) {
                    for (int jj = j; jj < j + block_size && jj < p; ++jj) {
                        c[ii][jj] += a[ii][k] * b[k][jj];
                    }
                }
            }
        }
    }

    // block_size=4;
    // #pragma omp parallel for collapse(2)
    // for (int i = 0; i < m; i += block_size) {
    //     for (int j = 0; j < p; j += block_size) {
    //         for (int ii = i; ii < i + block_size && ii < m; ++ii) {
    //             for (int k = 0; k < n; ++k) {
    //                 // for (int jj = j; jj < j + block_size && jj < p; ++jj) {
    //                 //     c[ii][jj] += a[ii][k] * b[k][jj];
    //                 // }
    //                 c[ii][j] += a[ii][k] * b[k][j];
    //                 c[ii][j+1] += a[ii][k] * b[k][j+1];
    //                 c[ii][j+2] += a[ii][k] * b[k][j+2];
    //                 c[ii][j+3] += a[ii][k] * b[k][j+3];
    //             }
    //         }
    //     }
    // }
}

// Function to allocate memory for a matrix
int** allocateMatrix(int rows, int cols) {
    int** matrix = (int**)malloc(rows * sizeof(int*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (int*)malloc(cols * sizeof(int));
    }
    return matrix;
}