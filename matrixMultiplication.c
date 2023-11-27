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
}

// Function to allocate memory for a matrix
int** allocateMatrix(int rows, int cols) {
    int** matrix = (int**)malloc(rows * sizeof(int*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (int*)malloc(cols * sizeof(int));
    }
    return matrix;
}