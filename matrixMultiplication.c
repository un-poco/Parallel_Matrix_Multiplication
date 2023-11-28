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
        for (int k = 0; k < n; k += block_size) {
            for (int j = 0; j < p; j += block_size) {
                for (int ii = i; ii < i + block_size && ii < m; ++ii) {
                    for (int kk = 0; kk < k + block_size && kk < n; ++kk) {
                        for (int jj = j; jj < j + block_size && jj < p; ++jj) {
                            c[ii][jj] += a[ii][kk] * b[kk][jj];
                        }
                    }
                }
            }
        }
    }
}
