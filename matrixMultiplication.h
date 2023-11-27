#ifndef MATRIX_MULTIPLY_H
#define MATRIX_MULTIPLY_H

#define BLOCK_SIZE 16

void matrixMultiplySerial(int **a, int **b, int **c, int m, int n, int p);
void matrixMultiplySerial2(int **a, int **b, int **c, int m, int n, int p);
void matrixMultiplyParallel(int **a, int **b, int **c, int m, int n, int p);
void matrixMultiplyParallelBlock(int **a, int **b, int **c, int m, int n, int p);

#endif
