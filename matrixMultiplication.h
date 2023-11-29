#ifndef MATRIX_MULTIPLY_H
#define MATRIX_MULTIPLY_H

void matrixMultiplySerial(int **a, int **b, int **c, int m, int n, int p);
void matrixMultiplySerial2(int **a, int **b, int **c, int m, int n, int p);
void matrixMultiplyParallel(int **a, int **b, int **c, int m, int n, int p);
void matrixMultiplyParallelBlock(int **a, int **b, int **c, int m, int n, int p, int block_size);
void matrixMultiplyParallelTranspose(int **a, int **b, int **c, int m, int n, int p);
void matrixMultiplyParallelBlockLoopUnrolling(int **a, int **b, int **c, int m, int n, int p, int block_size); 
#endif
