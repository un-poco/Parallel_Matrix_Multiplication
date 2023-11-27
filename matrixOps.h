#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

int** allocateMatrix(int rows, int cols);
void initializeMatrix(int **matrix, int rows, int cols);
void setMatrixZeros(int **matrix, int rows, int cols);
void printMatrix(int **matrix, int rows, int cols);

#endif