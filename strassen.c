#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "strassen.h"

// Function to print a matrix
void print(int n, int** mat) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", mat[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Function to allocate memory for a matrix
int** allocateMatrix2(int n) {
    int* data = (int*)malloc(n * n * sizeof(int));
    int** array = (int**)malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++) {
        array[i] = &(data[n * i]);
    }
    return array;
}

// Function to fill a matrix with random numbers
void fillMatrix(int n, int** mat) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mat[i][j] = rand() % 5;
        }
    }
}

// Function to free memory allocated for a matrix
void freeMatrix(int n, int** mat) {
    free(mat[0]);
    free(mat);
}

// Naive matrix multiplication function
int** naive(int n, int** mat1, int** mat2) {
    int** prod = allocateMatrix2(n);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            prod[i][j] = 0;
            for (int k = 0; k < n; k++) {
                prod[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }

    return prod;
}

// Function to get a submatrix (slice) from a matrix
int** getSlice(int n, int** mat, int offseti, int offsetj) {
    int m = n / 2;
    int** slice = allocateMatrix2(m);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            slice[i][j] = mat[offseti + i][offsetj + j];
        }
    }
    return slice;
}

// Function to add or subtract two matrices
int** addMatrices(int n, int** mat1, int** mat2, int add) {
    int** result = allocateMatrix2(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i][j] = mat1[i][j] + (add ? mat2[i][j] : -mat2[i][j]);
        }
    }

    return result;
}

// Function to combine four submatrices into one matrix
int** combineMatrices(int m, int** c11, int** c12, int** c21, int** c22) {
    int n = 2 * m;
    int** result = allocateMatrix2(n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i < m && j < m) {
                result[i][j] = c11[i][j];
            } else if (i < m) {
                result[i][j] = c12[i][j - m];
            } else if (j < m) {
                result[i][j] = c21[i - m][j];
            } else {
                result[i][j] = c22[i - m][j - m];
            }
        }
    }

    return result;
}

int** strassen(int n, int** mat1, int** mat2)
{

    if (n <= 32)
    {
        return naive(n, mat1, mat2);
    }

    int m = n / 2;

    int** a = getSlice(n, mat1, 0, 0);
    int** b = getSlice(n, mat1, 0, m);
    int** c = getSlice(n, mat1, m, 0);
    int** d = getSlice(n, mat1, m, m);
    int** e = getSlice(n, mat2, 0, 0);
    int** f = getSlice(n, mat2, 0, m);
    int** g = getSlice(n, mat2, m, 0);
    int** h = getSlice(n, mat2, m, m);

    int** s1;
    #pragma omp task shared(s1)
    {
        int** bds = addMatrices(m, b, d, 0);
        int** gha = addMatrices(m, g, h, 1);
        s1 = strassen(m, bds, gha);
        freeMatrix(m, bds);
        freeMatrix(m, gha);
    }

    int** s2;
    #pragma omp task shared(s2)
    {
        int** ada = addMatrices(m, a, d, 1);
        int** eha = addMatrices(m, e, h, 1);
        s2 = strassen(m, ada, eha);
        freeMatrix(m, ada);
        freeMatrix(m, eha);
    }

    int** s3;
    #pragma omp task shared(s3)
    {
        int** acs = addMatrices(m, a, c, 0);
        int** efa = addMatrices(m, e, f, 1);
        s3 = strassen(m, acs, efa);
        freeMatrix(m, acs);
        freeMatrix(m, efa);
    }

    int** s4;
    #pragma omp task shared(s4)
    {
        int** aba = addMatrices(m, a, b, 1);
        s4 = strassen(m, aba, h);
        freeMatrix(m, aba);
    }

    int** s5;
    #pragma omp task shared(s5)
    {
        int** fhs = addMatrices(m, f, h, 0);
        s5 = strassen(m, a, fhs);
        freeMatrix(m, fhs);
    }

    int** s6;
    #pragma omp task shared(s6)
    {
        int** ges = addMatrices(m, g, e, 0);
        s6 = strassen(m, d, ges);
        freeMatrix(m, ges);
    }

    int** s7;
    #pragma omp task shared(s7)
    {
        int** cda = addMatrices(m, c, d, 1);
        s7 = strassen(m, cda, e);
        freeMatrix(m, cda);
    }

    #pragma omp taskwait

    freeMatrix(m, a);
    freeMatrix(m, b);
    freeMatrix(m, c);
    freeMatrix(m, d);
    freeMatrix(m, e);
    freeMatrix(m, f);
    freeMatrix(m, g);
    freeMatrix(m, h);

    int** c11;
    #pragma omp task shared(c11)
    {
        int** s1s2a = addMatrices(m, s1, s2, 1);
        int** s6s4s = addMatrices(m, s6, s4, 0);
        c11 = addMatrices(m, s1s2a, s6s4s, 1);
        freeMatrix(m, s1s2a);
        freeMatrix(m, s6s4s);
    }

    int** c12;
    #pragma omp task shared(c12)
    {
        c12 = addMatrices(m, s4, s5, 1);
    }

    int** c21;
    #pragma omp task shared(c21)
    {
        c21 = addMatrices(m, s6, s7, 1);
    }

    int** c22;
    #pragma omp task shared(c22)
    {
        int** s2s3s = addMatrices(m, s2, s3, 0);
        int** s5s7s = addMatrices(m, s5, s7, 0);
        c22 = addMatrices(m, s2s3s, s5s7s, 1);
        freeMatrix(m, s2s3s);
        freeMatrix(m, s5s7s);
    }

    #pragma omp taskwait

    freeMatrix(m, s1);
    freeMatrix(m, s2);
    freeMatrix(m, s3);
    freeMatrix(m, s4);
    freeMatrix(m, s5);
    freeMatrix(m, s6);
    freeMatrix(m, s7);

    int** prod = combineMatrices(m, c11, c12, c21, c22);

    freeMatrix(m, c11);
    freeMatrix(m, c12);
    freeMatrix(m, c21);
    freeMatrix(m, c22);

    return prod;
}

int check(int n, int** prod1, int** prod2)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (prod1[i][j] != prod2[i][j])
                return 0;
        }
    }
    return 1;
}

// Serial matrix multiplication
void matrixMultiplySerial3(int **a, int **b, int **c, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < p; j++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

// int main(int argc, char *argv[]) 
int main2(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <n>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);

    int** mat1 = allocateMatrix2(n);
    fillMatrix(n, mat1);

    int** mat2 = allocateMatrix2(n);
    fillMatrix(n, mat2);

    double startParStrassen = omp_get_wtime();
    int** prod;

    omp_set_num_threads(8);

    #pragma omp parallel
    {
    #pragma omp single
        {
            prod = strassen(n, mat1, mat2);
        }
    }
    double endParStrassen = omp_get_wtime();
    printf("\nParallel Strassen Runtime (OMP): %f\n", endParStrassen - startParStrassen);

    //serial version to check result
    int** serialProd = allocateMatrix2(n);
    double startSerial = omp_get_wtime();
    matrixMultiplySerial3(mat1, mat2, serialProd, n, n, n);
    double endSerial = omp_get_wtime();
    printf("\nSerial Multiplication Runtime: %f\n", endSerial - startSerial);

    // Check if the results are the same
    int isSame = check(n, prod, serialProd);
    if (isSame) {
        printf("\nThe results of Strassen and Serial Multiplication are the same.\n");
    } else {
        printf("\nThe results of Strassen and Serial Multiplication are different.\n");
    }

    freeMatrix(n, mat1);
    freeMatrix(n, mat2);
    freeMatrix(n, prod);
    freeMatrix(n, serialProd);


    printf("\n");

    return 0;
}

