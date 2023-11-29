#include <stdio.h>
#include <omp.h>

int main() {
    int max_threads = omp_get_max_threads();
    printf("Maximum number of threads for next parallel region: %d\n", max_threads);
    
    // Parallel region or other code...
    return 0;
}