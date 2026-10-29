#include <stdio.h>

__global__ void f(void) {

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
            printf("Kernel executed\n");
            printf("Grid dims = (%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);
        }

        printf("Block = (%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z);
    }
    
    printf("Thread = (%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
}