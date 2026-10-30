#include <stdio.h>

__global__ void add(int * const a, int * const b, int * const result) {
    uint const i {blockDim.x * blockIdx.x + threadIdx.x};
    result[i] = a[i] + b[i];
}