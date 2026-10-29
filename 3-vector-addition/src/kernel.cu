#include <stdio.h>

__global__ void add(int * a, int * b, int * result) {
    uint const i {blockDim.x * blockIdx.x + threadIdx.x};
    result[i] = a[i] + b[i];
}