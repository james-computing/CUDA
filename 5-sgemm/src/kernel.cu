#include "../include/kernel.cuh"

#include <stdio.h>

// Each kernel computer alpha*A*B + beta*C

// A mxk
// B kxn
// C mxn

// Index of a 1D array that corresponds to row i and column j, for a column major matrix.
__device__ uint lexicographicIndex(uint const numCols, uint const i, uint const j) {
    return numCols * j + i;
}

__global__ void kernel1(
    uint const m,
    uint const n,
    uint const k,
    float const alpha,
    float const * const A,
    float const * const B,
    float const beta,
    float * const C
) {
    uint const i {blockDim.x * blockIdx.x + threadIdx.x};
    uint const j {blockDim.y * blockIdx.y + threadIdx.y};

    if (i < m && j < n) {
        float AB_ij {0};
        for (uint l {0}; l < k; ++l) {
            float const A_il {A[lexicographicIndex(m, i, l)]};
            float const B_lj {B[lexicographicIndex(k, l, j)]};
            AB_ij += A_il * B_lj;
        }
        uint const ij {lexicographicIndex(m, i, j)};
        C[ij] = alpha * AB_ij + beta * C[ij];
    }
};