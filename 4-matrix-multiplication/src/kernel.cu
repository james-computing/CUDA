#include <stdio.h>

// Given an m x n matrix encoded as a 1D array A, compute the index of the array corresponding to row i and column j.
// A is column major.
__device__ uint lexicographicIndex(uint const m, uint const i, uint const j) {
    return m * j + i;
}

__global__ void multiply(
    uint const m,
    uint const n,
    uint const p,
    int const * const a,
    int const * const b,
    int * const result
) {
    // Indices for the entry of result matrix that will be computed
    uint const i {blockDim.x * blockIdx.x + threadIdx.x};
    uint const j {blockDim.y * blockIdx.y + threadIdx.y};

    if (i < m && j < p) {
        int result_ij {0};
        for (uint k {0}; k < n; ++k) {
            uint const ik {lexicographicIndex(m, i, k)};
            uint const kj {lexicographicIndex(n, k, j)};
            result_ij += a[ik] * b[kj];
        }
        uint const ij {lexicographicIndex(m, i, j)};
        result[ij] = result_ij;
    }
}