#include "../include/kernel1.cuh"
#include "../include/lexicographicIndex.cuh"

// Each kernel computer alpha*A*B + beta*C

// A mxk
// B kxn
// C mxn

// kernel1 uses a single thread for each entry C_ij of C.
// It suffers from latency, each thread loads a row of A and a column of B from L2 (VRAM).

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