#include "../include/kernel2.cuh"
#include "../include/lexicographicIndex.cuh"

// Each kernel computer alpha*A*B + beta*C

// A mxk
// B kxn
// C mxn

// In kernel1 each entry C_ij of C was computed by a different thread.
// The problem with this approach is that it requires too much communication and each
// thread receives little work. Now I'll use a single thread to compute an entire block of C.

// It actually gave worse performance than kernel1.
// I think the problem is that using a thread for block serializes the computation.
// In any case, using blocks will be important for working with whole cache lines.
// This then points to using blocks for loading memory into shared memory,
// but keep using multiple threads for block.

__global__ void kernel2(
    uint const m,
    uint const n,
    uint const k,
    float const alpha,
    float const * const A,
    float const * const B,
    float const beta,
    float * const C
) {
    // We'll compute a block C_IJ of C.
    // (I,J) is indexing the block, not a single float of C.
    uint const I {blockDim.x * blockIdx.x + threadIdx.x};
    uint const J {blockDim.y * blockIdx.y + threadIdx.y};

    // Do the block multiplication
    for (uint i_block {0}; i_block < BLOCKSIZE_KERNEL2; ++i_block) {
        uint const i {I * BLOCKSIZE_KERNEL2 + i_block};

        if (i >= m) {
            break;
        }

        for (uint j_block {0}; j_block < BLOCKSIZE_KERNEL2; ++j_block) {
            uint const j {J * BLOCKSIZE_KERNEL2 + j_block};

            if (j >= n) {
                break;
            }

            // Compute C_ij
            float AB_ij {0};
            for (uint l {0}; l < k; ++l) {
                float A_il {A[lexicographicIndex(m, i, l)]};
                float B_lj {B[lexicographicIndex(k, l, j)]};
                AB_ij += A_il * B_lj;
            }

            uint const ij {lexicographicIndex(m, i, j)};
            C[ij] = alpha * AB_ij + beta * C[ij];
        }
    }
};