#include "../include/kernel3.cuh"
#include "../include/lexicographicIndex.cuh"
#include "../include/ceildiv.hpp"

// Each kernel computer alpha*A*B + beta*C

// A mxk
// B kxn
// C mxn

// Use shared memory to load entire cache lines. A cache line has size of 128B, while a float is 4B.
// Therefore, a cache line stores 128/4 = 32 floats. We need cache lines for A, which are in its columns.
// But the computation uses the rows of A, so we need to load a block of A. The natural solution would be
// to load at least a 32x32 block of A. These would correspond to 32 cache lines, one for each column of
// the block.
//
// We may have a problem if the number of rows of A isn't a multiple of 32. In this case, a single
// cache line may be storing elements from different columns. A solution would be to alloc matrices whose dimensions
// are multiples of 32, but use a submatrix if necessary.
//
// Since we'll be loading square blocks of A, it makes sense to multiply the matrices by blocks. Use square blocks
// whose sides are a multiple of 32, but small enough to fit in shared memory. My GPU supports at most 64.
// Also load a block of A and of B, of the same size. If BLOCKSIZE is the size (of the side) of the block of A
// to be loaded, the same for B, then we need 2 * BLOCKSIZE * BLOCKSIZE floats to store both blocks.
//
// The same block of A or of B will be loaded repeatedly for different block multiplications. I don't see how
// such loading could be avoided, if we are multiplying huge matrices.
//
// For my GPU
// shared memory = 32KB (default)
// = 32 * 1024 B = 32768B
//
// Each float uses 4B, so we can store up to 32768/4 = 8192 floats = 2 * 64 * 64 floats.
// Therefore, should use BLOCKSIZE = 64 or smaller.
//
// The matrices will be multiplied in blocks, but this doesn't say which thread will do what.
// I'll change the meaning of (i,j) to now indicate a block of C.
// That is, C_ij is now a block of C, not a single float.
// We may not be able to store a row of blocks of A, or a column of blocks of B.
// For this reason, unroll the summation over l in the block multiplication
// C_ij = sum_l A_il * B_lk.
// Again, C_ij, A_il and B_lk now denote blocks.
// Then a single thread will correspond to a triple (i,j,l), which will be reponsible to compute
// the block multiplication A_il * B_lk and add it to the block C_ij.


// kernel1 suffers from latency. Each thread computes an entry C_ij of C, and needs a row of A and a column of B.
// The cache lines are distributed in the columns. When we load a row of A, we are loading a whole block.
// For this reason, it makes more sense to work with blocks if we want to use the whole cache lines that are
// being loaded.
// kernel2 works with blocks, but uses a single thread for block, which serializes the code, which gave worse
// performance. The idea is then to keep using a single thread for each entry of C, as in kernel1,
// but organize the threads to compute the matrix multiplication in blocks. 
// We need lower latency, so we'll use shared memory to store a block of A and a block of B.
// If the matrices are too big, we can't expect to fit all necessary rows and columns in shared memory.
// Therefore, the blocks loaded in shared memory will have some small size, say 64x64.
// For big matrices, this won't be enough to compute an entry of C, because we don't have an entire row of A
// or column of B.
// I have two ideas to proceed.
//
// One is load all necessary blocks into shared memory until the threads
// of a thread block compute a block of C.
//
// Another idea is to load a single block of A and a single block of B into shared memory.
// Again, this may not be enough to compute an entry of C. Say the blocks of A and of B
// are indexed by an index L (similar to l, but for blocks). Then to compute the block of C
// I'll need to sum over L. I'll have to unroll the for loop of this sum, since I'm using
// shared memory to compute a single block multiplication. These block multiplications
// have to be added by different thread blocks. It can be done safely with atomicAdd.
// Then we have the problem of summing the term beta * C_ij. Since C is both input
// and output of sgemm, we have a race condition. If some thread modifies C by adding the
// block multiplication, the thread that add beta * C_ij may be adding the modified value.
// For this reason, this idea won't work if multiple thread blocks are responsible for the
// same entry of C. I won't use a single thread block, since the matrices may be large.

__global__ void kernel3(
    uint const m,
    uint const n,
    uint const k,
    float const alpha,
    float const * const A,
    float const * const B,
    float const beta,
    float * const C
) {
    __shared__ float sharedArray[2 * BLOCKSIZE_KERNEL3 * BLOCKSIZE_KERNEL3];

    // The indices for the entry C_ij that this thread will compute
    uint const i {blockDim.x * blockIdx.x + threadIdx.x};
    uint const j {blockDim.y * blockIdx.y + threadIdx.y};

    if (i < m && j < n) {
        float AB_ij {0};
        // For each block multiplication.
        // L is the index for the block.
        uint const numBlocks {CEIL_DIV(k, BLOCKSIZE_KERNEL3)};
        
        // If k is not a multiple of BLOCKSIZE_KERNEL3, the last inner product will use smaller vectors.
        uint const trimmedVectorSize {k % BLOCKSIZE_KERNEL3}; // size of the smaller vector, if it smaller
        // If we the size is 0, then we don't have to trim
        bool const trimLastVector {trimmedVectorSize != 0}; 
        uint const lastL {numBlocks -1};

        for (uint L {0}; L < numBlocks; ++L) {
            // Load a block of A and a block of B into shared memory cooperatively.

            // Indices to store the values in the shared array.
            // Just use the indices relative to the block.
            uint const index_A_store {lexicographicIndex(BLOCKSIZE_KERNEL3, threadIdx.x, threadIdx.y)};
            uint const index_B_store {BLOCKSIZE_KERNEL3 * BLOCKSIZE_KERNEL3 + index_A_store};

            // The thread loads the entries with indices (threadIdx.x, threadIdx.y) relative to the block.
            // The absolute row of A is i, and the absolute column of B is j.
            // The non trivial cases are the absolute column of A and absolute row of B.
            // These are the following.
            uint const j_A_load {L * BLOCKSIZE_KERNEL3 + threadIdx.y};
            uint const i_B_load {L * BLOCKSIZE_KERNEL3 + threadIdx.x};

            // Now store the values
            if (j_A_load < k) {
                sharedArray[index_A_store] = A[lexicographicIndex(m, i, j_A_load)];
            }

            if (i_B_load < k) {
                sharedArray[index_B_store] = B[lexicographicIndex(k, i_B_load, j)];
            }

            // Sync block threads to finish loading the blocks of A and B
            __syncthreads();

            // Now that the threads loaded a block of A and a block of B,
            // compute the block multiplication.
            // This threads needs the row threadIdx.x from the block of A and the column threadIdx.y from the block of B,
            // both relative.
            uint vectorSize {BLOCKSIZE_KERNEL3};
            if (L == lastL && trimLastVector) {
                vectorSize = trimmedVectorSize;
            }

            for (uint l {0}; l < vectorSize; ++l) {
                uint const index_A_load {lexicographicIndex(BLOCKSIZE_KERNEL3, threadIdx.x, l)};
                uint const index_B_load {BLOCKSIZE_KERNEL3 * BLOCKSIZE_KERNEL3 + lexicographicIndex(BLOCKSIZE_KERNEL3, l, threadIdx.y)};
                float const a {sharedArray[index_A_load]};
                float const b {sharedArray[index_B_load]};
                AB_ij += a * b;
            }

            // Sync threads so the sharedArray isn't modified before we finish using it
            __syncthreads();
        }

        uint const ij {lexicographicIndex(m, i, j)};
        C[ij] = alpha * AB_ij + beta * C[ij];
    }
};