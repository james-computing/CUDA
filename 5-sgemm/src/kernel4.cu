#include "../include/kernel4.cuh"
#include "../include/lexicographicIndex.cuh"
#include "../include/ceildiv.hpp"

// Each kernel computer alpha*A*B + beta*C

// A mxk
// B kxn
// C mxn

// kernel 4 will load the blocks of A and store them transposed in shared memory, for better cache locality.
// Also, we have space to store 4x the amount of blocks. Use it to compute 4 block multiplications in parallel
// before __syncthreads. This makes the code more parallel.
// Yet another possible optimization is to use vector instructions, to parallelize the inner product computation.

__global__ void kernel4(
    uint const m,
    uint const n,
    uint const k,
    float const alpha,
    float const * const A,
    float const * const B,
    float const beta,
    float * const C
) {
    // The indices for the entry C_ij that this thread will compute
    uint const i {blockDim.x * blockIdx.x + threadIdx.x};
    uint const j {blockDim.y * blockIdx.y + threadIdx.y};

    if (i < m && j < n) {
        // For 32KB of shared memory, we can store 8192 floats.
        uint constexpr maxFloatsInSharedMemory      {8192}; // 8192 = 2 * 64 * 64
        // Use as much shared memory as possible, up to maxFloatsInSharedMemory floats.
        uint constexpr blockSize                    {BLOCKSIZE_KERNEL4 * BLOCKSIZE_KERNEL4};
        uint constexpr blockPairSize                {2 * blockSize};
        uint constexpr maximumNumberOfBlockPairs    {maxFloatsInSharedMemory/blockPairSize}; // note that this is integer division
        uint constexpr sharedArraySize              {maximumNumberOfBlockPairs * blockPairSize}; // less than maxFloatsInSharedMemory if maxFloatsInSharedMemory isn't divisible by blockPairSize
        __shared__ float sharedArray[sharedArraySize];
        
        float AB_ij {0};
        // For each block multiplication.
        // L is the index for the block.
        uint const numBlocks {CEIL_DIV(k, BLOCKSIZE_KERNEL4)};
        
        // If k is not a multiple of BLOCKSIZE_KERNEL4, the last inner product will use smaller vectors.
        uint const trimmedVectorSize {k % BLOCKSIZE_KERNEL4}; // size of the smaller vector, if it smaller
        // If we the size is 0, then we don't have to trim
        bool const trimLastVector {trimmedVectorSize != 0}; 
        uint const lastL {numBlocks -1};

        // kernel4 changes this part.
        // Each L can be processed independently inside a single thread block,
        // as long as there is enough hardware resources.
        // Using more shared memory, we can load more blocks of A and B and multiple them in parallel.
        uint LFirstOfBlockGroup {0};
        uint const numBlockPairGroups {CEIL_DIV(numBlocks, maximumNumberOfBlockPairs)};
        for (uint blockPairGroup {0}; blockPairGroup < numBlockPairGroups; ++blockPairGroup) {
            // Load the block pairs for the current blockPairGroup
            uint L {LFirstOfBlockGroup};
            
            uint blockPairShift {0};
            for (uint blockPair {0}; blockPair < maximumNumberOfBlockPairs; ++blockPair) {
                // Load a block of A and a block of B into shared memory cooperatively.

                // In kernel4, in the loading stage we'll just change where to store the blocks in shared memory.
                // Indices to store the values in the shared array.
                // Just use the indices relative to the block.
                uint const index_A_store {blockPairShift + lexicographicIndex(BLOCKSIZE_KERNEL4, threadIdx.x, threadIdx.y)};
                uint const index_B_store {blockSize + index_A_store};

                // The thread loads the entries with indices (threadIdx.x, threadIdx.y) relative to the block.
                // The absolute row of A is i, and the absolute column of B is j.
                // The non trivial cases are the absolute column of A and absolute row of B.
                // These are the following.
                uint const j_A_load {L * BLOCKSIZE_KERNEL4 + threadIdx.y};
                uint const i_B_load {L * BLOCKSIZE_KERNEL4 + threadIdx.x};

                // Now store the values
                if (j_A_load < k) {
                    sharedArray[index_A_store] = A[lexicographicIndex(m, i, j_A_load)];
                }

                if (i_B_load < k) {
                    sharedArray[index_B_store] = B[lexicographicIndex(k, i_B_load, j)];
                }

                ++L;
                if (L >= numBlocks) {
                    break;
                }

                blockPairShift += blockPairSize;
            }
            __syncthreads();

            // Compute the block products for each block pair in the block group
            L = LFirstOfBlockGroup;
            blockPairShift = 0;

            for (uint blockPair {0}; blockPair < maximumNumberOfBlockPairs; ++blockPair) {
                // Now that the threads loaded a block of A and a block of B,
                // compute the block multiplication.
                // This threads needs the row threadIdx.x from the block of A and the column threadIdx.y from the block of B,
                // both relative.
                uint vectorSize {BLOCKSIZE_KERNEL4};
                if (L == lastL && trimLastVector) {
                    vectorSize = trimmedVectorSize;
                }
                
                uint const blockBShift {blockPairShift + blockSize};
                for (uint l {0}; l < vectorSize; ++l) {
                    uint const index_A_load {blockPairShift + lexicographicIndex(BLOCKSIZE_KERNEL4, threadIdx.x, l)};
                    uint const index_B_load {blockBShift + lexicographicIndex(BLOCKSIZE_KERNEL4, l, threadIdx.y)};
                    
                    float const a {sharedArray[index_A_load]};
                    float const b {sharedArray[index_B_load]};
                    AB_ij += a * b;
                }

                ++L;
                if (L >= numBlocks) {
                    break;
                }

                blockPairShift += blockPairSize;
            }
            // Sync threads so the sharedArray isn't modified before we finish using it
            __syncthreads();

            LFirstOfBlockGroup += maximumNumberOfBlockPairs;
        }

        uint const ij {lexicographicIndex(m, i, j)};
        C[ij] = alpha * AB_ij + beta * C[ij];
    }
};