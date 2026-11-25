#include "../include/kernel6.cuh"
#include "../include/lexicographicIndex.cuh"
#include "../include/ceildiv.hpp"

// Each kernel computer alpha*A*B + beta*C

// A mxk
// B kxn
// C mxn

// Use float4 in the k direction.
__global__ void kernel6(
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
        uint constexpr maxFloatsInSharedMemory          {8192}; // 8192 = 2 * 64 * 64
        uint constexpr maxFloatsInSharedMemoryFloat4    {maxFloatsInSharedMemory/4};
        // Use as much shared memory as possible, up to maxFloatsInSharedMemory floats.
        uint constexpr blockNumRowsFloat4               {BLOCKSIZE_KERNEL6/4};
        uint constexpr blockSizeFloat4                  {blockNumRowsFloat4 * BLOCKSIZE_KERNEL6};
        uint constexpr blockPairSizeFloat4              {2 * blockSizeFloat4};
        uint constexpr maximumNumberOfBlockPairs        {maxFloatsInSharedMemoryFloat4/blockPairSizeFloat4}; // note that this is integer division
        uint constexpr sharedArraySize                  {maximumNumberOfBlockPairs * blockPairSizeFloat4};
        __shared__ float4 sharedArray[sharedArraySize];

        //float4 const * const vB {reinterpret_cast<float4 const *>(B)};

        float AB_ij {0};
        // For each block multiplication.
        // L is the index for the block.
        uint const numBlocks {CEIL_DIV(k, BLOCKSIZE_KERNEL6)};
        
        // If k is not a multiple of BLOCKSIZE_KERNEL4, the last inner product will use smaller vectors.
        uint const trimmedVectorSizeFloat4 {(k/4) % blockNumRowsFloat4}; // size of the smaller vector, if it smaller
        // If we the size is 0, then we don't have to trim
        bool const trimLastVector {trimmedVectorSizeFloat4 != 0}; 
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
                uint const index_A_t_store {blockPairShift + lexicographicIndex(blockNumRowsFloat4, threadIdx.y/4, threadIdx.x)};
                uint const index_B_store {blockPairShift + blockSizeFloat4 + lexicographicIndex(blockNumRowsFloat4, threadIdx.x/4, threadIdx.y)};

                // The thread loads the entries with indices (threadIdx.x, threadIdx.y) relative to the block.
                // The absolute row of A is i, and the absolute column of B is j.
                // The non trivial cases are the absolute column of A and absolute row of B.
                // These are the following.
                uint const j_A_load {L * BLOCKSIZE_KERNEL6 + threadIdx.y};
                uint const i_B_load {L * BLOCKSIZE_KERNEL6 + threadIdx.x};

                // Now store the values
                if (j_A_load < k) {
                    uint const rest {threadIdx.y % 4};
                    float const valueToStore {A[lexicographicIndex(m, i, j_A_load)]};
                    switch(rest) {
                        case 0:
                            sharedArray[index_A_t_store].x = valueToStore;
                            break;
                        case 1:
                            sharedArray[index_A_t_store].y = valueToStore;
                            break;
                        case 2:
                            sharedArray[index_A_t_store].z = valueToStore;
                            break;
                        case 3:
                            sharedArray[index_A_t_store].w = valueToStore;
                            break;
                    }
                }

                if (i_B_load < k) {
                    uint const rest {threadIdx.x % 4};
                    float const valueToStore {B[lexicographicIndex(k, i_B_load, j)]};
                    switch(rest) {
                        case 0:
                            sharedArray[index_B_store].x = valueToStore;
                            break;
                        case 1:
                            sharedArray[index_B_store].y = valueToStore;
                            break;
                        case 2:
                            sharedArray[index_B_store].z = valueToStore;
                            break;
                        case 3:
                            sharedArray[index_B_store].w = valueToStore;
                            break;
                    }
                    
                }

                ++L;
                if (L >= numBlocks) {
                    break;
                }

                blockPairShift += blockPairSizeFloat4;
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
                uint vectorSizeFloat4 {blockNumRowsFloat4};
                if (L == lastL && trimLastVector) {
                    vectorSizeFloat4 = trimmedVectorSizeFloat4;
                }
                
                uint const blockBShift {blockPairShift + blockSizeFloat4};
                for (uint l {0}; l < vectorSizeFloat4; ++l) {
                    
                    uint const index_A_t_load {blockPairShift + lexicographicIndex(blockNumRowsFloat4, l, threadIdx.x)};
                    uint const index_B_load {blockBShift + lexicographicIndex(blockNumRowsFloat4, l, threadIdx.y)};
                    
                    float4 const as {sharedArray[index_A_t_load]};
                    float4 const bs {sharedArray[index_B_load]};
                    // Doesn't work if using too many threads per block
                    //AB_ij += as.x * bs.x + as.y * bs.y + as.z * bs.z + as.w * bs.w;
                    
                    AB_ij += as.x * bs.x;
                    AB_ij += as.y * bs.y;
                    AB_ij += as.z * bs.z;
                    AB_ij += as.w * bs.w;
                    
                }

                ++L;
                if (L >= numBlocks) {
                    break;
                }

                blockPairShift += blockPairSizeFloat4;
            }
            // Sync threads so the sharedArray isn't modified before we finish using it
            __syncthreads();

            LFirstOfBlockGroup += maximumNumberOfBlockPairs;
        }

        uint const ij {lexicographicIndex(m, i, j)};
        C[ij] = alpha * AB_ij + beta * C[ij];
        
    }
};