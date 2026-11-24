#include "../include/kernel.cuh"

#include <stdio.h>

// Each kernel computer alpha*A*B + beta*C

// A mxk
// B kxn
// C mxn

#define CEIL_DIV(a, b) ((a+b-1)/(b))

// Index of a 1D array that corresponds to row i and column j, for a column major matrix.
__device__ uint lexicographicIndex(uint const numRows, uint const i, uint const j) {
    return numRows * j + i;
}

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
        uint const numBlocks {(k + BLOCKSIZE_KERNEL3 - 1)/BLOCKSIZE_KERNEL3}; // CEIL_DIV(k, BLOCKSIZE_KERNEL3)
        
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


// Change by storing A transpose in shared memory.
// Performance is worse than kernel4

__global__ void kernel5(
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
        uint const numBlocks {(k + BLOCKSIZE_KERNEL4 - 1)/BLOCKSIZE_KERNEL4}; // CEIL_DIV(k, BLOCKSIZE_KERNEL4)
        
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
                uint const index_A_t_store {blockPairShift + lexicographicIndex(BLOCKSIZE_KERNEL4, threadIdx.y, threadIdx.x)};
                uint const index_B_store {blockPairShift + blockSize + lexicographicIndex(BLOCKSIZE_KERNEL4, threadIdx.x, threadIdx.y)};

                // The thread loads the entries with indices (threadIdx.x, threadIdx.y) relative to the block.
                // The absolute row of A is i, and the absolute column of B is j.
                // The non trivial cases are the absolute column of A and absolute row of B.
                // These are the following.
                uint const j_A_load {L * BLOCKSIZE_KERNEL4 + threadIdx.y};
                uint const i_B_load {L * BLOCKSIZE_KERNEL4 + threadIdx.x};

                // Now store the values
                if (j_A_load < k) {
                    sharedArray[index_A_t_store] = A[lexicographicIndex(m, i, j_A_load)]; // A transpose is stored in the shared array
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
                    uint const index_A_t_load {blockPairShift + lexicographicIndex(BLOCKSIZE_KERNEL4, l, threadIdx.x)};
                    uint const index_B_load {blockBShift + lexicographicIndex(BLOCKSIZE_KERNEL4, l, threadIdx.y)};
                    
                    float const a {sharedArray[index_A_t_load]};
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
        uint constexpr blockNumRowsFloat4               {BLOCKSIZE_KERNEL4/4};
        uint constexpr blockSizeFloat4                  {blockNumRowsFloat4 * BLOCKSIZE_KERNEL4};
        uint constexpr blockPairSizeFloat4              {2 * blockSizeFloat4};
        uint constexpr maximumNumberOfBlockPairs        {maxFloatsInSharedMemoryFloat4/blockPairSizeFloat4}; // note that this is integer division
        uint constexpr sharedArraySize                  {maximumNumberOfBlockPairs * blockPairSizeFloat4};
        __shared__ float4 sharedArray[sharedArraySize];

        //float4 const * const vB {reinterpret_cast<float4 const *>(B)};

        float AB_ij {0};
        // For each block multiplication.
        // L is the index for the block.
        uint const numBlocks {(k + BLOCKSIZE_KERNEL4 - 1)/BLOCKSIZE_KERNEL4}; // CEIL_DIV(k, BLOCKSIZE_KERNEL4)
        
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
                uint const j_A_load {L * BLOCKSIZE_KERNEL4 + threadIdx.y};
                uint const i_B_load {L * BLOCKSIZE_KERNEL4 + threadIdx.x};

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

/*
__global__ void kernel5(
    uint const m,
    uint const n,
    uint const k,
    float const alpha,
    float const * const A,
    float const * const B,
    float const beta,
    float * const C
) {
    // For 32KB of shared memory, we can store 8192 floats.
    uint constexpr maxFloatsInSharedMemory {8192}; // 8192 = 2 * 64 * 64
    // Use as much shared memory as possible, up to maxFloatsInSharedMemory floats.
    uint constexpr blockSize {BLOCKSIZE_KERNEL4 * BLOCKSIZE_KERNEL4};
    uint constexpr blockPairSize {2 * blockSize};
    uint constexpr maximumNumberOfBlockPairs {maxFloatsInSharedMemory/blockPairSize}; // note that this is integer division
    uint constexpr sharedArraySize {maximumNumberOfBlockPairs * blockPairSize}; // less than maxFloatsInSharedMemory if maxFloatsInSharedMemory isn't divisible by blockPairSize
    __shared__ float sharedArray[sharedArraySize];

    // The indices for the entry C_ij that this thread will compute
    uint const i {blockDim.x * blockIdx.x + threadIdx.x};
    uint const j {blockDim.y * blockIdx.y + threadIdx.y};

    if (i < m && j < n) {
        float AB_ij {0};
        // For each block multiplication.
        // L is the index for the block.
        uint const numBlocks {(k + BLOCKSIZE_KERNEL4 - 1)/BLOCKSIZE_KERNEL4}; // CEIL_DIV(k, BLOCKSIZE_KERNEL4)
        
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
        for (uint blockPairGroup {0}; blockPairGroup < numBlocks/maximumNumberOfBlockPairs; ++blockPairGroup) {
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
*/

// Data for shared shared array allocation
uint constexpr maxFloatsInSharedMemory  {8192}; // 8192 = 2 * 64 * 64
uint constexpr sizeBlockOfA             {BX_KERNEL7 * c1 * l}; // number of floats in the block
uint constexpr sizeBlockOfB             {BY_KERNEL7 * c2 * l};
uint constexpr blockPairSize            {sizeBlockOfA + sizeBlockOfB}; // total number of floats to load in shared memory

__global__ void kernel7(
    uint const m,
    uint const n,
    uint const k,
    float const alpha,
    float const * const A,
    float const * const B,
    float const beta,
    float * const C
) {
    // Divide the matrix C into blocks of size c1 x c2.
    // (I,J) is the row and column of one of these blocks, seeing C as a block matrix.
    // This thread will compute this block of C.
    uint const I {BX_KERNEL7 * blockIdx.x + threadIdx.x};
    uint const J {BY_KERNEL7 * blockIdx.y + threadIdx.y};

    // M is the height of C as a block matrix. Each block is c1 x c2 (except maybe the last blocks).
    //uint const M {CEIL_DIV(m, c1)};
    // N is the width of C as a block matrix.
    //uint const N {CEIL_DIV(n, c2)};

    // The shared array will store a block pair (a block of A and a block of B).
    uint constexpr sharedArraySize {blockPairSize};
    // Caution to not exceed shared memory capacity
    static_assert(sharedArraySize < maxFloatsInSharedMemory);
    __shared__ float sharedArray[sharedArraySize];

    // All threads will participate in loading data to shared memory,
    // but some of them may not compute entries of C.
    // For this reason, I can't immediately require that I < M and J < N.
    // This will only be required for the steps that compute an entry of C,
    // but not for loading data into shared memory.
    // Since I'll also check for individual entries with indices (i,j),
    // checking for I and J is redundant.

    // The thread will compute the multiplication of a block of A and a block of B,
    // resulting in a c1 x c2 block.
    float AB_block[c1][c2];

    // Initialize it with zeros
    for (uint dj {0}; dj < c2; ++dj) {
        for (uint di {0}; di < c1; ++di) {
            AB_block[di][dj] = 0;
        }
    }
    
    // To compute AB_block we'll repeatedly load a block pairs, a block of A and a block of B,
    // store it in shared memory, and then compute the multiplication of the blocks.
    // The result of each multiplication will be added to AB_block.
    // We need multiple blocks pairs, because shared memory may not have enough space.
    // numBlockPairs is the number of blocks pairs we need to load in total.
    // The blocks move in the k dimension of the each matrix, A and B.
    // Since the length of the loaded block in the k dimension is l,
    // the number of block pairs is:
    uint const numBlockPairs {CEIL_DIV(k, l)};
    uint vectorSize {l}; // l or k % l;
    uint const trimmedVectorSize {k % l};
    bool const trimVector {trimmedVectorSize != 0};
    uint const lastBlockPair {numBlockPairs - 1};
    
    // For each block pair, load them to shared memory and then compute their product.
    for (uint blockPair {0}; blockPair < numBlockPairs; ++blockPair) {
        // Load a block of A and a block of B cooperatively into shared memory
        
        // A, B and C are column major matrices.
        // For good cache locality, we should load the matrices by columns.
        // Since we need rows of A and columns of B, there is a difference in their loading patterns.

        // Each thread loads a c1 x l/BY_KERNEL7 block of A
        uint const iStartA {I * c1};
        uint const jStartA {blockPair * l + threadIdx.y * (l/BY_KERNEL7)};

        uint jLoadA {jStartA};
        uint jLoadRelativeA {threadIdx.y * (l/BY_KERNEL7)}; // index relative to the block of A loaded by the thread block
        for (uint dj {0}; dj < l/BY_KERNEL7; ++dj) {
            if (jLoadA >= k) {
                break;
            }
            uint iLoadA {iStartA};
            uint iLoadRelativeA {threadIdx.x * c1}; // index relative to the block of A loaded by the thread block
            for (uint di {0}; di < c1; ++di) {
                if (iLoadA >= m) {
                    break;
                }
                uint const indexStoreA = lexicographicIndex(l, jLoadRelativeA, iLoadRelativeA);
                sharedArray[indexStoreA] = A[lexicographicIndex(m, iLoadA, jLoadA)];
                ++iLoadA;
                ++iLoadRelativeA;
            }
            ++jLoadA;
            ++jLoadRelativeA;
        }

        // Each thread loads a l/BX_KERNEL7 x c2 block of B

        uint const iStartB {blockPair * l + threadIdx.x * (l/BX_KERNEL7)};
        uint const jStartB {J * c2};
        
        uint jLoadB {jStartB};
        uint jLoadRelativeB {threadIdx.y * c2};
        for (uint dj {0}; dj < c2; ++dj) {
            if (jLoadB >= n) {
                break;
            }
            uint iLoadB {iStartB};
            uint iLoadRelativeB {threadIdx.x * (l/BX_KERNEL7)};
            for (uint di {0}; di < l/BX_KERNEL7; ++di) {
                if (iLoadB >= k) {
                    break;
                }
                uint const indexStoreB {sizeBlockOfA + lexicographicIndex(l, iLoadRelativeB, jLoadRelativeB)};
                sharedArray[indexStoreB] = B[lexicographicIndex(k, iLoadB, jLoadB)];
                ++iLoadB;
                ++iLoadRelativeB;
            }
            ++jLoadB;
            ++jLoadRelativeB;
        }

        __syncthreads();
        
        // Multiply the block of A with the block of B
        // Do it by columns, since C is column major.
        if (blockPair == lastBlockPair && trimVector) {
            vectorSize = trimmedVectorSize;
        }

        for (uint dj {0}; dj < c2; ++dj) {
            uint const j {J * c2 + dj};
            if (j >= n) {
                break;
            }
            for (uint di {0}; di < c1; ++di) {
                uint const i {I * c1 + di};
                if (i >= m) {
                    break;
                }
                for (uint p {0}; p < vectorSize; ++p) {
                    uint const indexLoadA {lexicographicIndex(l, p, threadIdx.x * c1 + di)};
                    uint const indexLoadB {sizeBlockOfA + lexicographicIndex(l, p, threadIdx.y * c2 + dj)};
                    float const a {sharedArray[indexLoadA]};
                    float const b {sharedArray[indexLoadB]};
                    AB_block[di][dj] += a * b;
                }
            }
        }

        __syncthreads();
    }
    
    // Compute the block of C
    for (uint dj {0}; dj < c2; ++dj) {
        uint const j {J * c2 + dj};
        if (j >= n) {
            break;
        }
        for (uint di {0}; di < c1; ++di) {
            uint const i {I * c1 + di};
            if (i >= m) {
                break;
            }
            uint const ij {lexicographicIndex(m, i, j)};
            C[ij] = alpha * AB_block[di][dj] + beta * C[ij];
        }
    }
}