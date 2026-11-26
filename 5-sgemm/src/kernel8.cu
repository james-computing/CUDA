#include "../include/kernel8.cuh"
#include "../include/lexicographicIndex.cuh"
#include "../include/ceildiv.hpp"

#ifdef kernel_tuner
#include "../src/lexicographicIndex.cu"
#endif

// Each kernel computer alpha*A*B + beta*C

// A mxk
// B kxn
// C mxn

__device__ void kernel8StaticAsserts() {
    // Each thread will compute a block of C of size TX_KERNEL8 x TY_KERNEL8.
    // I'll work with multiples of 4, to simplify using float4 later.
    static_assert(TX_KERNEL8 > 0);
    //static_assert(TX_KERNEL8 % 4 == 0);

    // I want to pick a square block (TY_KERNEL8 = TX_KERNEL8) or a column (TY_KERNEL8 = 1).
    // A column may be better for cache locality.
    // I at least expect TY_KERNEL8 to divide TX_KERNEL8.
    // I see no good reason to take TY_KERNEL8 > TX_KERNEL8, because it may be worse for cache locality.
    static_assert(TY_KERNEL8 > 0);
    //static_assert(TX_KERNEL8 % 4 == 0);
    //static_assert(TX_KERNEL8 % TY_KERNEL8 == 0); // TY_KERNEL8 | TX_KERNEL8

    // Width of the block of A to be loaded in shared memory, in multiples of TX_KERNEL8.
    // That is, the actual width is L_KERNEL8 * TX_KERNEL8.
    // It is also the height for the block of B to be loaded in shared memory.
    static_assert(L_KERNEL8 > 0);
    // I want L_KERNEL8 to be a multiple of BX_KERNEL8 and BY_KERNEL8,
    // so we can divide nicely the data that will be loaded in shared memory.
    static_assert(L_KERNEL8 % BX_KERNEL8 == 0); // BX_KERNEL8 | L_KERNEL8
    static_assert(L_KERNEL8 % BY_KERNEL8 == 0); // BY_KERNEL8 | L_KERNEL8
}

// Like kernel7, but without the transposition of the block of A loaded to shared memory
__global__ void kernel8(
    uint const m,
    uint const n,
    uint const k,
    float const alpha,
    float const * const A,
    float const * const B,
    float const beta,
    float * const C
) {
    kernel8StaticAsserts();

    // Data for shared shared array allocation
    uint constexpr maxFloatsInSharedMemory  {8192}; // 8192 = 2 * 64 * 64
    uint constexpr sizeBlockOfA             {BX_KERNEL8 * TX_KERNEL8 * L_KERNEL8}; // number of floats in the block
    uint constexpr sizeBlockOfB             {BY_KERNEL8 * TY_KERNEL8 * L_KERNEL8};
    uint constexpr blockPairSize            {sizeBlockOfA + sizeBlockOfB}; // total number of floats to load in shared memory

    // Divide the matrix C into blocks of size TX_KERNEL8 x TY_KERNEL8.
    // (I,J) is the row and column of one of these blocks, seeing C as a block matrix.
    // This thread will compute this block of C.
    uint const I {BX_KERNEL8 * blockIdx.x + threadIdx.x};
    uint const J {BY_KERNEL8 * blockIdx.y + threadIdx.y};

    // M is the height of C as a block matrix. Each block is TX_KERNEL8 x TY_KERNEL8 (except maybe the last blocks).
    //uint const M {CEIL_DIV(m, TX_KERNEL8)};
    // N is the width of C as a block matrix.
    //uint const N {CEIL_DIV(n, TY_KERNEL8)};

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
    // resulting in a TX_KERNEL8 x TY_KERNEL8 block.
    float AB_block[TX_KERNEL8][TY_KERNEL8];

    // Initialize it with zeros
    for (uint dj {0}; dj < TY_KERNEL8; ++dj) {
        for (uint di {0}; di < TX_KERNEL8; ++di) {
            AB_block[di][dj] = 0;
        }
    }
    
    // To compute AB_block we'll repeatedly load a block pairs, a block of A and a block of B,
    // store it in shared memory, and then compute the multiplication of the blocks.
    // The result of each multiplication will be added to AB_block.
    // We need multiple blocks pairs, because shared memory may not have enough space.
    // numBlockPairs is the number of blocks pairs we need to load in total.
    // The blocks move in the k dimension of the each matrix, A and B.
    // Since the length of the loaded block in the k dimension is L_KERNEL8,
    // the number of block pairs is:
    uint const numBlockPairs {CEIL_DIV(k, L_KERNEL8)};
    uint vectorSize {L_KERNEL8}; // L_KERNEL8 or k % L_KERNEL8;
    uint const trimmedVectorSize {k % L_KERNEL8};
    bool const trimVector {trimmedVectorSize != 0};
    uint const lastBlockPair {numBlockPairs - 1};
    
    // For each block pair, load them to shared memory and then compute their product.
    for (uint blockPair {0}; blockPair < numBlockPairs; ++blockPair) {
        // Load a block of A and a block of B cooperatively into shared memory
        
        // A, B and C are column major matrices.
        // For good cache locality, we should load the matrices by columns.
        // Since we need rows of A and columns of B, there is a difference in their loading patterns.

        // Each thread loads a TX_KERNEL8 x L_KERNEL8/BY_KERNEL8 block of A
        uint const iStartA {I * TX_KERNEL8};
        uint const jStartA {blockPair * L_KERNEL8 + threadIdx.y * (L_KERNEL8/BY_KERNEL8)};

        uint jLoadA {jStartA};
        uint jLoadRelativeA {threadIdx.y * (L_KERNEL8/BY_KERNEL8)}; // index relative to the block of A loaded by the thread block
        for (uint dj {0}; dj < L_KERNEL8/BY_KERNEL8; ++dj) {
            if (jLoadA >= k) {
                break;
            }
            uint iLoadA {iStartA};
            uint iLoadRelativeA {threadIdx.x * TX_KERNEL8}; // index relative to the block of A loaded by the thread block
            for (uint di {0}; di < TX_KERNEL8; ++di) {
                if (iLoadA >= m) {
                    break;
                }
                uint const indexStoreA = lexicographicIndex(BX_KERNEL8 * TX_KERNEL8, iLoadRelativeA, jLoadRelativeA);
                sharedArray[indexStoreA] = A[lexicographicIndex(m, iLoadA, jLoadA)];
                ++iLoadA;
                ++iLoadRelativeA;
            }
            ++jLoadA;
            ++jLoadRelativeA;
        }

        // Each thread loads a L_KERNEL8/BX_KERNEL8 x TY_KERNEL8 block of B

        uint const iStartB {blockPair * L_KERNEL8 + threadIdx.x * (L_KERNEL8/BX_KERNEL8)};
        uint const jStartB {J * TY_KERNEL8};
        
        uint jLoadB {jStartB};
        uint jLoadRelativeB {threadIdx.y * TY_KERNEL8};
        for (uint dj {0}; dj < TY_KERNEL8; ++dj) {
            if (jLoadB >= n) {
                break;
            }
            uint iLoadB {iStartB};
            uint iLoadRelativeB {threadIdx.x * (L_KERNEL8/BX_KERNEL8)};
            for (uint di {0}; di < L_KERNEL8/BX_KERNEL8; ++di) {
                if (iLoadB >= k) {
                    break;
                }
                uint const indexStoreB {sizeBlockOfA + lexicographicIndex(L_KERNEL8, iLoadRelativeB, jLoadRelativeB)};
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

        for (uint dj {0}; dj < TY_KERNEL8; ++dj) {
            uint const j {J * TY_KERNEL8 + dj};
            if (j >= n) {
                break;
            }
            for (uint di {0}; di < TX_KERNEL8; ++di) {
                uint const i {I * TX_KERNEL8 + di};
                if (i >= m) {
                    break;
                }
                for (uint p {0}; p < vectorSize; ++p) {
                    uint const indexLoadA {lexicographicIndex(BX_KERNEL8 * TX_KERNEL8, threadIdx.x * TX_KERNEL8 + di, p)};
                    uint const indexLoadB {sizeBlockOfA + lexicographicIndex(L_KERNEL8, p, threadIdx.y * TY_KERNEL8 + dj)};
                    float const a {sharedArray[indexLoadA]};
                    float const b {sharedArray[indexLoadB]};
                    AB_block[di][dj] += a * b;
                }
            }
        }

        __syncthreads();
    }
    
    // Compute the block of C
    for (uint dj {0}; dj < TY_KERNEL8; ++dj) {
        uint const j {J * TY_KERNEL8 + dj};
        if (j >= n) {
            break;
        }
        for (uint di {0}; di < TX_KERNEL8; ++di) {
            uint const i {I * TX_KERNEL8 + di};
            if (i >= m) {
                break;
            }
            uint const ij {lexicographicIndex(m, i, j)};
            C[ij] = alpha * AB_block[di][dj] + beta * C[ij];
        }
    }
}