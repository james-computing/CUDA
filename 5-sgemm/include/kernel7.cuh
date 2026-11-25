#pragma once

#include "../include/lexicographicIndex.cuh"
#include "../include/ceildiv.hpp"

// Each kernel computer alpha*A*B + beta*C

// A mxk
// B kxn
// C mxn

// Compiler used by Kernel Tuner doesn't recognize the type uint
typedef unsigned int uint;

template <uint BX_KERNEL7, uint BY_KERNEL7, uint c1, uint c2, uint l>
__global__ void kernel7(
    uint const m,
    uint const n,
    uint const k,
    float const alpha,
    float const * const A,
    float const * const B,
    float const beta,
    float * const C
);

/*
uint constexpr BX_KERNEL7 {4};
uint constexpr BY_KERNEL7 {4};

// Each thread will compute a block of C of size c1 x c2.
// I'll work with multiples of 4, to simplify using float4 later.
uint constexpr c1 {4};
static_assert(c1 > 0);
static_assert(c1 % 4 == 0);

// I want to pick a square block (c2 = c1) or a column (c2 = 1).
// A column may be better for cache locality.
// I at least expect c2 to divide c1.
// I see no good reason to take c2 > c1, because it may be worse for cache locality.
uint constexpr c2 {4};
static_assert(c2 > 0);
//static_assert(c1 % c2 == 0); // c2 | c1

// Width of the block of A to be loaded in shared memory, in multiples of c1.
// That is, the actual width is l * c1.
// It is also the height for the block of B to be loaded in shared memory.
uint constexpr l {BX_KERNEL7};
static_assert(l > 0);
// I want l to be a multiple of BX_KERNEL7 and BY_KERNEL7,
// so we can divide nicely the data that will be loaded in shared memory.
static_assert(l % BX_KERNEL7 == 0); // BX_KERNEL7 | l
static_assert(l % BY_KERNEL7 == 0); // BY_KERNEL7 | l

// Data for shared shared array allocation
uint constexpr maxFloatsInSharedMemory  {8192}; // 8192 = 2 * 64 * 64
uint constexpr sizeBlockOfA             {BX_KERNEL7 * c1 * l}; // number of floats in the block
uint constexpr sizeBlockOfB             {BY_KERNEL7 * c2 * l};
uint constexpr blockPairSize            {sizeBlockOfA + sizeBlockOfB}; // total number of floats to load in shared memory
*/



template <uint BX_KERNEL7, uint BY_KERNEL7, uint c1, uint c2, uint l>
__device__
void kernel7StaticAsserts() {
    // Each thread will compute a block of C of size c1 x c2.
    // I'll work with multiples of 4, to simplify using float4 later.
    static_assert(c1 > 0);
    static_assert(c1 % 4 == 0);

    // I want to pick a square block (c2 = c1) or a column (c2 = 1).
    // A column may be better for cache locality.
    // I at least expect c2 to divide c1.
    // I see no good reason to take c2 > c1, because it may be worse for cache locality.
    static_assert(c2 > 0);
    //static_assert(c1 % c2 == 0); // c2 | c1

    // Width of the block of A to be loaded in shared memory, in multiples of c1.
    // That is, the actual width is l * c1.
    // It is also the height for the block of B to be loaded in shared memory.
    static_assert(l > 0);
    // I want l to be a multiple of BX_KERNEL7 and BY_KERNEL7,
    // so we can divide nicely the data that will be loaded in shared memory.
    static_assert(l % BX_KERNEL7 == 0); // BX_KERNEL7 | l
    static_assert(l % BY_KERNEL7 == 0); // BY_KERNEL7 | l
}

template <uint BX_KERNEL7, uint BY_KERNEL7, uint c1, uint c2, uint l>
__global__
void kernel7(
    uint const m,
    uint const n,
    uint const k,
    float const alpha,
    float const * const A,
    float const * const B,
    float const beta,
    float * const C
) {
    kernel7StaticAsserts<BX_KERNEL7, BY_KERNEL7, c1, c2, l>();

    // Data for shared shared array allocation
    uint constexpr maxFloatsInSharedMemory  {8192}; // 8192 = 2 * 64 * 64
    uint constexpr sizeBlockOfA             {BX_KERNEL7 * c1 * l}; // number of floats in the block
    uint constexpr sizeBlockOfB             {BY_KERNEL7 * c2 * l};
    uint constexpr blockPairSize            {sizeBlockOfA + sizeBlockOfB}; // total number of floats to load in shared memory

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