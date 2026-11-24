#pragma once

// Each kernel compute alpha*A*B + beta*C

uint constexpr numKernels {7};

uint constexpr BLOCKSIZE_KERNEL2 {1}; // Better performace was 1, which reduces to the idea of kernel1. That is, kernel2 is less performant any block size greater than 1.
uint constexpr BLOCKSIZE_KERNEL3 {32}; // Can be any multiple of 32, if the GPU has enough shared memory. Mine supports 32 and 64. Must be 32, because a thread block only supports up to 32 * 32 = 1024 threads.
uint constexpr BLOCKSIZE_KERNEL4 {32}; // 32, because 32 * 32 = 1024 is the maximum number of threads in a thread block.

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
static_assert(c1 % c2 == 0); // c2 | c1

// Width of the block of A to be loaded in shared memory, in multiples of c1.
// That is, the actual width is l * c1.
// It is also the height for the block of B to be loaded in shared memory.
uint constexpr l {BX_KERNEL7};
static_assert(l > 0);
// I want l to be a multiple of BX_KERNEL7 and BY_KERNEL7,
// so we can divide nicely the data that will be loaded in shared memory.
static_assert(l % BX_KERNEL7 == 0); // BX_KERNEL7 | l
static_assert(l % BY_KERNEL7 == 0); // BY_KERNEL7 | l

__global__ void kernel1(
    uint const m,
    uint const n,
    uint const k,
    float const alpha,
    float const * const A,
    float const * const B,
    float const beta,
    float * const C
);

__global__ void kernel2(
    uint const m,
    uint const n,
    uint const k,
    float const alpha,
    float const * const A,
    float const * const B,
    float const beta,
    float * const C
);

__global__ void kernel3(
    uint const m,
    uint const n,
    uint const k,
    float const alpha,
    float const * const A,
    float const * const B,
    float const beta,
    float * const C
);

__global__ void kernel4(
    uint const m,
    uint const n,
    uint const k,
    float const alpha,
    float const * const A,
    float const * const B,
    float const beta,
    float * const C
);

__global__ void kernel5(
    uint const m,
    uint const n,
    uint const k,
    float const alpha,
    float const * const A,
    float const * const B,
    float const beta,
    float * const C
);

__global__ void kernel6(
    uint const m,
    uint const n,
    uint const k,
    float const alpha,
    float const * const A,
    float const * const B,
    float const beta,
    float * const C
);

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
