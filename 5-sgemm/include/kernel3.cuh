#pragma once

// Each kernel computer alpha*A*B + beta*C

// A mxk
// B kxn
// C mxn

uint constexpr BLOCKSIZE_KERNEL3 {32}; // Can be any multiple of 32, if the GPU has enough shared memory. Mine supports 32 and 64. Must be 32, because a thread block only supports up to 32 * 32 = 1024 threads.

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