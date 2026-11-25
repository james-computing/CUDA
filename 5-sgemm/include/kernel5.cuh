#pragma once

// Each kernel computer alpha*A*B + beta*C

// A mxk
// B kxn
// C mxn

uint constexpr BLOCKSIZE_KERNEL5 {32}; // 32, because 32 * 32 = 1024 is the maximum number of threads in a thread block.

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