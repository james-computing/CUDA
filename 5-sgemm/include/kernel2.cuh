#pragma once

// Each kernel computer alpha*A*B + beta*C

// A mxk
// B kxn
// C mxn

uint constexpr BLOCKSIZE_KERNEL2 {1}; // Better performace was 1, which reduces to the idea of kernel1. That is, kernel2 is less performant any block size greater than 1.

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