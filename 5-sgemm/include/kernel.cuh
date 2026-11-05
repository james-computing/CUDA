#pragma once

// Each kernel computer alpha*A*B + beta*C

uint constexpr numKernels {1};

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