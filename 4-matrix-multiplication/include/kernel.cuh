#pragma once

__global__ void multiply(
    uint const m,
    uint const n,
    uint const p,
    int const * const a,
    int const * const b,
    int * const result
);