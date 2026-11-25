#pragma once

void printResult(uint const m, uint const n, float const * const h_C);

void printInput(
    uint const m,
    uint const n,
    uint const k,
    float const alpha,
    float const beta,
    float const * const d_A,
    float const * const d_B,
    float const * const d_C_initial
);