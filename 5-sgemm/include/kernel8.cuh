#pragma once

#ifdef kernel_tuner
#include "../include/uint.hpp" // for Kernel Tuner
#endif

// Each kernel computer alpha*A*B + beta*C

// A mxk
// B kxn
// C mxn

#ifndef kernel_tuner

uint constexpr BX_KERNEL8 {32};
uint constexpr BY_KERNEL8 {32};
uint constexpr TX_KERNEL8 {1};
uint constexpr TY_KERNEL8 {4};
uint constexpr L_KERNEL8 {32};

#else

uint constexpr BX_KERNEL8 {BX};
uint constexpr BY_KERNEL8 {BY};
uint constexpr TX_KERNEL8 {TX};
uint constexpr TY_KERNEL8 {TY};
uint constexpr L_KERNEL8 {L};

#endif

__global__ void kernel8(
    uint const m,
    uint const n,
    uint const k,
    float const alpha,
    float const * const A,
    float const * const B,
    float const beta,
    float * const C
);