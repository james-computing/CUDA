#pragma once

#ifdef kernel_tuner
#include "../include/uint.hpp" // for Kernel Tuner
#endif

// Each kernel computer alpha*A*B + beta*C

// A mxk
// B kxn
// C mxn

#ifndef kernel_tuner

uint constexpr BX_KERNEL7 {4};
uint constexpr BY_KERNEL7 {8};
uint constexpr TX_KERNEL7 {8};
uint constexpr TY_KERNEL7 {1};
uint constexpr L_KERNEL7 {8};

#else

uint constexpr BX_KERNEL7 {BX};
uint constexpr BY_KERNEL7 {BY};
uint constexpr TX_KERNEL7 {TX};
uint constexpr TY_KERNEL7 {TY};
uint constexpr L_KERNEL7 {L};

#endif

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