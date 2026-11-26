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
uint constexpr BY_KERNEL7 {4};
uint constexpr c1 {4};
uint constexpr c2 {4};
uint constexpr l {BX_KERNEL7};

#else

uint constexpr BX_KERNEL7 {BX};
uint constexpr BY_KERNEL7 {BY};
uint constexpr c1 {C1};
uint constexpr c2 {C2};
uint constexpr l {L};

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