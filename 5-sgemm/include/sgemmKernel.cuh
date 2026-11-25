#pragma once

#include "../include/profiling.hpp"

void sgemmKernel(
    uint const m,
    uint const n,
    uint const k,
    float const alpha,
    float const * const d_A, // device array A
    float const * const d_B, // device array B
    float const beta,
    float const * const d_C_initial, // device array to initialize d_C
    float * const d_C, // device array C 
    float * const h_C, // host array to store d_C
    float const * const h_C_reference,
    uint const kernelNumber
#if PROFILE_WITH_NSIGHT == false
    ,float const cublasElapsedTime
#endif
);