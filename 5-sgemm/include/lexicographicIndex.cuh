#pragma once

#ifdef kernel_tuner
#include "../include/uint.hpp" // for Kernel Tuner
#endif

// Index of a 1D array that corresponds to row i and column j, for a column major matrix.
__device__ uint lexicographicIndex(uint const numRows, uint const i, uint const j);