#pragma once

// How to initialize the matrices A, B and C.
#define INIT_RANDOMLY 0
#define INIT_AS_ZERO 1
#define INIT_AS_IDENTITY 2
#define INIT_AS_CONSTANT 3
#define INIT_AS_INCREASING_DIAGONAL 4
#define INIT_AS_LEXICOGRAPHIC_SEQUENCE 5
// Choose the init mode
#define INIT_MODE INIT_RANDOMLY

#if INIT_MODE == INIT_RANDOMLY
void initMatricesRandomly(
    uint const m,
    uint const n,
    uint const k,
    float * const d_A,
    float * const d_B,
    float * const d_C_initial
);

#elif INIT_MODE == INIT_AS_ZERO

void initMatricesAsZero(
    uint const m,
    uint const n,
    uint const k,
    float * const d_A,
    float * const d_B,
    float * const d_C_initial
);

#elif INIT_MODE == INIT_AS_IDENTITY

void initMatricesAsIdentity(
    uint const m,
    uint const n,
    uint const k,
    float * const d_A,
    float * const d_B,
    float * const h_C,
    float * const d_C_initial
);

#elif INIT_MODE == INIT_AS_CONSTANT

void initMatricesAsConstant(
    uint const m,
    uint const n,
    uint const k,
    float * const d_A,
    float * const d_B,
    float * const d_C_initial
);

#elif INIT_MODE == INIT_AS_INCREASING_DIAGONAL

void initMatricesAsIncreasingDiagonal(
    uint const m,
    uint const n,
    uint const k,
    float * const d_A,
    float * const d_B,
    float * const h_C,
    float * const d_C_initial
);

#elif INIT_MODE == INIT_AS_LEXICOGRAPHIC_SEQUENCE

void initMatricesAsLexicographicSequence(
    uint const m,
    uint const n,
    uint const k,
    float * const d_A,
    float * const d_B,
    float * const h_C,
    float * const d_C_initial
);

#endif