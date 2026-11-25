#include "../include/initMatrices.cuh"

#include <iostream>
#include <curand.h> // random numbers on the GPU with CUDA

#if INIT_MODE == INIT_RANDOMLY
void initMatricesRandomly(
    uint const m,
    uint const n,
    uint const k,
    float * const d_A,
    float * const d_B,
    float * const d_C_initial
) {
    // Initialize the matrices randomly on the GPU
    curandGenerator_t curandGenerator;
    curandStatus_t curand_status; // use underscore, because curandStatus already exists and is an enum
    curand_status = curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_DEFAULT);
    if (curand_status != CURAND_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to crate CUDA random number generator");
    }
    unsigned long long seed {0};
    curand_status = curandSetPseudoRandomGeneratorSeed(curandGenerator, seed);
    if (curand_status != CURAND_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to set CUDA random number generator seed");
    }

    curand_status = curandGenerateUniform(curandGenerator, d_A, m * k);
    if (curand_status != CURAND_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to initialize d_A with random numbers using curand");
    }
    curand_status = curandGenerateUniform(curandGenerator, d_B, k * n);
    if (curand_status != CURAND_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to initialize d_B with random numbers using curand");
    }
    curand_status = curandGenerateUniform(curandGenerator, d_C_initial, m * n);
    if (curand_status != CURAND_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to initialize d_C_initial with random numbers using curand");
    }

    // Cleanup
    curandDestroyGenerator(curandGenerator);
}

#elif INIT_MODE == INIT_AS_ZERO

void initMatricesAsZero(
    uint const m,
    uint const n,
    uint const k,
    float * const d_A,
    float * const d_B,
    float * const d_C_initial
) {
    size_t const size_A {m * k * sizeof(float)};
    size_t const size_B {k * n * sizeof(float)};
    size_t const size_C {m * n * sizeof(float)};
    cudaMemset(d_A, 0, size_A);
    cudaMemset(d_B, 0, size_B);
    cudaMemset(d_C_initial, 0, size_C);
}

#elif INIT_MODE == INIT_AS_IDENTITY

void initMatricesAsIdentity(
    uint const m,
    uint const n,
    uint const k,
    float * const d_A,
    float * const d_B,
    float * const h_C,
    float * const d_C_initial
) {
    float * h_A {new float[m * k]};
    float * h_B {new float[k * n]};

    uint index {0};
    for (uint j {0}; j < k; ++j) {
        for (uint i {0}; i < m; ++i) {
            if (i == j) {
                h_A[index] = 1;
            } else {
                h_A[index] = 0;
            }
            ++index;
        }
    }

    index = 0;
    for (uint j {0}; j < n; ++j) {
        for (uint i {0}; i < k; ++i) {
            if (i == j) {
                h_B[index] = 1;
            } else {
                h_B[index] = 0;
            }
            ++index;
        }
    }

    index = 0;
    for (uint j {0}; j < n; ++j) {
        for (uint i {0}; i < m; ++i) {
            if (i == j) {
                h_C[index] = 1;
            } else {
                h_C[index] = 0;
            }
            ++index;
        }
    }

    size_t const size_A {m * k * sizeof(float)};
    size_t const size_B {k * n * sizeof(float)};
    size_t const size_C {m * n * sizeof(float)};

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_initial, h_C, size_C, cudaMemcpyHostToDevice);

    // Cleanup
    delete[] h_A;
    h_A = nullptr;
    delete[] h_B;
    h_B = nullptr;
}

#elif INIT_MODE == INIT_AS_CONSTANT

void initMatricesAsConstant(
    uint const m,
    uint const n,
    uint const k,
    float * const d_A,
    float * const d_B,
    float * const d_C_initial
) {
    std::cout << "init matrices as constant" << std::endl;
    
    int constexpr constant {63}; // cudaMemset doesn't work with float
    size_t const size_A {m * k * sizeof(float)};
    size_t const size_B {k * n * sizeof(float)};
    size_t const size_C {m * n * sizeof(float)};
    cudaMemset(d_A, constant, size_A);
    cudaMemset(d_B, constant, size_B);
    cudaMemset(d_C_initial, constant, size_C);
}

#elif INIT_MODE == INIT_AS_INCREASING_DIAGONAL

void initMatricesAsIncreasingDiagonal(
    uint const m,
    uint const n,
    uint const k,
    float * const d_A,
    float * const d_B,
    float * const h_C,
    float * const d_C_initial
) {
    float * h_A {new float[m * k]};
    float * h_B {new float[k * n]};

    uint index {0};
    uint diagonalValue {1};
    for (uint j {0}; j < k; ++j) {
        for (uint i {0}; i < m; ++i) {
            if (i == j) {
                h_A[index] = diagonalValue;
                ++diagonalValue;
            } else {
                h_A[index] = 0;
            }
            ++index;
        }
    }

    index = 0;
    diagonalValue = 1;
    for (uint j {0}; j < n; ++j) {
        for (uint i {0}; i < k; ++i) {
            if (i == j) {
                h_B[index] = diagonalValue;
                ++diagonalValue;
            } else {
                h_B[index] = 0;
            }
            ++index;
        }
    }

    index = 0;
    diagonalValue = 1;
    for (uint j {0}; j < n; ++j) {
        for (uint i {0}; i < m; ++i) {
            if (i == j) {
                h_C[index] = diagonalValue;
                ++diagonalValue;
            } else {
                h_C[index] = 0;
            }
            ++index;
        }
    }

    size_t const size_A {m * k * sizeof(float)};
    size_t const size_B {k * n * sizeof(float)};
    size_t const size_C {m * n * sizeof(float)};

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_initial, h_C, size_C, cudaMemcpyHostToDevice);

    // Cleanup
    delete[] h_A;
    h_A = nullptr;
    delete[] h_B;
    h_B = nullptr;
}

#elif INIT_MODE == INIT_AS_LEXICOGRAPHIC_SEQUENCE

void initMatricesAsLexicographicSequence(
    uint const m,
    uint const n,
    uint const k,
    float * const d_A,
    float * const d_B,
    float * const h_C,
    float * const d_C_initial
) {
    float * h_A {new float[m * k]};
    float * h_B {new float[k * n]};

    // Matrices are column major, so A is initialized as
    // 1            height + 1      ...
    // 2            height + 2      ...
    // 3            height + 3      ...
    // ...          ...             ...
    // height       2 * height      ...

    float value {1};
    for (uint index {0}; index < m * k; ++index) {
        h_A[index] = value;
        ++value;
    }

    // For the next matrices I will keeping adding to value, so no value is repeated.
    for (uint index {0}; index < k * n; ++index) {
        h_B[index] = value;
        ++value;
    }

    for (uint index {0}; index < m * n; ++index) {
        h_C[index] = value;
        ++value;
    }

    size_t const size_A {m * k * sizeof(float)};
    size_t const size_B {k * n * sizeof(float)};
    size_t const size_C {m * n * sizeof(float)};

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_initial, h_C, size_C, cudaMemcpyHostToDevice);

    // Cleanup
    delete[] h_A;
    h_A = nullptr;
    delete[] h_B;
    h_B = nullptr;
}

#endif