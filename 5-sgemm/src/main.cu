#include <iostream>

#include "../include/initMatrices.cuh"
#include "../include/sgemmCublas.cuh"
#include "../include/sgemmKernel.cuh"

uint constexpr numKernels {7};

void run(uint const m, uint const n, uint const k) {
    float constexpr alpha {1};
    float constexpr beta {0};

    // Matrices A and B won't be changed by the kernels, but C will.
    // Since C is both input and output, make one more matrix to keep the
    // initial values.
    size_t const size_A {m * k * sizeof(float)};
    size_t const size_B {k * n * sizeof(float)};
    size_t const size_C {m * n * sizeof(float)};

    float * h_C = new float[m * n];
    if (h_C == nullptr)
        throw std::runtime_error("Failed to alloc h_C");
    // Make yet another array to store the result computed with cublas.
    // This will be used as the reference value.
    float * h_C_reference = new float[m * n];
    if (h_C_reference == nullptr)
        throw std::runtime_error("Failed to alloc h_C_reference");

    // Create device arrays
    float * d_A;
    float * d_B;
    float * d_C;
    float * d_C_initial; // to hold the initial values of C, since C is both input and output of the kernels.
    cudaError_t cuda_error; // use underscore, because cudaError already exists and is an enum
    cuda_error = cudaMalloc(&d_A, size_A);
    if (cuda_error != cudaSuccess)
        throw std::runtime_error("Failed to alloc d_A");
    cuda_error = cudaMalloc(&d_B, size_B);
    if (cuda_error != cudaSuccess)
        throw std::runtime_error("Failed to alloc d_B");
    cuda_error = cudaMalloc(&d_C, size_C);
    if (cuda_error != cudaSuccess)
        throw std::runtime_error("Failed to alloc d_C");
    cuda_error = cudaMalloc(&d_C_initial, size_C);
    if (cuda_error != cudaSuccess)
        throw std::runtime_error("Failed to alloc d_C_initial");

#if INIT_MODE == INIT_RANDOMLY
    initMatricesRandomly(m, n, k, d_A, d_B, d_C_initial);
#elif INIT_MODE == INIT_AS_ZERO
    initMatricesAsZero(m, n, k, d_A, d_B, d_C_initial);
#elif INIT_MODE == INIT_AS_IDENTITY
    initMatricesAsIdentity(m, n, k, d_A, d_B, h_C, d_C_initial);
#elif INIT_MODE == INIT_AS_CONSTANT
    initMatricesAsConstant(m, n, k, d_A, d_B, d_C_initial);
#elif INIT_MODE == INIT_AS_INCREASING_DIAGONAL
    initMatricesAsIncreasingDiagonal(m, n, k, d_A, d_B, h_C, d_C_initial);
#elif INIT_MODE == INIT_AS_LEXICOGRAPHIC_SEQUENCE
    initMatricesAsLexicographicSequence(m, n, k, d_A, d_B, h_C, d_C_initial);
#else
    throw std::runtime_error("Unset init mode");
#endif

    //printInput(m, n, k, alpha, beta, d_A, d_B, d_C_initial);

    float const cublasElapsedTime {sgemmCublas(m, n, k, alpha, d_A, d_B, beta, d_C_initial, d_C, h_C_reference)};

    for (uint kernelNumber {1}; kernelNumber <= numKernels; ++kernelNumber) {
        sgemmKernel(
            m,
            n,
            k,
            alpha,
            d_A,
            d_B,
            beta,
            d_C_initial,
            d_C,
            h_C,
            h_C_reference, // only needed if COMPARE_OUTPUT_TO_CUBLAS is true
            kernelNumber
#if PROFILE_WITH_NSIGHT == false
            ,cublasElapsedTime
#endif
        );
    }

    // Cleanup
    delete[] h_C;
    h_C = nullptr;
    delete[] h_C_reference;
    h_C_reference = nullptr;

    cudaFree(d_A);
    d_A = nullptr;
    cudaFree(d_B);
    d_B = nullptr;
    cudaFree(d_C);
    d_C = nullptr;
    cudaFree(d_C_initial);
    d_C_initial = nullptr;
}

int main() {
    // For better cache locality, take m, n and k to be multiples of 32
    
    
    uint constexpr m {1 << 11};
    uint constexpr n {1 << 11};
    uint constexpr k {1 << 11};
    

    /*
    uint constexpr m {1 << 9};
    uint constexpr n {1 << 9};
    uint constexpr k {1 << 9};
    */
    
    /*
    uint constexpr m {64};
    uint constexpr n {64};
    uint constexpr k {64};
    */
    
    /*
    uint constexpr m {32};
    uint constexpr n {32};
    uint constexpr k {32};
    */

    /*
    uint constexpr m {1};
    uint constexpr n {1};
    uint constexpr k {1};
    */

    /*
    uint constexpr m {2};
    uint constexpr n {m};
    uint constexpr k {m};
    */

    /*
    uint constexpr m {2};
    uint constexpr n {5};
    uint constexpr k {3};
    */

    std::cout << "(m, n, k) = (" << m << ", " << n << ", " << k << ")\n";
    std::cout << "m * n = " << m * n << std::endl;

    run(m, n, k);

    return 0;
}