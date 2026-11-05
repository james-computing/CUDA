#include <iostream>

#include "../include/kernel.cuh"
#include <cublas_v2.h> // basic linear algebra optimized for CUDA
#include <curand.h> // random numbers on the GPU with CUDA

#define CEIL_DIV(a,b) ((a+b-1)/b)

#define COMPARE_OUTPUT_TO_CUBLAS false

uint constexpr runs {100};

void initMatrices(
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

void printResult(uint const m, uint const n, float const * const h_C) {
    uint const N {m * n};
    std::cout << '\t';
    for (size_t i {0}; i < N; ++i) {
        std::cout << h_C[i] << ' ';
    }
    std::cout << std::endl;
}

void printInput(
    uint const m,
    uint const n,
    uint const k,
    float const alpha,
    float const beta,
    float const * const d_A,
    float const * const d_B,
    float const * const d_C_initial
) {
    std::cout << "\nInputs:\n";
    // First copy arrays to host
    float * h_A {new float[m * k]};
    if (h_A == nullptr)
        throw std::runtime_error("Failed to alloc h_A");
    float * h_B {new float[k * n]};
    if (h_B == nullptr)
        throw std::runtime_error("Failed to alloc h_B");
    float * h_C {new float[m * n]};
    if (h_C == nullptr)
        throw std::runtime_error("Failed to alloc h_C");

    cudaError_t cuda_error;
    cuda_error = cudaMemcpy(h_A, d_A, m * k * sizeof(float), cudaMemcpyDeviceToHost);
    if (cuda_error != cudaSuccess)
        throw std::runtime_error("Failed to copy d_A to h_A");
    cuda_error = cudaMemcpy(h_B, d_B, k * n* sizeof(float), cudaMemcpyDeviceToHost);
    if (cuda_error != cudaSuccess)
        throw std::runtime_error("Failed to copy d_B to h_B");
    cuda_error = cudaMemcpy(h_C, d_C_initial, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    if (cuda_error != cudaSuccess)
        throw std::runtime_error("Failed to copy d_C_initial to h_C");

    // Print the inputs
    std::cout << "alpha = " << alpha << '\n';
    std::cout << "beta = " << beta << '\n';
    std::cout << "A =\n";
    for (size_t i {0}; i < m * k; ++i) {
        std::cout << h_A[i] << ' ';
    }
    std::cout << std::endl;
    std::cout << "B =\n";
    for (size_t i {0}; i < k * n; ++i) {
        std::cout << h_B[i] << ' ';
    }
    std::cout << std::endl;
    std::cout << "C =\n";
    for (size_t i {0}; i < m * n; ++i) {
        std::cout << h_C[i] << ' ';
    }
    std::cout << '\n' <<  std::endl;

    // Cleanup
    delete[] h_A;
    h_A = nullptr;
    delete[] h_B;
    h_B = nullptr;
    delete[] h_C;
    h_C = nullptr;
}

inline float sgemmCublas(
    uint const m,
    uint const n,
    uint const k,
    float const alpha,
    float const * const d_A, // device array A
    float const * const d_B, // device array B
    float const beta,
    float const * const d_C_initial, // device array to initialize d_C
    float * const d_C, // device array C 
    float * const h_C // host array to store d_C
) {
    std::cout << "\ncublas" << std::endl;
    cudaError_t cuda_error;

    // d_C is initialized by copying d_C_initial
    size_t const size_C {m * n * sizeof(float)};
    cudaMemcpy(d_C, d_C_initial, size_C, cudaMemcpyDeviceToDevice);
    
    cublasHandle_t cublasHandle;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to create cublas handle");
    }

    cudaEvent_t start;
    cudaEvent_t end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Use cublas one time for a warm up
    cublasStatus = cublasSgemm(
        cublasHandle,
        CUBLAS_OP_N, CUBLAS_OP_N, // no transposition of matrices
        m, n, k,
        &alpha,
        d_A, m, // ld = leading dimension = number of rows of the whole matrix, if passing a submatrix
        d_B, k,
        &beta,
        d_C,
        m
    );
    cuda_error = cudaDeviceSynchronize(); // must synchronize before recording time
    if (cuda_error != cudaSuccess) {
        throw std::runtime_error("Failed to synchronize device");
    }

    // After the warm up, measure the time
    float elapsedTimeSum {0};
    float elapsedTimeSquaredSum {0};
    for (uint i {0}; i < runs; ++i) {
        cudaEventRecord(start);
        cublasStatus = cublasSgemm(
            cublasHandle,
            CUBLAS_OP_N, CUBLAS_OP_N, // no transposition of matrices
            m, n, k,
            &alpha,
            d_A, m, // ld = leading dimension = number of rows of the whole matrix, if passing a submatrix
            d_B, k,
            &beta,
            d_C,
            m
        );
        cuda_error = cudaDeviceSynchronize(); // must synchronize before recording time
        if (cuda_error != cudaSuccess) {
            throw std::runtime_error("Failed to synchronize device");
        }
        cudaEventRecord(end);
        if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasSgemm failed");
        }
        cudaEventSynchronize(end); // only need to synchronize the last event
        float elapsedTime; // in milliseconds
        cudaEventElapsedTime(&elapsedTime, start, end);
        elapsedTimeSum += elapsedTime;
        elapsedTimeSquaredSum += elapsedTime * elapsedTime;
    }

    float const elapsedTimeAverage {elapsedTimeSum/runs};
    float const elapsedTimeStardardDeviation {sqrt(elapsedTimeSquaredSum/runs - elapsedTimeAverage * elapsedTimeAverage)};
    std::cout << "Average elapsed time = " << elapsedTimeAverage << "ms" << std::endl;
    std::cout << "Standard deviation =   " << elapsedTimeStardardDeviation << "ms" << std::endl;

    // Copy the result form device to host
    cuda_error = cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    if (cuda_error != cudaSuccess)
        throw std::runtime_error("Failed to copy C from device to host");

    // Cleanup
    cublasStatus = cublasDestroy(cublasHandle);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to destroy cublas handle");
    }

    return elapsedTimeAverage;
}

//template <typename kernel>
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
    uint const kernelNumber,
    float const cublasElapsedTime
) {
    std::cout << "\nkernel" << kernelNumber << std::endl;

    cudaError_t cuda_error;

    #if COMPARE_OUTPUT_TO_CUBLAS == true
    float constexpr tolerance {1e-2};
    #endif

    dim3 const gridDim {CEIL_DIV(m, 32), CEIL_DIV(n, 32), 1};
    dim3 const blockDim {32, 32, 1};
    //std::cout << "gridDim = (" << gridDim.x << ", " << gridDim.y << ", " << gridDim.z << ")\n";
    //std::cout << "blockDim = (" << blockDim.x << ", " << blockDim.y << ", " << blockDim.z << ")\n";

    // d_C is initialized by copying d_C_initial
    size_t const size_C {m * n * sizeof(float)};
    cudaMemcpy(d_C, d_C_initial, size_C, cudaMemcpyDeviceToDevice);

    cudaEvent_t start;
    cudaEvent_t end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Warm up before measuring time
    switch(kernelNumber) {
        case 1:
            kernel1<<<gridDim,blockDim>>>(m, n, k, alpha, d_A, d_B, beta, d_C);
            break;
        default:
            throw std::runtime_error("Unknown kernel");
    }
    cuda_error = cudaDeviceSynchronize(); // must synchronize before recording time
    if (cuda_error != cudaSuccess) {
        throw std::runtime_error("Failed to synchronize device");
    }

    // After the warm up, measure time
    float elapsedTimeSum {0};
    float elapsedTimeSquaredSum {0};
    for (uint i {0}; i < runs; ++i) {
        cudaEventRecord(start);
        switch(kernelNumber) {
            case 1:
                kernel1<<<gridDim,blockDim>>>(m, n, k, alpha, d_A, d_B, beta, d_C);
                break;
            default:
                throw std::runtime_error("Unknown kernel");
        }
        cuda_error = cudaDeviceSynchronize(); // must synchronize before recording time
        if (cuda_error != cudaSuccess) {
            throw std::runtime_error("Failed to synchronize device");
        }
        cudaEventRecord(end);
        cudaEventSynchronize(end); // only need to synchronize the last event
        float elapsedTime; // in milliseconds
        cudaEventElapsedTime(&elapsedTime, start, end);
        elapsedTimeSum += elapsedTime;
        elapsedTimeSquaredSum += elapsedTime * elapsedTime;
    }

    float const elapsedTimeAverage {elapsedTimeSum/runs};
    float const elapsedTimeStardardDeviation {sqrt(elapsedTimeSquaredSum/runs - elapsedTimeAverage * elapsedTimeAverage)};
    std::cout << "Average elapsed time = " << elapsedTimeAverage << "ms" << std::endl;
    std::cout << "Standard deviation =   " << elapsedTimeStardardDeviation << "ms" << std::endl;
    std::cout << "Relative performance = " << 100 * cublasElapsedTime/elapsedTimeAverage << '%' << std::endl; // percentage of cublas performance


    // Copy the result form device to host
    cuda_error = cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    if (cuda_error != cudaSuccess)
        throw std::runtime_error("Failed to copy C from device to host");

    // compare with reference result from cublas
    #if COMPARE_OUTPUT_TO_CUBLAS == true
    bool correct {true};
    for (size_t i {0}; i < m * n; ++i) {
        float const err {abs(h_C[i] - h_C_reference[i])};
        if (err > tolerance) {
            size_t const row {i % m};
            size_t const col {i / m};
            std::cerr << "kernel1 doesn't match reference. (i, j) = (" << row << ", " << col  << "). Error = " << err << std::endl;
            correct = false;
            break;
        }
    }
    std::cout << std::boolalpha;
    std::cout << "Correct? " << correct << std::endl;
    #endif
}

int main() {
    uint constexpr m {1 << 11};
    uint constexpr n {1 << 11};
    uint constexpr k {1 << 11};
    std::cout << "(m, n, k) = (" << m << ", " << n << ", " << k << ")\n";
    std::cout << "m * n = " << m * n << std::endl;

    float constexpr alpha {2};
    float constexpr beta {3};

    // Matrices A and B won't be changed by the kernels, but C will.
    // Since C is both input and output, make one more matrix to keep the
    // initial values.
    size_t constexpr size_A {m * k * sizeof(float)};
    size_t constexpr size_B {k * n * sizeof(float)};
    size_t constexpr size_C {m * n * sizeof(float)};

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

    initMatrices(m, n, k, d_A, d_B, d_C_initial);

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
            kernelNumber,
            cublasElapsedTime
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

    return 0;
}