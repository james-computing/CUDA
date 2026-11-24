#include <iostream>

#include "../include/kernel.cuh"
#include <cublas_v2.h> // basic linear algebra optimized for CUDA
#include <curand.h> // random numbers on the GPU with CUDA

#define CEIL_DIV(a,b) ((a+b-1)/(b))

// How to initialize the matrices A, B and C.
#define INIT_RANDOMLY 0
#define INIT_AS_ZERO 1
#define INIT_AS_IDENTITY 2
#define INIT_AS_CONSTANT 3
#define INIT_AS_INCREASING_DIAGONAL 4
#define INIT_AS_LEXICOGRAPHIC_SEQUENCE 5
// Choose the init mode
#define INIT_MODE INIT_RANDOMLY

// For debugging
#define COMPARE_OUTPUT_TO_CUBLAS true
#define PRINT_RESULTS false
#define PRINT_LAST_RESULT false

// Set to true if profiling with Nsight Compute, so the profiler doesn't run more than necessary.
#define PROFILE_WITH_NSIGHT true
#if PROFILE_WITH_NSIGHT == false
uint constexpr runs {100};
#endif

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

void printResult(uint const m, uint const n, float const * const h_C) {
    uint ij;
    for (uint i {0}; i < m; ++i) {
        for (uint j {0}; j < n; ++j) {
            ij = m * j + i; // lexicographic index
            std::cout << h_C[ij] << ' ';
        }
        std::cout << std::endl;
    }
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

#if PRINT_RESULTS == true
    // Copy the result form device to host
    cuda_error = cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    if (cuda_error != cudaSuccess)
        throw std::runtime_error("Failed to copy C from device to host");

    std::cout << "First result" << std::endl;
    printResult(m, n, h_C);
#endif

#if PROFILE_WITH_NSIGHT == false
    // After the warm up, measure the time
    float elapsedTimeSum {0};
    float elapsedTimeSquaredSum {0};
    for (uint i {0}; i < runs; ++i) {
        // Initialize d_C again, since sgemm modifies uses the input d_C as output also.
        cudaMemcpy(d_C, d_C_initial, size_C, cudaMemcpyDeviceToDevice);
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
#endif

    // Copy the result form device to host
    cuda_error = cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    if (cuda_error != cudaSuccess)
        throw std::runtime_error("Failed to copy C from device to host");

#if PRINT_RESULTS == true && PRINT_LAST_RESULT == true
    std::cout << "Last result" << std::endl;
    printResult(m, n, h_C);
#endif

    // Cleanup
    cublasStatus = cublasDestroy(cublasHandle);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to destroy cublas handle");
    }

#if PROFILE_WITH_NSIGHT == false
    return elapsedTimeAverage;
#else
    return 0;
#endif
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
    uint const kernelNumber
#if PROFILE_WITH_NSIGHT == false
    ,float const cublasElapsedTime
#endif
) {
    std::cout << "\nkernel" << kernelNumber << std::endl;

    cudaError_t cuda_error;

    #if COMPARE_OUTPUT_TO_CUBLAS == true
    float constexpr tolerance {1e-2};
    #endif

    dim3 gridDim;
    dim3 blockDim; // supports up to 1024 = 32 * 32 threads for my GPU

    uint constexpr BLOCKSIZE {32};

    switch(kernelNumber) {
        case 1:
            gridDim = {CEIL_DIV(m, BLOCKSIZE), CEIL_DIV(n, BLOCKSIZE), 1};
            blockDim = {BLOCKSIZE, BLOCKSIZE, 1};
            break;
        case 2:
            gridDim = {CEIL_DIV(m, BLOCKSIZE * BLOCKSIZE_KERNEL2), CEIL_DIV(n, BLOCKSIZE * BLOCKSIZE_KERNEL2), 1};
            blockDim = {BLOCKSIZE, BLOCKSIZE, 1};
            break;
        case 3:
            gridDim = {CEIL_DIV(m, BLOCKSIZE_KERNEL3), CEIL_DIV(n, BLOCKSIZE_KERNEL3), 1};
            blockDim = {BLOCKSIZE_KERNEL3, BLOCKSIZE_KERNEL3, 1}; // would like to use 64 here, but can't, 32 is the max.
            break;
        case 4:
            gridDim = {CEIL_DIV(m, BLOCKSIZE_KERNEL4), CEIL_DIV(n, BLOCKSIZE_KERNEL4), 1};
            blockDim = {BLOCKSIZE_KERNEL4, BLOCKSIZE_KERNEL4, 1};
            break;
        case 5:
            gridDim = {CEIL_DIV(m, BLOCKSIZE_KERNEL4), CEIL_DIV(n, BLOCKSIZE_KERNEL4), 1};
            blockDim = {BLOCKSIZE_KERNEL4, BLOCKSIZE_KERNEL4, 1};
            break;
        case 6:
            gridDim = {CEIL_DIV(m, BLOCKSIZE_KERNEL4), CEIL_DIV(n, BLOCKSIZE_KERNEL4), 1};
            blockDim = {BLOCKSIZE_KERNEL4, BLOCKSIZE_KERNEL4, 1};
            break;
        case 7:
            // A thread computes a c1 x c2 block of C.
            // A thread block computes a BX*c1 x BY*c2 block of C.
            gridDim = {CEIL_DIV(m, BX_KERNEL7 * c1), CEIL_DIV(n, BY_KERNEL7 * c2), 1};
            blockDim = {BX_KERNEL7, BY_KERNEL7, 1};
            break;
        default:
            throw std::runtime_error("No definition of gridDim for kernel");
    }
            

    std::cout << "gridDim = (" << gridDim.x << ", " << gridDim.y << ", " << gridDim.z << ")\n";
    std::cout << "blockDim = (" << blockDim.x << ", " << blockDim.y << ", " << blockDim.z << ")\n";

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
        case 2:
            kernel2<<<gridDim,blockDim>>>(m, n, k, alpha, d_A, d_B, beta, d_C);
            break;
        case 3:
            kernel3<<<gridDim,blockDim>>>(m, n, k, alpha, d_A, d_B, beta, d_C);
            break;
        case 4:
            kernel4<<<gridDim,blockDim>>>(m, n, k, alpha, d_A, d_B, beta, d_C);
            break;
        case 5:
            kernel5<<<gridDim,blockDim>>>(m, n, k, alpha, d_A, d_B, beta, d_C);
            break;
        case 6:
            kernel6<<<gridDim,blockDim>>>(m, n, k, alpha, d_A, d_B, beta, d_C);
            break;
        case 7:
            kernel7<<<gridDim,blockDim>>>(m, n, k, alpha, d_A, d_B, beta, d_C);
            break;
        default:
            throw std::runtime_error("Unknown kernel");
    }
    cudaError_t lastError {cudaGetLastError()};
    if (lastError != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(lastError));
    }
    cuda_error = cudaDeviceSynchronize(); // must synchronize before recording time
    if (cuda_error != cudaSuccess) {
        std::cout << cudaGetErrorString(cuda_error) << std::endl;
        throw std::runtime_error("Failed to synchronize device after warm up");
    }

#if PRINT_RESULTS == true
    // Copy the result form device to host
    cuda_error = cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    if (cuda_error != cudaSuccess)
        throw std::runtime_error("Failed to copy C from device to host");
    
    std::cout << "First result" << std::endl;
    printResult(m, n, h_C);
#endif

    // After the warm up, measure time
#if PROFILE_WITH_NSIGHT == false
    float elapsedTimeSum {0};
    float elapsedTimeSquaredSum {0};
    for (uint i {0}; i < runs; ++i) {
        // Initialize d_C again, since sgemm modifies uses the input d_C as output also.
        cudaMemcpy(d_C, d_C_initial, size_C, cudaMemcpyDeviceToDevice);

        cudaEventRecord(start);

        switch(kernelNumber) {
            case 1:
                kernel1<<<gridDim,blockDim>>>(m, n, k, alpha, d_A, d_B, beta, d_C);
                break;
            case 2:
                kernel2<<<gridDim,blockDim>>>(m, n, k, alpha, d_A, d_B, beta, d_C);
                break;
            case 3:
                kernel3<<<gridDim,blockDim>>>(m, n, k, alpha, d_A, d_B, beta, d_C);
                break;
            case 4:
                kernel4<<<gridDim,blockDim>>>(m, n, k, alpha, d_A, d_B, beta, d_C);
                break;
            case 5:
                kernel5<<<gridDim,blockDim>>>(m, n, k, alpha, d_A, d_B, beta, d_C);
                break;
            case 6:
                kernel6<<<gridDim,blockDim>>>(m, n, k, alpha, d_A, d_B, beta, d_C);
                break;
            case 7:
                kernel7<<<gridDim,blockDim>>>(m, n, k, alpha, d_A, d_B, beta, d_C);
                break;
            default:
                throw std::runtime_error("Unknown kernel");
        }

        lastError = cudaGetLastError();
        if (lastError != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(lastError));
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
#endif

    // Copy the result form device to host
    cuda_error = cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    if (cuda_error != cudaSuccess)
        throw std::runtime_error("Failed to copy C from device to host");

#if PRINT_RESULTS == true && PRINT_LAST_RESULT == true
    std::cout << "Last result" << std::endl;
    printResult(m, n, h_C);
#endif

    // compare with reference result from cublas
#if COMPARE_OUTPUT_TO_CUBLAS == true
    bool correct {true};
    for (size_t i {0}; i < m * n; ++i) {
        float const err {abs(h_C[i] - h_C_reference[i])};
        if (err > tolerance) {
            size_t const row {i % m};
            size_t const col {i / m};
            std::cerr << "kernel" << kernelNumber << " doesn't match reference. (i, j) = (" << row << ", " << col  << "). Error = " << err << std::endl;
            correct = false;
            break;
        }
    }
    std::cout << std::boolalpha;
    std::cout << "Correct? " << correct << std::endl;
#endif
}

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