#include "../include/sgemmKernel.cuh"

#include "../include/kernel1.cuh"
#include "../include/kernel2.cuh"
#include "../include/kernel3.cuh"
#include "../include/kernel4.cuh"
#include "../include/kernel5.cuh"
#include "../include/kernel6.cuh"
#include "../include/kernel7.cuh"

#include "../include/ceildiv.hpp"

#include <iostream>

// For debugging
#define COMPARE_OUTPUT_TO_CUBLAS true
#define PRINT_RESULTS false
#define PRINT_LAST_RESULT false

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
            gridDim = {CEIL_DIV(m, BLOCKSIZE_KERNEL5), CEIL_DIV(n, BLOCKSIZE_KERNEL5), 1};
            blockDim = {BLOCKSIZE_KERNEL5, BLOCKSIZE_KERNEL5, 1};
            break;
        case 6:
            gridDim = {CEIL_DIV(m, BLOCKSIZE_KERNEL6), CEIL_DIV(n, BLOCKSIZE_KERNEL6), 1};
            blockDim = {BLOCKSIZE_KERNEL6, BLOCKSIZE_KERNEL6, 1};
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