#include "../include/sgemmCublas.cuh"

#include "../include/profiling.hpp"

#include <cublas_v2.h> // basic linear algebra optimized for CUDA
#include <iostream>

float sgemmCublas(
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