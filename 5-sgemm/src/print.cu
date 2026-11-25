#include "../include/print.cuh"

#include <iostream>

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