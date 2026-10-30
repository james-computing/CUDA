#include <iostream>

#include "../include/kernel.cuh"

int main() {
    // Matrix dimensions.
    // Will take a m x n matrix and multiply by a n x p matrix, resulting in a m x p matrix.
    uint constexpr m {2};
    uint constexpr n {3};
    uint constexpr p {2};
    
    std::cout << "m = " << m << std::endl;
    std::cout << "n = " << n << std::endl;
    std::cout << "p = " << p << '\n' << std::endl;

    // Will alloc the matrices as 1D arrays.
    // Use column major, that is, concatenate the columns to obtain the array.
    // Alloc on the host
    int * h_a {new int[m * n]};
    if (h_a == nullptr) {
        throw std::runtime_error("Failed to alloc h_a");
    }
    int * h_b {new int[n * p]};
    if (h_a == nullptr) {
        throw std::runtime_error("Failed to alloc h_b");
    }
    int * h_result {new int[m * p]};
    if (h_a == nullptr) {
        throw std::runtime_error("Failed to alloc h_result");
    }

    // Initialize matrices on the host.
    // Matrices are column major.
    std::cout << "a transpose =\n";
    uint l {0};
    for (uint j {0}; j < n; ++j) {
        for (uint i {0}; i < m; ++i) {
            h_a[l] = l;
            std::cout << h_a[l] << ' ';
            ++l;
        }
        std::cout << '\n';
    }
    std::cout << std::endl;

    std::cout << "b transpose =\n";
    l = 0;
    for (uint j {0}; j < p; ++j) {
        for (uint i {0}; i < n; ++i) {
            h_b[l] = l;
            std::cout << h_b[l] << ' ';
            ++l;
        }
        std::cout << '\n';
    }
    std::cout << std::endl;
    

    // Alloc the matrices on the device
    int * d_a;
    int * d_b;
    int * d_result;
    cudaError_t error;
    size_t constexpr size_a {m * n * sizeof(int)};
    size_t constexpr size_b {n * p * sizeof(int)};
    size_t constexpr size_result {m * p * sizeof(int)};
    error = cudaMalloc(&d_a, size_a);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to alloc d_a");
    }
    error = cudaMalloc(&d_b, size_b);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to alloc d_b");
    }
    error = cudaMalloc(&d_result, size_result);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to alloc d_result");
    }

    // copy data from host to device
    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    uint constexpr N {m*p};
    std::cout << "N = " << N << std::endl;

    dim3 constexpr blockDim {m, p, 1};
    dim3 constexpr gridDim {(m + blockDim.x -1)/blockDim.x, (p + blockDim.y -1)/blockDim.y, 1};
    std::cout << "gridDim = (" << gridDim.x << ", " << gridDim.y << ", " << gridDim.z << ')' << std::endl;
    std::cout << "blockDim = (" << blockDim.x << ", " << blockDim.y << ", " << blockDim.z << ')' << std::endl;

    std::cout << "Launch kernel\n" << std::endl;
    multiply<<<gridDim, blockDim>>>(m, n, p, d_a, d_b, d_result);
    cudaDeviceSynchronize();
    
    // copy result from device to host
    cudaMemcpy(h_result, d_result, size_result, cudaMemcpyDeviceToHost);

    // Print the result and test correctness
    bool correct {true};
    uint correctResult[] {10, 13, 28, 40}; // column major matrix
    l = 0;
    std::cout << "result transpose =\n";
    for (uint j {0}; j < p; ++j) {
        for (uint i {0}; i < m; ++i) {
            std::cout << h_result[l] << ' ';
            if (h_result[l] != correctResult[l]) {
                correct = false;
            }
            ++l;
        }
        std::cout << '\n';
    }
    std::cout << std::endl;

    std::cout << std::boolalpha;
    std::cout << "Results are correct? " << correct << std::endl;

    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_result;
    h_a = nullptr;
    h_b = nullptr;
    h_result = nullptr;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    d_a = nullptr;
    d_b = nullptr;
    d_result = nullptr;

    return 0;
}