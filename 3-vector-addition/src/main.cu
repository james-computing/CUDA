#include <iostream>

#include "../include/kernel.cuh"

int main() {

    uint constexpr N {1 << 6};
    
    std::cout << "N = " << N << std::endl;

    // Alloc the vectors on the host
    int * h_a {new int[N]};
    if (h_a == nullptr) {
        throw std::runtime_error("Failed to alloc h_a");
    }
    int * h_b {new int[N]};
    if (h_a == nullptr) {
        throw std::runtime_error("Failed to alloc h_b");
    }
    int * h_result {new int[N]};
    if (h_a == nullptr) {
        throw std::runtime_error("Failed to alloc h_result");
    }

    // Initialize vectors on the host
    // h_a = [0, 1,..., N-1]
    // h_b = [N-1,..., 1, 0]
    int b = N-1;
    for (size_t i {0}; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = b;
        --b;
    }

    // Alloc the vectors on the device
    int * d_a;
    int * d_b;
    int * d_result;
    size_t constexpr size {N * sizeof(int)};
    cudaError_t error;
    error = cudaMalloc(&d_a, size);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to alloc d_a");
    }
    error = cudaMalloc(&d_b, size);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to alloc d_b");
    }
    error = cudaMalloc(&d_result, size);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to alloc d_result");
    }

    // copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    uint constexpr blockDim {32}; // warp size is 32, so use the same for the block
    uint constexpr gridDim {N/blockDim};

    add<<<gridDim, blockDim>>>(d_a, d_b, d_result);
    cudaDeviceSynchronize();
    
    // copy result from device to host
    cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Print the result and test correctness
    bool correct {true};
    for (size_t i {0}; i < N; ++i) {
        std::cout << h_result[i] << std::endl;
        if (h_result[i] != 63) {
            correct = false;
        }
    }

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