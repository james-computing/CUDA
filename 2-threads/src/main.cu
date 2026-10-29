#include <iostream>

#include "../include/kernel.cuh"

int main() {
    std::cout << "Launch kernel" << std::endl;

    dim3 constexpr gridDim {2,3,5}; // number of blocks for each dimension of the grid
    dim3 constexpr blockDim {2,2,2}; // number of threads for each dimension of a block

    kernel<<<gridDim, blockDim>>>();

    // Wait for GPU to finish, so it uses the printf inside f.
    cudaDeviceSynchronize();

    return 0;
}