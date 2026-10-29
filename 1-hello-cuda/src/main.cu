#include <iostream>

#include "../include/kernel.cuh"

int main() {
    std::cout << "Launch kernel" << std::endl;

    kernel<<<1,1>>>();

    // Wait for GPU to finish, so it uses the printf inside f.
    cudaDeviceSynchronize();

    return 0;
}