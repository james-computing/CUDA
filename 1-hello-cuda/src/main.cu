#include <iostream>

#include "../include/kernel.cuh"

int main() {
    std::cout << "Lauch kernel" << std::endl;

    f<<<1,1>>>();

    // Wait for GPU to finish, so it uses the printf inside f.
    cudaDeviceSynchronize();

    return 0;
}