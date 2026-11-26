cmake_minimum_required(VERSION 4.2)

project(app LANGUAGES CXX CUDA)

# source files
target_sources(app
    PRIVATE
        initMatrices.cu
        main.cu
        kernel1.cu
        kernel2.cu
        kernel3.cu
        kernel4.cu
        kernel5.cu
        kernel6.cu
        kernel7.cu
        lexicographicIndex.cu
        print.cu
        sgemmCublas.cu
        sgemmKernel.cu
)