cmake_minimum_required(VERSION 4.2)

project(app LANGUAGES CXX CUDA)

# source files
target_sources(app
    PRIVATE
        main.cu
        kernel.cu
)