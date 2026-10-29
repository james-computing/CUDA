cmake_minimum_required(VERSION 4.2)

project(app LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES "75") # 75 = Turing

# If using cpp file instead of cu file
#find_package(CUDAToolkit REQUIRED)

add_executable(app)

# source files
target_sources(app
    PRIVATE
        src/main.cu
        src/kernel.cu
)

# header files
target_include_directories(app
    PRIVATE
        include
)

#target_link_libraries(app
#    PRIVATE
#        CUDA::cudart # if using cpp file instead of cu file
#)

# optional
set_target_properties(app PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)