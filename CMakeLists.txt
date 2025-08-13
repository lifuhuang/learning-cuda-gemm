cmake_minimum_required(VERSION 3.23)

# Set CUDA architecture (adjust as needed)
set(CMAKE_CUDA_ARCHITECTURES 90)

project(HelloGEMM LANGUAGES CXX CUDA)

# Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Packages
find_package(CUDAToolkit REQUIRED) 
find_package(Torch REQUIRED)

# Internal header-only utils
add_library(common_utils_headers INTERFACE)
target_include_directories(common_utils_headers INTERFACE
    ${PROJECT_SOURCE_DIR}/src/common
)

# Add subprojects
add_subdirectory(src/cublas_sgemm)
add_subdirectory(src/cute_sgemm_v1)

# Create the executable
add_executable(hello_gemm apps/main.cu)
target_include_directories(hello_gemm PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(hello_gemm PRIVATE 
    "${TORCH_LIBRARIES}" 
    cublas_sgemm 
    cute_sgemm_v1
    common_utils_headers)