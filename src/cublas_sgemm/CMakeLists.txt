# CUBLAS SGEMM
add_library(cublas_sgemm
    cublas_sgemm.cu
)
target_link_libraries(cublas_sgemm PRIVATE CUDA::cublas CUDA::cudart)

# header-only interface
target_include_directories(cublas_sgemm INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})