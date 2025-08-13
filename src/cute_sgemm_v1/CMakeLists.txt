
# SGEMM v1
add_library(cute_sgemm_v1
    cute_sgemm_v1.cu
)
target_include_directories(cute_sgemm_v1 PRIVATE $ENV{HOME}/src/cutlass/include)
target_link_libraries(cute_sgemm_v1 PRIVATE common_utils_headers)

# header-only interface
target_include_directories(cublas_sgemm INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})