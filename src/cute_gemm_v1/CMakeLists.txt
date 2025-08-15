
# SGEMM v1
add_library(cute_gemm_v1 STATIC
    cute_gemm_v1_inst.cu
)

target_include_directories(cute_gemm_v1 PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    $ENV{HOME}/src/cutlass/include
)

target_link_libraries(cute_gemm_v1 PUBLIC
  common_utils_headers
)

