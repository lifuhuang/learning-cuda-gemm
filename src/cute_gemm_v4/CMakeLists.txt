
# SGEMM v4
add_library(cute_gemm_v4 STATIC
    cute_gemm_v4_inst.cu
)

target_include_directories(cute_gemm_v4 PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    $ENV{HOME}/src/cutlass/include
)

target_link_libraries(cute_gemm_v4 PUBLIC
  common_utils_headers
)