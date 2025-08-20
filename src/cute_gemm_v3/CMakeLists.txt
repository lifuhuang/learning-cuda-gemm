
# SGEMM v3
add_library(cute_gemm_v3 STATIC
    cute_gemm_v3_inst.cu
)

target_include_directories(cute_gemm_v3 PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    $ENV{HOME}/src/cutlass/include
)

target_link_libraries(cute_gemm_v3 PUBLIC
  common_utils_headers
)