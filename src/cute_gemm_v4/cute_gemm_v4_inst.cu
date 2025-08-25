// Explicit instantiation for hot types to speed up compile times
#include "include/cute_gemm_v4.cuh"
template void cute_gemm_v4<float, float, float, float, float, cutlass::uint128_t>(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta);
template void cute_gemm_v4<float, float, float, float, float, uint64_t>(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta);
template void cute_gemm_v4<float, float, float, float, float, float>(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta);