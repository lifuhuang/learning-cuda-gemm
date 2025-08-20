// Explicit instantiation for hot types to speed up compile times
#include "include/cute_gemm_v3.cuh"
template void cute_gemm_v3<float, float, float, float, float, cutlass::uint128_t>(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta);
template void cute_gemm_v3<float, float, float, float, float, uint64_t>(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta);
template void cute_gemm_v3<float, float, float, float, float, float>(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta);