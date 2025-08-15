// Explicit instantiation for hot types to speed up compile times
#include "include/cute_gemm_v1.cuh"
template void cute_gemm_v1<float, float, float, float, float>(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta);