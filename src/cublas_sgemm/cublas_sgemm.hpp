#pragma once
#include <cublas_v2.h>

void cublas_sgemm(cublasHandle_t handle, const float *A, const float *B, float *C, int M, int N, int K);
