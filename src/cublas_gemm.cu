cublasHandle_t handle;

void cublas_sgemm(const float *A, const float *B, float *C, int M, int N, int K)
{
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_T,
                M, N, K,
                /*alpha=*/&alpha,
                A, M,
                B, N,
                /*beta=*/&beta,
                C, M);
}