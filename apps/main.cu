#include <iostream>
#include <torch/torch.h>

#include "cublas_sgemm.hpp"
#include "cute_gemm_v1.cuh"
#include "cute_gemm_v2.cuh"
// #include "cute_gemm_v3.cuh"
#include "utils.hpp"

struct Noop
{
    void operator()() const noexcept {}
};

template <class Work,
          class Prolog = Noop,
          class Epilog = Noop>
float benchmark(int repeat, int M, int N, int K,
                Work &&work,
                Prolog &&prolog = {},
                Epilog &&epilog = {},
                bool print_each = false)
{
    Event begin, end;
    float gflops = 0.0f;

    // warmup
    std::forward<Prolog>(prolog)();
    std::forward<Work>(work)();
    std::forward<Epilog>(epilog)();

    for (int i = 0; i < repeat; ++i)
    {
        std::forward<Prolog>(prolog)();
        CUDA_CHECK(cudaEventRecord(begin.e));
        std::forward<Work>(work)();
        CUDA_CHECK(cudaEventRecord(end.e));
        CUDA_CHECK(cudaEventSynchronize(end.e));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, begin.e, end.e));
        gflops += 2.0f * M * N * K / 1e6f / ms;
        if (print_each)
            std::cout << "Iteration " << i << ": " << ms << " ms\n";
        std::forward<Epilog>(epilog)();
    }
    return gflops / repeat;
}

void run_benchmark(cublasHandle_t handle, int repeat = 10)
{
    using namespace std;
    const int M = 5120;
    const int N = 5120;
    const int K = 4096;
    const float alpha = 2.56f;
    const float beta = 0.314f;

    auto A = torch::rand({M, K}, torch::kFloat32).t().contiguous().t().cuda();
    auto B = torch::rand({K, N}, torch::kFloat32).cuda();
    auto C = torch::rand({M, N}, torch::kFloat32).t().contiguous().t().cuda();
    auto C_copy = torch::clone(C);

    auto A_ptr = A.data_ptr<float>();
    auto B_ptr = B.data_ptr<float>();
    auto C_ptr = C_copy.data_ptr<float>();

    auto ans = alpha * torch::mm(A, B) + beta * C; 
    auto ans_ptr = ans.data_ptr<float>();

    auto prolog = [&]()
    {
        C_copy.copy_(C);
    };

    auto epilog = [&]()
    {
        CUDA_CHECK(cudaDeviceSynchronize());
        assert(torch::allclose(C_copy, ans));
    };

    auto cublas_gflops = benchmark(repeat, M, N, K, [&]()
                                   { cublas_sgemm(handle, A_ptr, B_ptr, C_ptr, M, N, K, alpha, beta); }, prolog, epilog);
    cout << "cuBLAS SGEMM: " << cublas_gflops << "GFLOPS" << endl;

    auto cute_v1_gflops = benchmark(repeat, M, N, K, [&]()
                                 { cute_gemm_v1(A_ptr, B_ptr, C_ptr, M, N, K, alpha, beta); }, prolog, epilog);
    cout << "CUTE_v1 SGEMM:  " << cute_v1_gflops << " GFLOPS" << endl;
    cout << "CUTE_v1 / cuBLAS: " << (cute_v1_gflops / cublas_gflops) * 100.0f << " %" << endl;

    auto cute_v2_gflops = benchmark(repeat, M, N, K, [&]()
                                   { cute_gemm_v2(A_ptr, B_ptr, C_ptr, M, N, K, alpha, beta); }, prolog, epilog);
    cout << "CUTE_v2 SGEMM:  " << cute_v2_gflops << " GFLOPS" << endl;
    cout << "CUTE_v2 / cuBLAS: " << (cute_v2_gflops / cublas_gflops) * 100.0f << " %" << endl;

    auto cute_v2_64bit_gflops = benchmark(repeat, M, N, K, [&]()
                                   { cute_gemm_v2<float, float, float, float, float, uint64_t>(A_ptr, B_ptr, C_ptr, M, N, K, alpha, beta); }, prolog, epilog);
    cout << "CUTE_v2 (64-bit) SGEMM:  " << cute_v2_64bit_gflops << " GFLOPS" << endl;
    cout << "CUTE_v2 (64-bit) / cuBLAS: " << (cute_v2_64bit_gflops / cublas_gflops) * 100.0f << " %" << endl;

    auto cute_v2_32bit_gflops = benchmark(repeat, M, N, K, [&]()
                                   { cute_gemm_v2<float, float, float, float, float, float>(A_ptr, B_ptr, C_ptr, M, N, K, alpha, beta); }, prolog, epilog);
    cout << "CUTE_v2 (32-bit) SGEMM:  " << cute_v2_32bit_gflops << " GFLOPS" << endl;
    cout << "CUTE_v2 (32-bit) / cuBLAS: " << (cute_v2_32bit_gflops / cublas_gflops) * 100.0f << " %" << endl;
}

int main()
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    run_benchmark(handle);
    cublasDestroy(handle);
    return 0;
}