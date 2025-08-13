#include <iostream>
#include <torch/torch.h>

#include "cublas_sgemm.hpp"
#include "cute_sgemm_v1.hpp"
#include "utils.hpp"


void validate_gemm(cublasHandle_t handle, int repeat = 10)
{
    using namespace std;
    const int M = 5120;
    const int N = 5120;
    const int K = 4096;

    auto A = torch::rand({M, K}, torch::kFloat32).t().contiguous().t().cuda();
    auto B = torch::rand({K, N}, torch::kFloat32).cuda();
    auto C = torch::zeros({M, N}, torch::kFloat32).t().contiguous().t().cuda();

    auto A_ptr = A.data_ptr<float>();
    auto B_ptr = B.data_ptr<float>();
    auto C_ptr = C.data_ptr<float>();

    cute_sgemm_v1(A_ptr, B_ptr, C_ptr, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto ans = torch::mm(A, B);
    auto ans_ptr = ans.data_ptr<float>();

    cout << "GEMM computation completed successfully." << endl;
    cout << "Max Diff:" << torch::abs(C - ans).max().item<float>() << endl;
    assert(torch::allclose(C, ans));
    cout << "GEMM result is correct." << endl;

    cublas_sgemm(handle, A_ptr, B_ptr, C_ptr, M, N, K);
    cudaDeviceSynchronize();
    cout << "cuBLAS GEMM computation completed successfully." << endl;
    cout << "Max Diff (cuBLAS):" << torch::abs(C - ans).max().item<float>() << endl;
    assert(torch::allclose(C, ans));
    cout << "cuBLAS GEMM result is correct." << endl;

    auto cute_gflops = benchmark(repeat, M, N, K, [&]()
                               { cute_sgemm_v1(A_ptr, B_ptr, C_ptr, M, N, K); });

    auto cublas_gflops = benchmark(repeat, M, N, K, [&]()
                                 { cublas_sgemm(handle, A_ptr, B_ptr, C_ptr, M, N, K); });

    cout << "CUTE SGEMM:  " << cute_gflops << " GFLOPS" << endl;
    cout << "cuBLAS SGEMM: " << cublas_gflops << "GFLOPS" << endl;
    cout << "CUTE / cuBLAS: " << (cute_gflops / cublas_gflops) * 100.0f << " %" << endl;
}

int main()
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    validate_gemm(handle);
    cublasDestroy(handle);
    return 0;
}