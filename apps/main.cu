#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <torch/torch.h>

#include "cublas_sgemm.hpp"
#include "cute_gemm_v1.cuh"
#include "cute_gemm_v2.cuh"
#include "cute_gemm_v3.cuh"
#include "cute_gemm_v4.cuh"
#include "utils.hpp"

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [repeat_count]" << std::endl;
    std::cout << "  repeat_count: Number of iterations to run each benchmark (default: 1)" << std::endl;
    std::cout << "Example: " << program_name << " 10" << std::endl;
}

struct BenchmarkStats {
    float avg_gflops;
    float p50_gflops;
    
    void print(const std::string& name) const {
        std::cout << name << " - Avg: " << avg_gflops << " GFLOPS, "
                  << "P50: " << p50_gflops << " GFLOPS" << std::endl;
    }
};

struct Noop
{
    void operator()() const noexcept {}
};

template <class Work,
          class Prolog = Noop,
          class Epilog = Noop>
BenchmarkStats benchmark(int repeat, int M, int N, int K,
                Work &&work,
                Prolog &&prolog = {},
                Epilog &&epilog = {},
                bool print_each = false)
{
    Event begin, end;
    std::vector<float> gflops_measurements;
    gflops_measurements.reserve(repeat);

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
        float gflops = 2.0f * M * N * K / 1e6f / ms;
        gflops_measurements.push_back(gflops);
        if (print_each)
            std::cout << "Iteration " << i << ": " << ms << " ms, " << gflops << " GFLOPS\n";
        std::forward<Epilog>(epilog)();
    }
    
    // Calculate statistics
    std::sort(gflops_measurements.begin(), gflops_measurements.end());
    
    // Average
    float avg = 0.0f;
    for (float gflops : gflops_measurements) {
        avg += gflops;
    }
    avg /= repeat;
    
    // P50 (median)
    float p50;
    if (repeat % 2 == 0) {
        p50 = (gflops_measurements[repeat/2 - 1] + gflops_measurements[repeat/2]) / 2.0f;
    } else {
        p50 = gflops_measurements[repeat/2];
    }
    
    return {avg, p50};
}

void run_benchmark(cublasHandle_t handle, int repeat)
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

    auto cublas_stats = benchmark(repeat, M, N, K, [&]()
                                   { cublas_sgemm(handle, A_ptr, B_ptr, C_ptr, M, N, K, alpha, beta); }, prolog, epilog);
    cublas_stats.print("cuBLAS SGEMM");
    cout << endl;

    // auto cute_v1_stats = benchmark(repeat, M, N, K, [&]()
    //                                 { cute_gemm_v1(A_ptr, B_ptr, C_ptr, M, N, K, alpha, beta); }, prolog, epilog);
    // cute_v1_stats.print("CUTE_v1 SGEMM");
    // cout << "CUTE_v1 / cuBLAS (avg): " << (cute_v1_stats.avg_gflops / cublas_stats.avg_gflops) * 100.0f << " %";
    // cout << ", (p50): " << (cute_v1_stats.p50_gflops / cublas_stats.p50_gflops) * 100.0f << " %" << endl << endl;

    // auto cute_v2_stats = benchmark(repeat, M, N, K, [&]()
    //                                 { cute_gemm_v2(A_ptr, B_ptr, C_ptr, M, N, K, alpha, beta); }, prolog, epilog);
    // cute_v2_stats.print("CUTE_v2 SGEMM");
    // cout << "CUTE_v2 / cuBLAS (avg): " << (cute_v2_stats.avg_gflops / cublas_stats.avg_gflops) * 100.0f << " %";
    // cout << ", (p50): " << (cute_v2_stats.p50_gflops / cublas_stats.p50_gflops) * 100.0f << " %" << endl << endl;

    // auto cute_v3_stats = benchmark(repeat, M, N, K, [&]()
    //                                 { cute_gemm_v3(A_ptr, B_ptr, C_ptr, M, N, K, alpha, beta); }, prolog, epilog);
    // cute_v3_stats.print("CUTE_v3 SGEMM");
    // cout << "CUTE_v3 / cuBLAS (avg): " << (cute_v3_stats.avg_gflops / cublas_stats.avg_gflops) * 100.0f << " %";
    // cout << ", (p50): " << (cute_v3_stats.p50_gflops / cublas_stats.p50_gflops) * 100.0f << " %" << endl << endl;

    auto cute_v4_stats = benchmark(repeat, M, N, K, [&]()
                                    { cute_gemm_v4(A_ptr, B_ptr, C_ptr, M, N, K, alpha, beta); }, prolog, epilog);
    cute_v4_stats.print("CUTE_v4 SGEMM");
    cout << "CUTE_v4 / cuBLAS (avg): " << (cute_v4_stats.avg_gflops / cublas_stats.avg_gflops) * 100.0f << " %";
    cout << ", (p50): " << (cute_v4_stats.p50_gflops / cublas_stats.p50_gflops) * 100.0f << " %" << endl << endl;
}

int main(int argc, char* argv[])
{
    int repeat = 10; // default repeat count
    
    // Parse command line arguments
    if (argc > 1) {
        std::string arg1(argv[1]);
        if (arg1 == "-h" || arg1 == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        
        repeat = std::atoi(argv[1]);
        if (repeat <= 0) {
            std::cerr << "Error: repeat count must be a positive integer" << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    std::cout << "Running benchmark with " << repeat << " iterations" << std::endl;
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    run_benchmark(handle, repeat);
    cublasDestroy(handle);
    return 0;
}