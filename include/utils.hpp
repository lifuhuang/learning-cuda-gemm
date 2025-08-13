#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

template <class T>
struct show;

__device__ __host__ inline int cdiv(int a, int b)
{
    return (a + b - 1) / b;
}

#define CUDA_CHECK(expr) do {                                  \
  cudaError_t _err = (expr);                                   \
  if (_err != cudaSuccess) {                                   \
    std::cerr << "CUDA error " << cudaGetErrorString(_err)     \
              << " at " << __FILE__ << ":" << __LINE__         \
              << std::endl;                                    \
    std::abort();                                              \
  }                                                            \
} while (0)

__device__ __host__ inline int cdiv(int a, int b) { return (a + b - 1) / b; }

struct Event {
  cudaEvent_t e{};
  explicit Event(unsigned flags = cudaEventDefault) { CUDA_CHECK(cudaEventCreateWithFlags(&e, flags)); }
  ~Event() { if (e) cudaEventDestroy(e); }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  Event(Event&& o) noexcept : e(o.e) { o.e = nullptr; }
};

inline float benchmark(int repeat, int M, int N, int K,
                       const std::function<void()>& work,
                       bool print_each = true) {
  Event begin, end;
  float gflops = 0.0f;
  work(); CUDA_CHECK(cudaDeviceSynchronize()); // warmup
  for (int i = 0; i < repeat; ++i) {
    CUDA_CHECK(cudaEventRecord(begin.e));
    work();
    CUDA_CHECK(cudaEventRecord(end.e));
    CUDA_CHECK(cudaEventSynchronize(end.e));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, begin.e, end.e));
    gflops += 2.0f * M * N * K / 1e6f / ms;
    if (print_each) std::cout << "Iteration " << i << ": " << ms << " ms\n";
  }
  return gflops / repeat;
}
