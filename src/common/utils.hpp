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

struct Event {
  cudaEvent_t e{};
  explicit Event(unsigned flags = cudaEventDefault) { CUDA_CHECK(cudaEventCreateWithFlags(&e, flags)); }
  ~Event() { if (e) cudaEventDestroy(e); }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  Event(Event&& o) noexcept : e(o.e) { o.e = nullptr; }
};