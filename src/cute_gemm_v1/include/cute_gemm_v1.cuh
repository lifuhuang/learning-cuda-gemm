#include "utils.hpp"
#include <cute/tensor.hpp>

template <class TA, class TB, class TC, class CtaTiler, class ALayout, class ASmemLayout, class AThreadLayout,
          class BLayout, class BSmemLayout, class BThreadLayout,
          class CLayout, class CSmemLayout, class CThreadLayout,
          class Alpha, class Beta>

__global__ static __launch_bounds__(decltype(size(CThreadLayout{}))::value) 
void gemm_kernel_v1(CtaTiler cta_tiler,
                    TA *A, ALayout layout_A, ASmemLayout layout_sA, AThreadLayout tA,
                    TB *B, BLayout layout_B, BSmemLayout layout_sB, BThreadLayout tB,
                    TC *C, CLayout layout_C, CSmemLayout, CThreadLayout tC,
                    Alpha alpha, Beta beta)
{
    using namespace cute;
    static_assert(is_static<ASmemLayout>::value, "ASmemLayout must be static");
    static_assert(is_static<BSmemLayout>::value, "BSmemLayout must be static");
    static_assert(is_static<CSmemLayout>::value, "CSmemLayout must be static");

    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});

    CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) == size<0>(layout_sA));
    CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) == size<0>(CSmemLayout{}));
    CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) == size<0>(layout_sB));

    CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) == size<1>(layout_sA));
    CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) == size<1>(layout_sB));
    CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) == size<1>(CSmemLayout{}));

    static_assert(is_static<AThreadLayout>::value, "ASmemLayout must be static");
    static_assert(is_static<BThreadLayout>::value, "BSmemLayout must be static");
    static_assert(is_static<AThreadLayout>::value, "CSmemLayout must be static");

    // show<decltype(size_v<AThreadLayout> == size_v<CThreadLayout>)> t1;
    // show<decltype(size(declval<CThreadLayout>()))> t2;
    static_assert(size_v<AThreadLayout> == size_v<CThreadLayout>);
    static_assert(size_v<BThreadLayout> == size_v<CThreadLayout>);

    int tid = threadIdx.x;

    Tensor mA = make_tensor(make_gmem_ptr(A), layout_A);
    Tensor mB = make_tensor(make_gmem_ptr(B), layout_B);
    Tensor mC = make_tensor(make_gmem_ptr(C), layout_C);

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // (bM, bK, K / bK)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // (bN, bK, K / bK)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (bM, bN)

    __shared__ remove_const_t<TA> smemA[size_v<ASmemLayout>];
    __shared__ remove_const_t<TB> smemB[size_v<BSmemLayout>];
    static_assert(size_v<ASmemLayout> == cosize_v<ASmemLayout>, "ASmemLayout must be coalesced");

    Tensor sA = make_tensor(make_smem_ptr(smemA), layout_sA);
    Tensor sB = make_tensor(make_smem_ptr(smemB), layout_sB);

    auto tAgA = local_partition(gA, tA, tid);
    auto tAsA = local_partition(sA, tA, tid);

    auto tBgB = local_partition(gB, tB, tid);
    auto tBsB = local_partition(sB, tB, tid);

    auto tCgC = local_partition(gC, tC, tid);
    auto tCrC = make_tensor_like(tCgC);
    clear(tCrC);

    auto tCsA = local_partition(sA, tC, tid, Step<_1, X>{});
    auto tCsB = local_partition(sB, tC, tid, Step<X, _1>{});

    // if(thread0()) {
    //     print("  mA : "); print(  mA); print("\n");
    //     print("  gA : "); print(  gA); print("\n");
    //     print("  sA : "); print(  sA); print("\n");
    //     print("tAgA : "); print(tAgA); print("\n");
    //     print("tAsA : "); print(tAsA); print("\n");
    //     print("  mB : "); print(  mB); print("\n");
    //     print("  gB : "); print(  gB); print("\n");
    //     print("  sB : "); print(  sB); print("\n");
    //     print("tBgB : "); print(tBgB); print("\n");
    //     print("tBsB : "); print(tBsB); print("\n");
    //     print("  mC : "); print(  mC); print("\n");
    //     print("  gC : "); print(  gC); print("\n");
    //     print("tCsA : "); print(tCsA); print("\n");
    //     print("tCsB : "); print(tCsB); print("\n");
    //     print("tCgC : "); print(tCgC); print("\n");
    //     print("tCrC : "); print(tCrC); print("\n");
    // }

    auto K_TILE_MAX = size<2>(tAgA);
    for (int k = 0; k < K_TILE_MAX; ++k)
    {
        copy(tAgA(_, _, k), tAsA);
        copy(tBgB(_, _, k), tBsB);

        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        gemm(tCsA, tCsB, tCrC);
        __syncthreads();
    }

    axpby(alpha, tCrC, beta, tCgC);
    return;
}

template <class TA, class TB, class TC, class Alpha, class Beta>
void cute_gemm_v1(TA *A, TB *B, TC *C, int M, int N, int K, Alpha alpha, Beta beta)
{
    using namespace cute;

    auto bM = _128{};
    auto bN = _128{};
    auto bK = _8{};

    auto MNK = make_shape(M, N, K);
    auto cta_tiler = make_shape(bM, bN, bK);

    auto layout_A = make_layout(select<0, 2>(MNK), make_stride(_1{}, M));
    auto layout_B = make_layout(select<1, 2>(MNK), make_stride(_1{}, N));
    auto layout_C = make_layout(select<0, 1>(MNK), make_stride(_1{}, M));

    auto layout_sA = make_layout(select<0, 2>(cta_tiler));
    auto layout_sB = make_layout(select<1, 2>(cta_tiler));
    auto layout_sC = make_layout(select<0, 1>(cta_tiler));

    auto tA = make_layout(make_shape(_32{}, _8{}));
    auto tB = make_layout(make_shape(_32{}, _8{}));
    auto tC = make_layout(make_shape(_16{}, _16{}));

    dim3 blockDim(size(tC));
    dim3 gridDim(cdiv(M, bM), cdiv(N, bN));
    gemm_kernel_v1<<<gridDim, blockDim>>>(
        cta_tiler,
        A, layout_A, layout_sA, tA,
        B, layout_B, layout_sB, tB,
        C, layout_C, layout_sC, tC,
        alpha, beta);
}

extern template void cute_gemm_v1<float, float, float, float, float>(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta);
