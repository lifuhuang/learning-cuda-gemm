#include "utils.hpp"
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cutlass/uint128.h>

template <class TA, class TB, class TC, class CtaTiler, class ALayout, class ASmemLayout, class G2SCopyA,
          class BLayout, class BSmemLayout, class G2SCopyB,
          class CLayout, class CSmemLayout, class TiledMma,
          class Alpha, class Beta>
__global__ static 
__launch_bounds__(decltype(size(TiledMma{}))::value)
void gemm_kernel_v3(CtaTiler cta_tiler,
                    const TA *A, ALayout layout_A, ASmemLayout layout_sA, G2SCopyA copy_A,
                    const TB *B, BLayout layout_B, BSmemLayout layout_sB, G2SCopyB copy_B,
                    TC *C, CLayout layout_C, CSmemLayout, TiledMma mma,
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

    // Copy Atom
    ThrCopy thr_layout_A = copy_A.get_slice(tid);
    Tensor tAgA = thr_layout_A.partition_S(gA);
    Tensor tAsA = thr_layout_A.partition_D(sA);
    Tensor tArA = make_fragment_like(tAsA);

    ThrCopy thr_layout_B = copy_B.get_slice(tid);
    Tensor tBgB = thr_layout_B.partition_S(gB);
    Tensor tBsB = thr_layout_B.partition_D(sB);
    Tensor tBrB = make_fragment_like(tBsB);

    // Pre-fetch: G -> R
    copy(copy_A, tAgA(_, _, _, 0), tArA);
    copy(copy_B, tBgB(_, _, _, 0), tBrB);

    // R -> S
    copy(tArA, tAsA);
    copy(tBrB, tBsB);
    __syncthreads();

    ThrMMA thr_layout_C = mma.get_slice(tid);
    Tensor tCsA = thr_layout_C.partition_A(sA);
    Tensor tCsB = thr_layout_C.partition_B(sB);
    Tensor tCgC = thr_layout_C.partition_C(gC);

    Tensor tCrA = thr_layout_C.make_fragment_A(tCsA);
    Tensor tCrB = thr_layout_C.make_fragment_B(tCsB);
    Tensor tCrC = thr_layout_C.make_fragment_C(tCgC);

    // Pre-fetch: S -> R
    copy(tCsA(_, _, _0{}), tCrA(_, _, _0{}));
    copy(tCsB(_, _, _0{}), tCrB(_, _, _0{}));
    clear(tCrC);

    auto K_TILE_MAX = size<3>(tAgA);
    auto K_BLOCK_MAX = size<2>(tCrA);

    CUTE_NO_UNROLL
    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
    {
        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
        {
            if (k_block == K_BLOCK_MAX - 1)
            {
                __syncthreads();
                copy(tArA, tAsA);
                copy(tBrB, tBsB);
                __syncthreads();
            }

            auto next_k_block = (k_block + 1) % K_BLOCK_MAX;
            copy(tCsA(_, _, next_k_block), tCrA(_, _, next_k_block));
            copy(tCsB(_, _, next_k_block), tCrB(_, _, next_k_block));

            if (k_block == 0) {
                auto next_k_tile = k_tile + 1 < K_TILE_MAX ? k_tile + 1 : k_tile;
                copy(copy_A, tAgA(_, _, _, next_k_tile), tArA);
                copy(copy_B, tBgB(_, _, _, next_k_tile), tBrB);
            }

            gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
        }
    }

    axpby(alpha, tCrC, beta, tCgC);
    return;
}

template <class TA, class TB, class TC, class Alpha, class Beta, class CopyT = cutlass::uint128_t>
void cute_gemm_v3(TA *A, TB *B, TC *C, int M, int N, int K, Alpha alpha, Beta beta)
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

    auto thr_layout_A = make_layout(make_shape(_32{}, _8{}));
    auto thr_layout_B = make_layout(make_shape(_32{}, _8{}));
    auto val_layout_A = make_layout(make_shape(_4{}, _1{}));
    auto val_layout_B = make_layout(make_shape(_4{}, _1{}));

    auto copy_A = make_tiled_copy(Copy_Atom<UniversalCopy<CopyT>, TA>{}, thr_layout_A, val_layout_A);
    auto copy_B = make_tiled_copy(Copy_Atom<UniversalCopy<CopyT>, TB>{}, thr_layout_B, val_layout_B);

    auto thr_layout_C = make_layout(make_shape(_16{}, _16{}, _1{}));
    auto mma = make_tiled_mma(UniversalFMA<TC, TA, TB>{}, thr_layout_C);

    dim3 blockDim(size(mma));
    dim3 gridDim(cdiv(M, bM), cdiv(N, bN));
    gemm_kernel_v3<<<gridDim, blockDim>>>(
        cta_tiler,
        A, layout_A, layout_sA, copy_A,
        B, layout_B, layout_sB, copy_B,
        C, layout_C, layout_sC, mma,
        alpha, beta);
}

extern template void cute_gemm_v3<float, float, float, float, float, cutlass::uint128_t>(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta);