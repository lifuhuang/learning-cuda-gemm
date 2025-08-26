#include "utils.hpp"
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cutlass/uint128.h>

template <class TA, class LayoutA, class TB, class LayoutB>
struct SharedStorage
{
    cute::ArrayEngine<TA, cute::cosize_v<LayoutA>> A;
    cute::ArrayEngine<TB, cute::cosize_v<LayoutB>> B;
};

template <class TA, class TB, class TC, class CtaTiler,
          class ALayout, class ASmemLayout, class G2SCopyA, class S2RCopyAtomA,
          class BLayout, class BSmemLayout, class G2SCopyB, class S2RCopyAtomB,
          class CLayout, class TiledMma,
          class Alpha, class Beta>
__global__ static __launch_bounds__(decltype(size(TiledMma{}))::value) void gemm_kernel_v4(CtaTiler cta_tiler,
                                                                                           const TA *A, ALayout layout_A, ASmemLayout layout_sA, G2SCopyA g2s_copy_A, S2RCopyAtomA s2r_copy_atom_A,
                                                                                           const TB *B, BLayout layout_B, BSmemLayout layout_sB, G2SCopyB g2s_copy_B, S2RCopyAtomB s2r_copy_atom_B,
                                                                                           TC *C, CLayout layout_C, TiledMma mma,
                                                                                           Alpha alpha, Beta beta)
{
    using namespace cute;
    static_assert(is_static<ASmemLayout>::value, "ASmemLayout must be static");
    static_assert(is_static<BSmemLayout>::value, "BSmemLayout must be static");

    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});

    CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) == size<0>(layout_sA));
    CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) == size<0>(layout_sB));

    CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) == size<1>(layout_sA));
    CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) == size<1>(layout_sB));

    int tid = threadIdx.x;

    Tensor mA = make_tensor(make_gmem_ptr(A), layout_A);
    Tensor mB = make_tensor(make_gmem_ptr(B), layout_B);
    Tensor mC = make_tensor(make_gmem_ptr(C), layout_C);

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // (bM, bK, K / bK)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // (bN, bK, K / bK)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (bM, bN)

    // __shared__ remove_const_t<TA> smemA[size_v<ASmemLayout>];
    // __shared__ remove_const_t<TB> smemB[size_v<BSmemLayout>];
    // static_assert(size_v<ASmemLayout> == cosize_v<ASmemLayout>, "ASmemLayout must be coalesced");
    // Tensor sA = make_tensor(make_smem_ptr(smemA), layout_sA);
    // Tensor sB = make_tensor(make_smem_ptr(smemB), layout_sB);

    extern __shared__ char smem[]; // Allocate shared memory dynamically
    SharedStorage<TA, ASmemLayout, TB, BSmemLayout> &shared = *reinterpret_cast<SharedStorage<TA, ASmemLayout, TB, BSmemLayout> *>(smem);
    Tensor sA = make_tensor(make_smem_ptr(shared.A.begin()), layout_sA); // (bM, bK, PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared.B.begin()), layout_sB); // (bN, bK, PIPE)
    constexpr auto N_PIPES = size<2>(layout_sA);                         // Number of pipeline stages

    // Global -> Shared
    ThrCopy thr_g2s_A = g2s_copy_A.get_slice(tid);
    Tensor tAgA = thr_g2s_A.partition_S(gA); // (CPY, CPY_M, CPY_K, K / bK)
    Tensor tAsA = thr_g2s_A.partition_D(sA); // (CPY, CPY_M, CPY_K, PIPES)

    ThrCopy thr_g2s_B = g2s_copy_B.get_slice(tid);
    Tensor tBgB = thr_g2s_B.partition_S(gB); // (CPY, CPY_N, CPY_K, K / bK)
    Tensor tBsB = thr_g2s_B.partition_D(sB); // (CPY, CPY_N, CPY_K, PIPES)

    // Shared -> Register
    ThrMMA thr_mma = mma.get_slice(tid);
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0)); // (MMA, MMA_M, MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0)); // (MMA, MMA_N, MMA_K)

    auto s2r_copy_A = make_tiled_copy_A(s2r_copy_atom_A, mma);
    ThrCopy thr_s2r_A = s2r_copy_A.get_slice(tid);
    Tensor tXsA = thr_s2r_A.partition_S(sA); // (CPY, MMA_M, MMA_K, PIPE)
    Tensor tXrA = thr_s2r_A.retile_D(tCrA);  // (CPY, MMA_M, MMA_K)

    auto s2r_copy_B = make_tiled_copy_B(s2r_copy_atom_B, mma);
    ThrCopy thr_s2r_B = s2r_copy_B.get_slice(tid);
    Tensor tXsB = thr_s2r_B.partition_S(sB); // (CPY, MMA_N, MMA_K, PIPE)
    Tensor tXrB = thr_s2r_B.retile_D(tCrB);  // (CPY, MMA_N, MMA_K)

    // // Pre-fetch: S -> R
    // __syncthreads()
    // copy(s2r_copy_A, tXsA(_, _, _, _0), tXrA);
    // copy(s2r_copy_B, tXsB(_, _, _, _0), tXrB);
    // pipe_read = 1;

    // Accumulator
    Tensor tCgC = thr_mma.partition_C(gC);       // (MMA, MMA_M, MMA_N)
    Tensor tCrC = thr_mma.make_fragment_C(tCgC); // (MMA, MMA_M, MMA_N)
    clear(tCrC);

    // Pre-fetch: G -> S
    auto k_tile_next = 0;
    auto K_TILE_MAX = size<3>(tAgA);

    CUTE_UNROLL
    for (int k_pipe = 0; k_pipe < N_PIPES - 1; ++k_pipe)
    {
        copy(g2s_copy_A, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, k_pipe));
        copy(g2s_copy_B, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, k_pipe));
        cp_async_fence();
        k_tile_next = (k_tile_next < K_TILE_MAX - 1) ? k_tile_next + 1 : k_tile_next;
    }

    auto pipe_read = 0;
    auto pipe_write = N_PIPES - 1;
    auto K_BLOCK_MAX = size<2>(tCrA);
    if (K_BLOCK_MAX > 1)
    {
        cp_async_wait<N_PIPES - 2>();
        __syncthreads();

        copy(s2r_copy_A, tXsA(_, _, _0{}, pipe_read), tXrA(_, _, _0{}));
        copy(s2r_copy_B, tXsB(_, _, _0{}, pipe_read), tXrB(_, _, _0{}));
    }

    CUTE_NO_UNROLL
    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
    {
        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
        {
            if (k_block == 0)
            {
                copy(g2s_copy_A, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, pipe_write));
                copy(g2s_copy_B, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, pipe_write));
                cp_async_fence();
                k_tile_next = (k_tile_next < K_TILE_MAX - 1) ? k_tile_next + 1 : k_tile_next;
                pipe_write = (pipe_write + 1) % N_PIPES;
            }

            if (k_block == K_BLOCK_MAX - 1)
            {
                cp_async_wait<N_PIPES - 2>();
                __syncthreads();
                pipe_read = (pipe_read + 1) % N_PIPES;
            }

            auto k_block_next = (k_block + 1) % K_BLOCK_MAX;
            copy(s2r_copy_A, tXsA(_, _, k_block_next, pipe_read), tXrA(_, _, k_block_next));
            copy(s2r_copy_B, tXsB(_, _, k_block_next, pipe_read), tXrB(_, _, k_block_next));

            gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
        }
    }

    axpby(alpha, tCrC, beta, tCgC);
    return;
}

template <class TA, class TB, class TC, class Alpha, class Beta, class CopyT = cutlass::uint128_t>
void cute_gemm_v4(const TA *A, const TB *B, TC *C, int M, int N, int K, Alpha alpha, Beta beta)
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

    auto N_PIPES = _3{};
    auto layout_sA = make_layout(append(select<0, 2>(cta_tiler), N_PIPES));
    auto layout_sB = make_layout(append(select<1, 2>(cta_tiler), N_PIPES));
    auto layout_sC = make_layout(append(select<0, 1>(cta_tiler), N_PIPES));

    auto thr_layout_A = make_layout(make_shape(_32{}, _8{}));
    auto thr_layout_B = make_layout(make_shape(_32{}, _8{}));
    auto val_layout_A = make_layout(make_shape(_4{}, _1{}));
    auto val_layout_B = make_layout(make_shape(_4{}, _1{}));

    // auto g2s_copy_A = make_tiled_copy(Copy_Atom<UniversalCopy<CopyT>, TA>{}, thr_layout_A, val_layout_A);
    // auto g2s_copy_B = make_tiled_copy(Copy_Atom<UniversalCopy<CopyT>, TB>{}, thr_layout_B, val_layout_B);

    TiledCopy g2s_copy_A = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TA>{},
                                           thr_layout_A,  // Thr layout 32x8 m-major
                                           val_layout_A); // Val layout  4x1 m-major
    TiledCopy g2s_copy_B = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TB>{},
                                           thr_layout_B,  // Thr layout 32x8 n-major
                                           val_layout_B); // Val layout  4x1 n-major

    auto s2r_copy_atom_A = Copy_Atom<AutoVectorizingCopy, TA>{};

    // Copy_Atom<DefaultCopy, half_t> s2r_atom_B;
    // Copy_Atom<UniversalCopy<half_t>, half_t> s2r_atom_B;
    // Copy_Atom<SM75_U32x1_LDSM_N, half_t> s2r_atom_B;
    // Copy_Atom<SM75_U32x2_LDSM_N, half_t> s2r_atom_B;
    auto s2r_copy_atom_B = Copy_Atom<AutoVectorizingCopy, TB>{};

    auto thr_mma = make_layout(make_shape(_16{}, _16{}, _1{}));
    // auto permutation = Tile<_128, _128, _8>{};
    auto mma = make_tiled_mma(UniversalFMA<TC, TA, TB>{}, thr_mma);

    dim3 blockDim(size(mma));
    dim3 gridDim(cdiv(M, bM), cdiv(N, bN));
    auto smem_size = int(sizeof(SharedStorage<TA, decltype(layout_sA), TB, decltype(layout_sB)>));

    gemm_kernel_v4<<<gridDim, blockDim, smem_size, 0>>>(
        cta_tiler,
        A, layout_A, layout_sA, g2s_copy_A, s2r_copy_atom_A,
        B, layout_B, layout_sB, g2s_copy_B, s2r_copy_atom_B,
        C, layout_C, mma,
        alpha, beta);
}

extern template void cute_gemm_v4<float, float, float, float, float, cutlass::uint128_t>(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta);