#include "hgemm_lab.hpp"

#include "ai_system/cuda/runtime.hpp"
#include "ai_system/profiling/nvtx.hpp"

#include <cute/tensor.hpp>

#include <cuda_fp16.h>

#include <cstddef>
#include <string>

namespace ai_system::labs::hgemm {
namespace {

using cute::_;
using cute::Int;
using cute::Shape;
using cute::Stride;
using cute::X;
using cute::_1;
using cute::_2;
using cute::_4;
using cute::_8;
using cute::_16;
using cute::_32;

constexpr int kBlockM = 256;
constexpr int kBlockN = 128;
constexpr int kBlockK = 16;
constexpr int kThreads = 256;

template <class SmemLayoutA, class SmemLayoutB>
struct CuteHgemmSharedStorage {
    cute::ArrayEngine<cute::half_t, cute::cosize_v<SmemLayoutA>> a;
    cute::ArrayEngine<cute::half_t, cute::cosize_v<SmemLayoutB>> b;
};

__host__ __device__ int ceil_div(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

bool validate_problem(
    const half* a,
    const half* b,
    half* c,
    int m,
    int n,
    int k,
    int stages,
    bool swizzle,
    int swizzle_stride,
    std::string& error
) {
    if(a == nullptr || b == nullptr || c == nullptr) {
        error = "CuTe HGEMM device pointers must be non-null.";
        return false;
    }
    if(m <= 0 || n <= 0 || k <= 0) {
        error = "CuTe HGEMM expects positive M, N, and K.";
        return false;
    }
    if(stages < 2 || stages > 4) {
        error = "cute_hgemm_tn_v01 supports --stages 2, 3, or 4.";
        return false;
    }
    if(swizzle && swizzle_stride <= 0) {
        error = "cute_hgemm_tn_v01 expects a positive --swizzle-stride.";
        return false;
    }

    cudaDeviceProp properties {};
    if(!ai_system::cuda_utils::check_status(
           cudaGetDeviceProperties(&properties, 0),
           "cudaGetDeviceProperties(cute_hgemm_tn_v01)",
           error
       )) {
        return false;
    }
    if(properties.major < 8) {
        error = "cute_hgemm_tn_v01 requires SM80 or newer for cp.async and Tensor Core MMA.";
        return false;
    }
    return true;
}

template <bool BlockSwizzle>
__device__ int cta_n_coordinate() {
    if constexpr(BlockSwizzle) {
        return static_cast<int>(blockIdx.z * gridDim.x + blockIdx.x);
    }
    return static_cast<int>(blockIdx.x);
}

template <
    int Stages,
    bool BlockSwizzle,
    class ProblemShape,
    class CtaTiler,
    class AStride,
    class ASmemLayout,
    class TiledCopyA,
    class S2RAtomA,
    class BStride,
    class BSmemLayout,
    class TiledCopyB,
    class S2RAtomB,
    class CStride,
    class TiledMma>
__global__ __launch_bounds__(kThreads) void cute_hgemm_tn_v01_kernel(
    ProblemShape problem_shape,
    CtaTiler cta_tiler,
    const cute::half_t* a,
    AStride d_a,
    ASmemLayout s_a_layout,
    TiledCopyA copy_a,
    S2RAtomA s2r_atom_a,
    const cute::half_t* b,
    BStride d_b,
    BSmemLayout s_b_layout,
    TiledCopyB copy_b,
    S2RAtomB s2r_atom_b,
    cute::half_t* c,
    CStride d_c,
    TiledMma mma,
    int n_tile_count
) {
    static_assert(Stages >= 2 && Stages <= 4);

    const int cta_n = cta_n_coordinate<BlockSwizzle>();
    if(cta_n >= n_tile_count) {
        return;
    }

    // Full logical tensors: A(M,K), B(N,K), and row-major C(M,N).
    cute::Tensor m_a = cute::make_tensor(cute::make_gmem_ptr(a), cute::select<0, 2>(problem_shape), d_a);
    cute::Tensor m_b = cute::make_tensor(cute::make_gmem_ptr(b), cute::select<1, 2>(problem_shape), d_b);
    cute::Tensor m_c = cute::make_tensor(cute::make_gmem_ptr(c), cute::select<0, 1>(problem_shape), d_c);

    const auto cta_coord = cute::make_coord(static_cast<int>(blockIdx.y), cta_n, _);
    cute::Tensor g_a = cute::local_tile(m_a, cta_tiler, cta_coord, cute::Step<_1, X, _1> {});
    cute::Tensor g_b = cute::local_tile(m_b, cta_tiler, cta_coord, cute::Step<X, _1, _1> {});
    cute::Tensor g_c = cute::local_tile(m_c, cta_tiler, cta_coord, cute::Step<_1, _1, X> {});

    extern __shared__ char shared_memory[];
    using SharedStorage = CuteHgemmSharedStorage<ASmemLayout, BSmemLayout>;
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(shared_memory);
    cute::Tensor s_a = cute::make_tensor(cute::make_smem_ptr(storage.a.begin()), s_a_layout);
    cute::Tensor s_b = cute::make_tensor(cute::make_smem_ptr(storage.b.begin()), s_b_layout);

    // TiledCopy maps every thread to matching global and shared subtensors.
    cute::ThrCopy g2s_thr_a = copy_a.get_slice(threadIdx.x);
    cute::Tensor t_ag_a = g2s_thr_a.partition_S(g_a);
    cute::Tensor t_as_a = g2s_thr_a.partition_D(s_a);
    cute::ThrCopy g2s_thr_b = copy_b.get_slice(threadIdx.x);
    cute::Tensor t_bg_b = g2s_thr_b.partition_S(g_b);
    cute::Tensor t_bs_b = g2s_thr_b.partition_D(s_b);

    int k_tile_count = cute::size<3>(t_ag_a);
    int k_tile_next = 0;

    // Prologue: fill all shared-memory stages except the stage reserved for the
    // next producer step. copy() dispatches to 16-byte SM80 cp.async.
    CUTE_UNROLL
    for(int pipe = 0; pipe < Stages - 1; ++pipe) {
        cute::copy(copy_a, t_ag_a(_, _, _, k_tile_next), t_as_a(_, _, _, pipe));
        cute::copy(copy_b, t_bg_b(_, _, _, k_tile_next), t_bs_b(_, _, _, pipe));
        cute::cp_async_fence();
        --k_tile_count;
        if(k_tile_count > 0) {
            ++k_tile_next;
        }
    }

    // ThrMMA projects the CTA tensors onto this thread's A/B/C fragments.
    cute::ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    cute::Tensor t_cg_c = thr_mma.partition_C(g_c);
    cute::Tensor t_cr_a = thr_mma.partition_fragment_A(s_a(_, _, 0));
    cute::Tensor t_cr_b = thr_mma.partition_fragment_B(s_b(_, _, 0));
    cute::Tensor t_cr_c = thr_mma.make_fragment_C(t_cg_c);
    cute::clear(t_cr_c);

    // Retile the MMA fragments to the layouts required by ldmatrix.
    cute::TiledCopy s2r_copy_a = cute::make_tiled_copy_A(s2r_atom_a, mma);
    cute::ThrCopy s2r_thr_a = s2r_copy_a.get_slice(threadIdx.x);
    cute::Tensor t_xs_a = s2r_thr_a.partition_S(s_a);
    cute::Tensor t_xr_a = s2r_thr_a.retile_D(t_cr_a);

    cute::TiledCopy s2r_copy_b = cute::make_tiled_copy_B(s2r_atom_b, mma);
    cute::ThrCopy s2r_thr_b = s2r_copy_b.get_slice(threadIdx.x);
    cute::Tensor t_xs_b = s2r_thr_b.partition_S(s_b);
    cute::Tensor t_xr_b = s2r_thr_b.retile_D(t_cr_b);

    int smem_read = 0;
    int smem_write = Stages - 1;
    cute::Tensor t_xs_a_pipe = t_xs_a(_, _, _, smem_read);
    cute::Tensor t_xs_b_pipe = t_xs_b(_, _, _, smem_read);

    constexpr int kRegisterBlocks = kBlockK / 16;
    static_assert(kRegisterBlocks == 1);

    if constexpr(kRegisterBlocks > 1) {
        cute::cp_async_wait<Stages - 2>();
        __syncthreads();
        cute::copy(s2r_atom_a, t_xs_a_pipe(_, _, Int<0> {}), t_xr_a(_, _, Int<0> {}));
        cute::copy(s2r_atom_b, t_xs_b_pipe(_, _, Int<0> {}), t_xr_b(_, _, Int<0> {}));
    }

    // Steady state overlaps G2S cp.async, S2R ldmatrix, and register MMA.
    CUTE_NO_UNROLL
    while(k_tile_count > -(Stages - 1)) {
        CUTE_UNROLL
        for(int k_block = 0; k_block < kRegisterBlocks; ++k_block) {
            if(k_block == kRegisterBlocks - 1) {
                t_xs_a_pipe = t_xs_a(_, _, _, smem_read);
                t_xs_b_pipe = t_xs_b(_, _, _, smem_read);
                cute::cp_async_wait<Stages - 2>();
                __syncthreads();
            }

            const int k_block_next = (k_block + 1) % kRegisterBlocks;
            cute::copy(s2r_atom_a, t_xs_a_pipe(_, _, k_block_next), t_xr_a(_, _, k_block_next));
            cute::copy(s2r_atom_b, t_xs_b_pipe(_, _, k_block_next), t_xr_b(_, _, k_block_next));

            if(k_block == 0) {
                cute::copy(copy_a, t_ag_a(_, _, _, k_tile_next), t_as_a(_, _, _, smem_write));
                cute::copy(copy_b, t_bg_b(_, _, _, k_tile_next), t_bs_b(_, _, _, smem_write));
                cute::cp_async_fence();

                --k_tile_count;
                if(k_tile_count > 0) {
                    ++k_tile_next;
                }
                smem_write = smem_read;
                smem_read = smem_read == Stages - 1 ? 0 : smem_read + 1;
            }

            cute::gemm(mma, t_cr_a(_, _, k_block), t_cr_b(_, _, k_block), t_cr_c);
        }
    }

    // The benchmark contract is C = A * B with FP16 accumulation and FP16 C.
    cute::axpby(1.0F, t_cr_c, 0.0F, t_cg_c);
}

__global__ void cute_hgemm_tn_v01_edge_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k,
    int fast_m,
    int fast_n,
    bool fast_path_ran
) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int element_count = m * n;
    if(index >= element_count) {
        return;
    }

    const int row = index / n;
    const int col = index % n;
    if(fast_path_ran && row < fast_m && col < fast_n) {
        return;
    }

    half accumulator = __float2half_rn(0.0F);
    for(int inner = 0; inner < k; ++inner) {
        accumulator = __hfma(a[row * k + inner], b[col * k + inner], accumulator);
    }
    c[index] = accumulator;
}

template <int Stages, bool BlockSwizzle>
bool launch_cute_hgemm_tn_v01_fast(
    const half* a,
    const half* b,
    half* c,
    int m,
    int n,
    int k,
    int m_tiles,
    int n_tiles,
    int swizzle_stride,
    std::string& error
) {
    using namespace cute;

    auto problem_shape = make_shape(m, n, k);
    auto cta_tiler = make_shape(Int<kBlockM> {}, Int<kBlockN> {}, Int<kBlockK> {});
    auto d_a = make_stride(k, Int<1> {});
    auto d_b = make_stride(k, Int<1> {});
    auto d_c = make_stride(n, Int<1> {});

    // This is the half-precision form of a 32-byte K-major swizzle atom:
    // eight rows by sixteen half values. Four A+B stages use 48 KiB.
    auto smem_atom = composition(
        Swizzle<1, 4, 3> {},
        Layout<Shape<_8, _16>, Stride<_16, _1>> {}
    );
    auto s_a = tile_to_shape(
        smem_atom,
        make_shape(Int<kBlockM> {}, Int<kBlockK> {}, Int<Stages> {})
    );
    auto s_b = tile_to_shape(
        smem_atom,
        make_shape(Int<kBlockN> {}, Int<kBlockK> {}, Int<Stages> {})
    );

    auto copy_a = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t> {},
        Layout<Shape<cute::_128, _2>, Stride<_2, _1>> {},
        Layout<Shape<_1, _8>> {}
    );
    auto copy_b = copy_a;

    auto s2r_atom_a = Copy_Atom<SM75_U32x4_LDSM_N, half_t> {};
    auto s2r_atom_b = Copy_Atom<SM75_U32x4_LDSM_N, half_t> {};
    auto mma = make_tiled_mma(
        SM80_16x8x16_F16F16F16F16_TN {},
        Layout<Shape<_4, _2>> {},
        Tile<cute::_64, _32, _16> {}
    );
    static_assert(decltype(size(mma))::value == kThreads);

    auto kernel = cute_hgemm_tn_v01_kernel<
        Stages,
        BlockSwizzle,
        decltype(problem_shape),
        decltype(cta_tiler),
        decltype(d_a),
        decltype(s_a),
        decltype(copy_a),
        decltype(s2r_atom_a),
        decltype(d_b),
        decltype(s_b),
        decltype(copy_b),
        decltype(s2r_atom_b),
        decltype(d_c),
        decltype(mma)>;

    using SharedStorage = CuteHgemmSharedStorage<decltype(s_a), decltype(s_b)>;
    const int shared_bytes = static_cast<int>(sizeof(SharedStorage));
    if(!ai_system::cuda_utils::check_status(
           cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes),
           "cudaFuncSetAttribute(cute_hgemm_tn_v01_kernel)",
           error
       ) ||
       !ai_system::cuda_utils::check_status(
           cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100),
           "cudaFuncSetAttribute(cute_hgemm_tn_v01 carveout)",
           error
       )) {
        return false;
    }

    dim3 grid;
    if constexpr(BlockSwizzle) {
        const int tiles_per_group = max(1, swizzle_stride / kBlockN);
        grid = dim3(
            static_cast<unsigned int>(min(n_tiles, tiles_per_group)),
            static_cast<unsigned int>(m_tiles),
            static_cast<unsigned int>(ceil_div(n_tiles, tiles_per_group))
        );
    } else {
        grid = dim3(static_cast<unsigned int>(n_tiles), static_cast<unsigned int>(m_tiles));
    }

    kernel<<<grid, kThreads, shared_bytes>>>(
        problem_shape,
        cta_tiler,
        reinterpret_cast<const cute::half_t*>(a),
        d_a,
        s_a,
        copy_a,
        s2r_atom_a,
        reinterpret_cast<const cute::half_t*>(b),
        d_b,
        s_b,
        copy_b,
        s2r_atom_b,
        reinterpret_cast<cute::half_t*>(c),
        d_c,
        mma,
        n_tiles
    );
    return ai_system::cuda_utils::check_last_launch(error);
}

template <bool BlockSwizzle>
bool dispatch_stages(
    const half* a,
    const half* b,
    half* c,
    int m,
    int n,
    int k,
    int stages,
    int m_tiles,
    int n_tiles,
    int swizzle_stride,
    std::string& error
) {
    switch(stages) {
        case 2:
            return launch_cute_hgemm_tn_v01_fast<2, BlockSwizzle>(
                a, b, c, m, n, k, m_tiles, n_tiles, swizzle_stride, error
            );
        case 3:
            return launch_cute_hgemm_tn_v01_fast<3, BlockSwizzle>(
                a, b, c, m, n, k, m_tiles, n_tiles, swizzle_stride, error
            );
        case 4:
            return launch_cute_hgemm_tn_v01_fast<4, BlockSwizzle>(
                a, b, c, m, n, k, m_tiles, n_tiles, swizzle_stride, error
            );
        default:
            error = "cute_hgemm_tn_v01 supports --stages 2, 3, or 4.";
            return false;
    }
}

}  // namespace

bool hgemm_cute_tn_v01(
    const half* a,
    const half* b,
    half* c,
    int m,
    int n,
    int k,
    int stages,
    bool swizzle,
    int swizzle_stride,
    std::string& error
) {
    const ai_system::profiling::ScopedNvtxRange launch_range("cute_hgemm_tn_v01_launch");
    if(!validate_problem(a, b, c, m, n, k, stages, swizzle, swizzle_stride, error)) {
        return false;
    }

    const int m_tiles = m / kBlockM;
    const int n_tiles = n / kBlockN;
    const bool fast_path_ran = m_tiles > 0 && n_tiles > 0 && (k % kBlockK) == 0;

    if(fast_path_ran) {
        const bool launched = swizzle
            ? dispatch_stages<true>(
                  a, b, c, m, n, k, stages, m_tiles, n_tiles, swizzle_stride, error
              )
            : dispatch_stages<false>(
                  a, b, c, m, n, k, stages, m_tiles, n_tiles, swizzle_stride, error
              );
        if(!launched) {
            return false;
        }
    }

    const int fast_m = m_tiles * kBlockM;
    const int fast_n = n_tiles * kBlockN;
    if(!fast_path_ran || fast_m != m || fast_n != n) {
        constexpr int kEdgeThreads = 256;
        const std::size_t element_count = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
        const auto edge_blocks = static_cast<unsigned int>(
            (element_count + kEdgeThreads - 1) / kEdgeThreads
        );
        cute_hgemm_tn_v01_edge_kernel<<<edge_blocks, kEdgeThreads>>>(
            a, b, c, m, n, k, fast_m, fast_n, fast_path_ran
        );
        if(!ai_system::cuda_utils::check_last_launch(error)) {
            return false;
        }
    }
    return true;
}

}  // namespace ai_system::labs::hgemm
