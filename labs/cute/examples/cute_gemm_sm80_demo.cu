#include "cute_gemm_demo_common.hpp"

#include <cute/tensor.hpp>

using namespace cute;

namespace cute_gemm_sm80_demo {

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorage {
  ArrayEngine<ElementA, cosize_v<SmemLayoutA>> a;
  ArrayEngine<ElementB, cosize_v<SmemLayoutB>> b;
};

template <class ProblemShape, class CtaTiler,
          class AStride, class ASmemLayout, class TiledCopyA, class S2RAtomA,
          class BStride, class BSmemLayout, class TiledCopyB, class S2RAtomB,
          class CStride, class TiledMma>
__global__ __launch_bounds__(decltype(size(TiledMma {}))::value)
void sm80_gemm_kernel(
    ProblemShape problem_shape,
    CtaTiler cta_tiler,
    half_t const* a,
    AStride d_a,
    ASmemLayout s_a_layout,
    TiledCopyA copy_a,
    S2RAtomA s2r_atom_a,
    half_t const* b,
    BStride d_b,
    BSmemLayout s_b_layout,
    TiledCopyB copy_b,
    S2RAtomB s2r_atom_b,
    half_t* c,
    CStride d_c,
    TiledMma mma) {
  Tensor m_a = make_tensor(make_gmem_ptr(a), select<0, 2>(problem_shape), d_a);
  Tensor m_b = make_tensor(make_gmem_ptr(b), select<1, 2>(problem_shape), d_b);
  Tensor m_c = make_tensor(make_gmem_ptr(c), select<0, 1>(problem_shape), d_c);

  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor g_a = local_tile(m_a, cta_tiler, cta_coord, Step<_1, X, _1> {});
  Tensor g_b = local_tile(m_b, cta_tiler, cta_coord, Step<X, _1, _1> {});
  Tensor g_c = local_tile(m_c, cta_tiler, cta_coord, Step<_1, _1, X> {});

  extern __shared__ char shared_memory[];
  using Storage = SharedStorage<half_t, half_t, ASmemLayout, BSmemLayout>;
  Storage& storage = *reinterpret_cast<Storage*>(shared_memory);
  Tensor s_a = make_tensor(make_smem_ptr(storage.a.begin()), s_a_layout);
  Tensor s_b = make_tensor(make_smem_ptr(storage.b.begin()), s_b_layout);

  ThrCopy g2s_thr_a = copy_a.get_slice(threadIdx.x);
  Tensor t_ag_a = g2s_thr_a.partition_S(g_a);
  Tensor t_as_a = g2s_thr_a.partition_D(s_a);
  ThrCopy g2s_thr_b = copy_b.get_slice(threadIdx.x);
  Tensor t_bg_b = g2s_thr_b.partition_S(g_b);
  Tensor t_bs_b = g2s_thr_b.partition_D(s_b);

  int const pipe_count = size<3>(t_as_a);
  int k_tile_count = size<3>(t_ag_a);
  int k_tile_next = 0;

  CUTE_UNROLL
  for (int pipe = 0; pipe < pipe_count - 1; ++pipe) {
    copy(copy_a, t_ag_a(_, _, _, k_tile_next), t_as_a(_, _, _, pipe));
    copy(copy_b, t_bg_b(_, _, _, k_tile_next), t_bs_b(_, _, _, pipe));
    cp_async_fence();
    --k_tile_count;
    if (k_tile_count > 0) {
      ++k_tile_next;
    }
  }

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor t_cg_c = thr_mma.partition_C(g_c);
  Tensor t_cr_a = thr_mma.partition_fragment_A(s_a(_, _, 0));
  Tensor t_cr_b = thr_mma.partition_fragment_B(s_b(_, _, 0));
  Tensor t_cr_c = thr_mma.make_fragment_C(t_cg_c);
  clear(t_cr_c);

  TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
  ThrCopy s2r_thr_a = s2r_copy_a.get_slice(threadIdx.x);
  Tensor t_xs_a = s2r_thr_a.partition_S(s_a);
  Tensor t_xr_a = s2r_thr_a.retile_D(t_cr_a);

  TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
  ThrCopy s2r_thr_b = s2r_copy_b.get_slice(threadIdx.x);
  Tensor t_xs_b = s2r_thr_b.partition_S(s_b);
  Tensor t_xr_b = s2r_thr_b.retile_D(t_cr_b);

  int smem_read = 0;
  int smem_write = pipe_count - 1;
  Tensor t_xs_a_pipe = t_xs_a(_, _, _, smem_read);
  Tensor t_xs_b_pipe = t_xs_b(_, _, _, smem_read);

  int const k_block_count = size<2>(t_cr_a);
  if (k_block_count > 1) {
    cp_async_wait<decltype(size<2>(ASmemLayout {}))::value - 2>();
    __syncthreads();
    copy(s2r_atom_a, t_xs_a_pipe(_, _, Int<0> {}), t_xr_a(_, _, Int<0> {}));
    copy(s2r_atom_b, t_xs_b_pipe(_, _, Int<0> {}), t_xr_b(_, _, Int<0> {}));
  }

  // Three concurrent levels: G2S cp.async, S2R ldmatrix, and register MMA.
  CUTE_NO_UNROLL
  while (k_tile_count > -(pipe_count - 1)) {
    CUTE_UNROLL
    for (int k_block = 0; k_block < k_block_count; ++k_block) {
      if (k_block == k_block_count - 1) {
        t_xs_a_pipe = t_xs_a(_, _, _, smem_read);
        t_xs_b_pipe = t_xs_b(_, _, _, smem_read);
        cp_async_wait<decltype(size<2>(ASmemLayout {}))::value - 2>();
        __syncthreads();
      }

      int k_block_next = (k_block + 1) % k_block_count;
      copy(s2r_atom_a, t_xs_a_pipe(_, _, k_block_next), t_xr_a(_, _, k_block_next));
      copy(s2r_atom_b, t_xs_b_pipe(_, _, k_block_next), t_xr_b(_, _, k_block_next));

      if (k_block == 0) {
        copy(copy_a, t_ag_a(_, _, _, k_tile_next), t_as_a(_, _, _, smem_write));
        copy(copy_b, t_bg_b(_, _, _, k_tile_next), t_bs_b(_, _, _, smem_write));
        cp_async_fence();

        --k_tile_count;
        if (k_tile_count > 0) {
          ++k_tile_next;
        }
        smem_write = smem_read;
        smem_read = smem_read == pipe_count - 1 ? 0 : smem_read + 1;
      }

      gemm(mma, t_cr_a(_, _, k_block), t_cr_b(_, _, k_block), t_cr_c);
    }
  }

  axpby(1.0f, t_cr_c, 0.0f, t_cg_c);
}

template <class AStride, class BStride, class ASmemLayout, class BSmemLayout,
          class TiledCopyA, class TiledCopyB, class S2RAtomA, class S2RAtomB,
          class TiledMma>
void launch_configured(
    half_t const* a,
    AStride d_a,
    ASmemLayout s_a,
    TiledCopyA copy_a,
    S2RAtomA s2r_a,
    half_t const* b,
    BStride d_b,
    BSmemLayout s_b,
    TiledCopyB copy_b,
    S2RAtomB s2r_b,
    half_t* c,
    int m,
    int n,
    int k,
    TiledMma mma,
    cudaStream_t stream) {
  auto problem_shape = make_shape(m, n, k);
  auto cta_tiler = make_shape(Int<128> {}, Int<128> {}, Int<64> {});
  auto d_c = make_stride(Int<1> {}, m);

  auto kernel = sm80_gemm_kernel<
      decltype(problem_shape), decltype(cta_tiler),
      AStride, ASmemLayout, TiledCopyA, S2RAtomA,
      BStride, BSmemLayout, TiledCopyB, S2RAtomB,
      decltype(d_c), TiledMma>;
  int smem_bytes = int(sizeof(SharedStorage<half_t, half_t, ASmemLayout, BSmemLayout>));
  cute_gemm_demo::check_cuda(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes),
      "set SM80 dynamic shared-memory size");
  cute_gemm_demo::check_cuda(
      cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100),
      "set SM80 shared-memory carveout");

  dim3 block(size(mma));
  dim3 grid(ceil_div(m, 128), ceil_div(n, 128));
  kernel<<<grid, block, smem_bytes, stream>>>(
      problem_shape, cta_tiler,
      a, d_a, s_a, copy_a, s2r_a,
      b, d_b, s_b, copy_b, s2r_b,
      c, d_c, mma);
}

void launch_sm80(
    cute_gemm_demo::LayoutMode mode,
    half_t const* a,
    half_t const* b,
    half_t* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  auto mma = make_tiled_mma(
      SM80_16x8x16_F32F16F16F32_TN {},
      Layout<Shape<_2, _2>> {},
      Tile<_32, _32, _16> {});

  if (mode == cute_gemm_demo::LayoutMode::TN) {
    // K-contiguous vectors stay contiguous through the swizzle; LDSM_N reads them.
    auto d_a = make_stride(k, Int<1> {});
    auto d_b = make_stride(k, Int<1> {});
    auto swizzle = composition(
        Swizzle<3, 3, 3> {},
        Layout<Shape<_8, Shape<_8, _8>>, Stride<_8, Stride<_1, _64>>> {});
    auto s_a = tile_to_shape(swizzle, make_shape(Int<128> {}, Int<64> {}, Int<3> {}));
    auto s_b = tile_to_shape(swizzle, make_shape(Int<128> {}, Int<64> {}, Int<3> {}));
    auto copy_a = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t> {},
        Layout<Shape<_16, _8>, Stride<_8, _1>> {},
        Layout<Shape<_1, _8>> {});
    auto copy_b = copy_a;
    auto s2r_a = Copy_Atom<SM75_U32x4_LDSM_N, half_t> {};
    auto s2r_b = Copy_Atom<SM75_U32x4_LDSM_N, half_t> {};
    launch_configured(
        a, d_a, s_a, copy_a, s2r_a,
        b, d_b, s_b, copy_b, s2r_b,
        c, m, n, k, mma, stream);
  } else {
    // M/N-contiguous vectors use the transposed swizzle and LDSM_T register mapping.
    auto d_a = make_stride(Int<1> {}, m);
    auto d_b = make_stride(Int<1> {}, n);
    auto swizzle = composition(
        Swizzle<3, 3, 3> {},
        Layout<Shape<Shape<_8, _8>, _8>, Stride<Stride<_1, _64>, _8>> {});
    auto s_a = tile_to_shape(swizzle, make_shape(Int<128> {}, Int<64> {}, Int<3> {}));
    auto s_b = tile_to_shape(swizzle, make_shape(Int<128> {}, Int<64> {}, Int<3> {}));
    auto copy_a = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t> {},
        Layout<Shape<_16, _8>> {},
        Layout<Shape<_8, _1>> {});
    auto copy_b = copy_a;
    auto s2r_a = Copy_Atom<SM75_U16x8_LDSM_T, half_t> {};
    auto s2r_b = Copy_Atom<SM75_U16x8_LDSM_T, half_t> {};
    launch_configured(
        a, d_a, s_a, copy_a, s2r_a,
        b, d_b, s_b, copy_b, s2r_b,
        c, m, n, k, mma, stream);
  }
}

}  // namespace cute_gemm_sm80_demo

int main(int argc, char** argv) {
  cudaDeviceProp props {};
  cute_gemm_demo::check_cuda(cudaGetDeviceProperties(&props, 0), "cudaGetDeviceProperties");
  return cute_gemm_demo::run_main(
      argc,
      argv,
      "cute_gemm_sm80_mma",
      128,
      128,
      64,
      props.major >= 8,
      "SM80 MMA, cp.async, and ldmatrix require an Ampere-or-newer GPU",
      cute_gemm_sm80_demo::launch_sm80);
}
