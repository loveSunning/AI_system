#include "cute_gemm_demo_common.hpp"

#include <cute/tensor.hpp>

using namespace cute;

namespace cute_gemm_sm70_demo {

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorage {
  ArrayEngine<ElementA, cosize_v<SmemLayoutA>> a;
  ArrayEngine<ElementB, cosize_v<SmemLayoutB>> b;
};

template <class ProblemShape, class CtaTiler,
          class AStride, class ASmemLayout, class TiledCopyA,
          class BStride, class BSmemLayout, class TiledCopyB,
          class CStride, class TiledMma>
__global__ __launch_bounds__(decltype(size(TiledMma {}))::value)
void sm70_gemm_kernel(
    ProblemShape problem_shape,
    CtaTiler cta_tiler,
    half_t const* a,
    AStride d_a,
    ASmemLayout s_a_layout,
    TiledCopyA copy_a,
    half_t const* b,
    BStride d_b,
    BSmemLayout s_b_layout,
    TiledCopyB copy_b,
    half_t* c,
    CStride d_c,
    TiledMma mma) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700) && (__CUDA_ARCH__ < 800)
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
  Tensor t_ar_a = make_fragment_like(t_as_a(_, _, _, 0));

  ThrCopy g2s_thr_b = copy_b.get_slice(threadIdx.x);
  Tensor t_bg_b = g2s_thr_b.partition_S(g_b);
  Tensor t_bs_b = g2s_thr_b.partition_D(s_b);
  Tensor t_br_b = make_fragment_like(t_bs_b(_, _, _, 0));

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor t_cg_c = thr_mma.partition_C(g_c);
  Tensor t_cr_c = thr_mma.make_fragment_C(t_cg_c);
  clear(t_cr_c);

  int const k_tile_count = size<3>(t_ag_a);
  copy(copy_a, t_ag_a(_, _, _, 0), t_ar_a);
  copy(copy_b, t_bg_b(_, _, _, 0), t_br_b);
  copy(t_ar_a, t_as_a(_, _, _, 0));
  copy(t_br_b, t_bs_b(_, _, _, 0));
  __syncthreads();

  int smem_read = 0;
  int smem_write = 1;

  // Volta has no cp.async: stage through G2R registers, then alternate two SMEM tiles.
  CUTE_NO_UNROLL
  for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
    Tensor t_cs_a = thr_mma.partition_A(s_a(_, _, smem_read));
    Tensor t_cs_b = thr_mma.partition_B(s_b(_, _, smem_read));
    Tensor t_cr_a = thr_mma.make_fragment_A(t_cs_a);
    Tensor t_cr_b = thr_mma.make_fragment_B(t_cs_b);

    int const k_block_count = size<2>(t_cr_a);
    copy(t_cs_a(_, _, 0), t_cr_a(_, _, 0));
    copy(t_cs_b(_, _, 0), t_cr_b(_, _, 0));

    if (k_tile + 1 < k_tile_count) {
      copy(copy_a, t_ag_a(_, _, _, k_tile + 1), t_ar_a);
      copy(copy_b, t_bg_b(_, _, _, k_tile + 1), t_br_b);
      copy(t_ar_a, t_as_a(_, _, _, smem_write));
      copy(t_br_b, t_bs_b(_, _, _, smem_write));
    }

    CUTE_UNROLL
    for (int k_block = 0; k_block < k_block_count; ++k_block) {
      int k_block_next = (k_block + 1) % k_block_count;
      copy(t_cs_a(_, _, k_block_next), t_cr_a(_, _, k_block_next));
      copy(t_cs_b(_, _, k_block_next), t_cr_b(_, _, k_block_next));
      gemm(mma, t_cr_a(_, _, k_block), t_cr_b(_, _, k_block), t_cr_c);
    }

    __syncthreads();
    smem_read = smem_write;
    smem_write = smem_write == 0 ? 1 : 0;
  }

  axpby(1.0f, t_cr_c, 0.0f, t_cg_c);
#else
  (void)problem_shape;
  (void)cta_tiler;
  (void)a;
  (void)d_a;
  (void)s_a_layout;
  (void)copy_a;
  (void)b;
  (void)d_b;
  (void)s_b_layout;
  (void)copy_b;
  (void)c;
  (void)d_c;
  (void)mma;
#endif
}

template <class MmaOperation, class AStride, class BStride,
          class ASmemLayout, class BSmemLayout,
          class TiledCopyA, class TiledCopyB>
void launch_configured(
    half_t const* a,
    AStride d_a,
    ASmemLayout s_a,
    TiledCopyA copy_a,
    half_t const* b,
    BStride d_b,
    BSmemLayout s_b,
    TiledCopyB copy_b,
    half_t* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  auto problem_shape = make_shape(m, n, k);
  auto cta_tiler = make_shape(Int<128> {}, Int<128> {}, Int<32> {});
  auto d_c = make_stride(Int<1> {}, m);
  auto mma = make_tiled_mma(MmaOperation {}, Layout<Shape<_4, _4, _1>> {});

  auto kernel = sm70_gemm_kernel<
      decltype(problem_shape), decltype(cta_tiler),
      AStride, ASmemLayout, TiledCopyA,
      BStride, BSmemLayout, TiledCopyB,
      decltype(d_c), decltype(mma)>;

  int smem_bytes = int(sizeof(SharedStorage<half_t, half_t, ASmemLayout, BSmemLayout>));
  dim3 block(size(mma));
  dim3 grid(ceil_div(m, 128), ceil_div(n, 128));
  kernel<<<grid, block, smem_bytes, stream>>>(
      problem_shape, cta_tiler,
      a, d_a, s_a, copy_a,
      b, d_b, s_b, copy_b,
      c, d_c, mma);
}

void launch_sm70(
    cute_gemm_demo::LayoutMode mode,
    half_t const* a,
    half_t const* b,
    half_t* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  if (mode == cute_gemm_demo::LayoutMode::TN) {
    auto d_a = make_stride(k, Int<1> {});
    auto d_b = make_stride(k, Int<1> {});
    auto s_a = make_layout(
        make_shape(Int<128> {}, Int<32> {}, Int<2> {}),
        make_stride(Int<32> {}, Int<1> {}, Int<4096> {}));
    auto s_b = s_a;
    auto copy_a = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, half_t> {},
        Layout<Shape<_32, _4>, Stride<_4, _1>> {},
        Layout<Shape<_1, _8>> {});
    auto copy_b = copy_a;
    launch_configured<SM70_8x8x4_F32F16F16F32_TN>(
        a, d_a, s_a, copy_a,
        b, d_b, s_b, copy_b,
        c, m, n, k, stream);
  } else {
    auto d_a = make_stride(Int<1> {}, m);
    auto d_b = make_stride(Int<1> {}, n);
    auto s_a = make_layout(
        make_shape(Int<128> {}, Int<32> {}, Int<2> {}),
        make_stride(Int<1> {}, Int<128> {}, Int<4096> {}));
    auto s_b = s_a;
    auto copy_a = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, half_t> {},
        Layout<Shape<_16, _8>> {},
        Layout<Shape<_8, _1>> {});
    auto copy_b = copy_a;
    launch_configured<SM70_8x8x4_F32F16F16F32_NT>(
        a, d_a, s_a, copy_a,
        b, d_b, s_b, copy_b,
        c, m, n, k, stream);
  }
}

}  // namespace cute_gemm_sm70_demo

int main(int argc, char** argv) {
  cudaDeviceProp props {};
  cute_gemm_demo::check_cuda(cudaGetDeviceProperties(&props, 0), "cudaGetDeviceProperties");
  bool is_volta_family = props.major == 7;
  return cute_gemm_demo::run_main(
      argc,
      argv,
      "cute_gemm_sm70_mma",
      128,
      128,
      32,
      is_volta_family,
      "the SM70 m8n8k4 MMA demo must run on a compute-capability 7.x GPU",
      cute_gemm_sm70_demo::launch_sm70);
}
