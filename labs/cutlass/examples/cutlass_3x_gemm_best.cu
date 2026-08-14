#include "cute/tensor.hpp"

#include "gemm_lab_common.hpp"

#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/collective/sm80_mma_multistage.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/layout/matrix.h"

#include <cstdint>
#include <iostream>

namespace {

using namespace cute;

using ElementInput = cutlass::half_t;
using ElementOutput = float;
using ElementAccumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;

// RTX 5060 / SM120, M=8192, N=8192, K=4096 profiler result:
//   cutlass_tensorop_s16816gemm_f16_128x128_32x3_tn_align8
//
// Five confirmation runs measured a 13.0291 ms median (42.205 TFLOP/s).
// This 3.x kernel maps the winning algorithm topology into explicit CuTe types.
// The profiler operation is a legacy generated kernel, so the two binaries are
// expected to be close experiments rather than instruction-for-instruction clones.
using TileShape = Shape<_128, _128, _32>;
using DispatchPolicy = cutlass::gemm::MainloopSm80CpAsync<3>;

// Warp count = 2 x 2 x 1. Combined with the CTA tile above, each warp owns a
// conceptual 64 x 64 x 32 warp tile. The MMA atom is m16n8k16 FP16->FP32.
using TiledMma = TiledMMA<
    MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
    Layout<Shape<_2, _2, _1>>,
    Tile<_32, _32, _16>>;

using SmemLayoutAtomA = decltype(composition(
    Swizzle<2, 3, 3>{},
    Layout<Shape<_8, _32>, Stride<_32, _1>>{}));
using SmemLayoutAtomB = SmemLayoutAtomA;
using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementInput>;
using SmemCopyAtomB = Copy_Atom<SM75_U32x4_LDSM_N, ElementInput>;

// align8 means 8 FP16 elements = 128 bits per global-memory copy operation.
using GmemTiledCopyA = decltype(make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, ElementInput>{},
    Layout<Shape<_32, _4>, Stride<_4, _1>>{},
    Layout<Shape<_1, _8>>{}));
using GmemTiledCopyB = GmemTiledCopyA;

using StrideA = cutlass::gemm::TagToStrideA_t<LayoutA>;
using StrideB = cutlass::gemm::TagToStrideB_t<LayoutB>;
using StrideC = cutlass::gemm::TagToStrideC_t<LayoutC>;

using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    DispatchPolicy,
    TileShape,
    ElementInput, StrideA,
    ElementInput, StrideB,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity>;

using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
    ElementOutput,
    StrideC,
    StrideC,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 4, ElementAccumulator, ElementOutput>,
    cutlass::gemm::EpilogueDefault>;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue>;
using Gemm3x = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

cutlass_lab::Options best_problem_defaults() {
  cutlass_lab::Options options;
  options.m = 8192;
  options.n = 8192;
  options.k = 4096;
  options.warmup = 5;
  options.iterations = 20;
  return options;
}

int run(int argc, char const** argv) {
  cutlass_lab::Options defaults = best_problem_defaults();
  cutlass_lab::Options options = cutlass_lab::parse_options(argc, argv, defaults);
  if (options.help) {
    cutlass_lab::print_usage(argv[0], "CUTLASS 3.x profiler-selected GEMM", defaults);
    return 0;
  }

  // The profiled kernel family uses column-major C/D. This storage mode keeps
  // the profiler experiment and this implementation layout-compatible.
  cutlass_lab::ProblemStorage storage(options, true);
  cutlass_lab::initialize_problem(storage, options);
  cutlass_lab::print_environment(
      "CUTLASS 3.x profiler-selected GEMM for RTX 5060", options, storage);
  std::cout << "Selected profiler  : cutlass_tensorop_s16816gemm_f16_128x128_32x3_tn_align8\n"
            << "CTA tile           : 128x128x32\n"
            << "Warp topology      : 2x2x1 warps -> conceptual warp tile 64x64x32\n"
            << "MMA instruction    : m16n8k16, FP16 inputs, FP32 accumulate\n"
            << "Pipeline           : MainloopSm80CpAsync<3>\n"
            << "A/B alignment      : 8 FP16 elements = 128 bits\n"
            << "Split-K            : 1 (normal GEMM mode)\n"
            << "Architecture path  : SM80-compatible mma.sync compiled for SM120\n";

  StrideA stride_a{static_cast<std::int64_t>(storage.lda), _1{}, std::int64_t{0}};
  StrideB stride_b{static_cast<std::int64_t>(storage.ldb), _1{}, std::int64_t{0}};
  StrideC stride_c{_1{}, static_cast<std::int64_t>(storage.ldc), std::int64_t{0}};
  StrideC stride_d{_1{}, static_cast<std::int64_t>(storage.ldd), std::int64_t{0}};

  typename Gemm3x::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {storage.padded_m, storage.padded_n, storage.padded_k, 1},
      {storage.a.get(), stride_a, storage.b.get(), stride_b},
      {{options.alpha, options.beta}, storage.c.get(), stride_c, storage.d.get(), stride_d}};

  cutlass_lab::check_cutlass(Gemm3x::can_implement(arguments), "best GEMM can_implement");
  cutlass_lab::DeviceBuffer<std::uint8_t> workspace(Gemm3x::get_workspace_size(arguments));
  Gemm3x gemm;
  cutlass_lab::check_cutlass(
      gemm.initialize(arguments, workspace.get()), "best GEMM initialize");

  float average_ms = cutlass_lab::benchmark(options, [&] {
    cutlass_lab::check_cutlass(gemm.run(), "best GEMM launch");
  });
  cutlass_lab::print_performance(options, average_ms);
  return !options.verify || cutlass_lab::verify_result(storage, options) ? 0 : 2;
}

}  // namespace

int main(int argc, char const** argv) {
  try {
    return run(argc, argv);
  } catch (std::exception const& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
