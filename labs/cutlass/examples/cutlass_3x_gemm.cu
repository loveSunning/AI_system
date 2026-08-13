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
using LayoutC = cutlass::layout::RowMajor;

// CUTLASS 3.x describes the same SM80-compatible Tensor Core algorithm with
// composable collectives and CuTe layouts rather than 2.x iterator classes.
// The shared learning harness defaults to the naturally aligned 4096^3 problem.
using TileShape = Shape<_128, _128, _32>;
using DispatchPolicy = cutlass::gemm::MainloopSm80CpAsync<3>;

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
        ElementOutput, 1, ElementAccumulator, ElementOutput>,
    cutlass::gemm::EpilogueDefault>;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue>;
using Gemm3x = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

int run(int argc, char const** argv) {
  cutlass_lab::Options options = cutlass_lab::parse_options(argc, argv);
  if (options.help) {
    cutlass_lab::print_usage(argv[0], "CUTLASS 3.x Collective API");
    return 0;
  }

  cutlass_lab::ProblemStorage storage(options);
  cutlass_lab::initialize_problem(storage, options);
  cutlass_lab::print_environment("CUTLASS 3.x Collective/CuTe API GEMM", options, storage);
  std::cout << "API hierarchy      : Adapter -> Kernel -> Collective -> TiledMMA/Copy -> Atom\n"
            << "CuTe hierarchy     : TileShape 128x128x32 -> TiledMMA -> MMA_Atom m16n8k16\n"
            << "Pipeline           : MainloopSm80CpAsync<3>, explicit TiledCopy\n"
            << "Architecture path  : SM80-compatible mma.sync; runs on SM89 and SM120\n";

  StrideA stride_a{static_cast<std::int64_t>(storage.lda), _1{}, std::int64_t{0}};
  StrideB stride_b{static_cast<std::int64_t>(storage.ldb), _1{}, std::int64_t{0}};
  StrideC stride_c{static_cast<std::int64_t>(storage.ldc), _1{}, std::int64_t{0}};
  StrideC stride_d{static_cast<std::int64_t>(storage.ldd), _1{}, std::int64_t{0}};
  typename Gemm3x::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {storage.padded_m, storage.padded_n, storage.padded_k, 1},
      {storage.a.get(), stride_a, storage.b.get(), stride_b},
      {{options.alpha, options.beta}, storage.c.get(), stride_c, storage.d.get(), stride_d}};

  cutlass_lab::check_cutlass(Gemm3x::can_implement(arguments), "CUTLASS 3.x can_implement");
  cutlass_lab::DeviceBuffer<std::uint8_t> workspace(Gemm3x::get_workspace_size(arguments));
  Gemm3x gemm;
  cutlass_lab::check_cutlass(gemm.initialize(arguments, workspace.get()), "CUTLASS 3.x initialize");

  float average_ms = cutlass_lab::benchmark(options, [&] {
    cutlass_lab::check_cutlass(gemm.run(), "CUTLASS 3.x launch");
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
