#include "cute/tensor.hpp"

#include "bias_relu_lab_common.hpp"

#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/collective/sm80_mma_multistage.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/layout/matrix.h"

#include <cstdint>
#include <exception>
#include <iostream>

namespace {

using namespace cute;

using ElementInput = cutlass::half_t;
using ElementOutput = float;
using ElementAccumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

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
using StrideOutput = cutlass::gemm::TagToStrideC_t<LayoutOutput>;

using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    DispatchPolicy,
    TileShape,
    ElementInput, StrideA,
    ElementInput, StrideB,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity>;

using UnfusedThreadEpilogue = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, 1, ElementAccumulator, ElementOutput>;

// The SM80-compatible DefaultEpilogue expects the modern ElementD alias. The
// legacy LinearCombinationRelu functor predates that alias, so this thin adapter
// supplies metadata only; all math and parameters remain the CUTLASS functor.
class FusedThreadEpilogue
    : public cutlass::epilogue::thread::LinearCombinationRelu<
          ElementOutput,
          1,
          ElementAccumulator,
          ElementOutput,
          cutlass::epilogue::thread::ScaleType::NoBetaScaling> {
 public:
  using Base = cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      1,
      ElementAccumulator,
      ElementOutput,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
  using ElementD = ElementOutput;

  CUTLASS_HOST_DEVICE
  explicit FusedThreadEpilogue(typename Base::Params const& params) : Base(params) {}

  // This collective's DefaultEpilogue visits one scalar accumulator at a time.
  // Adapt that call to the one-element fragment interface of the legacy CUTLASS
  // ReLU functor.
  CUTLASS_HOST_DEVICE
  ElementD operator()(ElementAccumulator const accumulator, ElementOutput const source) const {
    typename Base::FragmentAccumulator accumulator_fragment;
    typename Base::FragmentOutput source_fragment;
    accumulator_fragment[0] = accumulator;
    source_fragment[0] = source;
    return Base::operator()(accumulator_fragment, source_fragment)[0];
  }

  CUTLASS_HOST_DEVICE
  ElementD operator()(ElementAccumulator const accumulator) const {
    typename Base::FragmentAccumulator accumulator_fragment;
    accumulator_fragment[0] = accumulator;
    return Base::operator()(accumulator_fragment)[0];
  }
};

template <class ThreadEpilogue>
using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
    ElementOutput,
    StrideOutput,
    StrideOutput,
    ThreadEpilogue,
    cutlass::gemm::EpilogueDefault>;

template <class ThreadEpilogue>
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue<ThreadEpilogue>>;

template <class ThreadEpilogue>
using Gemm3x = cutlass::gemm::device::GemmUniversalAdapter<
    GemmKernel<ThreadEpilogue>>;

using UnfusedGemm = Gemm3x<UnfusedThreadEpilogue>;
using FusedGemm = Gemm3x<FusedThreadEpilogue>;

int run(int argc, char const** argv) {
  cutlass_lab::Options options = cutlass_lab::parse_options(argc, argv);
  if (options.help) {
    cutlass_lab::bias_relu::print_usage(
        argv[0], "CUTLASS 3.x Collective API bias+ReLU");
    return 0;
  }
  cutlass_lab::bias_relu::validate_options(options);

  cutlass_lab::bias_relu::ProblemStorage storage(options);
  cutlass_lab::bias_relu::initialize_problem(storage, options);
  cutlass_lab::bias_relu::print_environment(
      "CUTLASS 3.x Collective/CuTe API fused bias+ReLU", options, storage);
  std::cout << "API hierarchy      : Adapter -> Kernel -> Collective -> TiledMMA/Copy -> Atom\n"
            << "Tile hierarchy     : CTA 128x128x32, TiledMMA, MMA_Atom m16n8k16\n"
            << "Bias broadcast     : StrideC{0, 1, 0} projects away M and batch\n"
            << "Epilogue functor   : DefaultEpilogue<LinearCombinationRelu<...>>\n";

  StrideA stride_a{static_cast<std::int64_t>(storage.lda), _1{}, std::int64_t{0}};
  StrideB stride_b{static_cast<std::int64_t>(storage.ldb), _1{}, std::int64_t{0}};
  StrideOutput output_stride{
      static_cast<std::int64_t>(storage.ldd), _1{}, std::int64_t{0}};
  StrideOutput bias_stride{std::int64_t{0}, _1{}, std::int64_t{0}};

  typename UnfusedGemm::Arguments unfused_arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {storage.padded_m, storage.padded_n, storage.padded_k, 1},
      {storage.a.get(), stride_a, storage.b.get(), stride_b},
      {{options.alpha, 0.0f},
       storage.temporary.get(), output_stride,
       storage.temporary.get(), output_stride}};

  typename FusedGemm::Arguments fused_arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {storage.padded_m, storage.padded_n, storage.padded_k, 1},
      {storage.a.get(), stride_a, storage.b.get(), stride_b},
      {{options.alpha},
       storage.bias.get(), bias_stride,
       storage.fused_output.get(), output_stride}};

  cutlass_lab::check_cutlass(
      UnfusedGemm::can_implement(unfused_arguments),
      "CUTLASS 3.x unfused can_implement");
  cutlass_lab::check_cutlass(
      FusedGemm::can_implement(fused_arguments),
      "CUTLASS 3.x fused can_implement");

  cutlass_lab::DeviceBuffer<std::uint8_t> unfused_workspace(
      UnfusedGemm::get_workspace_size(unfused_arguments));
  cutlass_lab::DeviceBuffer<std::uint8_t> fused_workspace(
      FusedGemm::get_workspace_size(fused_arguments));
  UnfusedGemm unfused_gemm;
  FusedGemm fused_gemm;
  cutlass_lab::check_cutlass(
      unfused_gemm.initialize(unfused_arguments, unfused_workspace.get()),
      "CUTLASS 3.x unfused initialize");
  cutlass_lab::check_cutlass(
      fused_gemm.initialize(fused_arguments, fused_workspace.get()),
      "CUTLASS 3.x fused initialize");

  float unfused_ms = cutlass_lab::benchmark(options, [&] {
    cutlass_lab::check_cutlass(
        unfused_gemm.run(), "CUTLASS 3.x unfused GEMM launch");
    cutlass_lab::bias_relu::launch_unfused_epilogue(storage, options);
  });
  float fused_ms = cutlass_lab::benchmark(options, [&] {
    cutlass_lab::check_cutlass(
        fused_gemm.run(), "CUTLASS 3.x fused GEMM launch");
  });

  cutlass_lab::bias_relu::print_performance_comparison(options, unfused_ms, fused_ms);
  return !options.verify || cutlass_lab::bias_relu::verify_result(storage, options) ? 0 : 2;
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
