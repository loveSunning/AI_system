#include "bias_relu_lab_common.hpp"

#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"

#include <cstdint>
#include <exception>
#include <iostream>

namespace {

using ElementInput = cutlass::half_t;
using ElementOutput = float;
using ElementAccumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
constexpr int kStages = 3;
constexpr int kAlignmentA = 8;
constexpr int kAlignmentB = 8;
constexpr int kOutputElementsPerAccess = 4;

using UnfusedEpilogue = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    kOutputElementsPerAccess,
    ElementAccumulator,
    ElementOutput>;

using FusedEpilogue = cutlass::epilogue::thread::LinearCombinationRelu<
    ElementOutput,
    kOutputElementsPerAccess,
    ElementAccumulator,
    ElementOutput,
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

template <class Epilogue>
using Gemm2x = cutlass::gemm::device::Gemm<
    ElementInput, LayoutA,
    ElementInput, LayoutB,
    ElementOutput, LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    Epilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    kStages,
    kAlignmentA,
    kAlignmentB>;

using UnfusedGemm = Gemm2x<UnfusedEpilogue>;
using FusedGemm = Gemm2x<FusedEpilogue>;

int run(int argc, char const** argv) {
  cutlass_lab::Options options = cutlass_lab::parse_options(argc, argv);
  if (options.help) {
    cutlass_lab::bias_relu::print_usage(
        argv[0], "CUTLASS 2.x Device API bias+ReLU");
    return 0;
  }
  cutlass_lab::bias_relu::validate_options(options);

  cutlass_lab::bias_relu::ProblemStorage storage(options);
  cutlass_lab::bias_relu::initialize_problem(storage, options);
  cutlass_lab::bias_relu::print_environment(
      "CUTLASS 2.x Device API fused bias+ReLU", options, storage);
  std::cout << "API hierarchy      : DeviceGemm -> threadblock -> warp -> mma.sync\n"
            << "Tile hierarchy     : CTA 128x128x32, warp 64x64x32, instruction 16x8x16\n"
            << "Bias broadcast     : RowMajor TensorRef(bias, leading_dimension=0)\n"
            << "Epilogue functor   : thread::LinearCombinationRelu<..., NoBetaScaling>\n";

  typename UnfusedGemm::Arguments unfused_arguments(
      {storage.padded_m, storage.padded_n, storage.padded_k},
      {storage.a.get(), storage.lda},
      {storage.b.get(), storage.ldb},
      {storage.temporary.get(), storage.ldd},
      {storage.temporary.get(), storage.ldd},
      {options.alpha, 0.0f});

  typename FusedGemm::Arguments fused_arguments(
      {storage.padded_m, storage.padded_n, storage.padded_k},
      {storage.a.get(), storage.lda},
      {storage.b.get(), storage.ldb},
      {storage.bias.get(), 0},
      {storage.fused_output.get(), storage.ldd},
      {options.alpha});

  cutlass_lab::check_cutlass(
      UnfusedGemm::can_implement(unfused_arguments),
      "CUTLASS 2.x unfused can_implement");
  cutlass_lab::check_cutlass(
      FusedGemm::can_implement(fused_arguments),
      "CUTLASS 2.x fused can_implement");

  cutlass_lab::DeviceBuffer<std::uint8_t> unfused_workspace(
      UnfusedGemm::get_workspace_size(unfused_arguments));
  cutlass_lab::DeviceBuffer<std::uint8_t> fused_workspace(
      FusedGemm::get_workspace_size(fused_arguments));
  UnfusedGemm unfused_gemm;
  FusedGemm fused_gemm;
  cutlass_lab::check_cutlass(
      unfused_gemm.initialize(unfused_arguments, unfused_workspace.get()),
      "CUTLASS 2.x unfused initialize");
  cutlass_lab::check_cutlass(
      fused_gemm.initialize(fused_arguments, fused_workspace.get()),
      "CUTLASS 2.x fused initialize");

  float unfused_ms = cutlass_lab::benchmark(options, [&] {
    cutlass_lab::check_cutlass(unfused_gemm(), "CUTLASS 2.x unfused GEMM launch");
    cutlass_lab::bias_relu::launch_unfused_epilogue(storage, options);
  });
  float fused_ms = cutlass_lab::benchmark(options, [&] {
    cutlass_lab::check_cutlass(fused_gemm(), "CUTLASS 2.x fused GEMM launch");
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
