#include "gemm_lab_common.hpp"

#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"

#include <cstdint>
#include <iostream>

namespace {

using ElementInput = cutlass::half_t;
using ElementOutput = float;
using ElementAccumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

// CUTLASS 2.x mirrors the hardware hierarchy directly.
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
constexpr int kStages = 3;
constexpr int kAlignmentA = 8;  // 8 FP16 values = one 128-bit access.
constexpr int kAlignmentB = 8;
constexpr int kOutputElementsPerAccess = 4;  // 4 FP32 values = 128 bits.

using Epilogue = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    kOutputElementsPerAccess,
    ElementAccumulator,
    ElementOutput>;

using Gemm2x = cutlass::gemm::device::Gemm<
    ElementInput, LayoutA,
    ElementInput, LayoutB,
    ElementOutput, LayoutC,
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

int run(int argc, char const** argv) {
  cutlass_lab::Options options = cutlass_lab::parse_options(argc, argv);
  if (options.help) {
    cutlass_lab::print_usage(argv[0], "CUTLASS 2.x Device API");
    return 0;
  }

  cutlass_lab::ProblemStorage storage(options);
  cutlass_lab::initialize_problem(storage, options);
  cutlass_lab::print_environment("CUTLASS 2.x Device API GEMM", options, storage);
  std::cout << "API hierarchy      : DeviceGemm -> threadblock -> warp -> mma.sync\n"
            << "Tile hierarchy     : CTA 128x128x32, warp 64x64x32, instruction 16x8x16\n"
            << "Pipeline           : MmaMultistage, 3 shared-memory stages\n";

  typename Gemm2x::Arguments arguments(
      {storage.padded_m, storage.padded_n, storage.padded_k},
      {storage.a.get(), storage.lda},
      {storage.b.get(), storage.ldb},
      {storage.c.get(), storage.ldc},
      {storage.d.get(), storage.ldd},
      {options.alpha, options.beta});

  cutlass_lab::check_cutlass(Gemm2x::can_implement(arguments), "CUTLASS 2.x can_implement");
  cutlass_lab::DeviceBuffer<std::uint8_t> workspace(Gemm2x::get_workspace_size(arguments));
  Gemm2x gemm;
  cutlass_lab::check_cutlass(gemm.initialize(arguments, workspace.get()), "CUTLASS 2.x initialize");

  float average_ms = cutlass_lab::benchmark(options, [&] {
    cutlass_lab::check_cutlass(gemm(), "CUTLASS 2.x launch");
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
