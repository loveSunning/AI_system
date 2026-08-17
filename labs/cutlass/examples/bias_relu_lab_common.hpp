#pragma once

#include "gemm_lab_common.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace cutlass_lab::bias_relu {

struct ProblemStorage {
  explicit ProblemStorage(Options const& options)
      : padded_m(round_up(options.m, 128)),
        padded_n(round_up(options.n, 128)),
        padded_k(round_up(options.k, 32)),
        lda(padded_k),
        ldb(padded_k),
        ldd(padded_n),
        a(static_cast<std::size_t>(padded_m) * lda),
        b(static_cast<std::size_t>(padded_n) * ldb),
        bias(padded_n),
        temporary(static_cast<std::size_t>(padded_m) * ldd),
        unfused_output(static_cast<std::size_t>(padded_m) * ldd),
        fused_output(static_cast<std::size_t>(padded_m) * ldd) {}

  int padded_m;
  int padded_n;
  int padded_k;
  int lda;
  int ldb;
  int ldd;
  DeviceBuffer<cutlass::half_t> a;
  DeviceBuffer<cutlass::half_t> b;
  DeviceBuffer<float> bias;
  DeviceBuffer<float> temporary;
  DeviceBuffer<float> unfused_output;
  DeviceBuffer<float> fused_output;
};

__global__ void fill_bias(
    float* bias, int logical_n, int padded_n, float accumulator_term) {
  int column = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (column < padded_n) {
    // With A=B=1, alpha*AB is accumulator_term. Logical columns therefore
    // alternate between pre-activation -1 and +1, exercising both ReLU paths.
    bias[column] = column < logical_n
        ? -accumulator_term + ((column & 1) ? 1.0f : -1.0f)
        : 0.0f;
  }
}

__global__ void apply_bias_relu(
    float const* input,
    float const* bias,
    float* output,
    int rows,
    int columns,
    int leading_dimension) {
  std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::size_t total = static_cast<std::size_t>(rows) * leading_dimension;
  if (index < total) {
    int column = static_cast<int>(index % leading_dimension);
    float value = input[index] + (column < columns ? bias[column] : 0.0f);
    output[index] = value > 0.0f ? value : 0.0f;
  }
}

__global__ void count_mismatches(
    float const* unfused,
    float const* fused,
    int rows,
    int columns,
    int leading_dimension,
    float tolerance,
    unsigned long long* mismatch_count) {
  std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::size_t total = static_cast<std::size_t>(rows) * columns;
  if (index < total) {
    int row = static_cast<int>(index / columns);
    int column = static_cast<int>(index % columns);
    std::size_t offset = static_cast<std::size_t>(row) * leading_dimension + column;
    float expected = (column & 1) ? 1.0f : 0.0f;
    float unfused_value = unfused[offset];
    float fused_value = fused[offset];
    bool invalid = !isfinite(unfused_value) || !isfinite(fused_value) ||
                   fabsf(unfused_value - expected) > tolerance ||
                   fabsf(fused_value - expected) > tolerance ||
                   fabsf(unfused_value - fused_value) > tolerance;
    if (invalid) {
      atomicAdd(mismatch_count, 1ULL);
    }
  }
}

inline void initialize_problem(ProblemStorage& storage, Options const& options) {
  constexpr int threads = 256;
  fill_row_major_half<<<grid_for(storage.a.size()), threads>>>(
      storage.a.get(), storage.padded_m, options.k, storage.lda, 1.0f);
  fill_column_major_half<<<grid_for(storage.b.size()), threads>>>(
      storage.b.get(), options.k, storage.padded_n, storage.ldb, 1.0f);
  fill_bias<<<grid_for(storage.bias.size()), threads>>>(
      storage.bias.get(), options.n, storage.padded_n,
      options.alpha * static_cast<float>(options.k));
  check_cuda(cudaMemset(
      storage.temporary.get(), 0, storage.temporary.size() * sizeof(float)),
      "clear temporary output");
  check_cuda(cudaMemset(
      storage.unfused_output.get(), 0, storage.unfused_output.size() * sizeof(float)),
      "clear unfused output");
  check_cuda(cudaMemset(
      storage.fused_output.get(), 0, storage.fused_output.size() * sizeof(float)),
      "clear fused output");
  check_cuda(cudaGetLastError(), "bias+ReLU initialization kernels");
  check_cuda(cudaDeviceSynchronize(), "bias+ReLU initialization synchronization");
}

inline void launch_unfused_epilogue(ProblemStorage const& storage, Options const& options) {
  constexpr int threads = 256;
  apply_bias_relu<<<grid_for(storage.unfused_output.size()), threads>>>(
      storage.temporary.get(), storage.bias.get(), storage.unfused_output.get(),
      storage.padded_m, options.n, storage.ldd);
  check_cuda(cudaGetLastError(), "unfused bias+ReLU launch");
}

inline bool verify_result(ProblemStorage const& storage, Options const& options) {
  DeviceBuffer<unsigned long long> mismatch_count(1);
  check_cuda(cudaMemset(mismatch_count.get(), 0, sizeof(unsigned long long)),
             "clear bias+ReLU mismatch count");

  float tolerance = 1.0e-3f;
  std::size_t logical_elements = static_cast<std::size_t>(options.m) * options.n;
  count_mismatches<<<grid_for(logical_elements), 256>>>(
      storage.unfused_output.get(), storage.fused_output.get(),
      options.m, options.n, storage.ldd, tolerance, mismatch_count.get());
  check_cuda(cudaGetLastError(), "bias+ReLU verification kernel");

  unsigned long long host_mismatches = 0;
  check_cuda(cudaMemcpy(
      &host_mismatches, mismatch_count.get(), sizeof(host_mismatches),
      cudaMemcpyDeviceToHost), "copy bias+ReLU verification result");

  std::cout << "Verification       : "
            << (host_mismatches == 0 ? "PASSED" : "FAILED")
            << " (expected columns alternate 0/1, tolerance " << tolerance
            << ", mismatches " << host_mismatches << ")\n";
  return host_mismatches == 0;
}

inline void print_usage(char const* executable, char const* api_name) {
  Options defaults;
  std::cout
      << api_name << " fused epilogue learning example\n\n"
      << "Usage: " << executable << " [options]\n\n"
      << "  --m=<int>             logical M (default " << defaults.m << ")\n"
      << "  --n=<int>             logical N (default " << defaults.n << ")\n"
      << "  --k=<int>             logical K (default " << defaults.k << ")\n"
      << "  --warmup=<int>        untimed launches per path (default "
      << defaults.warmup << ")\n"
      << "  --iterations=<int>    timed launches per path (default "
      << defaults.iterations << ")\n"
      << "  --alpha=<float>       GEMM accumulator scale (default 1)\n"
      << "  --no-verify           skip full logical-output verification\n"
      << "  --help                show this message\n\n"
      << "Compares two equivalent paths:\n"
      << "  unfused: T = alpha*A*B; D = relu(T + bias)\n"
      << "  fused:   D = relu(alpha*A*B + bias) in the CUTLASS epilogue\n";
}

inline cudaDeviceProp print_environment(
    char const* api_name, Options const& options, ProblemStorage const& storage) {
  int device = 0;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
  if (properties.major * 10 + properties.minor < 80) {
    throw std::runtime_error("this example requires SM80 or newer for mma.sync m16n8k16");
  }

  bool const is_padded = storage.padded_m != options.m ||
                         storage.padded_n != options.n ||
                         storage.padded_k != options.k;
  double logical_saved_mib =
      2.0 * static_cast<double>(options.m) * options.n * sizeof(float) /
      (1024.0 * 1024.0);
  double executed_saved_mib =
      2.0 * static_cast<double>(storage.padded_m) * storage.padded_n * sizeof(float) /
      (1024.0 * 1024.0);

  std::cout << api_name << "\n"
            << "CUTLASS version    : " << CUTLASS_MAJOR << '.' << CUTLASS_MINOR
            << '.' << CUTLASS_PATCH << '\n'
            << "GPU                : " << properties.name << " (SM"
            << properties.major << properties.minor << ")\n"
            << "Logical problem    : " << options.m << " x " << options.n
            << " x " << options.k << '\n'
            << "Executed problem   : " << storage.padded_m << " x "
            << storage.padded_n << " x " << storage.padded_k
            << (is_padded ? " (padded)\n" : " (no padding)\n")
            << "Output layout      : row-major; bias[N] broadcasts across M\n"
            << "Data path          : FP16 inputs -> FP32 Tensor Core accumulate -> FP32 output\n"
            << "Fused epilogue     : D = ReLU(" << options.alpha << " * AB + bias)\n"
            << std::fixed << std::setprecision(3)
            << "Traffic model      : saves " << logical_saved_mib
            << " MiB for logical MxN";
  if (is_padded) {
    std::cout << "; " << executed_saved_mib << " MiB for executed padded MxN";
  }
  std::cout << '\n';
  return properties;
}

inline void print_performance_comparison(
    Options const& options, float unfused_ms, float fused_ms) {
  double operations = 2.0 * static_cast<double>(options.m) * options.n * options.k;
  double unfused_tflops = operations / (static_cast<double>(unfused_ms) * 1.0e9);
  double fused_tflops = operations / (static_cast<double>(fused_ms) * 1.0e9);
  double speedup = static_cast<double>(unfused_ms) / fused_ms;
  std::cout << std::fixed << std::setprecision(3)
            << "Unfused runtime    : " << unfused_ms << " ms (GEMM + separate kernel)\n"
            << "Fused runtime      : " << fused_ms << " ms (single GEMM kernel)\n"
            << "Unfused throughput : " << unfused_tflops << " TFLOP/s\n"
            << "Fused throughput   : " << fused_tflops << " TFLOP/s\n"
            << "End-to-end speedup : " << speedup << "x\n";
}

inline void validate_options(Options const& options) {
  if (options.beta != 0.0f) {
    throw std::runtime_error(
        "--beta is not used by bias+ReLU; keep --beta=0 so the source is an unscaled bias");
  }
}

}  // namespace cutlass_lab::bias_relu
