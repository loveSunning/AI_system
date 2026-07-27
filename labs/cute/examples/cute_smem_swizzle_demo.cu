#include <cute/swizzle.hpp>
#include <cute/tensor.hpp>
#include <cute/util/print.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kRows = 32;
constexpr int kColumns = 32;
constexpr int kElements = kRows * kColumns;
constexpr int kThreads = 32;
constexpr int kReadIterations = 4096;
constexpr int kBenchmarkIterations = 100;

using PlainLayout = cute::Layout<
    cute::Shape<cute::_32, cute::_32>,
    cute::Stride<cute::_32, cute::_1>>;
using PaddedLayout = cute::Layout<
    cute::Shape<cute::_32, cute::_32>,
    cute::Stride<cute::Int<33>, cute::_1>>;
using SwizzledLayout =
    decltype(cute::composition(cute::Swizzle<5, 0, 5>{}, PlainLayout{}));

#define CUDA_CHECK(call)                                                                      \
  do {                                                                                        \
    cudaError_t status_ = (call);                                                             \
    if (status_ != cudaSuccess) {                                                             \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "                 \
                << cudaGetErrorString(status_) << "\n";                                      \
      std::exit(EXIT_FAILURE);                                                                \
    }                                                                                         \
  } while (false)

float make_value(int row, int column) {
  return float((row * 7 + column * 3) % 17);
}

template <class Layout>
__global__ void shared_column_read_kernel(
    float const* input,
    float* output,
    Layout layout) {
  using namespace cute;

  __shared__ float storage[cosize_v<Layout>];
  Tensor tile = make_tensor(make_smem_ptr(storage), layout);

  for (int linear = int(threadIdx.x); linear < kElements; linear += int(blockDim.x)) {
    int row = linear / kColumns;
    int column = linear % kColumns;
    tile(row, column) = input[linear];
  }
  __syncthreads();

  // Every loop iteration makes the warp read one logical column. The plain row-major
  // layout maps all 32 lanes to one bank; padding and swizzle spread them over 32 banks.
  volatile float* volatile_storage = storage;
  int row = int(threadIdx.x);
  float sum = 0.0f;
  for (int iteration = 0; iteration < kReadIterations; ++iteration) {
    int column = iteration & (kColumns - 1);
    int physical_offset = int(layout(row, column));
    sum += volatile_storage[physical_offset];
  }
  output[row] = sum;
}

template <class Layout>
std::array<int, 32> bank_histogram(Layout layout, int column) {
  std::array<int, 32> histogram{};
  for (int row = 0; row < kRows; ++row) {
    int offset = int(layout(row, column));
    int bank = offset % 32;
    ++histogram[bank];
  }
  return histogram;
}

template <class Layout>
std::pair<int, int> bank_summary(Layout layout, int column) {
  auto histogram = bank_histogram(layout, column);
  int active = 0;
  int maximum = 0;
  for (int count : histogram) {
    active += count > 0 ? 1 : 0;
    maximum = std::max(maximum, count);
  }
  return {active, maximum};
}

template <class Layout>
float benchmark_microseconds(float const* input, float* output, Layout layout) {
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  CUDA_CHECK(cudaEventCreate(&begin));
  CUDA_CHECK(cudaEventCreate(&end));
  for (int i = 0; i < 5; ++i) {
    shared_column_read_kernel<<<1, kThreads>>>(input, output, layout);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(begin));
  for (int i = 0; i < kBenchmarkIterations; ++i) {
    shared_column_read_kernel<<<1, kThreads>>>(input, output, layout);
  }
  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, begin, end));
  CUDA_CHECK(cudaEventDestroy(begin));
  CUDA_CHECK(cudaEventDestroy(end));
  return elapsed_ms * 1000.0f / float(kBenchmarkIterations);
}

template <class Layout>
bool run_and_verify(
    char const* name,
    float const* device_input,
    float* device_output,
    Layout layout) {
  shared_column_read_kernel<<<1, kThreads>>>(device_input, device_output, layout);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> output(kRows);
  CUDA_CHECK(cudaMemcpy(
      output.data(), device_output, sizeof(float) * output.size(), cudaMemcpyDeviceToHost));

  int errors = 0;
  for (int row = 0; row < kRows; ++row) {
    float one_cycle = 0.0f;
    for (int column = 0; column < kColumns; ++column) {
      one_cycle += make_value(row, column);
    }
    float expected = one_cycle * float(kReadIterations / kColumns);
    if (output[row] != expected) {
      if (errors < 5) {
        std::cerr << name << " row " << row << ": actual=" << output[row]
                  << " expected=" << expected << "\n";
      }
      ++errors;
    }
  }
  bool ok = errors == 0;
  std::cout << std::left << std::setw(12) << name << " data validation: "
            << (ok ? "PASS" : "FAIL") << "\n";
  return ok;
}

}  // namespace

int main() {
  using namespace cute;

  PlainLayout plain{};
  PaddedLayout padded{};
  SwizzledLayout swizzled{};

  std::cout << "CuTe W16 shared-memory bank-conflict and swizzle demo\n";
  std::cout << "Logical tile               : 32 x 32 float\n";
  std::cout << "Warp access                : lane t reads logical (t, column)\n";
  std::cout << "Bank formula               : (physical_offset * 4 / 4) % 32\n";
  std::cout << "Read iterations            : " << kReadIterations << "\n\n";

  std::cout << "plain layout               : ";
  print(plain);
  std::cout << "\n";
  std::cout << "padded layout              : ";
  print(padded);
  std::cout << "\n";
  std::cout << "swizzled layout            : ";
  print(swizzled);
  std::cout << "\n";
  std::cout << "cosize plain/padded/swizzle: " << cosize(plain) << "/" << cosize(padded)
            << "/" << cosize(swizzled) << "\n";

  std::cout << "\n[column 0 physical mapping]\n";
  std::cout << "lane | plain off bank | padded off bank | swizzle off bank\n";
  for (int lane = 0; lane < kThreads; ++lane) {
    int plain_offset = int(plain(lane, 0));
    int padded_offset = int(padded(lane, 0));
    int swizzled_offset = int(swizzled(lane, 0));
    std::cout << std::setw(4) << lane << " | " << std::setw(9) << plain_offset << " "
              << std::setw(4) << plain_offset % 32 << " | " << std::setw(10) << padded_offset
              << " " << std::setw(4) << padded_offset % 32 << " | " << std::setw(11)
              << swizzled_offset << " " << std::setw(4) << swizzled_offset % 32 << "\n";
  }

  auto [plain_active, plain_max] = bank_summary(plain, 0);
  auto [padded_active, padded_max] = bank_summary(padded, 0);
  auto [swizzled_active, swizzled_max] = bank_summary(swizzled, 0);
  std::cout << "\n[bank summary for one warp column read]\n";
  std::cout << "layout   | active banks | max lanes per bank | estimated conflict\n";
  std::cout << "plain    | " << std::setw(12) << plain_active << " | " << std::setw(18)
            << plain_max << " | " << plain_max << "-way\n";
  std::cout << "padded   | " << std::setw(12) << padded_active << " | " << std::setw(18)
            << padded_max << " | " << padded_max << "-way\n";
  std::cout << "swizzled | " << std::setw(12) << swizzled_active << " | " << std::setw(18)
            << swizzled_max << " | " << swizzled_max << "-way\n";

  std::vector<float> host_input(kElements);
  for (int row = 0; row < kRows; ++row) {
    for (int column = 0; column < kColumns; ++column) {
      host_input[row * kColumns + column] = make_value(row, column);
    }
  }

  float* device_input = nullptr;
  float* device_output = nullptr;
  CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&device_input), sizeof(float) * host_input.size()));
  CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&device_output), sizeof(float) * kRows));
  CUDA_CHECK(cudaMemcpy(
      device_input,
      host_input.data(),
      sizeof(float) * host_input.size(),
      cudaMemcpyHostToDevice));

  std::cout << "\n[correctness]\n";
  bool ok = true;
  ok = run_and_verify("plain", device_input, device_output, plain) && ok;
  ok = run_and_verify("padded", device_input, device_output, padded) && ok;
  ok = run_and_verify("swizzled", device_input, device_output, swizzled) && ok;

  float plain_us = benchmark_microseconds(device_input, device_output, plain);
  float padded_us = benchmark_microseconds(device_input, device_output, padded);
  float swizzled_us = benchmark_microseconds(device_input, device_output, swizzled);
  std::cout << "\n[microbenchmark]\n";
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "plain     : " << plain_us << " us\n";
  std::cout << "padded    : " << padded_us << " us\n";
  std::cout << "swizzled  : " << swizzled_us << " us\n";
  std::cout << "Relative padded/plain   : " << plain_us / padded_us << "x\n";
  std::cout << "Relative swizzle/plain  : " << plain_us / swizzled_us << "x\n";
  std::cout << "Use Nsight Compute to confirm hardware bank-conflict counters.\n";

  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_output));

  bool expected_bank_mapping =
      plain_active == 1 && plain_max == 32 && padded_active == 32 && padded_max == 1 &&
      swizzled_active == 32 && swizzled_max == 1;
  std::cout << "\nExpected bank mapping       : "
            << (expected_bank_mapping ? "PASS" : "FAIL") << "\n";
  ok = expected_bank_mapping && ok;

  if (!ok) {
    std::cerr << "W16 shared-memory swizzle checks failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "W16 shared-memory swizzle checks passed\n";
  return EXIT_SUCCESS;
}
