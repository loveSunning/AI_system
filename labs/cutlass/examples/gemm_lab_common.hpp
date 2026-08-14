#pragma once

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/version.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace cutlass_lab {

// The default 4096^3 problem is naturally aligned. Custom non-aligned shapes
// are zero-padded so the vectorized Tensor Core kernels remain valid.
struct Options {
  int m = 4096;
  int n = 4096;
  int k = 4096;
  int warmup = 2;
  int iterations = 10;
  float alpha = 1.0f;
  float beta = 0.0f;
  bool verify = true;
  bool help = false;
};

inline int parse_int(std::string const& text, char const* name) {
  try {
    std::size_t consumed = 0;
    int value = std::stoi(text, &consumed);
    if (consumed != text.size() || value <= 0) {
      throw std::invalid_argument("not a positive integer");
    }
    return value;
  } catch (std::exception const&) {
    throw std::runtime_error(std::string("invalid value for ") + name + ": " + text);
  }
}

inline float parse_float(std::string const& text, char const* name) {
  try {
    std::size_t consumed = 0;
    float value = std::stof(text, &consumed);
    if (consumed != text.size() || !std::isfinite(value)) {
      throw std::invalid_argument("not finite");
    }
    return value;
  } catch (std::exception const&) {
    throw std::runtime_error(std::string("invalid value for ") + name + ": " + text);
  }
}

inline Options parse_options(int argc, char const** argv, Options options = Options{}) {
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    auto value_after = [&](char const* prefix) -> std::string {
      return arg.substr(std::string(prefix).size());
    };

    if (arg == "--help" || arg == "-h") {
      options.help = true;
    } else if (arg == "--no-verify") {
      options.verify = false;
    } else if (arg.rfind("--m=", 0) == 0) {
      options.m = parse_int(value_after("--m="), "--m");
    } else if (arg.rfind("--n=", 0) == 0) {
      options.n = parse_int(value_after("--n="), "--n");
    } else if (arg.rfind("--k=", 0) == 0) {
      options.k = parse_int(value_after("--k="), "--k");
    } else if (arg.rfind("--warmup=", 0) == 0) {
      options.warmup = parse_int(value_after("--warmup="), "--warmup");
    } else if (arg.rfind("--iterations=", 0) == 0) {
      options.iterations = parse_int(value_after("--iterations="), "--iterations");
    } else if (arg.rfind("--alpha=", 0) == 0) {
      options.alpha = parse_float(value_after("--alpha="), "--alpha");
    } else if (arg.rfind("--beta=", 0) == 0) {
      options.beta = parse_float(value_after("--beta="), "--beta");
    } else {
      throw std::runtime_error("unknown option: " + arg);
    }
  }
  return options;
}

inline void print_usage(
    char const* executable,
    char const* api_name,
    Options const& defaults = Options{}) {
  std::cout
      << api_name << " Tensor Core GEMM learning example\n\n"
      << "Usage: " << executable << " [options]\n\n"
      << "  --m=<int>             logical M (default " << defaults.m << ")\n"
      << "  --n=<int>             logical N (default " << defaults.n << ")\n"
      << "  --k=<int>             logical K (default " << defaults.k << ")\n"
      << "  --warmup=<int>        untimed launches (default " << defaults.warmup << ")\n"
      << "  --iterations=<int>    timed launches (default " << defaults.iterations << ")\n"
      << "  --alpha=<float>       epilogue alpha (default 1)\n"
      << "  --beta=<float>        epilogue beta (default 0)\n"
      << "  --no-verify           skip full output verification\n"
      << "  --help                show this message\n\n"
      << "Computes D = alpha * A * B + beta * C with A=B=C=1.\n";
}

inline void check_cuda(cudaError_t status, char const* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

inline void check_cutlass(cutlass::Status status, char const* operation) {
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cutlassGetStatusString(status));
  }
}

template <class T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) { reset(count); }
  ~DeviceBuffer() { release(); }

  DeviceBuffer(DeviceBuffer const&) = delete;
  DeviceBuffer& operator=(DeviceBuffer const&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)), count_(std::exchange(other.count_, 0)) {}

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release();
      data_ = std::exchange(other.data_, nullptr);
      count_ = std::exchange(other.count_, 0);
    }
    return *this;
  }

  void reset(std::size_t count) {
    release();
    if (count != 0) {
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)), "cudaMalloc");
    }
    count_ = count;
  }

  T* get() const { return data_; }
  std::size_t size() const { return count_; }

 private:
  void release() noexcept {
    if (data_ != nullptr) {
      cudaFree(data_);
      data_ = nullptr;
      count_ = 0;
    }
  }

  T* data_ = nullptr;
  std::size_t count_ = 0;
};

inline int round_up(int value, int alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

struct ProblemStorage {
  explicit ProblemStorage(Options const& options, bool use_column_major_output = false)
      : padded_m(round_up(options.m, 128)),
        padded_n(round_up(options.n, 128)),
        padded_k(round_up(options.k, 32)),
        lda(padded_k),
        ldb(padded_k),
        ldc(use_column_major_output ? padded_m : padded_n),
        ldd(ldc),
        column_major_output(use_column_major_output),
        a(static_cast<std::size_t>(padded_m) * lda),
        b(static_cast<std::size_t>(padded_n) * ldb),
        c(static_cast<std::size_t>(use_column_major_output ? padded_n : padded_m) * ldc),
        d(static_cast<std::size_t>(use_column_major_output ? padded_n : padded_m) * ldd) {}

  int padded_m;
  int padded_n;
  int padded_k;
  int lda;
  int ldb;
  int ldc;
  int ldd;
  bool column_major_output;
  DeviceBuffer<cutlass::half_t> a;
  DeviceBuffer<cutlass::half_t> b;
  DeviceBuffer<float> c;
  DeviceBuffer<float> d;
};

__global__ void fill_row_major_half(
    cutlass::half_t* data, int rows, int columns, int leading_dimension, float value) {
  std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::size_t total = static_cast<std::size_t>(rows) * leading_dimension;
  if (index < total) {
    int column = static_cast<int>(index % leading_dimension);
    data[index] = cutlass::half_t(column < columns ? value : 0.0f);
  }
}

__global__ void fill_column_major_half(
    cutlass::half_t* data, int rows, int columns, int leading_dimension, float value) {
  std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::size_t total = static_cast<std::size_t>(columns) * leading_dimension;
  if (index < total) {
    int row = static_cast<int>(index % leading_dimension);
    data[index] = cutlass::half_t(row < rows ? value : 0.0f);
  }
}

__global__ void fill_row_major_float(
    float* data, int rows, int columns, int leading_dimension, float value) {
  std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::size_t total = static_cast<std::size_t>(rows) * leading_dimension;
  if (index < total) {
    int column = static_cast<int>(index % leading_dimension);
    data[index] = column < columns ? value : 0.0f;
  }
}

__global__ void fill_column_major_float(
    float* data, int rows, int columns, int leading_dimension, float value) {
  std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::size_t total = static_cast<std::size_t>(columns) * leading_dimension;
  if (index < total) {
    int row = static_cast<int>(index % leading_dimension);
    data[index] = row < rows ? value : 0.0f;
  }
}

__global__ void count_mismatches(
    float const* data,
    int rows,
    int columns,
    int leading_dimension,
    float expected,
    float tolerance,
    unsigned long long* mismatch_count) {
  std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::size_t total = static_cast<std::size_t>(rows) * columns;
  if (index < total) {
    int row = static_cast<int>(index / columns);
    int column = static_cast<int>(index % columns);
    float actual = data[static_cast<std::size_t>(row) * leading_dimension + column];
    if (!isfinite(actual) || fabsf(actual - expected) > tolerance) {
      atomicAdd(mismatch_count, 1ULL);
    }
  }
}

__global__ void count_mismatches_column_major(
    float const* data,
    int rows,
    int columns,
    int leading_dimension,
    float expected,
    float tolerance,
    unsigned long long* mismatch_count) {
  std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::size_t total = static_cast<std::size_t>(rows) * columns;
  if (index < total) {
    int row = static_cast<int>(index / columns);
    int column = static_cast<int>(index % columns);
    float actual = data[static_cast<std::size_t>(column) * leading_dimension + row];
    if (!isfinite(actual) || fabsf(actual - expected) > tolerance) {
      atomicAdd(mismatch_count, 1ULL);
    }
  }
}

inline dim3 grid_for(std::size_t count, int threads = 256) {
  return dim3(static_cast<unsigned int>((count + threads - 1) / threads));
}

inline void initialize_problem(ProblemStorage& storage, Options const& options) {
  constexpr int threads = 256;
  fill_row_major_half<<<grid_for(storage.a.size()), threads>>>(
      storage.a.get(), storage.padded_m, options.k, storage.lda, 1.0f);
  fill_column_major_half<<<grid_for(storage.b.size()), threads>>>(
      storage.b.get(), options.k, storage.padded_n, storage.ldb, 1.0f);
  if (storage.column_major_output) {
    fill_column_major_float<<<grid_for(storage.c.size()), threads>>>(
        storage.c.get(), options.m, storage.padded_n, storage.ldc, 1.0f);
    fill_column_major_float<<<grid_for(storage.d.size()), threads>>>(
        storage.d.get(), options.m, storage.padded_n, storage.ldd, 0.0f);
  } else {
    fill_row_major_float<<<grid_for(storage.c.size()), threads>>>(
        storage.c.get(), storage.padded_m, options.n, storage.ldc, 1.0f);
    fill_row_major_float<<<grid_for(storage.d.size()), threads>>>(
        storage.d.get(), storage.padded_m, options.n, storage.ldd, 0.0f);
  }
  check_cuda(cudaGetLastError(), "initialize kernels");
  check_cuda(cudaDeviceSynchronize(), "initialize synchronization");
}

inline bool verify_result(ProblemStorage const& storage, Options const& options) {
  DeviceBuffer<unsigned long long> mismatch_count(1);
  check_cuda(cudaMemset(mismatch_count.get(), 0, sizeof(unsigned long long)), "clear mismatch count");

  float expected = options.alpha * static_cast<float>(options.k) + options.beta;
  float tolerance = std::max(1.0e-3f, std::abs(expected) * 1.0e-4f);
  std::size_t logical_elements = static_cast<std::size_t>(options.m) * options.n;
  if (storage.column_major_output) {
    count_mismatches_column_major<<<grid_for(logical_elements), 256>>>(
        storage.d.get(), options.m, options.n, storage.ldd,
        expected, tolerance, mismatch_count.get());
  } else {
    count_mismatches<<<grid_for(logical_elements), 256>>>(
        storage.d.get(), options.m, options.n, storage.ldd,
        expected, tolerance, mismatch_count.get());
  }
  check_cuda(cudaGetLastError(), "verification kernel");

  unsigned long long host_mismatches = 0;
  check_cuda(
      cudaMemcpy(&host_mismatches, mismatch_count.get(), sizeof(host_mismatches), cudaMemcpyDeviceToHost),
      "copy verification result");

  std::cout << "Verification       : " << (host_mismatches == 0 ? "PASSED" : "FAILED")
            << " (expected " << expected << ", tolerance " << tolerance
            << ", mismatches " << host_mismatches << ")\n";
  return host_mismatches == 0;
}

inline cudaDeviceProp print_environment(char const* api_name, Options const& options, ProblemStorage const& storage) {
  int device = 0;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
  if (properties.major * 10 + properties.minor < 80) {
    throw std::runtime_error("this example requires SM80 or newer for mma.sync m16n8k16");
  }

  bool const is_padded = storage.padded_m != options.m || storage.padded_n != options.n ||
                         storage.padded_k != options.k;

  std::cout << api_name << "\n"
            << "CUTLASS version    : " << CUTLASS_MAJOR << '.' << CUTLASS_MINOR << '.' << CUTLASS_PATCH << '\n'
            << "GPU                : " << properties.name << " (SM" << properties.major << properties.minor << ")\n"
            << "Logical problem    : " << options.m << " x " << options.n << " x " << options.k << '\n'
            << "Executed problem   : " << storage.padded_m << " x " << storage.padded_n
            << " x " << storage.padded_k << (is_padded ? " (padded)\n" : " (no padding)\n")
            << "Physical strides   : lda=" << storage.lda << ", ldb=" << storage.ldb
            << ", ldc=" << storage.ldc << ", ldd=" << storage.ldd << '\n'
            << "Output layout      : " << (storage.column_major_output ? "column-major" : "row-major") << '\n'
            << "Data path          : FP16 inputs -> FP32 Tensor Core accumulate -> FP32 output\n"
            << "Epilogue           : D = " << options.alpha << " * AB + " << options.beta << " * C\n";
  return properties;
}

template <class Launch>
float benchmark(Options const& options, Launch&& launch) {
  for (int i = 0; i < options.warmup; ++i) {
    launch();
  }
  check_cuda(cudaDeviceSynchronize(), "warmup synchronization");

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
  try {
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int i = 0; i < options.iterations; ++i) {
      launch();
    }
    check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return elapsed_ms / static_cast<float>(options.iterations);
  } catch (...) {
    if (stop != nullptr) {
      cudaEventDestroy(stop);
    }
    cudaEventDestroy(start);
    throw;
  }
}

inline void print_performance(Options const& options, float average_ms) {
  double operations = 2.0 * static_cast<double>(options.m) * options.n * options.k;
  double tflops = operations / (static_cast<double>(average_ms) * 1.0e9);
  std::cout << std::fixed << std::setprecision(3)
            << "Average runtime    : " << average_ms << " ms\n"
            << "Tensor throughput  : " << tflops << " TFLOP/s\n";
}

}  // namespace cutlass_lab
