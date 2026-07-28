#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cute_gemm_demo {

using Element = cute::half_t;

enum class LayoutMode {
  NT,
  TN,
};

struct Options {
  int m = 4096;
  int n = 4096;
  int k = 4096;
  int warmups = 5;
  int iterations = 20;
  std::string layout = "both";
};

inline char const* layout_name(LayoutMode mode) {
  return mode == LayoutMode::NT ? "NT" : "TN";
}

inline void check_cuda(cudaError_t status, char const* context) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
  }
}

inline void check_cublas(cublasStatus_t status, char const* context) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        std::string(context) + ": cuBLAS status " + std::to_string(int(status)));
  }
}

template <class T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(std::size_t count) : count_(count) {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)), "cudaMalloc");
  }

  DeviceBuffer(DeviceBuffer const&) = delete;
  DeviceBuffer& operator=(DeviceBuffer const&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(other.data_), count_(other.count_) {
    other.data_ = nullptr;
    other.count_ = 0;
  }

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  T* data() { return data_; }
  T const* data() const { return data_; }
  std::size_t size() const { return count_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0;
};

class CudaEvent {
 public:
  CudaEvent() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
  CudaEvent(CudaEvent const&) = delete;
  CudaEvent& operator=(CudaEvent const&) = delete;
  ~CudaEvent() { cudaEventDestroy(event_); }
  operator cudaEvent_t() const { return event_; }

 private:
  cudaEvent_t event_ {};
};

class CublasHandle {
 public:
  CublasHandle() {
    check_cublas(cublasCreate(&handle_), "cublasCreate");
    check_cublas(cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode");
  }
  CublasHandle(CublasHandle const&) = delete;
  CublasHandle& operator=(CublasHandle const&) = delete;
  ~CublasHandle() { cublasDestroy(handle_); }
  operator cublasHandle_t() const { return handle_; }

 private:
  cublasHandle_t handle_ {};
};

inline Options parse_options(int argc, char** argv) {
  Options options;
  if (argc > 1) options.m = std::atoi(argv[1]);
  if (argc > 2) options.n = std::atoi(argv[2]);
  if (argc > 3) options.k = std::atoi(argv[3]);
  if (argc > 4) options.layout = argv[4];
  if (argc > 5) options.iterations = std::atoi(argv[5]);
  if (argc > 6) options.warmups = std::atoi(argv[6]);

  std::transform(options.layout.begin(), options.layout.end(), options.layout.begin(),
                 [](unsigned char c) { return char(std::tolower(c)); });

  if (options.m <= 0 || options.n <= 0 || options.k <= 0 ||
      options.iterations <= 0 || options.warmups < 0) {
    throw std::invalid_argument("M, N, K and iterations must be positive");
  }
  if (options.layout != "nt" && options.layout != "tn" && options.layout != "both") {
    throw std::invalid_argument("layout must be nt, tn, or both");
  }
  return options;
}

inline std::vector<LayoutMode> selected_layouts(std::string const& layout) {
  if (layout == "nt") return {LayoutMode::NT};
  if (layout == "tn") return {LayoutMode::TN};
  return {LayoutMode::NT, LayoutMode::TN};
}

inline std::size_t a_offset(LayoutMode mode, int m, int k, int M, int K) {
  return mode == LayoutMode::NT ? std::size_t(m) + std::size_t(k) * M
                                : std::size_t(k) + std::size_t(m) * K;
}

inline std::size_t b_offset(LayoutMode mode, int n, int k, int N, int K) {
  return mode == LayoutMode::NT ? std::size_t(n) + std::size_t(k) * N
                                : std::size_t(k) + std::size_t(n) * K;
}

inline void initialize_inputs(
    LayoutMode mode,
    int m,
    int n,
    int k,
    std::vector<Element>& a,
    std::vector<Element>& b) {
  a.resize(std::size_t(m) * k);
  b.resize(std::size_t(n) * k);

  for (int row = 0; row < m; ++row) {
    for (int kk = 0; kk < k; ++kk) {
      int value = ((row * 17 + kk * 13) % 17) - 8;
      a[a_offset(mode, row, kk, m, k)] = Element(float(value) * 0.0625f);
    }
  }
  for (int col = 0; col < n; ++col) {
    for (int kk = 0; kk < k; ++kk) {
      int value = ((col * 11 + kk * 7) % 19) - 9;
      b[b_offset(mode, col, kk, n, k)] = Element(float(value) * 0.0625f);
    }
  }
}

inline void launch_cublas(
    cublasHandle_t handle,
    LayoutMode mode,
    Element const* a,
    Element const* b,
    Element* c,
    int m,
    int n,
    int k) {
  float alpha = 1.0f;
  float beta = 0.0f;

  cublasOperation_t op_a = mode == LayoutMode::NT ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t op_b = mode == LayoutMode::NT ? CUBLAS_OP_T : CUBLAS_OP_N;
  int lda = mode == LayoutMode::NT ? m : k;
  int ldb = mode == LayoutMode::NT ? n : k;

  check_cublas(
      cublasGemmEx(
          handle,
          op_a,
          op_b,
          m,
          n,
          k,
          &alpha,
          a,
          CUDA_R_16F,
          lda,
          b,
          CUDA_R_16F,
          ldb,
          &beta,
          c,
          CUDA_R_16F,
          m,
          CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      "cublasGemmEx");
}

template <class Launch>
float time_launch(Launch&& launch, int warmups, int iterations) {
  for (int i = 0; i < warmups; ++i) {
    launch();
  }
  check_cuda(cudaGetLastError(), "warmup launch");
  check_cuda(cudaDeviceSynchronize(), "warmup synchronize");

  CudaEvent begin;
  CudaEvent end;
  check_cuda(cudaEventRecord(begin), "cudaEventRecord(begin)");
  for (int i = 0; i < iterations; ++i) {
    launch();
  }
  check_cuda(cudaEventRecord(end), "cudaEventRecord(end)");
  check_cuda(cudaEventSynchronize(end), "cudaEventSynchronize(end)");
  check_cuda(cudaGetLastError(), "timed launch");

  float elapsed_ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&elapsed_ms, begin, end), "cudaEventElapsedTime");
  return elapsed_ms / float(iterations);
}

struct ErrorStats {
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  std::size_t mismatches = 0;
};

inline ErrorStats compare(
    std::vector<Element> const& actual,
    std::vector<Element> const& reference) {
  ErrorStats stats;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    float got = float(actual[i]);
    float expected = float(reference[i]);
    float abs_error = std::abs(got - expected);
    float rel_error = abs_error / std::max(std::abs(expected), 1.0e-3f);
    stats.max_abs = std::max(stats.max_abs, abs_error);
    stats.max_rel = std::max(stats.max_rel, rel_error);
    if (!std::isfinite(got) || abs_error > 0.5f + 0.05f * std::abs(expected)) {
      ++stats.mismatches;
    }
  }
  return stats;
}

inline double tflops(int m, int n, int k, float milliseconds) {
  return 2.0 * double(m) * double(n) * double(k) / (double(milliseconds) * 1.0e9);
}

template <class KernelLaunch>
bool run_layout(
    char const* demo_name,
    Options const& options,
    LayoutMode mode,
    KernelLaunch&& launch_kernel) {
  std::vector<Element> host_a;
  std::vector<Element> host_b;
  initialize_inputs(mode, options.m, options.n, options.k, host_a, host_b);

  std::size_t c_count = std::size_t(options.m) * options.n;
  DeviceBuffer<Element> device_a(host_a.size());
  DeviceBuffer<Element> device_b(host_b.size());
  DeviceBuffer<Element> device_c(c_count);
  DeviceBuffer<Element> device_reference(c_count);

  check_cuda(cudaMemcpy(device_a.data(), host_a.data(), host_a.size() * sizeof(Element),
                        cudaMemcpyHostToDevice),
             "copy A to device");
  check_cuda(cudaMemcpy(device_b.data(), host_b.data(), host_b.size() * sizeof(Element),
                        cudaMemcpyHostToDevice),
             "copy B to device");

  CublasHandle handle;
  launch_cublas(handle, mode, device_a.data(), device_b.data(), device_reference.data(),
                options.m, options.n, options.k);
  check_cuda(cudaDeviceSynchronize(), "cuBLAS reference synchronize");

  launch_kernel(mode, device_a.data(), device_b.data(), device_c.data(),
                options.m, options.n, options.k, cudaStream_t {});
  check_cuda(cudaGetLastError(), "demo correctness launch");
  check_cuda(cudaDeviceSynchronize(), "demo correctness synchronize");

  std::vector<Element> host_c(c_count);
  std::vector<Element> host_reference(c_count);
  check_cuda(cudaMemcpy(host_c.data(), device_c.data(), c_count * sizeof(Element),
                        cudaMemcpyDeviceToHost),
             "copy result to host");
  check_cuda(cudaMemcpy(host_reference.data(), device_reference.data(), c_count * sizeof(Element),
                        cudaMemcpyDeviceToHost),
             "copy reference to host");

  ErrorStats errors = compare(host_c, host_reference);

  float demo_ms = time_launch(
      [&] {
        launch_kernel(mode, device_a.data(), device_b.data(), device_c.data(),
                      options.m, options.n, options.k, cudaStream_t {});
      },
      options.warmups,
      options.iterations);

  float cublas_ms = time_launch(
      [&] {
        launch_cublas(handle, mode, device_a.data(), device_b.data(), device_reference.data(),
                      options.m, options.n, options.k);
      },
      options.warmups,
      options.iterations);

  std::printf(
      "%-24s layout=%s  %8.3f ms  %8.3f TFLOP/s  "
      "cuBLAS=%8.3f ms %8.3f TFLOP/s  %5.1f%%  "
      "max_abs=%g max_rel=%g mismatches=%zu\n",
      demo_name,
      layout_name(mode),
      demo_ms,
      tflops(options.m, options.n, options.k, demo_ms),
      cublas_ms,
      tflops(options.m, options.n, options.k, cublas_ms),
      100.0f * cublas_ms / demo_ms,
      errors.max_abs,
      errors.max_rel,
      errors.mismatches);

  return errors.mismatches == 0;
}

inline void print_configuration(char const* demo_name, Options const& options) {
  cudaDeviceProp props {};
  check_cuda(cudaGetDeviceProperties(&props, 0), "cudaGetDeviceProperties");
  std::cout << demo_name << '\n'
            << "device: " << props.name << " (SM" << props.major << props.minor << ")\n"
            << "shape: M=" << options.m << " N=" << options.n << " K=" << options.k
            << " layout=" << options.layout
            << " warmups=" << options.warmups
            << " iterations=" << options.iterations << '\n';
}

template <class KernelLaunch>
inline int run_main(
    int argc,
    char** argv,
    char const* demo_name,
    int tile_m,
    int tile_n,
    int tile_k,
    bool architecture_supported,
    char const* architecture_message,
    KernelLaunch&& launch_kernel) {
  try {
    Options options = parse_options(argc, argv);
    print_configuration(demo_name, options);

    if (!architecture_supported) {
      std::cout << "SKIPPED: " << architecture_message << '\n';
      return 0;
    }

    if (options.m % tile_m != 0 || options.n % tile_n != 0 || options.k % tile_k != 0) {
      std::cerr << "This teaching kernel requires M%" << tile_m << "==0, N%"
                << tile_n << "==0, and K%" << tile_k << "==0.\n";
      return 2;
    }

    bool passed = true;
    for (LayoutMode mode : selected_layouts(options.layout)) {
      passed = run_layout(demo_name, options, mode, launch_kernel) && passed;
    }
    return passed ? 0 : 1;
  } catch (std::exception const& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}

}  // namespace cute_gemm_demo
