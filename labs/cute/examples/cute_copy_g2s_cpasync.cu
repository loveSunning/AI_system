#include <cute/tensor.hpp>
#include <cute/util/print.hpp>

#include <cuda_runtime.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

using Element = cute::half_t;

constexpr int kM = 2048;
constexpr int kK = 2048;
constexpr int kBlockM = 128;
constexpr int kBlockK = 32;
constexpr int kThreads = 128;
constexpr int kTileM = 3;
constexpr int kTileK = 7;
constexpr int kTileElements = kBlockM * kBlockK;
constexpr int kBenchmarkIterations = 200;

using BlockShape = cute::Shape<cute::Int<kBlockM>, cute::Int<kBlockK>>;
using SmemLayout = cute::Layout<
    cute::Shape<cute::Int<kBlockM>, cute::Int<kBlockK>>,
    cute::Stride<cute::Int<kBlockK>, cute::_1>>;
using CopyThreadLayout =
    cute::Layout<cute::Shape<cute::_32, cute::_4>, cute::Stride<cute::_4, cute::_1>>;
using CopyValueLayout = cute::Layout<cute::Shape<cute::_1, cute::_8>>;

#define CUDA_CHECK(call)                                                                      \
  do {                                                                                        \
    cudaError_t status_ = (call);                                                             \
    if (status_ != cudaSuccess) {                                                             \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "                 \
                << cudaGetErrorString(status_) << "\n";                                      \
      std::exit(EXIT_FAILURE);                                                                \
    }                                                                                         \
  } while (false)

Element make_value(int m, int k) {
  return Element(float((m * 37 + k * 13) % 1024));
}

bool same_value(Element lhs, Element rhs) {
  return static_cast<float>(lhs) == static_cast<float>(rhs);
}

template <bool Predicated, class TiledCopy>
__global__ void cpasync_g2s_kernel(
    Element const* input,
    Element* output,
    int m_extent,
    int k_extent,
    int leading_dim,
    int tile_m,
    int tile_k,
    TiledCopy tiled_copy) {
  using namespace cute;

  __shared__ Element storage[cosize_v<SmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(storage), SmemLayout{});

  for (int i = int(threadIdx.x); i < kTileElements; i += int(blockDim.x)) {
    sA(i) = Element{};
  }
  __syncthreads();

  auto problem_shape = make_shape(m_extent, k_extent);
  Tensor mA = make_tensor(
      make_gmem_ptr(input),
      make_layout(problem_shape, make_stride(leading_dim, Int<1>{})));
  Tensor gA = local_tile(mA, BlockShape{}, make_coord(tile_m, tile_k));

  ThrCopy thr_copy = tiled_copy.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy.partition_S(gA);
  Tensor tAsA = thr_copy.partition_D(sA);

  if constexpr (Predicated) {
    Tensor cA = make_identity_tensor(problem_shape);
    Tensor pA = cute::lazy::transform(cA, [&](auto coord) {
      return elem_less(coord, problem_shape);
    });
    Tensor gP = local_tile(pA, BlockShape{}, make_coord(tile_m, tile_k));
    Tensor tAgP = thr_copy.partition_S(gP);
    copy_if(tiled_copy, tAgP, tAgA, tAsA);
  } else {
    copy(tiled_copy, tAgA, tAsA);
  }

  // fence closes the issued cp.async group; wait<0> waits for all groups issued by this
  // thread; __syncthreads makes every thread's shared-memory writes visible to the CTA.
  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();

  for (int i = int(threadIdx.x); i < kTileElements; i += int(blockDim.x)) {
    int local_m = i / kBlockK;
    int local_k = i % kBlockK;
    output[i] = sA(local_m, local_k);
  }
}

std::vector<Element> make_problem(int m_extent, int k_extent, int leading_dim, int allocated_rows) {
  std::vector<Element> data(std::size_t(allocated_rows) * std::size_t(leading_dim), Element{});
  for (int m = 0; m < m_extent; ++m) {
    for (int k = 0; k < k_extent; ++k) {
      data[std::size_t(m) * std::size_t(leading_dim) + std::size_t(k)] = make_value(m, k);
    }
  }
  return data;
}

bool verify_tile(
    char const* name,
    std::vector<Element> const& output,
    int m_extent,
    int k_extent,
    int tile_m,
    int tile_k) {
  int errors = 0;
  int valid = 0;
  int zero_filled = 0;
  int out_of_range_nonzero = 0;
  for (int local_m = 0; local_m < kBlockM; ++local_m) {
    for (int local_k = 0; local_k < kBlockK; ++local_k) {
      int global_m = tile_m * kBlockM + local_m;
      int global_k = tile_k * kBlockK + local_k;
      Element expected{};
      if (global_m < m_extent && global_k < k_extent) {
        expected = make_value(global_m, global_k);
        ++valid;
      } else {
        ++zero_filled;
      }
      Element actual = output[local_m * kBlockK + local_k];
      if (global_m >= m_extent || global_k >= k_extent) {
        out_of_range_nonzero += static_cast<float>(actual) != 0.0f ? 1 : 0;
      }
      if (!same_value(actual, expected)) {
        if (errors < 5) {
          std::cerr << name << " mismatch at (" << local_m << "," << local_k << "): actual "
                    << static_cast<float>(actual) << " expected " << static_cast<float>(expected)
                    << "\n";
        }
        ++errors;
      }
    }
  }
  bool ok = errors == 0;
  std::cout << std::left << std::setw(30) << name << ": " << (ok ? "PASS" : "FAIL")
            << "  valid=" << valid << " zero-filled=" << zero_filled << "\n";
  if (zero_filled > 0) {
    std::cout << "  logical out-of-range nonzero writes: " << out_of_range_nonzero << "\n";
  }
  return ok;
}

template <class Launch>
bool run_and_verify(
    char const* name,
    Launch const& launch,
    Element* device_output,
    int m_extent,
    int k_extent,
    int tile_m,
    int tile_k) {
  CUDA_CHECK(cudaMemset(device_output, 0, sizeof(Element) * kTileElements));
  launch();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<Element> output(kTileElements);
  CUDA_CHECK(cudaMemcpy(
      output.data(), device_output, sizeof(Element) * kTileElements, cudaMemcpyDeviceToHost));
  return verify_tile(name, output, m_extent, k_extent, tile_m, tile_k);
}

template <class Launch>
float benchmark_microseconds(Launch const& launch) {
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  CUDA_CHECK(cudaEventCreate(&begin));
  CUDA_CHECK(cudaEventCreate(&end));
  for (int i = 0; i < 10; ++i) {
    launch();
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(begin));
  for (int i = 0; i < kBenchmarkIterations; ++i) {
    launch();
  }
  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, begin, end));
  CUDA_CHECK(cudaEventDestroy(begin));
  CUDA_CHECK(cudaEventDestroy(end));
  return elapsed_ms * 1000.0f / float(kBenchmarkIterations);
}

}  // namespace

int main() {
  using namespace cute;

  auto cache_always_copy = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, Element>{},
      CopyThreadLayout{},
      CopyValueLayout{});
  auto cache_global_copy = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, Element>{},
      CopyThreadLayout{},
      CopyValueLayout{});

  std::cout << "CuTe W16 cp.async global-to-shared demo\n";
  std::cout << "Problem                   : M=" << kM << " K=" << kK << "\n";
  std::cout << "CTA tile                  : " << kBlockM << " x " << kBlockK << "\n";
  std::cout << "Threads                   : " << kThreads << "\n";
  std::cout << "Copy width                : 128 bits (8 half values)\n";
  std::cout << "Async groups              : 1\n";
  std::cout << "CACHEALWAYS               : cache at all available levels\n";
  std::cout << "CACHEGLOBAL               : prefer global/L2 caching for 16-byte copy\n";
  std::cout << "Synchronization           : copy -> fence -> wait<0> -> __syncthreads\n";
  std::cout << "\nTiledCopy layout:\n";
  print(cache_always_copy);
  std::cout << "\n";

  std::vector<Element> host_input = make_problem(kM, kK, kK, kM);
  Element* device_input = nullptr;
  Element* device_output = nullptr;
  CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&device_input), sizeof(Element) * host_input.size()));
  CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&device_output), sizeof(Element) * kTileElements));
  CUDA_CHECK(cudaMemcpy(
      device_input,
      host_input.data(),
      sizeof(Element) * host_input.size(),
      cudaMemcpyHostToDevice));

  auto launch_always = [&] {
    cpasync_g2s_kernel<false><<<1, kThreads>>>(
        device_input,
        device_output,
        kM,
        kK,
        kK,
        kTileM,
        kTileK,
        cache_always_copy);
  };
  auto launch_global = [&] {
    cpasync_g2s_kernel<false><<<1, kThreads>>>(
        device_input,
        device_output,
        kM,
        kK,
        kK,
        kTileM,
        kTileK,
        cache_global_copy);
  };

  bool ok = true;
  std::cout << "\n[aligned correctness]\n";
  ok = run_and_verify(
           "cp.async CACHEALWAYS",
           launch_always,
           device_output,
           kM,
           kK,
           kTileM,
           kTileK) &&
       ok;
  ok = run_and_verify(
           "cp.async CACHEGLOBAL",
           launch_global,
           device_output,
           kM,
           kK,
           kTileM,
           kTileK) &&
       ok;

  float always_us = benchmark_microseconds(launch_always);
  float global_us = benchmark_microseconds(launch_global);
  double round_trip_bytes = 2.0 * double(kTileElements) * sizeof(Element);
  auto bandwidth = [&](float us) { return round_trip_bytes / (double(us) * 1.0e3); };
  std::cout << "\n[one-CTA microbenchmark]\n";
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "CACHEALWAYS  : " << always_us << " us, " << bandwidth(always_us) << " GB/s\n";
  std::cout << "CACHEGLOBAL  : " << global_us << " us, " << bandwidth(global_us) << " GB/s\n";
  std::cout << "Note: this isolates copy semantics; a multi-stage GEMM is needed to hide latency.\n";

  CUDA_CHECK(cudaFree(device_input));

  constexpr int kRaggedM = 2053;
  constexpr int kRaggedK = 2051;
  constexpr int kRaggedLeadingDim = 2056;
  constexpr int kRaggedAllocatedRows = 2176;
  constexpr int kRaggedTileM = 16;
  constexpr int kRaggedTileK = 64;

  std::vector<Element> ragged_input =
      make_problem(kRaggedM, kRaggedK, kRaggedLeadingDim, kRaggedAllocatedRows);
  CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&device_input), sizeof(Element) * ragged_input.size()));
  CUDA_CHECK(cudaMemcpy(
      device_input,
      ragged_input.data(),
      sizeof(Element) * ragged_input.size(),
      cudaMemcpyHostToDevice));

  auto launch_ragged = [&] {
    cpasync_g2s_kernel<true><<<1, kThreads>>>(
        device_input,
        device_output,
        kRaggedM,
        kRaggedK,
        kRaggedLeadingDim,
        kRaggedTileM,
        kRaggedTileK,
        cache_always_copy);
  };

  std::cout << "\n[ragged predication]\n";
  std::cout << "Logical problem           : M=" << kRaggedM << " K=" << kRaggedK << "\n";
  std::cout << "Padded allocation         : lda=" << kRaggedLeadingDim
            << ", invalid vector lanes contain zero\n";
  ok = run_and_verify(
           "predicated cp.async",
           launch_ragged,
           device_output,
           kRaggedM,
           kRaggedK,
           kRaggedTileM,
           kRaggedTileK) &&
       ok;

  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_output));

  if (!ok) {
    std::cerr << "W16 cp.async checks failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "\nW16 cp.async checks passed\n";
  return EXIT_SUCCESS;
}
