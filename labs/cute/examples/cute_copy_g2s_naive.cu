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

using Element = cute::half_t;

constexpr int kM = 2048;
constexpr int kK = 2048;
constexpr int kBlockM = 128;
constexpr int kBlockK = 32;
constexpr int kThreads = 128;
constexpr int kTileM = 3;
constexpr int kTileK = 7;
constexpr int kTileElements = kBlockM * kBlockK;
constexpr int kElementsPerThread = kTileElements / kThreads;
constexpr int kBenchmarkIterations = 200;

using BlockShape = cute::Shape<cute::Int<kBlockM>, cute::Int<kBlockK>>;
using SmemLayout = cute::Layout<
    cute::Shape<cute::Int<kBlockM>, cute::Int<kBlockK>>,
    cute::Stride<cute::Int<kBlockK>, cute::_1>>;
using ScalarThreadLayout =
    cute::Layout<cute::Shape<cute::_4, cute::_32>, cute::Stride<cute::_32, cute::_1>>;

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

template <bool Predicated>
__global__ void g2s_local_partition_kernel(
    Element const* input,
    Element* output,
    int m_extent,
    int k_extent,
    int leading_dim,
    int tile_m,
    int tile_k) {
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

  Tensor tAgA = local_partition(gA, ScalarThreadLayout{}, threadIdx.x);
  Tensor tAsA = local_partition(sA, ScalarThreadLayout{}, threadIdx.x);

  if constexpr (Predicated) {
    Tensor cA = make_identity_tensor(problem_shape);
    Tensor pA = cute::lazy::transform(cA, [&](auto coord) {
      return elem_less(coord, problem_shape);
    });
    Tensor gP = local_tile(pA, BlockShape{}, make_coord(tile_m, tile_k));
    Tensor tAgP = local_partition(gP, ScalarThreadLayout{}, threadIdx.x);
    copy_if(tAgP, tAgA, tAsA);
  } else {
    copy(tAgA, tAsA);
  }

  __syncthreads();
  for (int i = int(threadIdx.x); i < kTileElements; i += int(blockDim.x)) {
    int local_m = i / kBlockK;
    int local_k = i % kBlockK;
    output[i] = sA(local_m, local_k);
  }
}

template <bool Predicated, class TiledCopy>
__global__ void g2s_tiled_copy_kernel(
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

  __syncthreads();
  for (int i = int(threadIdx.x); i < kTileElements; i += int(blockDim.x)) {
    int local_m = i / kBlockK;
    int local_k = i % kBlockK;
    output[i] = sA(local_m, local_k);
  }
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
  bool ok = true;
  int errors = 0;
  int valid_count = 0;
  int zero_fill_count = 0;
  int out_of_range_nonzero = 0;
  for (int local_m = 0; local_m < kBlockM; ++local_m) {
    for (int local_k = 0; local_k < kBlockK; ++local_k) {
      int global_m = tile_m * kBlockM + local_m;
      int global_k = tile_k * kBlockK + local_k;
      Element expected{};
      if (global_m < m_extent && global_k < k_extent) {
        expected = make_value(global_m, global_k);
        ++valid_count;
      } else {
        ++zero_fill_count;
      }
      Element actual = output[local_m * kBlockK + local_k];
      if (global_m >= m_extent || global_k >= k_extent) {
        out_of_range_nonzero += static_cast<float>(actual) != 0.0f ? 1 : 0;
      }
      if (!same_value(actual, expected)) {
        if (errors < 5) {
          std::cerr << name << " mismatch at local (" << local_m << "," << local_k
                    << "), global (" << global_m << "," << global_k << "): actual "
                    << static_cast<float>(actual) << " expected " << static_cast<float>(expected)
                    << "\n";
        }
        ++errors;
        ok = false;
      }
    }
  }

  std::cout << std::left << std::setw(29) << name << ": " << (ok ? "PASS" : "FAIL")
            << "  valid=" << valid_count << " zero-filled=" << zero_fill_count << "\n";
  if (zero_fill_count > 0) {
    std::cout << "  logical out-of-range nonzero writes: " << out_of_range_nonzero << "\n";
  }
  return ok;
}

template <class TiledCopy>
bool print_and_check_tv_mapping(TiledCopy tiled_copy) {
  using namespace cute;

  std::cout << "\n[128-bit TiledCopy layouts]\n";
  std::cout << "TiledCopy:\n";
  print(tiled_copy);
  std::cout << "\n";

  Tensor coordinates = make_identity_tensor(BlockShape{});
  std::vector<int> visits(kTileElements, 0);
  constexpr std::array<int, 5> selected_threads{0, 1, 31, 32, 127};

  for (int tid = 0; tid < kThreads; ++tid) {
    auto thr_copy = tiled_copy.get_slice(tid);
    Tensor thread_coordinates = thr_copy.partition_S(coordinates);

    bool selected =
        std::find(selected_threads.begin(), selected_threads.end(), tid) != selected_threads.end();
    if (selected) {
      std::cout << "thread " << std::setw(3) << tid << " -> ";
    }

    for (int value = 0; value < size(thread_coordinates); ++value) {
      auto coord = thread_coordinates(value);
      int m = int(get<0>(coord));
      int k = int(get<1>(coord));
      if (m >= 0 && m < kBlockM && k >= 0 && k < kBlockK) {
        ++visits[m * kBlockK + k];
      }
      if (selected) {
        std::cout << "(" << m << "," << k << ")";
        if (value + 1 != size(thread_coordinates)) {
          std::cout << " ";
        }
      }
    }
    if (selected) {
      std::cout << "\n";
    }
  }

  int covered = 0;
  int duplicates = 0;
  int missing = 0;
  for (int count : visits) {
    covered += count > 0 ? 1 : 0;
    duplicates += count > 1 ? count - 1 : 0;
    missing += count == 0 ? 1 : 0;
  }
  bool ok = covered == kTileElements && duplicates == 0 && missing == 0;
  std::cout << "coverage=" << covered << "/" << kTileElements << " duplicates=" << duplicates
            << " missing=" << missing << " -> " << (ok ? "PASS" : "FAIL") << "\n";
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

}  // namespace

int main() {
  using namespace cute;

  auto scalar_tiled_copy = make_tiled_copy(
      Copy_Atom<UniversalCopy<Element>, Element>{},
      ScalarThreadLayout{},
      Layout<Shape<_1, _1>>{});

  auto vector_tiled_copy = make_tiled_copy(
      Copy_Atom<UniversalCopy<uint128_t>, Element>{},
      Layout<Shape<_32, _4>, Stride<_4, _1>>{},
      Layout<Shape<_1, _8>>{});

  std::cout << "CuTe W16 global-to-shared copy demo\n";
  std::cout << "Problem                   : M=" << kM << " K=" << kK << "\n";
  std::cout << "CTA tile                  : " << kBlockM << " x " << kBlockK << "\n";
  std::cout << "Tile elements             : " << kTileElements << "\n";
  std::cout << "Threads                   : " << kThreads << "\n";
  std::cout << "Elements per thread       : " << kElementsPerThread << "\n";
  std::cout << "Vector copy atom width    : 16 bytes\n";
  std::cout << "Elements per instruction  : 8 half values\n";
  std::cout << "Instructions per thread   : 4\n";

  bool ok = print_and_check_tv_mapping(vector_tiled_copy);

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

  auto launch_local = [&] {
    g2s_local_partition_kernel<false><<<1, kThreads>>>(
        device_input, device_output, kM, kK, kK, kTileM, kTileK);
  };
  auto launch_scalar_tiled = [&] {
    g2s_tiled_copy_kernel<false><<<1, kThreads>>>(
        device_input,
        device_output,
        kM,
        kK,
        kK,
        kTileM,
        kTileK,
        scalar_tiled_copy);
  };
  auto launch_vector_tiled = [&] {
    g2s_tiled_copy_kernel<false><<<1, kThreads>>>(
        device_input,
        device_output,
        kM,
        kK,
        kK,
        kTileM,
        kTileK,
        vector_tiled_copy);
  };

  std::cout << "\n[aligned correctness]\n";
  ok = run_and_verify(
           "scalar local_partition",
           launch_local,
           device_output,
           kM,
           kK,
           kTileM,
           kTileK) &&
       ok;
  ok = run_and_verify(
           "scalar TiledCopy",
           launch_scalar_tiled,
           device_output,
           kM,
           kK,
           kTileM,
           kTileK) &&
       ok;
  ok = run_and_verify(
           "128-bit TiledCopy",
           launch_vector_tiled,
           device_output,
           kM,
           kK,
           kTileM,
           kTileK) &&
       ok;

  std::cout << "\n[one-CTA microbenchmark]\n";
  float local_us = benchmark_microseconds(launch_local);
  float scalar_us = benchmark_microseconds(launch_scalar_tiled);
  float vector_us = benchmark_microseconds(launch_vector_tiled);
  double round_trip_bytes = 2.0 * double(kTileElements) * sizeof(Element);
  auto bandwidth = [&](float us) { return round_trip_bytes / (double(us) * 1.0e3); };
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "scalar local_partition : " << local_us << " us, " << bandwidth(local_us)
            << " GB/s\n";
  std::cout << "scalar TiledCopy       : " << scalar_us << " us, " << bandwidth(scalar_us)
            << " GB/s\n";
  std::cout << "128-bit TiledCopy      : " << vector_us << " us, " << bandwidth(vector_us)
            << " GB/s\n";
  std::cout << "Note: this is a one-CTA teaching microbenchmark; launch overhead dominates.\n";

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

  auto launch_ragged_local = [&] {
    g2s_local_partition_kernel<true><<<1, kThreads>>>(
        device_input,
        device_output,
        kRaggedM,
        kRaggedK,
        kRaggedLeadingDim,
        kRaggedTileM,
        kRaggedTileK);
  };
  auto launch_ragged_vector = [&] {
    g2s_tiled_copy_kernel<true><<<1, kThreads>>>(
        device_input,
        device_output,
        kRaggedM,
        kRaggedK,
        kRaggedLeadingDim,
        kRaggedTileM,
        kRaggedTileK,
        vector_tiled_copy);
  };

  std::cout << "\n[ragged predication]\n";
  std::cout << "Logical problem           : M=" << kRaggedM << " K=" << kRaggedK << "\n";
  std::cout << "Padded leading dimension  : " << kRaggedLeadingDim
            << " (keeps 128-bit accesses in allocated memory)\n";
  ok = run_and_verify(
           "ragged scalar copy_if",
           launch_ragged_local,
           device_output,
           kRaggedM,
           kRaggedK,
           kRaggedTileM,
           kRaggedTileK) &&
       ok;
  ok = run_and_verify(
           "ragged vector copy_if",
           launch_ragged_vector,
           device_output,
           kRaggedM,
           kRaggedK,
           kRaggedTileM,
           kRaggedTileK) &&
       ok;

  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_output));

  if (!ok) {
    std::cerr << "W16 global-to-shared copy checks failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "\nW16 global-to-shared copy checks passed\n";
  return EXIT_SUCCESS;
}
