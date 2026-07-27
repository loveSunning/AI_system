#include <cute/tensor.hpp>
#include <cute/util/print.hpp>

#include <cuda_runtime.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

using Element = cute::half_t;

constexpr int kBlockM = 128;
constexpr int kBlockK = 32;
constexpr int kThreads = 128;
constexpr int kTileElements = kBlockM * kBlockK;
constexpr int kValuesPerThread = kTileElements / kThreads;
constexpr int kSelectedThread = 45;

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

template <class TiledCopy>
__global__ void shared_to_register_kernel(
    Element const* global_input,
    Element* fragment_output,
    int2* coordinate_output,
    TiledCopy tiled_copy) {
  using namespace cute;

  __shared__ Element storage[cosize_v<SmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(storage), SmemLayout{});

  for (int linear = int(threadIdx.x); linear < kTileElements; linear += int(blockDim.x)) {
    int m = linear / kBlockK;
    int k = linear % kBlockK;
    sA(m, k) = global_input[linear];
  }
  __syncthreads();

  ThrCopy thr_copy = tiled_copy.get_slice(threadIdx.x);
  Tensor tXsA = thr_copy.partition_S(sA);
  Tensor tXrA = make_fragment_like(tXsA);
  copy(tiled_copy, tXsA, tXrA);

  Tensor coordinates = make_identity_tensor(BlockShape{});
  Tensor tXcA = thr_copy.partition_S(coordinates);
  static_assert(decltype(size(tXrA))::value == kValuesPerThread);
  static_assert(decltype(size(tXcA))::value == kValuesPerThread);

  for (int value = 0; value < size(tXrA); ++value) {
    int output_index = int(threadIdx.x) * kValuesPerThread + value;
    auto coord = tXcA(value);
    fragment_output[output_index] = tXrA(value);
    coordinate_output[output_index] = make_int2(int(get<0>(coord)), int(get<1>(coord)));
  }
}

bool verify_fragments(
    std::vector<Element> const& values,
    std::vector<int2> const& coordinates) {
  int errors = 0;
  std::vector<int> visits(kTileElements, 0);
  for (int tid = 0; tid < kThreads; ++tid) {
    for (int value = 0; value < kValuesPerThread; ++value) {
      int index = tid * kValuesPerThread + value;
      int m = coordinates[index].x;
      int k = coordinates[index].y;
      if (m < 0 || m >= kBlockM || k < 0 || k >= kBlockK) {
        ++errors;
        continue;
      }
      ++visits[m * kBlockK + k];
      float actual = static_cast<float>(values[index]);
      float expected = static_cast<float>(make_value(m, k));
      if (actual != expected) {
        if (errors < 5) {
          std::cerr << "fragment mismatch: thread=" << tid << " value=" << value
                    << " coord=(" << m << "," << k << ") actual=" << actual
                    << " expected=" << expected << "\n";
        }
        ++errors;
      }
    }
  }

  int missing = 0;
  int duplicates = 0;
  for (int count : visits) {
    missing += count == 0 ? 1 : 0;
    duplicates += count > 1 ? count - 1 : 0;
  }
  bool ok = errors == 0 && missing == 0 && duplicates == 0;
  std::cout << "All fragments validation    : " << (ok ? "PASS" : "FAIL") << "\n";
  std::cout << "Coordinate coverage         : " << kTileElements - missing << "/"
            << kTileElements << ", duplicates=" << duplicates << "\n";
  return ok;
}

}  // namespace

int main() {
  using namespace cute;

  auto tiled_copy = make_tiled_copy(
      Copy_Atom<UniversalCopy<uint128_t>, Element>{},
      CopyThreadLayout{},
      CopyValueLayout{});

  std::cout << "CuTe W16 shared-to-register fragment demo\n";
  std::cout << "Data path                  : global tile -> shared tile -> register fragment\n";
  std::cout << "Tile                       : " << kBlockM << " x " << kBlockK << "\n";
  std::cout << "Threads                    : " << kThreads << "\n";
  std::cout << "Values per thread          : " << kValuesPerThread << "\n";
  std::cout << "Copy width                 : 128 bits (8 half values)\n";
  std::cout << "\nTiledCopy layout:\n";
  print(tiled_copy);
  std::cout << "\n";

  std::vector<Element> host_input(kTileElements);
  for (int m = 0; m < kBlockM; ++m) {
    for (int k = 0; k < kBlockK; ++k) {
      host_input[m * kBlockK + k] = make_value(m, k);
    }
  }

  Element* device_input = nullptr;
  Element* device_fragments = nullptr;
  int2* device_coordinates = nullptr;
  CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&device_input), sizeof(Element) * kTileElements));
  CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&device_fragments), sizeof(Element) * kTileElements));
  CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&device_coordinates), sizeof(int2) * kTileElements));
  CUDA_CHECK(cudaMemcpy(
      device_input,
      host_input.data(),
      sizeof(Element) * kTileElements,
      cudaMemcpyHostToDevice));

  shared_to_register_kernel<<<1, kThreads>>>(
      device_input, device_fragments, device_coordinates, tiled_copy);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<Element> host_fragments(kTileElements);
  std::vector<int2> host_coordinates(kTileElements);
  CUDA_CHECK(cudaMemcpy(
      host_fragments.data(),
      device_fragments,
      sizeof(Element) * kTileElements,
      cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(
      host_coordinates.data(),
      device_coordinates,
      sizeof(int2) * kTileElements,
      cudaMemcpyDeviceToHost));

  bool ok = verify_fragments(host_fragments, host_coordinates);

  auto null_smem =
      make_tensor(make_smem_ptr(static_cast<Element*>(nullptr)), SmemLayout{});
  auto selected_copy = tiled_copy.get_slice(kSelectedThread);
  auto selected_source = selected_copy.partition_S(null_smem);
  auto selected_fragment = make_fragment_like(selected_source);

  std::cout << "\n[selected thread]\n";
  std::cout << "thread id                  : " << kSelectedThread << "\n";
  std::cout << "shared partition layout    : ";
  print(selected_source.layout());
  std::cout << "\n";
  std::cout << "register fragment layout   : ";
  print(selected_fragment.layout());
  std::cout << "\n";
  std::cout << "value | logical coord | shared offset | register value\n";

  for (int value = 0; value < kValuesPerThread; ++value) {
    int index = kSelectedThread * kValuesPerThread + value;
    int m = host_coordinates[index].x;
    int k = host_coordinates[index].y;
    int shared_offset = int(SmemLayout{}(m, k));
    std::cout << std::setw(5) << value << " | (" << std::setw(3) << m << "," << std::setw(2)
              << k << ")       | " << std::setw(13) << shared_offset << " | "
              << static_cast<float>(host_fragments[index]) << "\n";
  }

  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_fragments));
  CUDA_CHECK(cudaFree(device_coordinates));

  if (!ok) {
    std::cerr << "W16 shared-to-register checks failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "\nW16 shared-to-register checks passed\n";
  return EXIT_SUCCESS;
}
