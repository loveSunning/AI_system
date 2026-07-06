#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/version.h>

#include <cuda_runtime_api.h>

#include <cstdlib>
#include <iostream>

int main() {
  int device_count = 0;
  cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess) {
    std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(status) << "\n";
    return EXIT_FAILURE;
  }

  std::cout << "CUTLASS header probe\n";
  std::cout << "CUTLASS version: " << CUTLASS_MAJOR << "." << CUTLASS_MINOR << "." << CUTLASS_PATCH << "\n";
  std::cout << "CUDA devices: " << device_count << "\n";

  for (int device = 0; device < device_count; ++device) {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);
    std::cout << "device " << device << ": " << prop.name << " sm_" << prop.major << prop.minor << "\n";
  }

  cutlass::half_t value = cutlass::half_t(2.25f);
  std::cout << "cutlass::half_t smoke value converted to float: " << static_cast<float>(value) << "\n";
  return EXIT_SUCCESS;
}
