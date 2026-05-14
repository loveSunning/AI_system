#include "gemm_lab_kernels.hpp"

#include "ai_system/cuda/runtime.hpp"
#include "ai_system/profiling/nvtx.hpp"

#include <string>

namespace ai_system::labs::gemm {

namespace {

constexpr int kTiledGemmV1TileSize = 16;

// Flip this to true after implementing tiled_gemm_v1_kernel below.
constexpr bool kTiledGemmV1KernelImplemented = false;

__global__ void tiled_gemm_v1_kernel(
    const float* lhs,
    const float* rhs,
    float* out,
    std::size_t m,
    std::size_t n,
    std::size_t k
) {
    // TODO: Implement shared-memory tiled GEMM v1 here.
    // Contract: row-major lhs[m,k], rhs[k,n], out[m,n], one output element per thread.
    (void)lhs;
    (void)rhs;
    (void)out;
    (void)m;
    (void)n;
    (void)k;
}

}  // namespace

namespace detail {

bool launch_tiled_gemm_v1(
    const float* lhs,
    const float* rhs,
    float* out,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    std::string& error
) {
    const ai_system::profiling::ScopedNvtxRange launch_range("tiled_gemm_v1_kernel_launch");

    if(!kTiledGemmV1KernelImplemented) {
        error = "tiled_gemm_v1_kernel is declared but not implemented yet.";
        return false;
    }

    const dim3 block(kTiledGemmV1TileSize, kTiledGemmV1TileSize);
    const dim3 grid(
        static_cast<unsigned int>((n + block.x - 1) / block.x),
        static_cast<unsigned int>((m + block.y - 1) / block.y)
    );

    tiled_gemm_v1_kernel<<<grid, block>>>(lhs, rhs, out, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool tiled_gemm_v1_kernel_available() {
    return kTiledGemmV1KernelImplemented;
}

}  // namespace detail

}  // namespace ai_system::labs::gemm
