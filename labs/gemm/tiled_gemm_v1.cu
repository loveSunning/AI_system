#include "gemm_lab_kernels.hpp"

#include "ai_system/cuda/runtime.hpp"
#include "ai_system/profiling/nvtx.hpp"

#include <string>

namespace ai_system::labs::gemm {

namespace {

// Flip this to true after implementing tiled_gemm_v1_kernel below.
constexpr bool kTiledGemmV1KernelImplemented = true;

template <int BlockM, int BlockN, int BlockK>
__global__ void tiled_gemm_v1_kernel(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    float* __restrict__ out,
    std::size_t m,
    std::size_t n,
    std::size_t k
) {
    static_assert(BlockM > 0 && BlockN > 0 && BlockK > 0, "tiled_gemm_v1 tile dimensions must be positive.");
    static_assert(BlockM * BlockN <= 1024, "tiled_gemm_v1 block size exceeds CUDA's 1024-thread block limit.");

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int num_threads = blockDim.x * blockDim.y;

    const std::size_t block_row = static_cast<std::size_t>(blockIdx.y) * BlockM;
    const std::size_t block_col = static_cast<std::size_t>(blockIdx.x) * BlockN;
    const std::size_t row = block_row + static_cast<std::size_t>(ty);
    const std::size_t col = block_col + static_cast<std::size_t>(tx);

    __shared__ float lhs_tile[BlockM][BlockK];
    __shared__ float rhs_tile[BlockK][BlockN];

    float accumulator = 0.0f;

    const std::size_t num_k_tiles = (k + BlockK - 1) / BlockK;
    for(std::size_t tile = 0; tile < num_k_tiles; ++tile) {
        const std::size_t k_base = tile * BlockK;

        for(int idx = tid; idx < BlockM * BlockK; idx += num_threads) {
            const int local_row = idx / BlockK;
            const int local_col = idx % BlockK;
            const std::size_t global_row = block_row + static_cast<std::size_t>(local_row);
            const std::size_t global_col = k_base + static_cast<std::size_t>(local_col);

            lhs_tile[local_row][local_col] =
                (global_row < m && global_col < k) ? lhs[global_row * k + global_col] : 0.0f;
        }

        for(int idx = tid; idx < BlockK * BlockN; idx += num_threads) {
            const int local_row = idx / BlockN;
            const int local_col = idx % BlockN;
            const std::size_t global_row = k_base + static_cast<std::size_t>(local_row);
            const std::size_t global_col = block_col + static_cast<std::size_t>(local_col);

            rhs_tile[local_row][local_col] =
                (global_row < k && global_col < n) ? rhs[global_row * n + global_col] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for(int inner = 0; inner < BlockK; ++inner) {
            accumulator += lhs_tile[ty][inner] * rhs_tile[inner][tx];
        }

        __syncthreads();
    }

    if(row < m && col < n) {
        out[row * n + col] = accumulator;
    }
}

template <int BlockM, int BlockN, int BlockK>
bool launch_tiled_gemm_v1_kernel(
    const float* lhs,
    const float* rhs,
    float* out,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    std::string& error
) {
    const dim3 block(BlockN, BlockM);
    const dim3 grid(
        static_cast<unsigned int>((n + BlockN - 1) / BlockN),
        static_cast<unsigned int>((m + BlockM - 1) / BlockM)
    );

    tiled_gemm_v1_kernel<BlockM, BlockN, BlockK><<<grid, block>>>(lhs, rhs, out, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

template <int BlockM, int BlockN>
bool dispatch_tiled_gemm_v1_block_k(
    const float* lhs,
    const float* rhs,
    float* out,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    int block_k,
    std::string& error
) {
    switch(block_k) {
        case 8:
            return launch_tiled_gemm_v1_kernel<BlockM, BlockN, 8>(lhs, rhs, out, m, n, k, error);
        case 16:
            return launch_tiled_gemm_v1_kernel<BlockM, BlockN, 16>(lhs, rhs, out, m, n, k, error);
        case 32:
            return launch_tiled_gemm_v1_kernel<BlockM, BlockN, 32>(lhs, rhs, out, m, n, k, error);
    }

    error = "tiled_gemm_v1 block_k must be one of 8, 16, or 32.";
    return false;
}

template <int BlockM>
bool dispatch_tiled_gemm_v1_block_n(
    const float* lhs,
    const float* rhs,
    float* out,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    int block_n,
    int block_k,
    std::string& error
) {
    switch(block_n) {
        case 8:
            return dispatch_tiled_gemm_v1_block_k<BlockM, 8>(lhs, rhs, out, m, n, k, block_k, error);
        case 16:
            return dispatch_tiled_gemm_v1_block_k<BlockM, 16>(lhs, rhs, out, m, n, k, block_k, error);
        case 32:
            return dispatch_tiled_gemm_v1_block_k<BlockM, 32>(lhs, rhs, out, m, n, k, block_k, error);
    }

    error = "tiled_gemm_v1 block_n must be one of 8, 16, or 32.";
    return false;
}

bool dispatch_tiled_gemm_v1(
    const float* lhs,
    const float* rhs,
    float* out,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    GemmLabTileConfig tile_config,
    std::string& error
) {
    switch(tile_config.block_m) {
        case 8:
            return dispatch_tiled_gemm_v1_block_n<8>(
                lhs,
                rhs,
                out,
                m,
                n,
                k,
                tile_config.block_n,
                tile_config.block_k,
                error
            );
        case 16:
            return dispatch_tiled_gemm_v1_block_n<16>(
                lhs,
                rhs,
                out,
                m,
                n,
                k,
                tile_config.block_n,
                tile_config.block_k,
                error
            );
        case 32:
            return dispatch_tiled_gemm_v1_block_n<32>(
                lhs,
                rhs,
                out,
                m,
                n,
                k,
                tile_config.block_n,
                tile_config.block_k,
                error
            );
    }

    error = "tiled_gemm_v1 block_m must be one of 8, 16, or 32.";
    return false;
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
    GemmLabTileConfig tile_config,
    std::string& error
) {
    const ai_system::profiling::ScopedNvtxRange launch_range("tiled_gemm_v1_kernel_launch");

    if(!kTiledGemmV1KernelImplemented) {
        error = "tiled_gemm_v1_kernel is declared but not implemented yet.";
        return false;
    }

    return dispatch_tiled_gemm_v1(lhs, rhs, out, m, n, k, tile_config, error);
}

bool is_tiled_gemm_v1_kernel_implemented() {
    return kTiledGemmV1KernelImplemented;
}

}  // namespace detail

}  // namespace ai_system::labs::gemm
