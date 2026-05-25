#include "gemm_lab_kernels.hpp"

#include "ai_system/cuda/runtime.hpp"
#include "ai_system/profiling/nvtx.hpp"

#include <string>

namespace ai_system::labs::gemm {

namespace {

constexpr bool kTiledGemmRegisterKernelImplemented = true;

template <int BlockK>
__device__ __forceinline__ int swizzled_lhs_tile_col(int row, int col) {
    return col ^ (row & (BlockK - 1));
}

template <int BlockM, int BlockN, int BlockK, int RegisterM, int RegisterN>
__global__ void tiled_gemm_register_kernel(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    float* __restrict__ out,
    std::size_t m,
    std::size_t n,
    std::size_t k
) {
    static_assert(BlockM > 0 && BlockN > 0 && BlockK > 0, "tiled_gemm_register tile dimensions must be positive.");
    static_assert(RegisterM > 0 && RegisterN > 0, "tiled_gemm_register register tile dimensions must be positive.");
    static_assert(BlockM % RegisterM == 0, "tiled_gemm_register block_m must be divisible by the row register tile.");
    static_assert(BlockN % RegisterN == 0, "tiled_gemm_register block_n must be divisible by the column register tile.");
    static_assert((BlockK & (BlockK - 1)) == 0, "tiled_gemm_register block_k must be a power of two for lhs swizzling.");

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int num_threads = blockDim.x * blockDim.y;

    const int local_row_base = ty * RegisterM;
    const int local_col_base = tx * RegisterN;
    const std::size_t block_row = static_cast<std::size_t>(blockIdx.y) * BlockM;
    const std::size_t block_col = static_cast<std::size_t>(blockIdx.x) * BlockN;

    __shared__ float lhs_tile[BlockM][BlockK];
    __shared__ float rhs_tile[BlockK][BlockN];

    float accumulator[RegisterM][RegisterN];

    #pragma unroll
    for(int row_item = 0; row_item < RegisterM; ++row_item) {
        #pragma unroll
        for(int col_item = 0; col_item < RegisterN; ++col_item) {
            accumulator[row_item][col_item] = 0.0f;
        }
    }

    const std::size_t num_k_tiles = (k + BlockK - 1) / BlockK;
    for(std::size_t tile = 0; tile < num_k_tiles; ++tile) {
        const std::size_t k_base = tile * BlockK;

        for(int idx = tid; idx < BlockM * BlockK; idx += num_threads) {
            const int local_row = idx / BlockK;
            const int local_col = idx % BlockK;
            const std::size_t global_row = block_row + static_cast<std::size_t>(local_row);
            const std::size_t global_col = k_base + static_cast<std::size_t>(local_col);

            lhs_tile[local_row][swizzled_lhs_tile_col<BlockK>(local_row, local_col)] =
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
            float lhs_fragment[RegisterM];
            float rhs_fragment[RegisterN];

            #pragma unroll
            for(int row_item = 0; row_item < RegisterM; ++row_item) {
                const int local_row = local_row_base + row_item;
                lhs_fragment[row_item] =
                    lhs_tile[local_row][swizzled_lhs_tile_col<BlockK>(local_row, inner)];
            }

            #pragma unroll
            for(int col_item = 0; col_item < RegisterN; ++col_item) {
                rhs_fragment[col_item] = rhs_tile[inner][local_col_base + col_item];
            }

            #pragma unroll
            for(int row_item = 0; row_item < RegisterM; ++row_item) {
                #pragma unroll
                for(int col_item = 0; col_item < RegisterN; ++col_item) {
                    accumulator[row_item][col_item] += lhs_fragment[row_item] * rhs_fragment[col_item];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for(int row_item = 0; row_item < RegisterM; ++row_item) {
        const std::size_t row = block_row + static_cast<std::size_t>(local_row_base + row_item);

        #pragma unroll
        for(int col_item = 0; col_item < RegisterN; ++col_item) {
            const std::size_t col = block_col + static_cast<std::size_t>(local_col_base + col_item);
            if(row < m && col < n) {
                out[row * n + col] = accumulator[row_item][col_item];
            }
        }
    }
}

template <int BlockM, int BlockN, int BlockK, int RegisterM, int RegisterN>
bool launch_tiled_gemm_register_kernel(
    const float* lhs,
    const float* rhs,
    float* out,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    std::string& error
) {
    const dim3 block(BlockN / RegisterN, BlockM / RegisterM);
    const dim3 grid(
        static_cast<unsigned int>((n + BlockN - 1) / BlockN),
        static_cast<unsigned int>((m + BlockM - 1) / BlockM)
    );

    tiled_gemm_register_kernel<BlockM, BlockN, BlockK, RegisterM, RegisterN><<<grid, block>>>(lhs, rhs, out, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

template <int BlockM, int BlockN, int RegisterM, int RegisterN>
bool dispatch_tiled_gemm_register_k(
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
            return launch_tiled_gemm_register_kernel<BlockM, BlockN, 8, RegisterM, RegisterN>(
                lhs,
                rhs,
                out,
                m,
                n,
                k,
                error
            );
        case 16:
            return launch_tiled_gemm_register_kernel<BlockM, BlockN, 16, RegisterM, RegisterN>(
                lhs,
                rhs,
                out,
                m,
                n,
                k,
                error
            );
        case 32:
            return launch_tiled_gemm_register_kernel<BlockM, BlockN, 32, RegisterM, RegisterN>(
                lhs,
                rhs,
                out,
                m,
                n,
                k,
                error
            );
    }

    error = "tiled_gemm_register block_k must be one of 8, 16, or 32.";
    return false;
}

template <int BlockM, int BlockN, int RegisterM>
bool dispatch_tiled_gemm_register_register_n(
    const float* lhs,
    const float* rhs,
    float* out,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    int register_n,
    int block_k,
    std::string& error
) {
    switch(register_n) {
        case 1:
            return dispatch_tiled_gemm_register_k<BlockM, BlockN, RegisterM, 1>(
                lhs,
                rhs,
                out,
                m,
                n,
                k,
                block_k,
                error
            );
        case 2:
            return dispatch_tiled_gemm_register_k<BlockM, BlockN, RegisterM, 2>(
                lhs,
                rhs,
                out,
                m,
                n,
                k,
                block_k,
                error
            );
        case 4:
            return dispatch_tiled_gemm_register_k<BlockM, BlockN, RegisterM, 4>(
                lhs,
                rhs,
                out,
                m,
                n,
                k,
                block_k,
                error
            );
    }

    error = "tiled_gemm_register register_n must be one of 1, 2, or 4.";
    return false;
}

template <int BlockM, int BlockN>
bool dispatch_tiled_gemm_register_register_m(
    const float* lhs,
    const float* rhs,
    float* out,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    int register_m,
    int register_n,
    int block_k,
    std::string& error
) {
    switch(register_m) {
        case 1:
            return dispatch_tiled_gemm_register_register_n<BlockM, BlockN, 1>(
                lhs,
                rhs,
                out,
                m,
                n,
                k,
                register_n,
                block_k,
                error
            );
        case 2:
            return dispatch_tiled_gemm_register_register_n<BlockM, BlockN, 2>(
                lhs,
                rhs,
                out,
                m,
                n,
                k,
                register_n,
                block_k,
                error
            );
        case 4:
            return dispatch_tiled_gemm_register_register_n<BlockM, BlockN, 4>(
                lhs,
                rhs,
                out,
                m,
                n,
                k,
                register_n,
                block_k,
                error
            );
    }

    error = "tiled_gemm_register register_m must be one of 1, 2, or 4.";
    return false;
}

template <int BlockM>
bool dispatch_tiled_gemm_register_n(
    const float* lhs,
    const float* rhs,
    float* out,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    int block_n,
    int register_m,
    int register_n,
    int block_k,
    std::string& error
) {
    switch(block_n) {
        case 8:
            return dispatch_tiled_gemm_register_register_m<BlockM, 8>(
                lhs,
                rhs,
                out,
                m,
                n,
                k,
                register_m,
                register_n,
                block_k,
                error
            );
        case 16:
            return dispatch_tiled_gemm_register_register_m<BlockM, 16>(
                lhs,
                rhs,
                out,
                m,
                n,
                k,
                register_m,
                register_n,
                block_k,
                error
            );
        case 32:
            return dispatch_tiled_gemm_register_register_m<BlockM, 32>(
                lhs,
                rhs,
                out,
                m,
                n,
                k,
                register_m,
                register_n,
                block_k,
                error
            );
    }

    error = "tiled_gemm_register block_n must be one of 8, 16, or 32.";
    return false;
}

bool dispatch_tiled_gemm_register(
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
            return dispatch_tiled_gemm_register_n<8>(
                lhs,
                rhs,
                out,
                m,
                n,
                k,
                tile_config.block_n,
                tile_config.register_m,
                tile_config.register_n,
                tile_config.block_k,
                error
            );
        case 16:
            return dispatch_tiled_gemm_register_n<16>(
                lhs,
                rhs,
                out,
                m,
                n,
                k,
                tile_config.block_n,
                tile_config.register_m,
                tile_config.register_n,
                tile_config.block_k,
                error
            );
        case 32:
            return dispatch_tiled_gemm_register_n<32>(
                lhs,
                rhs,
                out,
                m,
                n,
                k,
                tile_config.block_n,
                tile_config.register_m,
                tile_config.register_n,
                tile_config.block_k,
                error
            );
    }

    error = "tiled_gemm_register block_m must be one of 8, 16, or 32.";
    return false;
}

}  // namespace

namespace detail {

bool launch_tiled_gemm_register(
    const float* lhs,
    const float* rhs,
    float* out,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    GemmLabTileConfig tile_config,
    std::string& error
) {
    const ai_system::profiling::ScopedNvtxRange launch_range("tiled_gemm_register_kernel_launch");

    if(!kTiledGemmRegisterKernelImplemented) {
        error = "tiled_gemm_register kernel is not implemented yet.";
        return false;
    }

    return dispatch_tiled_gemm_register(lhs, rhs, out, m, n, k, tile_config, error);
}

bool is_tiled_gemm_register_kernel_implemented() {
    return kTiledGemmRegisterKernelImplemented;
}

}  // namespace detail

}  // namespace ai_system::labs::gemm
