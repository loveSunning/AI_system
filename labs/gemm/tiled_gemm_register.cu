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
__global__ void __launch_bounds__(1024, 1) tiled_gemm_register_kernel(
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
bool launch_tiled_gemm_register_instance(
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

using TiledGemmRegisterLaunchFn = bool (*)(
    const float* lhs,
    const float* rhs,
    float* out,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    std::string& error
);

struct TiledGemmRegisterSpec {
    int block_m;
    int block_n;
    int block_k;
    int register_m;
    int register_n;
    TiledGemmRegisterLaunchFn launch;
};

#define AI_SYSTEM_TILED_GEMM_REGISTER_SPEC(BLOCK_M, BLOCK_N, BLOCK_K, REGISTER_M, REGISTER_N) \
    { \
        BLOCK_M, \
        BLOCK_N, \
        BLOCK_K, \
        REGISTER_M, \
        REGISTER_N, \
        &launch_tiled_gemm_register_instance<BLOCK_M, BLOCK_N, BLOCK_K, REGISTER_M, REGISTER_N> \
    }

#define AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS_FOR_K(BLOCK_M, BLOCK_N, BLOCK_K) \
    AI_SYSTEM_TILED_GEMM_REGISTER_SPEC(BLOCK_M, BLOCK_N, BLOCK_K, 2, 2), \
    AI_SYSTEM_TILED_GEMM_REGISTER_SPEC(BLOCK_M, BLOCK_N, BLOCK_K, 4, 4), \
    AI_SYSTEM_TILED_GEMM_REGISTER_SPEC(BLOCK_M, BLOCK_N, BLOCK_K, 4, 8), \
    AI_SYSTEM_TILED_GEMM_REGISTER_SPEC(BLOCK_M, BLOCK_N, BLOCK_K, 8, 4), \
    AI_SYSTEM_TILED_GEMM_REGISTER_SPEC(BLOCK_M, BLOCK_N, BLOCK_K, 8, 8)

#define AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS(BLOCK_M, BLOCK_N) \
    AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS_FOR_K(BLOCK_M, BLOCK_N, 8), \
    AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS_FOR_K(BLOCK_M, BLOCK_N, 16), \
    AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS_FOR_K(BLOCK_M, BLOCK_N, 32)

#define AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS_NO_2X2_FOR_K(BLOCK_M, BLOCK_N, BLOCK_K) \
    AI_SYSTEM_TILED_GEMM_REGISTER_SPEC(BLOCK_M, BLOCK_N, BLOCK_K, 4, 4), \
    AI_SYSTEM_TILED_GEMM_REGISTER_SPEC(BLOCK_M, BLOCK_N, BLOCK_K, 4, 8), \
    AI_SYSTEM_TILED_GEMM_REGISTER_SPEC(BLOCK_M, BLOCK_N, BLOCK_K, 8, 4), \
    AI_SYSTEM_TILED_GEMM_REGISTER_SPEC(BLOCK_M, BLOCK_N, BLOCK_K, 8, 8)

#define AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS_NO_2X2(BLOCK_M, BLOCK_N) \
    AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS_NO_2X2_FOR_K(BLOCK_M, BLOCK_N, 8), \
    AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS_NO_2X2_FOR_K(BLOCK_M, BLOCK_N, 16), \
    AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS_NO_2X2_FOR_K(BLOCK_M, BLOCK_N, 32)

constexpr TiledGemmRegisterSpec kTiledGemmRegisterSpecs[] = {
    AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS(8, 8),
    AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS(8, 16),
    AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS(8, 32),
    AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS(8, 64),
    AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS(8, 128),
    AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS(16, 8),
    AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS(16, 16),
    AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS(16, 32),
    AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS(16, 64),
    AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS(16, 128),
    AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS(32, 8),
    AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS(32, 16),
    AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS(32, 32),
    AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS(32, 64),
    AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS(32, 128),
    AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS(64, 8),
    AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS(64, 16),
    AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS(64, 32),
    AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS(64, 64),
    AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS_NO_2X2(64, 128),
    AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS(128, 8),
    AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS(128, 16),
    AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS(128, 32),
    AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS_NO_2X2(128, 64),
    AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS_NO_2X2(128, 128)
};

#undef AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS_NO_2X2
#undef AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS_NO_2X2_FOR_K
#undef AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS
#undef AI_SYSTEM_TILED_GEMM_REGISTER_REG_PAIRS_FOR_K
#undef AI_SYSTEM_TILED_GEMM_REGISTER_SPEC

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
    for(const auto& spec : kTiledGemmRegisterSpecs) {
        if(spec.block_m == tile_config.block_m && spec.block_n == tile_config.block_n &&
           spec.block_k == tile_config.block_k && spec.register_m == tile_config.register_m &&
           spec.register_n == tile_config.register_n) {
            return spec.launch(lhs, rhs, out, m, n, k, error);
        }
    }

    error = "tiled_gemm_register shape is not compiled into the launcher table: block=" +
        std::to_string(tile_config.block_m) + "x" + std::to_string(tile_config.block_n) + "x" +
        std::to_string(tile_config.block_k) + ", register=" + std::to_string(tile_config.register_m) + "x" +
        std::to_string(tile_config.register_n) +
        ". Compiled register tiles are 2x2, 4x4, 4x8, 8x4, and 8x8 and obey the 1024-thread limit.";
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
