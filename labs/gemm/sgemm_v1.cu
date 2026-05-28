#include "gemm_lab_kernels.hpp"

#include "ai_system/cuda/runtime.hpp"
#include "ai_system/profiling/nvtx.hpp"

#include <string>

namespace ai_system::labs::gemm {

namespace {

constexpr bool kSgemmV1KernelImplemented = true;
constexpr int kMaxSgemmV1ThreadsPerBlock = 32 * 32;
constexpr int kFloat4Width = 4;

template <int BlockM, int BlockN, int RegisterM, int RegisterN>
constexpr int sgemm_v1_threads_per_cta() {
    return (BlockM / RegisterM) * (BlockN / RegisterN);
}

template <int BlockM, int BlockK>
__device__ __forceinline__ int lhs_tile_offset(int buffer, int row, int col) {
    return (buffer * BlockK + col) * BlockM + row;
}

template <int BlockK, int BlockN>
__device__ __forceinline__ int rhs_tile_offset(int buffer, int row, int col) {
    return (buffer * BlockK + row) * BlockN + col;
}

__device__ __forceinline__ float4 zero_float4() {
    return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

__device__ __forceinline__ bool can_load_float4(std::size_t element_offset, std::size_t vector_col, std::size_t extent) {
    return vector_col + (kFloat4Width - 1) < extent && (element_offset & (kFloat4Width - 1)) == 0;
}

template <int BlockM, int BlockK>
__device__ __forceinline__ void store_lhs_vector(
    float* lhs_tiles,
    int buffer,
    int local_row,
    int local_col,
    float4 value
) {
    lhs_tiles[lhs_tile_offset<BlockM, BlockK>(buffer, local_row, local_col + 0)] = value.x;
    lhs_tiles[lhs_tile_offset<BlockM, BlockK>(buffer, local_row, local_col + 1)] = value.y;
    lhs_tiles[lhs_tile_offset<BlockM, BlockK>(buffer, local_row, local_col + 2)] = value.z;
    lhs_tiles[lhs_tile_offset<BlockM, BlockK>(buffer, local_row, local_col + 3)] = value.w;
}

template <int BlockK, int BlockN>
__device__ __forceinline__ void store_rhs_vector(
    float* rhs_tiles,
    int buffer,
    int local_row,
    int local_col,
    float4 value
) {
    rhs_tiles[rhs_tile_offset<BlockK, BlockN>(buffer, local_row, local_col + 0)] = value.x;
    rhs_tiles[rhs_tile_offset<BlockK, BlockN>(buffer, local_row, local_col + 1)] = value.y;
    rhs_tiles[rhs_tile_offset<BlockK, BlockN>(buffer, local_row, local_col + 2)] = value.z;
    rhs_tiles[rhs_tile_offset<BlockK, BlockN>(buffer, local_row, local_col + 3)] = value.w;
}

template <int BlockM, int BlockN, int BlockK>
__device__ __forceinline__ void load_global_tiles_float4(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    float* lhs_tiles,
    float* rhs_tiles,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    std::size_t block_row,
    std::size_t block_col,
    std::size_t k_base,
    int buffer,
    int tid,
    int num_threads
) {
    static_assert(BlockK % kFloat4Width == 0, "sgemm_v1 requires block_k to be divisible by 4.");
    static_assert(BlockN % kFloat4Width == 0, "sgemm_v1 requires block_n to be divisible by 4.");

    constexpr int kLhsVectors = BlockM * BlockK / kFloat4Width;
    for(int idx = tid; idx < kLhsVectors; idx += num_threads) {
        const int local_row = idx / (BlockK / kFloat4Width);
        const int local_col = (idx % (BlockK / kFloat4Width)) * kFloat4Width;
        const std::size_t global_row = block_row + static_cast<std::size_t>(local_row);
        const std::size_t global_col = k_base + static_cast<std::size_t>(local_col);

        float4 value = zero_float4();
        if(global_row < m) {
            const std::size_t element_offset = global_row * k + global_col;
            if(can_load_float4(element_offset, global_col, k)) {
                value = *reinterpret_cast<const float4*>(lhs + element_offset);
            } else {
                value.x = global_col + 0 < k ? lhs[global_row * k + global_col + 0] : 0.0f;
                value.y = global_col + 1 < k ? lhs[global_row * k + global_col + 1] : 0.0f;
                value.z = global_col + 2 < k ? lhs[global_row * k + global_col + 2] : 0.0f;
                value.w = global_col + 3 < k ? lhs[global_row * k + global_col + 3] : 0.0f;
            }
        }

        store_lhs_vector<BlockM, BlockK>(lhs_tiles, buffer, local_row, local_col, value);
    }

    constexpr int kRhsVectors = BlockK * BlockN / kFloat4Width;
    for(int idx = tid; idx < kRhsVectors; idx += num_threads) {
        const int local_row = idx / (BlockN / kFloat4Width);
        const int local_col = (idx % (BlockN / kFloat4Width)) * kFloat4Width;
        const std::size_t global_row = k_base + static_cast<std::size_t>(local_row);
        const std::size_t global_col = block_col + static_cast<std::size_t>(local_col);

        float4 value = zero_float4();
        if(global_row < k) {
            const std::size_t element_offset = global_row * n + global_col;
            if(can_load_float4(element_offset, global_col, n)) {
                value = *reinterpret_cast<const float4*>(rhs + element_offset);
            } else {
                value.x = global_col + 0 < n ? rhs[global_row * n + global_col + 0] : 0.0f;
                value.y = global_col + 1 < n ? rhs[global_row * n + global_col + 1] : 0.0f;
                value.z = global_col + 2 < n ? rhs[global_row * n + global_col + 2] : 0.0f;
                value.w = global_col + 3 < n ? rhs[global_row * n + global_col + 3] : 0.0f;
            }
        }

        store_rhs_vector<BlockK, BlockN>(rhs_tiles, buffer, local_row, local_col, value);
    }
}

template <int BlockM, int BlockN, int BlockK, int RegisterM, int RegisterN>
__device__ __forceinline__ void load_register_fragments(
    const float* lhs_tiles,
    const float* rhs_tiles,
    float (&lhs_fragment)[2][RegisterM],
    float (&rhs_fragment)[2][RegisterN],
    int shared_buffer,
    int register_buffer,
    int inner,
    int local_row_base,
    int local_col_base
) {
    #pragma unroll
    for(int row_item = 0; row_item < RegisterM; ++row_item) {
        lhs_fragment[register_buffer][row_item] =
            lhs_tiles[lhs_tile_offset<BlockM, BlockK>(shared_buffer, local_row_base + row_item, inner)];
    }

    #pragma unroll
    for(int col_item = 0; col_item < RegisterN; ++col_item) {
        rhs_fragment[register_buffer][col_item] =
            rhs_tiles[rhs_tile_offset<BlockK, BlockN>(shared_buffer, inner, local_col_base + col_item)];
    }
}

template <int RegisterM, int RegisterN>
__device__ __forceinline__ void accumulate_register_tile(
    float (&accumulator)[RegisterM][RegisterN],
    const float (&lhs_fragment)[2][RegisterM],
    const float (&rhs_fragment)[2][RegisterN],
    int register_buffer
) {
    #pragma unroll
    for(int row_item = 0; row_item < RegisterM; ++row_item) {
        const float lhs_value = lhs_fragment[register_buffer][row_item];

        #pragma unroll
        for(int col_item = 0; col_item < RegisterN; ++col_item) {
            accumulator[row_item][col_item] += lhs_value * rhs_fragment[register_buffer][col_item];
        }
    }
}

template <int RegisterM, int RegisterN>
__device__ __forceinline__ void store_output_tile(
    const float (&accumulator)[RegisterM][RegisterN],
    float* __restrict__ out,
    std::size_t m,
    std::size_t n,
    std::size_t block_row,
    std::size_t block_col,
    int local_row_base,
    int local_col_base
) {
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
__global__ void __launch_bounds__(sgemm_v1_threads_per_cta<BlockM, BlockN, RegisterM, RegisterN>(), 1) sgemm_v1_kernel(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    float* __restrict__ out,
    std::size_t m,
    std::size_t n,
    std::size_t k
) {
    static_assert(BlockM == 32 || BlockM == 64 || BlockM == 128, "sgemm_v1 block_m must be 32, 64, or 128.");
    static_assert(BlockN == 32 || BlockN == 64 || BlockN == 128, "sgemm_v1 block_n must be 32, 64, or 128.");
    static_assert(BlockK == 8 || BlockK == 16 || BlockK == 32, "sgemm_v1 block_k must be 8, 16, or 32.");
    static_assert(RegisterM == RegisterN, "sgemm_v1 currently supports square register tiles.");
    static_assert(RegisterM == 4 || RegisterM == 8, "sgemm_v1 register tile must be 4x4 or 8x8.");
    static_assert(BlockM % RegisterM == 0, "sgemm_v1 block_m must be divisible by register_m.");
    static_assert(BlockN % RegisterN == 0, "sgemm_v1 block_n must be divisible by register_n.");

    constexpr int kThreadBlockDimX = BlockN / RegisterN;
    constexpr int kThreadsPerCta = sgemm_v1_threads_per_cta<BlockM, BlockN, RegisterM, RegisterN>();
    static_assert(kThreadsPerCta <= kMaxSgemmV1ThreadsPerBlock, "sgemm_v1 derived block size must fit within 1024.");

    const int tid = static_cast<int>(threadIdx.y) * kThreadBlockDimX + static_cast<int>(threadIdx.x);
    const int local_row_base = static_cast<int>(threadIdx.y) * RegisterM;
    const int local_col_base = static_cast<int>(threadIdx.x) * RegisterN;
    const std::size_t block_row = static_cast<std::size_t>(blockIdx.y) * BlockM;
    const std::size_t block_col = static_cast<std::size_t>(blockIdx.x) * BlockN;

    extern __shared__ float shared_storage[];
    float* lhs_tiles = shared_storage;
    float* rhs_tiles = lhs_tiles + 2 * BlockK * BlockM;

    float accumulator[RegisterM][RegisterN];
    float lhs_fragment[2][RegisterM];
    float rhs_fragment[2][RegisterN];

    #pragma unroll
    for(int row_item = 0; row_item < RegisterM; ++row_item) {
        #pragma unroll
        for(int col_item = 0; col_item < RegisterN; ++col_item) {
            accumulator[row_item][col_item] = 0.0f;
        }
    }

    const std::size_t num_k_tiles = (k + BlockK - 1) / BlockK;
    load_global_tiles_float4<BlockM, BlockN, BlockK>(
        lhs,
        rhs,
        lhs_tiles,
        rhs_tiles,
        m,
        n,
        k,
        block_row,
        block_col,
        0,
        0,
        tid,
        kThreadsPerCta
    );
    __syncthreads();

    for(std::size_t tile = 0; tile < num_k_tiles; ++tile) {
        const int shared_buffer = static_cast<int>(tile & 1);
        const int next_shared_buffer = shared_buffer ^ 1;

        if(tile + 1 < num_k_tiles) {
            load_global_tiles_float4<BlockM, BlockN, BlockK>(
                lhs,
                rhs,
                lhs_tiles,
                rhs_tiles,
                m,
                n,
                k,
                block_row,
                block_col,
                (tile + 1) * BlockK,
                next_shared_buffer,
                tid,
                kThreadsPerCta
            );
        }

        load_register_fragments<BlockM, BlockN, BlockK, RegisterM, RegisterN>(
            lhs_tiles,
            rhs_tiles,
            lhs_fragment,
            rhs_fragment,
            shared_buffer,
            0,
            0,
            local_row_base,
            local_col_base
        );

        #pragma unroll
        for(int inner = 0; inner < BlockK; ++inner) {
            const int register_buffer = inner & 1;
            const int next_register_buffer = register_buffer ^ 1;

            if(inner + 1 < BlockK) {
                load_register_fragments<BlockM, BlockN, BlockK, RegisterM, RegisterN>(
                    lhs_tiles,
                    rhs_tiles,
                    lhs_fragment,
                    rhs_fragment,
                    shared_buffer,
                    next_register_buffer,
                    inner + 1,
                    local_row_base,
                    local_col_base
                );
            }

            accumulate_register_tile<RegisterM, RegisterN>(accumulator, lhs_fragment, rhs_fragment, register_buffer);
        }

        __syncthreads();
    }

    store_output_tile<RegisterM, RegisterN>(
        accumulator,
        out,
        m,
        n,
        block_row,
        block_col,
        local_row_base,
        local_col_base
    );
}

template <int BlockM, int BlockN, int BlockK, int RegisterM, int RegisterN>
bool launch_sgemm_v1_instance(
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
    constexpr int kSharedStorageFloats = 2 * (BlockK * BlockM + BlockK * BlockN);
    constexpr std::size_t kSharedStorageBytes = static_cast<std::size_t>(kSharedStorageFloats) * sizeof(float);

    static bool attributes_configured = false;
    if(!attributes_configured) {
        const auto max_dynamic_shared_status = cudaFuncSetAttribute(
            sgemm_v1_kernel<BlockM, BlockN, BlockK, RegisterM, RegisterN>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(kSharedStorageBytes)
        );
        if(!ai_system::cuda_utils::check_status(
               max_dynamic_shared_status,
               "cudaFuncSetAttribute(MaxDynamicSharedMemorySize)",
               error
           )) {
            return false;
        }

        attributes_configured = true;
    }

    sgemm_v1_kernel<BlockM, BlockN, BlockK, RegisterM, RegisterN>
        <<<grid, block, kSharedStorageBytes>>>(lhs, rhs, out, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

using SgemmV1LaunchFn = bool (*)(
    const float* lhs,
    const float* rhs,
    float* out,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    std::string& error
);

struct SgemmV1Spec {
    int block_m;
    int block_n;
    int block_k;
    int register_m;
    int register_n;
    SgemmV1LaunchFn launch;
};

#define AI_SYSTEM_SGEMM_V1_SPEC(BLOCK_M, BLOCK_N, BLOCK_K, REGISTER_M, REGISTER_N) \
    { \
        BLOCK_M, \
        BLOCK_N, \
        BLOCK_K, \
        REGISTER_M, \
        REGISTER_N, \
        &launch_sgemm_v1_instance<BLOCK_M, BLOCK_N, BLOCK_K, REGISTER_M, REGISTER_N> \
    }

#define AI_SYSTEM_SGEMM_V1_REG_PAIRS_FOR_K(BLOCK_M, BLOCK_N, BLOCK_K) \
    AI_SYSTEM_SGEMM_V1_SPEC(BLOCK_M, BLOCK_N, BLOCK_K, 4, 4), \
    AI_SYSTEM_SGEMM_V1_SPEC(BLOCK_M, BLOCK_N, BLOCK_K, 8, 8)

#define AI_SYSTEM_SGEMM_V1_REG_PAIRS(BLOCK_M, BLOCK_N) \
    AI_SYSTEM_SGEMM_V1_REG_PAIRS_FOR_K(BLOCK_M, BLOCK_N, 8), \
    AI_SYSTEM_SGEMM_V1_REG_PAIRS_FOR_K(BLOCK_M, BLOCK_N, 16), \
    AI_SYSTEM_SGEMM_V1_REG_PAIRS_FOR_K(BLOCK_M, BLOCK_N, 32)

#define AI_SYSTEM_SGEMM_V1_N_VALUES(BLOCK_M) \
    AI_SYSTEM_SGEMM_V1_REG_PAIRS(BLOCK_M, 32), \
    AI_SYSTEM_SGEMM_V1_REG_PAIRS(BLOCK_M, 64), \
    AI_SYSTEM_SGEMM_V1_REG_PAIRS(BLOCK_M, 128)

constexpr SgemmV1Spec kSgemmV1Specs[] = {
    AI_SYSTEM_SGEMM_V1_N_VALUES(32),
    AI_SYSTEM_SGEMM_V1_N_VALUES(64),
    AI_SYSTEM_SGEMM_V1_N_VALUES(128)
};

#undef AI_SYSTEM_SGEMM_V1_N_VALUES
#undef AI_SYSTEM_SGEMM_V1_REG_PAIRS
#undef AI_SYSTEM_SGEMM_V1_REG_PAIRS_FOR_K
#undef AI_SYSTEM_SGEMM_V1_SPEC

bool dispatch_sgemm_v1(
    const float* lhs,
    const float* rhs,
    float* out,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    GemmLabTileConfig tile_config,
    std::string& error
) {
    for(const auto& spec : kSgemmV1Specs) {
        if(spec.block_m == tile_config.block_m && spec.block_n == tile_config.block_n &&
           spec.block_k == tile_config.block_k && spec.register_m == tile_config.register_m &&
           spec.register_n == tile_config.register_n) {
            return spec.launch(lhs, rhs, out, m, n, k, error);
        }
    }

    error = "sgemm_v1 shape is not compiled into the launcher table: block=" +
        std::to_string(tile_config.block_m) + "x" + std::to_string(tile_config.block_n) + "x" +
        std::to_string(tile_config.block_k) + ", register=" + std::to_string(tile_config.register_m) + "x" +
        std::to_string(tile_config.register_n) +
        ". Compiled block_m/block_n values are 32, 64, and 128; block_k values are 8, 16, and 32; " +
        "register tiles are 4x4 and 8x8.";
    return false;
}

}  // namespace

namespace detail {

bool launch_sgemm_v1(
    const float* lhs,
    const float* rhs,
    float* out,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    GemmLabTileConfig tile_config,
    std::string& error
) {
    const ai_system::profiling::ScopedNvtxRange launch_range("sgemm_v1_kernel_launch");

    if(!kSgemmV1KernelImplemented) {
        error = "sgemm_v1 kernel is not implemented yet.";
        return false;
    }

    return dispatch_sgemm_v1(lhs, rhs, out, m, n, k, tile_config, error);
}

bool is_sgemm_v1_kernel_implemented() {
    return kSgemmV1KernelImplemented;
}

}  // namespace detail

}  // namespace ai_system::labs::gemm
