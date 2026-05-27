#include "gemm_lab_kernels.hpp"

#include "ai_system/cuda/runtime.hpp"
#include "ai_system/profiling/nvtx.hpp"

#include <string>

namespace ai_system::labs::gemm {

namespace {

constexpr bool kGemmDbufferVloadKernelImplemented = true;
constexpr int kThreadBlockDimX = 16;
constexpr int kThreadBlockDimY = 16;
constexpr int kThreadsPerBlock = kThreadBlockDimX * kThreadBlockDimY;
constexpr int kFloat4Width = 4;

template <int BlockK>
__device__ __forceinline__ int swizzled_lhs_tile_col(int row, int col) {
    return col ^ (row & (BlockK - 1));
}

template <int BlockM, int BlockK>
__device__ __forceinline__ int lhs_tile_offset(int buffer, int row, int col) {
    return (buffer * BlockM + row) * BlockK + col;
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
    lhs_tiles[lhs_tile_offset<BlockM, BlockK>(
        buffer,
        local_row,
        swizzled_lhs_tile_col<BlockK>(local_row, local_col + 0)
    )] = value.x;
    lhs_tiles[lhs_tile_offset<BlockM, BlockK>(
        buffer,
        local_row,
        swizzled_lhs_tile_col<BlockK>(local_row, local_col + 1)
    )] = value.y;
    lhs_tiles[lhs_tile_offset<BlockM, BlockK>(
        buffer,
        local_row,
        swizzled_lhs_tile_col<BlockK>(local_row, local_col + 2)
    )] = value.z;
    lhs_tiles[lhs_tile_offset<BlockM, BlockK>(
        buffer,
        local_row,
        swizzled_lhs_tile_col<BlockK>(local_row, local_col + 3)
    )] = value.w;
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
    int tid
) {
    static_assert(BlockK % kFloat4Width == 0, "gemm_dbuffer_vload requires block_k to be divisible by 4.");
    static_assert(BlockN % kFloat4Width == 0, "gemm_dbuffer_vload requires block_n to be divisible by 4.");

    constexpr int kLhsVectors = BlockM * BlockK / kFloat4Width;
    for(int idx = tid; idx < kLhsVectors; idx += kThreadsPerBlock) {
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
    for(int idx = tid; idx < kRhsVectors; idx += kThreadsPerBlock) {
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

template <int BlockM, int BlockN, int BlockK, int RegisterM, int RegisterN, int WorkTiles>
__device__ __forceinline__ void load_register_fragments(
    const float* lhs_tiles,
    const float* rhs_tiles,
    float (&lhs_fragment)[WorkTiles][2][RegisterM],
    float (&rhs_fragment)[WorkTiles][2][RegisterN],
    int shared_buffer,
    int register_buffer,
    int inner,
    int tid
) {
    constexpr int kThreadTilesN = BlockN / RegisterN;
    constexpr int kThreadTilesPerCta = (BlockM / RegisterM) * kThreadTilesN;

    #pragma unroll
    for(int work_item = 0; work_item < WorkTiles; ++work_item) {
        const int thread_tile = tid + work_item * kThreadsPerBlock;
        if(thread_tile < kThreadTilesPerCta) {
            const int tile_row = thread_tile / kThreadTilesN;
            const int tile_col = thread_tile % kThreadTilesN;
            const int local_row_base = tile_row * RegisterM;
            const int local_col_base = tile_col * RegisterN;

            #pragma unroll
            for(int row_item = 0; row_item < RegisterM; ++row_item) {
                const int local_row = local_row_base + row_item;
                lhs_fragment[work_item][register_buffer][row_item] =
                    lhs_tiles[lhs_tile_offset<BlockM, BlockK>(
                        shared_buffer,
                        local_row,
                        swizzled_lhs_tile_col<BlockK>(local_row, inner)
                    )];
            }

            #pragma unroll
            for(int col_item = 0; col_item < RegisterN; ++col_item) {
                rhs_fragment[work_item][register_buffer][col_item] =
                    rhs_tiles[rhs_tile_offset<BlockK, BlockN>(shared_buffer, inner, local_col_base + col_item)];
            }
        }
    }
}

template <int BlockM, int BlockN, int RegisterM, int RegisterN, int WorkTiles>
__device__ __forceinline__ void accumulate_register_tile(
    float (&accumulator)[WorkTiles][RegisterM][RegisterN],
    const float (&lhs_fragment)[WorkTiles][2][RegisterM],
    const float (&rhs_fragment)[WorkTiles][2][RegisterN],
    int register_buffer,
    int tid
) {
    constexpr int kThreadTilesN = BlockN / RegisterN;
    constexpr int kThreadTilesPerCta = (BlockM / RegisterM) * kThreadTilesN;

    #pragma unroll
    for(int work_item = 0; work_item < WorkTiles; ++work_item) {
        const int thread_tile = tid + work_item * kThreadsPerBlock;
        if(thread_tile < kThreadTilesPerCta) {
            #pragma unroll
            for(int row_item = 0; row_item < RegisterM; ++row_item) {
                const float lhs_value = lhs_fragment[work_item][register_buffer][row_item];

                #pragma unroll
                for(int col_item = 0; col_item < RegisterN; ++col_item) {
                    accumulator[work_item][row_item][col_item] +=
                        lhs_value * rhs_fragment[work_item][register_buffer][col_item];
                }
            }
        }
    }
}

template <int BlockM, int BlockN, int RegisterM, int RegisterN, int WorkTiles>
__device__ __forceinline__ void store_output_tile(
    const float (&accumulator)[WorkTiles][RegisterM][RegisterN],
    float* __restrict__ out,
    std::size_t m,
    std::size_t n,
    std::size_t block_row,
    std::size_t block_col,
    int tid
) {
    constexpr int kThreadTilesN = BlockN / RegisterN;
    constexpr int kThreadTilesPerCta = (BlockM / RegisterM) * kThreadTilesN;

    #pragma unroll
    for(int work_item = 0; work_item < WorkTiles; ++work_item) {
        const int thread_tile = tid + work_item * kThreadsPerBlock;
        if(thread_tile < kThreadTilesPerCta) {
            const int tile_row = thread_tile / kThreadTilesN;
            const int tile_col = thread_tile % kThreadTilesN;
            const int local_row_base = tile_row * RegisterM;
            const int local_col_base = tile_col * RegisterN;

            #pragma unroll
            for(int row_item = 0; row_item < RegisterM; ++row_item) {
                const std::size_t row = block_row + static_cast<std::size_t>(local_row_base + row_item);

                #pragma unroll
                for(int col_item = 0; col_item < RegisterN; ++col_item) {
                    const std::size_t col = block_col + static_cast<std::size_t>(local_col_base + col_item);
                    if(row < m && col < n) {
                        out[row * n + col] = accumulator[work_item][row_item][col_item];
                    }
                }
            }
        }
    }
}

template <int BlockM, int BlockN, int BlockK, int RegisterM, int RegisterN>
__global__ void __launch_bounds__(kThreadsPerBlock, 1) gemm_dbuffer_vload_kernel(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    float* __restrict__ out,
    std::size_t m,
    std::size_t n,
    std::size_t k
) {
    static_assert(BlockM == 32 || BlockM == 64 || BlockM == 128, "gemm_dbuffer_vload block_m must be 32, 64, or 128.");
    static_assert(BlockN == 32 || BlockN == 64 || BlockN == 128, "gemm_dbuffer_vload block_n must be 32, 64, or 128.");
    static_assert(BlockK == 8 || BlockK == 16 || BlockK == 32, "gemm_dbuffer_vload block_k must be 8, 16, or 32.");
    static_assert(RegisterM == RegisterN, "gemm_dbuffer_vload currently supports square register tiles.");
    static_assert(RegisterM == 4 || RegisterM == 8, "gemm_dbuffer_vload register tile must be 4x4 or 8x8.");
    static_assert(BlockM % RegisterM == 0, "gemm_dbuffer_vload block_m must be divisible by register_m.");
    static_assert(BlockN % RegisterN == 0, "gemm_dbuffer_vload block_n must be divisible by register_n.");
    static_assert((BlockK & (BlockK - 1)) == 0, "gemm_dbuffer_vload block_k must be a power of two.");

    constexpr int kThreadTilesPerCta = (BlockM / RegisterM) * (BlockN / RegisterN);
    constexpr int kWorkTilesPerThread = (kThreadTilesPerCta + kThreadsPerBlock - 1) / kThreadsPerBlock;

    const int tid = static_cast<int>(threadIdx.y) * kThreadBlockDimX + static_cast<int>(threadIdx.x);
    const std::size_t block_row = static_cast<std::size_t>(blockIdx.y) * BlockM;
    const std::size_t block_col = static_cast<std::size_t>(blockIdx.x) * BlockN;

    extern __shared__ float shared_storage[];
    float* lhs_tiles = shared_storage;
    float* rhs_tiles = lhs_tiles + 2 * BlockM * BlockK;

    float accumulator[kWorkTilesPerThread][RegisterM][RegisterN];
    float lhs_fragment[kWorkTilesPerThread][2][RegisterM];
    float rhs_fragment[kWorkTilesPerThread][2][RegisterN];

    #pragma unroll
    for(int work_item = 0; work_item < kWorkTilesPerThread; ++work_item) {
        #pragma unroll
        for(int row_item = 0; row_item < RegisterM; ++row_item) {
            #pragma unroll
            for(int col_item = 0; col_item < RegisterN; ++col_item) {
                accumulator[work_item][row_item][col_item] = 0.0f;
            }
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
        tid
    );
    __syncthreads();

    for(std::size_t tile = 0; tile < num_k_tiles; ++tile) {
        const int shared_buffer = static_cast<int>(tile & 1);
        const int next_shared_buffer = shared_buffer ^ 1;

        if(tile + 1 < num_k_tiles) {
            const std::size_t next_k_base = (tile + 1) * BlockK;
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
                next_k_base,
                next_shared_buffer,
                tid
            );
        }

        load_register_fragments<BlockM, BlockN, BlockK, RegisterM, RegisterN, kWorkTilesPerThread>(
            lhs_tiles,
            rhs_tiles,
            lhs_fragment,
            rhs_fragment,
            shared_buffer,
            0,
            0,
            tid
        );

        #pragma unroll
        for(int inner = 0; inner < BlockK; ++inner) {
            const int register_buffer = inner & 1;
            const int next_register_buffer = register_buffer ^ 1;

            if(inner + 1 < BlockK) {
                load_register_fragments<BlockM, BlockN, BlockK, RegisterM, RegisterN, kWorkTilesPerThread>(
                    lhs_tiles,
                    rhs_tiles,
                    lhs_fragment,
                    rhs_fragment,
                    shared_buffer,
                    next_register_buffer,
                    inner + 1,
                    tid
                );
            }

            accumulate_register_tile<BlockM, BlockN, RegisterM, RegisterN, kWorkTilesPerThread>(
                accumulator,
                lhs_fragment,
                rhs_fragment,
                register_buffer,
                tid
            );
        }

        __syncthreads();
    }

    store_output_tile<BlockM, BlockN, RegisterM, RegisterN, kWorkTilesPerThread>(
        accumulator,
        out,
        m,
        n,
        block_row,
        block_col,
        tid
    );
}

template <int BlockM, int BlockN, int BlockK, int RegisterM, int RegisterN>
bool launch_gemm_dbuffer_vload_instance(
    const float* lhs,
    const float* rhs,
    float* out,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    std::string& error
) {
    const dim3 block(kThreadBlockDimX, kThreadBlockDimY);
    const dim3 grid(
        static_cast<unsigned int>((n + BlockN - 1) / BlockN),
        static_cast<unsigned int>((m + BlockM - 1) / BlockM)
    );
    constexpr int kSharedStorageFloats = 2 * (BlockM * BlockK + BlockK * BlockN);
    constexpr std::size_t kSharedStorageBytes = static_cast<std::size_t>(kSharedStorageFloats) * sizeof(float);

    static bool attributes_configured = false;
    if(!attributes_configured) {
        const auto max_dynamic_shared_status = cudaFuncSetAttribute(
            gemm_dbuffer_vload_kernel<BlockM, BlockN, BlockK, RegisterM, RegisterN>,
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

        const auto carveout_status = cudaFuncSetAttribute(
            gemm_dbuffer_vload_kernel<BlockM, BlockN, BlockK, RegisterM, RegisterN>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared
        );
        if(!ai_system::cuda_utils::check_status(
               carveout_status,
               "cudaFuncSetAttribute(PreferredSharedMemoryCarveout)",
               error
           )) {
            return false;
        }

        attributes_configured = true;
    }

    gemm_dbuffer_vload_kernel<BlockM, BlockN, BlockK, RegisterM, RegisterN>
        <<<grid, block, kSharedStorageBytes>>>(lhs, rhs, out, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

using GemmDbufferVloadLaunchFn = bool (*)(
    const float* lhs,
    const float* rhs,
    float* out,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    std::string& error
);

struct GemmDbufferVloadSpec {
    int block_m;
    int block_n;
    int block_k;
    int register_m;
    int register_n;
    GemmDbufferVloadLaunchFn launch;
};

#define AI_SYSTEM_GEMM_DBUFFER_VLOAD_SPEC(BLOCK_M, BLOCK_N, BLOCK_K, REGISTER_M, REGISTER_N) \
    { \
        BLOCK_M, \
        BLOCK_N, \
        BLOCK_K, \
        REGISTER_M, \
        REGISTER_N, \
        &launch_gemm_dbuffer_vload_instance<BLOCK_M, BLOCK_N, BLOCK_K, REGISTER_M, REGISTER_N> \
    }

#define AI_SYSTEM_GEMM_DBUFFER_VLOAD_REG_PAIRS_FOR_K(BLOCK_M, BLOCK_N, BLOCK_K) \
    AI_SYSTEM_GEMM_DBUFFER_VLOAD_SPEC(BLOCK_M, BLOCK_N, BLOCK_K, 4, 4), \
    AI_SYSTEM_GEMM_DBUFFER_VLOAD_SPEC(BLOCK_M, BLOCK_N, BLOCK_K, 8, 8)

#define AI_SYSTEM_GEMM_DBUFFER_VLOAD_REG_PAIRS(BLOCK_M, BLOCK_N) \
    AI_SYSTEM_GEMM_DBUFFER_VLOAD_REG_PAIRS_FOR_K(BLOCK_M, BLOCK_N, 8), \
    AI_SYSTEM_GEMM_DBUFFER_VLOAD_REG_PAIRS_FOR_K(BLOCK_M, BLOCK_N, 16), \
    AI_SYSTEM_GEMM_DBUFFER_VLOAD_REG_PAIRS_FOR_K(BLOCK_M, BLOCK_N, 32)

#define AI_SYSTEM_GEMM_DBUFFER_VLOAD_N_VALUES(BLOCK_M) \
    AI_SYSTEM_GEMM_DBUFFER_VLOAD_REG_PAIRS(BLOCK_M, 32), \
    AI_SYSTEM_GEMM_DBUFFER_VLOAD_REG_PAIRS(BLOCK_M, 64), \
    AI_SYSTEM_GEMM_DBUFFER_VLOAD_REG_PAIRS(BLOCK_M, 128)

constexpr GemmDbufferVloadSpec kGemmDbufferVloadSpecs[] = {
    AI_SYSTEM_GEMM_DBUFFER_VLOAD_N_VALUES(32),
    AI_SYSTEM_GEMM_DBUFFER_VLOAD_N_VALUES(64),
    AI_SYSTEM_GEMM_DBUFFER_VLOAD_N_VALUES(128)
};

#undef AI_SYSTEM_GEMM_DBUFFER_VLOAD_N_VALUES
#undef AI_SYSTEM_GEMM_DBUFFER_VLOAD_REG_PAIRS
#undef AI_SYSTEM_GEMM_DBUFFER_VLOAD_REG_PAIRS_FOR_K
#undef AI_SYSTEM_GEMM_DBUFFER_VLOAD_SPEC

bool dispatch_gemm_dbuffer_vload(
    const float* lhs,
    const float* rhs,
    float* out,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    GemmLabTileConfig tile_config,
    std::string& error
) {
    for(const auto& spec : kGemmDbufferVloadSpecs) {
        if(spec.block_m == tile_config.block_m && spec.block_n == tile_config.block_n &&
           spec.block_k == tile_config.block_k && spec.register_m == tile_config.register_m &&
           spec.register_n == tile_config.register_n) {
            return spec.launch(lhs, rhs, out, m, n, k, error);
        }
    }

    error = "gemm_dbuffer_vload shape is not compiled into the launcher table: block=" +
        std::to_string(tile_config.block_m) + "x" + std::to_string(tile_config.block_n) + "x" +
        std::to_string(tile_config.block_k) + ", register=" + std::to_string(tile_config.register_m) + "x" +
        std::to_string(tile_config.register_n) +
        ". Compiled block_m/block_n values are 32, 64, and 128; block_k values are 8, 16, and 32; " +
        "register tiles are 4x4 and 8x8.";
    return false;
}

}  // namespace

namespace detail {

bool launch_gemm_dbuffer_vload(
    const float* lhs,
    const float* rhs,
    float* out,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    GemmLabTileConfig tile_config,
    std::string& error
) {
    const ai_system::profiling::ScopedNvtxRange launch_range("gemm_dbuffer_vload_kernel_launch");

    if(!kGemmDbufferVloadKernelImplemented) {
        error = "gemm_dbuffer_vload kernel is not implemented yet.";
        return false;
    }

    return dispatch_gemm_dbuffer_vload(lhs, rhs, out, m, n, k, tile_config, error);
}

bool is_gemm_dbuffer_vload_kernel_implemented() {
    return kGemmDbufferVloadKernelImplemented;
}

}  // namespace detail

}  // namespace ai_system::labs::gemm
