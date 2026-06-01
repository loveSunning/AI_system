#include "gemm_lab_kernels.hpp"

#include "ai_system/cuda/runtime.hpp"
#include "ai_system/profiling/nvtx.hpp"

#include <string>

namespace ai_system::labs::gemm {

namespace {

constexpr bool kGemmWrapTileKernelImplemented = true;
constexpr int kMaxGemmWrapTileThreadsPerBlock = 32 * 32;
constexpr int kFloat4Width = 4;

struct ThreadTileCoord {
    int local_row_base;
    int local_col_base;
};

template <int BlockM, int BlockN, int RegisterM, int RegisterN>
struct GemmWrapTileLayout {
    static constexpr int kThreadTileRows = BlockM / RegisterM;
    static constexpr int kThreadTileCols = BlockN / RegisterN;
    static constexpr int kLogicalThreads = kThreadTileRows * kThreadTileCols;

    static constexpr bool kUsesWarpQuadrants =
        RegisterM == 8 && RegisterN == 8 && BlockM % 32 == 0 && BlockN % 64 == 0;
    static constexpr int kWarpsPerCta = (kLogicalThreads + 31) / 32;
    static constexpr int kLaunchedThreads = kWarpsPerCta * 32;
    static constexpr int kBlockDimX = 32;
    static constexpr int kBlockDimY = kWarpsPerCta;
    static constexpr bool kAllThreadsCompute = kLogicalThreads == kLaunchedThreads;
};

template <int BlockM, int BlockN, int RegisterM, int RegisterN>
constexpr int gemm_wrap_tile_threads_per_cta() {
    return GemmWrapTileLayout<BlockM, BlockN, RegisterM, RegisterN>::kLaunchedThreads;
}

template <int BlockM, int BlockN, int RegisterM, int RegisterN>
__device__ __forceinline__ ThreadTileCoord compute_thread_tile_coord(int tid) {
    using Layout = GemmWrapTileLayout<BlockM, BlockN, RegisterM, RegisterN>;

    int thread_tile_row = 0;
    int thread_tile_col = 0;
    if constexpr(Layout::kUsesWarpQuadrants) {
        // The v3-style quadrant path does not use a single contiguous thread
        // tile coordinate. Register fragments and stores remap row/column
        // coordinates directly from warp_id/lane_id.
    } else {
        constexpr bool kUse4x8WarpTile = Layout::kThreadTileRows % 4 == 0 && Layout::kThreadTileCols % 8 == 0;
        constexpr bool kUse8x4WarpTile =
            !kUse4x8WarpTile && Layout::kThreadTileRows % 8 == 0 && Layout::kThreadTileCols % 4 == 0;
        constexpr bool kUse2x16WarpTile =
            !kUse4x8WarpTile && !kUse8x4WarpTile && Layout::kThreadTileRows % 2 == 0 &&
            Layout::kThreadTileCols % 16 == 0;
        constexpr bool kUse16x2WarpTile =
            !kUse4x8WarpTile && !kUse8x4WarpTile && !kUse2x16WarpTile &&
            Layout::kThreadTileRows % 16 == 0 && Layout::kThreadTileCols % 2 == 0;
        constexpr bool kUsesWarpTiles =
            Layout::kLogicalThreads >= 32 &&
            (kUse4x8WarpTile || kUse8x4WarpTile || kUse2x16WarpTile || kUse16x2WarpTile);

        if constexpr(kUsesWarpTiles) {
            constexpr int kWarpThreadRows =
                kUse4x8WarpTile ? 4 : (kUse8x4WarpTile ? 8 : (kUse2x16WarpTile ? 2 : 16));
            constexpr int kWarpThreadCols = 32 / kWarpThreadRows;
            constexpr int kWarpsPerRow = Layout::kThreadTileCols / kWarpThreadCols;

            const int warp_id = tid >> 5;
            const int lane_id = tid & 31;
            const int warp_row = warp_id / kWarpsPerRow;
            const int warp_col = warp_id - warp_row * kWarpsPerRow;
            const int lane_row = lane_id / kWarpThreadCols;
            const int lane_col = lane_id - lane_row * kWarpThreadCols;

            thread_tile_row = warp_row * kWarpThreadRows + lane_row;
            thread_tile_col = warp_col * kWarpThreadCols + lane_col;
        } else {
            thread_tile_row = tid / Layout::kThreadTileCols;
            thread_tile_col = tid - thread_tile_row * Layout::kThreadTileCols;
        }
    }

    return ThreadTileCoord {thread_tile_row * RegisterM, thread_tile_col * RegisterN};
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
    lhs_tiles[lhs_tile_offset<BlockM, BlockK>(
        buffer,
        local_row,
        local_col + 0
    )] = value.x;
    lhs_tiles[lhs_tile_offset<BlockM, BlockK>(
        buffer,
        local_row,
        local_col + 1
    )] = value.y;
    lhs_tiles[lhs_tile_offset<BlockM, BlockK>(
        buffer,
        local_row,
        local_col + 2
    )] = value.z;
    lhs_tiles[lhs_tile_offset<BlockM, BlockK>(
        buffer,
        local_row,
        local_col + 3
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
    int tid,
    int num_threads
) {
    static_assert(BlockK % kFloat4Width == 0, "gemm_wrap_tile requires block_k to be divisible by 4.");
    static_assert(BlockN % kFloat4Width == 0, "gemm_wrap_tile requires block_n to be divisible by 4.");

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
    int local_col_base,
    int tid
) {
    using Layout = GemmWrapTileLayout<BlockM, BlockN, RegisterM, RegisterN>;

    if constexpr(Layout::kUsesWarpQuadrants) {
        constexpr int kWarpsPerNGroup = BlockN / 64;
        constexpr int kHalfBlockM = BlockM / 2;
        constexpr int kHalfBlockN = BlockN / 2;

        const int warp_id = tid >> 5;
        const int lane_id = tid & 31;
        const int warp_m = warp_id / kWarpsPerNGroup;
        const int warp_n = warp_id - warp_m * kWarpsPerNGroup;
        const int a_tile_index = warp_m * 16 + (lane_id / 8) * 4;
        const int b_tile_index = warp_n * 32 + (lane_id & 7) * 4;

        #pragma unroll
        for(int row_item = 0; row_item < 4; ++row_item) {
            const int top_row = a_tile_index + row_item;
            const int bottom_row = a_tile_index + kHalfBlockM + row_item;
            lhs_fragment[register_buffer][row_item] =
                lhs_tiles[lhs_tile_offset<BlockM, BlockK>(
                    shared_buffer,
                    top_row,
                    inner
                )];
            lhs_fragment[register_buffer][row_item + 4] =
                lhs_tiles[lhs_tile_offset<BlockM, BlockK>(
                    shared_buffer,
                    bottom_row,
                    inner
                )];
        }

        #pragma unroll
        for(int col_item = 0; col_item < 4; ++col_item) {
            rhs_fragment[register_buffer][col_item] =
                rhs_tiles[rhs_tile_offset<BlockK, BlockN>(shared_buffer, inner, b_tile_index + col_item)];
            rhs_fragment[register_buffer][col_item + 4] =
                rhs_tiles[rhs_tile_offset<BlockK, BlockN>(
                    shared_buffer,
                    inner,
                    b_tile_index + kHalfBlockN + col_item
                )];
        }
    } else {
        #pragma unroll
        for(int row_item = 0; row_item < RegisterM; ++row_item) {
            const int local_row = local_row_base + row_item;
            lhs_fragment[register_buffer][row_item] =
                lhs_tiles[lhs_tile_offset<BlockM, BlockK>(
                    shared_buffer,
                    local_row,
                    inner
                )];
        }

        #pragma unroll
        for(int col_item = 0; col_item < RegisterN; ++col_item) {
            rhs_fragment[register_buffer][col_item] =
                rhs_tiles[rhs_tile_offset<BlockK, BlockN>(shared_buffer, inner, local_col_base + col_item)];
        }
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

template <int BlockM, int BlockN, int RegisterM, int RegisterN>
__device__ __forceinline__ void store_output_tile(
    const float (&accumulator)[RegisterM][RegisterN],
    float* __restrict__ out,
    std::size_t m,
    std::size_t n,
    std::size_t block_row,
    std::size_t block_col,
    int local_row_base,
    int local_col_base,
    int tid
) {
    using Layout = GemmWrapTileLayout<BlockM, BlockN, RegisterM, RegisterN>;

    if constexpr(Layout::kUsesWarpQuadrants) {
        constexpr int kWarpsPerNGroup = BlockN / 64;
        constexpr int kHalfBlockM = BlockM / 2;
        constexpr int kHalfBlockN = BlockN / 2;

        const int warp_id = tid >> 5;
        const int lane_id = tid & 31;
        const int warp_m = warp_id / kWarpsPerNGroup;
        const int warp_n = warp_id - warp_m * kWarpsPerNGroup;
        const int c_row_base = warp_m * 16 + (lane_id / 8) * 4;
        const int c_col_base = warp_n * 32 + (lane_id & 7) * 4;

        #pragma unroll
        for(int row_item = 0; row_item < 8; ++row_item) {
            const int local_row = row_item < 4 ? c_row_base + row_item : c_row_base + kHalfBlockM + row_item - 4;
            const std::size_t row = block_row + static_cast<std::size_t>(local_row);

            #pragma unroll
            for(int col_item = 0; col_item < 8; ++col_item) {
                const int local_col = col_item < 4 ? c_col_base + col_item : c_col_base + kHalfBlockN + col_item - 4;
                const std::size_t col = block_col + static_cast<std::size_t>(local_col);
                if(row < m && col < n) {
                    out[row * n + col] = accumulator[row_item][col_item];
                }
            }
        }
    } else {
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
}

template <int BlockM, int BlockN, int BlockK, int RegisterM, int RegisterN>
__global__ void __launch_bounds__(gemm_wrap_tile_threads_per_cta<BlockM, BlockN, RegisterM, RegisterN>(), 1)
    gemm_wrap_tile_kernel(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    float* __restrict__ out,
    std::size_t m,
    std::size_t n,
    std::size_t k
) {
    static_assert(BlockM == 32 || BlockM == 64 || BlockM == 128, "gemm_wrap_tile block_m must be 32, 64, or 128.");
    static_assert(BlockN == 32 || BlockN == 64 || BlockN == 128, "gemm_wrap_tile block_n must be 32, 64, or 128.");
    static_assert(BlockK == 8 || BlockK == 16 || BlockK == 32, "gemm_wrap_tile block_k must be 8, 16, or 32.");
    static_assert(RegisterM == RegisterN, "gemm_wrap_tile currently supports square register tiles.");
    static_assert(RegisterM == 4 || RegisterM == 8, "gemm_wrap_tile register tile must be 4x4 or 8x8.");
    static_assert(BlockM % RegisterM == 0, "gemm_wrap_tile block_m must be divisible by register_m.");
    static_assert(BlockN % RegisterN == 0, "gemm_wrap_tile block_n must be divisible by register_n.");
    static_assert((BlockK & (BlockK - 1)) == 0, "gemm_wrap_tile block_k must be a power of two.");

    using Layout = GemmWrapTileLayout<BlockM, BlockN, RegisterM, RegisterN>;
    static_assert(
        Layout::kLaunchedThreads <= kMaxGemmWrapTileThreadsPerBlock,
        "gemm_wrap_tile derived thread block size must fit within 32 * 32 threads."
    );

    const int tid = static_cast<int>(threadIdx.y) * Layout::kBlockDimX + static_cast<int>(threadIdx.x);
    [[maybe_unused]] const bool computes_output = tid < Layout::kLogicalThreads;
    const ThreadTileCoord thread_tile =
        compute_thread_tile_coord<BlockM, BlockN, RegisterM, RegisterN>(tid);

    const std::size_t block_row = static_cast<std::size_t>(blockIdx.y) * BlockM;
    const std::size_t block_col = static_cast<std::size_t>(blockIdx.x) * BlockN;

    extern __shared__ float shared_storage[];
    float* lhs_tiles = shared_storage;
    float* rhs_tiles = lhs_tiles + 2 * BlockM * BlockK;

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
        Layout::kLaunchedThreads
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
                Layout::kLaunchedThreads
            );
        }

        if constexpr(Layout::kAllThreadsCompute) {
            load_register_fragments<BlockM, BlockN, BlockK, RegisterM, RegisterN>(
                lhs_tiles,
                rhs_tiles,
                lhs_fragment,
                rhs_fragment,
                shared_buffer,
                0,
                0,
                thread_tile.local_row_base,
                thread_tile.local_col_base,
                tid
            );
        } else if(computes_output) {
            load_register_fragments<BlockM, BlockN, BlockK, RegisterM, RegisterN>(
                lhs_tiles,
                rhs_tiles,
                lhs_fragment,
                rhs_fragment,
                shared_buffer,
                0,
                0,
                thread_tile.local_row_base,
                thread_tile.local_col_base,
                tid
            );
        }

        #pragma unroll
        for(int inner = 0; inner < BlockK; ++inner) {
            if constexpr(Layout::kAllThreadsCompute) {
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
                        thread_tile.local_row_base,
                        thread_tile.local_col_base,
                        tid
                    );
                }

                accumulate_register_tile<RegisterM, RegisterN>(
                    accumulator,
                    lhs_fragment,
                    rhs_fragment,
                    register_buffer
                );
            } else if(computes_output) {
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
                        thread_tile.local_row_base,
                        thread_tile.local_col_base,
                        tid
                    );
                }

                accumulate_register_tile<RegisterM, RegisterN>(
                    accumulator,
                    lhs_fragment,
                    rhs_fragment,
                    register_buffer
                );
            }
        }

        __syncthreads();
    }

    if constexpr(Layout::kAllThreadsCompute) {
        store_output_tile<BlockM, BlockN, RegisterM, RegisterN>(
            accumulator,
            out,
            m,
            n,
            block_row,
            block_col,
            thread_tile.local_row_base,
            thread_tile.local_col_base,
            tid
        );
    } else if(computes_output) {
        store_output_tile<BlockM, BlockN, RegisterM, RegisterN>(
            accumulator,
            out,
            m,
            n,
            block_row,
            block_col,
            thread_tile.local_row_base,
            thread_tile.local_col_base,
            tid
        );
    }
}

template <int BlockM, int BlockN, int BlockK, int RegisterM, int RegisterN>
bool launch_gemm_wrap_tile_instance(
    const float* lhs,
    const float* rhs,
    float* out,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    std::string& error
) {
    using Layout = GemmWrapTileLayout<BlockM, BlockN, RegisterM, RegisterN>;
    const dim3 block(Layout::kBlockDimX, Layout::kBlockDimY);
    const dim3 grid(
        static_cast<unsigned int>((n + BlockN - 1) / BlockN),
        static_cast<unsigned int>((m + BlockM - 1) / BlockM)
    );
    constexpr int kSharedStorageFloats = 2 * (BlockM * BlockK + BlockK * BlockN);
    constexpr std::size_t kSharedStorageBytes = static_cast<std::size_t>(kSharedStorageFloats) * sizeof(float);

    static bool attributes_configured = false;
    if(!attributes_configured) {
        const auto max_dynamic_shared_status = cudaFuncSetAttribute(
            gemm_wrap_tile_kernel<BlockM, BlockN, BlockK, RegisterM, RegisterN>,
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
            gemm_wrap_tile_kernel<BlockM, BlockN, BlockK, RegisterM, RegisterN>,
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

    gemm_wrap_tile_kernel<BlockM, BlockN, BlockK, RegisterM, RegisterN>
        <<<grid, block, kSharedStorageBytes>>>(lhs, rhs, out, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

using GemmWrapTileLaunchFn = bool (*)(
    const float* lhs,
    const float* rhs,
    float* out,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    std::string& error
);

struct GemmWrapTileSpec {
    int block_m;
    int block_n;
    int block_k;
    int register_m;
    int register_n;
    GemmWrapTileLaunchFn launch;
};

#define AI_SYSTEM_GEMM_WRAP_TILE_SPEC(BLOCK_M, BLOCK_N, BLOCK_K, REGISTER_M, REGISTER_N) \
    { \
        BLOCK_M, \
        BLOCK_N, \
        BLOCK_K, \
        REGISTER_M, \
        REGISTER_N, \
        &launch_gemm_wrap_tile_instance<BLOCK_M, BLOCK_N, BLOCK_K, REGISTER_M, REGISTER_N> \
    }

#define AI_SYSTEM_GEMM_WRAP_TILE_REG_PAIRS_FOR_K(BLOCK_M, BLOCK_N, BLOCK_K) \
    AI_SYSTEM_GEMM_WRAP_TILE_SPEC(BLOCK_M, BLOCK_N, BLOCK_K, 4, 4), \
    AI_SYSTEM_GEMM_WRAP_TILE_SPEC(BLOCK_M, BLOCK_N, BLOCK_K, 8, 8)

#define AI_SYSTEM_GEMM_WRAP_TILE_REG_PAIRS(BLOCK_M, BLOCK_N) \
    AI_SYSTEM_GEMM_WRAP_TILE_REG_PAIRS_FOR_K(BLOCK_M, BLOCK_N, 8), \
    AI_SYSTEM_GEMM_WRAP_TILE_REG_PAIRS_FOR_K(BLOCK_M, BLOCK_N, 16), \
    AI_SYSTEM_GEMM_WRAP_TILE_REG_PAIRS_FOR_K(BLOCK_M, BLOCK_N, 32)

#define AI_SYSTEM_GEMM_WRAP_TILE_N_VALUES(BLOCK_M) \
    AI_SYSTEM_GEMM_WRAP_TILE_REG_PAIRS(BLOCK_M, 32), \
    AI_SYSTEM_GEMM_WRAP_TILE_REG_PAIRS(BLOCK_M, 64), \
    AI_SYSTEM_GEMM_WRAP_TILE_REG_PAIRS(BLOCK_M, 128)

constexpr GemmWrapTileSpec kGemmWrapTileSpecs[] = {
    AI_SYSTEM_GEMM_WRAP_TILE_N_VALUES(32),
    AI_SYSTEM_GEMM_WRAP_TILE_N_VALUES(64),
    AI_SYSTEM_GEMM_WRAP_TILE_N_VALUES(128)
};

#undef AI_SYSTEM_GEMM_WRAP_TILE_N_VALUES
#undef AI_SYSTEM_GEMM_WRAP_TILE_REG_PAIRS
#undef AI_SYSTEM_GEMM_WRAP_TILE_REG_PAIRS_FOR_K
#undef AI_SYSTEM_GEMM_WRAP_TILE_SPEC

bool dispatch_gemm_wrap_tile(
    const float* lhs,
    const float* rhs,
    float* out,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    GemmLabTileConfig tile_config,
    std::string& error
) {
    for(const auto& spec : kGemmWrapTileSpecs) {
        if(spec.block_m == tile_config.block_m && spec.block_n == tile_config.block_n &&
           spec.block_k == tile_config.block_k && spec.register_m == tile_config.register_m &&
           spec.register_n == tile_config.register_n) {
            return spec.launch(lhs, rhs, out, m, n, k, error);
        }
    }

    error = "gemm_wrap_tile shape is not compiled into the launcher table: block=" +
        std::to_string(tile_config.block_m) + "x" + std::to_string(tile_config.block_n) + "x" +
        std::to_string(tile_config.block_k) + ", register=" + std::to_string(tile_config.register_m) + "x" +
        std::to_string(tile_config.register_n) +
        ". Compiled block_m/block_n values are 32, 64, and 128; block_k values are 8, 16, and 32; " +
        "register tiles are 4x4 and 8x8.";
    return false;
}

}  // namespace

namespace detail {

bool launch_gemm_wrap_tile(
    const float* lhs,
    const float* rhs,
    float* out,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    GemmLabTileConfig tile_config,
    std::string& error
) {
    const ai_system::profiling::ScopedNvtxRange launch_range("gemm_wrap_tile_kernel_launch");

    if(!kGemmWrapTileKernelImplemented) {
        error = "gemm_wrap_tile kernel is not implemented yet.";
        return false;
    }

    return dispatch_gemm_wrap_tile(lhs, rhs, out, m, n, k, tile_config, error);
}

bool is_gemm_wrap_tile_kernel_implemented() {
    return kGemmWrapTileKernelImplemented;
}

}  // namespace detail

}  // namespace ai_system::labs::gemm
