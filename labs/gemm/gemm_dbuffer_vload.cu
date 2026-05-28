#include "gemm_lab_kernels.hpp"

#include "ai_system/cuda/runtime.hpp"
#include "ai_system/profiling/nvtx.hpp"

#include <string>

namespace ai_system::labs::gemm {

namespace {

// This file implements a CUDA-core SGEMM kernel with three main ideas:
//
// 1. CTA-level tiling:
//    Each CTA computes one BlockM x BlockN tile of C.
//
// 2. Shared-memory double buffering:
//    Two shared-memory stages are allocated for A and B. While the CTA computes
//    from one stage, it preloads the next K tile into the other stage.
//
// 3. Register-fragment double buffering:
//    Each thread owns one RegisterM x RegisterN output tile. For each inner K
//    step, the thread preloads the next A/B fragments into one register buffer
//    while computing from the other buffer.
//
// The kernel intentionally stays on the CUDA-core FFMA path. It is useful for
// studying global coalescing, shared-memory layout, bank behavior, register
// tiling, and launch-bound/register-pressure tradeoffs before moving to Tensor
// Core GEMM.
constexpr bool kGemmDbufferVloadKernelImplemented = true;
constexpr int kMaxGemmDbufferVloadThreadsPerBlock = 32 * 32;
constexpr int kFloat4Width = 4;

// The logical thread block is derived from the CTA tile and the per-thread
// output tile. For example, BlockM=128, BlockN=128, RegisterM=8, RegisterN=8
// maps to a 16 x 16 CUDA block, or 256 threads per CTA.
template <int BlockM, int BlockN, int RegisterM, int RegisterN>
constexpr int gemm_dbuffer_vload_threads_per_cta() {
    return (BlockM / RegisterM) * (BlockN / RegisterN);
}

// A-side shared-memory swizzle.
//
// The A tile is logically stored as [stage][m][k]. Threads in a warp often
// access the same inner-k column for different rows when loading A fragments.
// XORing the column with row low bits changes the physical bank mapping and can
// reduce repeated bank conflicts for those column-wise shared-memory reads.
template <int BlockK>
__device__ __forceinline__ int swizzled_lhs_tile_col(int row, int col) {
    return col ^ (row & (BlockK - 1));
}

// Offset into the A shared-memory tile:
//
//   lhs_tiles[buffer][row][col]
//
// where each buffer contains a BlockM x BlockK tile.
template <int BlockM, int BlockK>
__device__ __forceinline__ int lhs_tile_offset(int buffer, int row, int col) {
    return (buffer * BlockM + row) * BlockK + col;
}

// Offset into the B shared-memory tile:
//
//   rhs_tiles[buffer][row][col]
//
// where each buffer contains a BlockK x BlockN tile.
template <int BlockK, int BlockN>
__device__ __forceinline__ int rhs_tile_offset(int buffer, int row, int col) {
    return (buffer * BlockK + row) * BlockN + col;
}

__device__ __forceinline__ float4 zero_float4() {
    return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

// A float4 load is only legal when both conditions hold:
// - the requested four elements stay inside the row extent
// - the element offset is 16-byte aligned, because float4 has four floats
//
// Boundary tiles and non-4-aligned leading dimensions fall back to scalar loads
// and explicitly zero-fill out-of-bounds lanes.
__device__ __forceinline__ bool can_load_float4(std::size_t element_offset, std::size_t vector_col, std::size_t extent) {
    return vector_col + (kFloat4Width - 1) < extent && (element_offset & (kFloat4Width - 1)) == 0;
}

// Store one vectorized A load into shared memory. A is stored in row-major
// logical order, but each K coordinate is swizzled before computing the physical
// shared-memory address.
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

// Store one vectorized B load into shared memory. B keeps the straightforward
// [stage][k][n] layout because each inner step reads contiguous N coordinates.
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

// Load the current CTA's A and B tiles from global memory into the requested
// shared-memory stage.
//
// A global tile shape:
//   lhs[block_row : block_row + BlockM, k_base : k_base + BlockK]
//
// B global tile shape:
//   rhs[k_base : k_base + BlockK, block_col : block_col + BlockN]
//
// Work distribution:
//   The tile is viewed as a list of float4 vectors. Each thread takes vector
//   indices tid, tid + num_threads, tid + 2*num_threads, ...
//
// Boundary behavior:
//   This kernel supports arbitrary M/N/K. Full aligned vectors use float4
//   loads, and boundary/non-aligned vectors use scalar loads with zero padding.
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
    static_assert(BlockK % kFloat4Width == 0, "gemm_dbuffer_vload requires block_k to be divisible by 4.");
    static_assert(BlockN % kFloat4Width == 0, "gemm_dbuffer_vload requires block_n to be divisible by 4.");

    constexpr int kLhsVectors = BlockM * BlockK / kFloat4Width;
    for(int idx = tid; idx < kLhsVectors; idx += num_threads) {
        // Map a linear float4 vector index back to a row and a four-column
        // segment inside the local A tile.
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
        // B is also loaded as float4 vectors, but its contiguous dimension is N.
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

// Load one thread's A and B fragments from shared memory into one of the two
// register fragment buffers.
//
// For a fixed inner K index:
//   A fragment = RegisterM values from this thread's output rows
//   B fragment = RegisterN values from this thread's output columns
//
// These fragments are exactly the operands needed to update the thread-owned
// RegisterM x RegisterN accumulator tile for one K step.
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
        const int local_row = local_row_base + row_item;
        // The same swizzle used when storing A must be used when reading it
        // back from shared memory.
        lhs_fragment[register_buffer][row_item] =
            lhs_tiles[lhs_tile_offset<BlockM, BlockK>(
                shared_buffer,
                local_row,
                swizzled_lhs_tile_col<BlockK>(local_row, inner)
            )];
    }

    #pragma unroll
    for(int col_item = 0; col_item < RegisterN; ++col_item) {
        rhs_fragment[register_buffer][col_item] =
            rhs_tiles[rhs_tile_offset<BlockK, BlockN>(shared_buffer, inner, local_col_base + col_item)];
    }
}

// Update the thread-local C register tile with one outer product:
//
//   accumulator[row][col] += A_fragment[row] * B_fragment[col]
//
// The compiler fully unrolls these tiny loops for the supported 4x4 and 8x8
// register tiles.
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

// Store the final per-thread register tile back to C. Boundary checks are kept
// here so the main compute loop can operate on zero-padded shared-memory tiles
// without caring whether the CTA is on the matrix edge.
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

// Main SGEMM kernel:
//
//   C[M, N] = A[M, K] * B[K, N]
//
// Each CTA computes one BlockM x BlockN tile of C. Each thread computes one
// RegisterM x RegisterN sub-tile inside that CTA tile. The K dimension is
// processed in BlockK chunks.
//
// Template parameters are compile-time constants so the compiler can unroll the
// inner loops and size shared/register arrays statically.
template <int BlockM, int BlockN, int BlockK, int RegisterM, int RegisterN>
__global__ void __launch_bounds__(gemm_dbuffer_vload_threads_per_cta<BlockM, BlockN, RegisterM, RegisterN>(), 1)
    gemm_dbuffer_vload_kernel(
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

    constexpr int kThreadBlockDimX = BlockN / RegisterN;
    constexpr int kThreadsPerCta = gemm_dbuffer_vload_threads_per_cta<BlockM, BlockN, RegisterM, RegisterN>();
    // Keep the launch bound tied to the real template instance. A too-large
    // launch bound can force the compiler to cap registers/thread and spill
    // accumulator fragments to local memory.
    static_assert(
        kThreadsPerCta <= kMaxGemmDbufferVloadThreadsPerBlock,
        "gemm_dbuffer_vload derived thread block size must fit within 32 * 32 threads."
    );

    // Linear thread id is used for cooperative global-to-shared loading.
    const int tid = static_cast<int>(threadIdx.y) * kThreadBlockDimX + static_cast<int>(threadIdx.x);

    // Each thread owns the C tile starting at this local row/column inside the
    // CTA output tile.
    const int local_row_base = static_cast<int>(threadIdx.y) * RegisterM;
    const int local_col_base = static_cast<int>(threadIdx.x) * RegisterN;
    const std::size_t block_row = static_cast<std::size_t>(blockIdx.y) * BlockM;
    const std::size_t block_col = static_cast<std::size_t>(blockIdx.x) * BlockN;

    // Dynamic shared memory layout:
    //
    //   lhs_tiles: 2 stages * BlockM * BlockK floats
    //   rhs_tiles: 2 stages * BlockK * BlockN floats
    //
    // The two stages implement shared-memory double buffering.
    extern __shared__ float shared_storage[];
    float* lhs_tiles = shared_storage;
    float* rhs_tiles = lhs_tiles + 2 * BlockM * BlockK;

    // Thread-local state:
    //
    //   accumulator: the final RegisterM x RegisterN C sub-tile
    //   lhs_fragment/rhs_fragment: two register buffers used to pipeline
    //   shared-memory loads with FFMA work across inner K steps
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

    // Prime shared-memory stage 0 before entering the main K-tile loop.
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
        // Read from the current shared stage and write the next global tile
        // into the opposite stage.
        const int shared_buffer = static_cast<int>(tile & 1);
        const int next_shared_buffer = shared_buffer ^ 1;

        // Preload the next K tile into shared memory. This is logically the
        // shared-memory double-buffer step. The following compute section still
        // consumes shared_buffer.
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
                tid,
                kThreadsPerCta
            );
        }

        // Prime register-fragment buffer 0 for inner K step 0.
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
            // Ping-pong register fragments on alternating inner K steps:
            // compute from register_buffer while preloading inner+1 into the
            // other register buffer when it exists.
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

            accumulate_register_tile<RegisterM, RegisterN>(
                accumulator,
                lhs_fragment,
                rhs_fragment,
                register_buffer
            );
        }

        // At this point every thread has consumed the current shared stage.
        // Synchronize before the next loop iteration can read from the stage
        // that may have just been overwritten by the preload above.
        __syncthreads();
    }

    // Write the completed C sub-tile owned by this thread.
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

// Launch one compiled template instance. The public runtime configuration is
// still dynamic, but dispatch selects a matching compile-time specialization so
// the kernel keeps static loop bounds and static shared-memory sizes.
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
    const dim3 block(BlockN / RegisterN, BlockM / RegisterM);
    const dim3 grid(
        static_cast<unsigned int>((n + BlockN - 1) / BlockN),
        static_cast<unsigned int>((m + BlockM - 1) / BlockM)
    );
    constexpr int kSharedStorageFloats = 2 * (BlockM * BlockK + BlockK * BlockN);
    constexpr std::size_t kSharedStorageBytes = static_cast<std::size_t>(kSharedStorageFloats) * sizeof(float);

    // Dynamic shared memory can exceed the default per-block limit for some
    // shapes, so configure the opt-in size once per template specialization.
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

        // Prefer a larger shared-memory carveout because this kernel's hot data
        // lives in shared memory rather than relying heavily on L1 caching.
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

// The launcher table explicitly instantiates all supported tile shapes:
//
//   BlockM/BlockN: 32, 64, 128
//   BlockK:        8, 16, 32
//   Register tile: 4x4, 8x8
//
// Adding a new shape requires adding it here and matching the validation logic
// in gemm_lab.cu.
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
    // Runtime dispatch is intentionally simple: find the exact tile/register
    // tuple requested by the benchmark and call the corresponding template
    // specialization.
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
