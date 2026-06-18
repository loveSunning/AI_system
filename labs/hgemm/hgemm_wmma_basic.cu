#include "hgemm_lab.hpp"

#include "ai_system/cuda/runtime.hpp"
#include "ai_system/profiling/nvtx.hpp"

#include <cuda_fp16.h>
#include <mma.h>

#include <cstdint>
#include <limits>
#include <string>

namespace ai_system::labs::hgemm {
namespace {

constexpr int kWarpSize = 32;
constexpr int kWmmaTileM = 16;
constexpr int kWmmaTileN = 16;

bool checked_count(std::size_t lhs, std::size_t rhs, std::size_t& out) {
    if(lhs != 0 && rhs > (std::numeric_limits<std::size_t>::max)() / lhs) {
        return false;
    }
    out = lhs * rhs;
    return true;
}

bool validate_problem(std::size_t m, std::size_t n, std::size_t k, std::string& error) {
    if(m == 0 || n == 0 || k == 0) {
        error = "HGEMM requires non-zero M, N, and K.";
        return false;
    }
    if(m > static_cast<std::size_t>((std::numeric_limits<int>::max)()) ||
       n > static_cast<std::size_t>((std::numeric_limits<int>::max)()) ||
       k > static_cast<std::size_t>((std::numeric_limits<int>::max)())) {
        error = "HGEMM dimensions exceed int limits.";
        return false;
    }

    std::size_t ignored = 0;
    if(!checked_count(m, k, ignored) || !checked_count(k, n, ignored) || !checked_count(m, n, ignored)) {
        error = "HGEMM matrix size overflows size_t.";
        return false;
    }

    return true;
}

bool validate_device_problem(
    const half* a,
    const half* b,
    half* c,
    int m,
    int n,
    int k,
    std::string& error
) {
    if(a == nullptr || b == nullptr || c == nullptr) {
        error = "HGEMM device pointers must be non-null.";
        return false;
    }
    return validate_problem(
        static_cast<std::size_t>(m),
        static_cast<std::size_t>(n),
        static_cast<std::size_t>(k),
        error
    );
}

bool validate_stage_options(int stages, bool swizzle, int swizzle_stride, std::string& error) {
    if(stages < 1 || stages > 8) {
        error = "HGEMM staged kernels expect stages in [1, 8].";
        return false;
    }
    if(swizzle && swizzle_stride <= 0) {
        error = "HGEMM staged kernels expect a positive swizzle_stride when swizzle is enabled.";
        return false;
    }
    return true;
}

__device__ __forceinline__ half zero_half() {
    return __float2half_rn(0.0f);
}

__device__ __forceinline__ float hgemm_load_as_float(const half* values, int row, int col, int rows, int cols) {
    if(row < rows && col < cols) {
        return __half2float(values[row * cols + col]);
    }
    return 0.0f;
}

__device__ __forceinline__ half hgemm_load_or_zero(const half* values, int row, int col, int rows, int cols) {
    if(row < rows && col < cols) {
        return values[row * cols + col];
    }
    return zero_half();
}

__device__ __forceinline__ bool hgemm_is_aligned(const void* pointer, std::uintptr_t alignment) {
    return (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1u)) == 0u;
}

template <int Count, int PackBits>
__device__ __forceinline__ void hgemm_copy_contiguous_half(const half* source, half* destination) {
    static_assert(
        Count == 2 || Count == 4 || Count == 8 || Count == 16 || Count == 32,
        "Unsupported HGEMM half vector width."
    );

    if constexpr(PackBits >= 128 && Count >= 8) {
        if(hgemm_is_aligned(source, 16u) && hgemm_is_aligned(destination, 16u)) {
#pragma unroll
            for(int offset = 0; offset < Count; offset += 8) {
                *reinterpret_cast<float4*>(destination + offset) =
                    *reinterpret_cast<const float4*>(source + offset);
            }
            return;
        }
    }
    if constexpr(PackBits >= 64 && Count >= 4) {
        if(hgemm_is_aligned(source, 8u) && hgemm_is_aligned(destination, 8u)) {
#pragma unroll
            for(int offset = 0; offset < Count; offset += 4) {
                *reinterpret_cast<float2*>(destination + offset) =
                    *reinterpret_cast<const float2*>(source + offset);
            }
            return;
        }
    }
    if constexpr(PackBits >= 32 && Count >= 2) {
        if(hgemm_is_aligned(source, 4u) && hgemm_is_aligned(destination, 4u)) {
#pragma unroll
            for(int offset = 0; offset < Count; offset += 2) {
                *reinterpret_cast<half2*>(destination + offset) =
                    *reinterpret_cast<const half2*>(source + offset);
            }
            return;
        }
    }

#pragma unroll
    for(int index = 0; index < Count; ++index) {
        destination[index] = source[index];
    }
}

template <int Count, int PackBits>
__device__ __forceinline__ void hgemm_load_contiguous_half(
    const half* values,
    int row,
    int col,
    int rows,
    int cols,
    half* destination
) {
    if(row < rows && col + Count <= cols) {
        hgemm_copy_contiguous_half<Count, PackBits>(values + row * cols + col, destination);
        return;
    }

#pragma unroll
    for(int index = 0; index < Count; ++index) {
        destination[index] = hgemm_load_or_zero(values, row, col + index, rows, cols);
    }
}

__device__ __forceinline__ void hgemm_cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int Groups>
__device__ __forceinline__ void hgemm_cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" : : "n"(Groups));
}

template <int Bytes>
__device__ __forceinline__ void hgemm_cp_async_cg(std::uint32_t shared_address, const half* global_address) {
    static_assert(Bytes == 16, "cp.async.cg in this lab copies 16 bytes.");
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
        :
        : "r"(shared_address), "l"(global_address), "n"(Bytes)
    );
}

template <int Bytes>
__device__ __forceinline__ void hgemm_cp_async_ca(std::uint32_t shared_address, const half* global_address) {
    static_assert(Bytes == 16, "cp.async.ca in this lab copies 16 bytes.");
    asm volatile(
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
        :
        : "r"(shared_address), "l"(global_address), "n"(Bytes)
    );
}

template <int BlockM, int BlockN>
__device__ void hgemm_wmma_scalar_fallback_tile_at(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k,
    int block_row,
    int block_col
) {
    const int tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    const int thread_count = blockDim.x * blockDim.y * blockDim.z;

    // Fallback path for boundary tiles and unsupported WMMA alignment cases.
    // Threads flatten the BlockM x BlockN output tile and stride by the CTA's
    // total thread count, so every in-bounds C element is computed exactly once.
    for(int index = tid; index < BlockM * BlockN; index += thread_count) {
        const int row = block_row + index / BlockN;
        const int col = block_col + index % BlockN;
        if(row < m && col < n) {
            float accumulator = 0.0f;
            for(int inner = 0; inner < k; ++inner) {
                accumulator += hgemm_load_as_float(a, row, inner, m, k) * hgemm_load_as_float(b, inner, col, k, n);
            }
            c[row * n + col] = __float2half_rn(accumulator);
        }
    }
}

template <int BlockM, int BlockN>
__device__ void hgemm_wmma_scalar_fallback_tile(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_wmma_scalar_fallback_tile_at<BlockM, BlockN>(
        a,
        b,
        c,
        m,
        n,
        k,
        static_cast<int>(blockIdx.y) * BlockM,
        static_cast<int>(blockIdx.x) * BlockN
    );
}

template <int WmmaM, int WmmaN, int WmmaK>
__device__ __forceinline__ void hgemm_wmma_store_float_fragment(
    const nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WmmaM, WmmaN, WmmaK, float>& fragment,
    float* shared_fragment,
    half* __restrict__ c,
    int row_base,
    int col_base,
    int m,
    int n,
    int lane
) {
    // WMMA fragments cannot be directly indexed in a portable way.  Store the
    // float accumulator fragment to shared memory, then let warp lanes write
    // row-major C elements.  For a 16x16 fragment, each lane writes 8 elements
    // (256 values / 32 lanes) with boundary masking.
    nvcuda::wmma::store_matrix_sync(shared_fragment, fragment, WmmaN, nvcuda::wmma::mem_row_major);
    __syncwarp();

    for(int index = lane; index < WmmaM * WmmaN; index += kWarpSize) {
        const int row = row_base + index / WmmaN;
        const int col = col_base + index % WmmaN;
        if(row < m && col < n) {
            c[row * n + col] = __float2half_rn(shared_fragment[index]);
        }
    }
    __syncwarp();
}

__device__ __forceinline__ void hgemm_wmma_load_half8_async(
    const half* __restrict__ source,
    half* __restrict__ destination
) {
    // One cp.async copies 16 bytes, exactly 8 half values.  The fallback keeps
    // the helper usable in non-aligned cases.
    if(hgemm_is_aligned(source, 16u) && hgemm_is_aligned(destination, 16u)) {
        const auto shared_address = static_cast<std::uint32_t>(__cvta_generic_to_shared(destination));
        hgemm_cp_async_cg<16>(shared_address, source);
        return;
    }

    hgemm_copy_contiguous_half<8, 128>(source, destination);
}

__device__ void hgemm_wmma_m16n16k16_naive_body(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    constexpr int kBlockM = 16;
    constexpr int kBlockN = 16;
    constexpr int kBlockK = 16;
    // One warp computes one 16x16 output tile.  Each K iteration feeds one
    // 16x16x16 WMMA operation with A and B loaded directly from global memory.
    const int block_row = static_cast<int>(blockIdx.y) * kBlockM;
    const int block_col = static_cast<int>(blockIdx.x) * kBlockN;
    const int lane = threadIdx.x & (kWarpSize - 1);

    // WMMA load/store requirements are stricter than the scalar kernels.  Use a
    // readable scalar fallback for partial M/N tiles, non-multiple K, or B/C
    // leading dimensions that do not satisfy the half8 alignment assumptions.
    if(block_row + kBlockM > m || block_col + kBlockN > n || (k % kBlockK) != 0 || (n % 8) != 0) {
        hgemm_wmma_scalar_fallback_tile<kBlockM, kBlockN>(a, b, c, m, n, k);
        return;
    }

    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, kBlockM, kBlockN, kBlockK, half, wmma::row_major> a_fragment;
    wmma::fragment<wmma::matrix_b, kBlockM, kBlockN, kBlockK, half, wmma::row_major> b_fragment;
    wmma::fragment<wmma::accumulator, kBlockM, kBlockN, kBlockK, float> c_fragment;
    wmma::fill_fragment(c_fragment, 0.0f);

    for(int k_base = 0; k_base < k; k_base += kBlockK) {
        wmma::load_matrix_sync(a_fragment, a + block_row * k + k_base, k);
        wmma::load_matrix_sync(b_fragment, b + k_base * n + block_col, n);
        wmma::mma_sync(c_fragment, a_fragment, b_fragment, c_fragment);
    }

    __shared__ float shared_c[kBlockM * kBlockN];
    hgemm_wmma_store_float_fragment<kBlockM, kBlockN, kBlockK>(
        c_fragment,
        shared_c,
        c,
        block_row,
        block_col,
        m,
        n,
        lane
    );
}

__device__ void hgemm_wmma_m16n16k16_mma4x2_body(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    constexpr int kWmmaM = 16;
    constexpr int kWmmaN = 16;
    constexpr int kWmmaK = 16;
    constexpr int kWmmaTileM = 4;
    constexpr int kWmmaTileN = 2;
    constexpr int kBlockM = kWmmaM * kWmmaTileM;
    constexpr int kBlockN = kWmmaN * kWmmaTileN;
    constexpr int kWarpCount = kWmmaTileM * kWmmaTileN;
    // CTA tile is 64x32 with 8 warps.  Each warp owns one 16x16 WMMA output
    // tile.  warp_id -> (warp_m, warp_n) in a 4x2 warp grid:
    //   warp_m = warp_id / 2, warp_n = warp_id % 2.
    //
    // Per K slice:
    //   A shared tile = 64x16 half = 1024 half; each thread loads 4 half.
    //   B shared tile = 16x32 half = 512 half; each thread loads 2 half.
    const int block_row = static_cast<int>(blockIdx.y) * kBlockM;
    const int block_col = static_cast<int>(blockIdx.x) * kBlockN;

    if(block_row + kBlockM > m || block_col + kBlockN > n || (k % kWmmaK) != 0) {
        hgemm_wmma_scalar_fallback_tile<kBlockM, kBlockN>(a, b, c, m, n, k);
        return;
    }

    using namespace nvcuda;
    __shared__ half shared_a[kBlockM][kWmmaK];
    __shared__ half shared_b[kWmmaK][kBlockN];
    __shared__ float shared_c[kWarpCount][kWmmaM * kWmmaN];

    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane = tid & (kWarpSize - 1);
    const int warp_m = warp_id / kWmmaTileN;
    const int warp_n = warp_id % kWmmaTileN;

    // 256 threads cooperatively load A and B into shared memory.
    const int load_a_smem_m = tid / 4;
    const int load_a_smem_k = (tid % 4) * 4;
    const int load_b_smem_k = tid / 16;
    const int load_b_smem_n = (tid % 16) * 2;

    wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_fragment;
    wmma::fill_fragment(c_fragment, 0.0f);

    for(int k_base = 0; k_base < k; k_base += kWmmaK) {
        hgemm_load_contiguous_half<4, 64>(
            a,
            block_row + load_a_smem_m,
            k_base + load_a_smem_k,
            m,
            k,
            &shared_a[load_a_smem_m][load_a_smem_k]
        );
        hgemm_load_contiguous_half<2, 32>(
            b,
            k_base + load_b_smem_k,
            block_col + load_b_smem_n,
            k,
            n,
            &shared_b[load_b_smem_k][load_b_smem_n]
        );
        __syncthreads();

        // Each warp loads the 16x16 A/B tile matching its warp_m/warp_n and
        // performs one WMMA mma_sync for this K slice.
        wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> a_fragment;
        wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> b_fragment;
        wmma::load_matrix_sync(a_fragment, &shared_a[warp_m * kWmmaM][0], kWmmaK);
        wmma::load_matrix_sync(b_fragment, &shared_b[0][warp_n * kWmmaN], kBlockN);
        wmma::mma_sync(c_fragment, a_fragment, b_fragment, c_fragment);
        __syncthreads();
    }

    hgemm_wmma_store_float_fragment<kWmmaM, kWmmaN, kWmmaK>(
        c_fragment,
        shared_c[warp_id],
        c,
        block_row + warp_m * kWmmaM,
        block_col + warp_n * kWmmaN,
        m,
        n,
        lane
    );
}

template <
    int WmmaM,
    int WmmaN,
    int WmmaTileM,
    int WmmaTileN,
    int WarpTileM,
    int WarpTileN>
__device__ void hgemm_wmma_warp2x4_body(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    constexpr int kWmmaK = 16;
    constexpr int kBlockM = WmmaM * WmmaTileM * WarpTileM;
    constexpr int kBlockN = WmmaN * WmmaTileN * WarpTileN;
    constexpr int kWarpCount = WmmaTileM * WmmaTileN;
    static_assert(kBlockM == 128 && kBlockN == 128, "This WMMA body expects a 128x128 CTA tile.");
    static_assert(kWarpCount == 8, "This WMMA body expects 8 warps per block.");

    // 8-warp CTA for 128x128 output tiles.  WmmaTileM x WmmaTileN is the warp
    // grid inside the CTA; WarpTileM x WarpTileN is the number of WMMA fragments
    // computed by each warp.
    //
    // For hgemm_wmma_m16n16k16_mma4x2_warp2x4:
    //   WmmaM/N/K = 16/16/16, WmmaTileM/N = 4/2, WarpTileM/N = 2/4.
    //   each warp computes 2x4 fragments = a 32x64 C region.
    //
    // For hgemm_wmma_m32n8k16_mma2x4_warp2x4:
    //   WmmaM/N/K = 32/8/16, WmmaTileM/N = 2/4, WarpTileM/N = 2/4.
    //   each warp computes 2x4 fragments = a 64x32 C region.
    //
    // Per K slice for both:
    //   A shared tile = 128x16 half = 2048 half; each thread loads 8 half.
    //   B shared tile = 16x128 half = 2048 half; each thread loads 8 half.
    const int block_row = static_cast<int>(blockIdx.y) * kBlockM;
    const int block_col = static_cast<int>(blockIdx.x) * kBlockN;
    if(block_row + kBlockM > m || block_col + kBlockN > n || (k % kWmmaK) != 0) {
        hgemm_wmma_scalar_fallback_tile<kBlockM, kBlockN>(a, b, c, m, n, k);
        return;
    }

    using namespace nvcuda;
    __shared__ half shared_a[kBlockM][kWmmaK];
    __shared__ half shared_b[kWmmaK][kBlockN];
    __shared__ float shared_c[kWarpCount][WmmaM * WmmaN];

    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane = tid & (kWarpSize - 1);
    const int warp_m = warp_id / WmmaTileN;
    const int warp_n = warp_id % WmmaTileN;

    // A: two half8 vectors cover each M row of the 16-wide K slice.
    // B: sixteen half8 vectors cover each K row of the 128-wide N tile.
    const int load_a_smem_m = tid / 2;
    const int load_a_smem_k = (tid % 2) * 8;
    const int load_b_smem_k = tid / 16;
    const int load_b_smem_n = (tid % 16) * 8;

    wmma::fragment<wmma::accumulator, WmmaM, WmmaN, kWmmaK, float> c_fragments[WarpTileM][WarpTileN];
#pragma unroll
    for(int i = 0; i < WarpTileM; ++i) {
#pragma unroll
        for(int j = 0; j < WarpTileN; ++j) {
            wmma::fill_fragment(c_fragments[i][j], 0.0f);
        }
    }

    for(int k_base = 0; k_base < k; k_base += kWmmaK) {
        hgemm_load_contiguous_half<8, 128>(
            a,
            block_row + load_a_smem_m,
            k_base + load_a_smem_k,
            m,
            k,
            &shared_a[load_a_smem_m][load_a_smem_k]
        );
        hgemm_load_contiguous_half<8, 128>(
            b,
            k_base + load_b_smem_k,
            block_col + load_b_smem_n,
            k,
            n,
            &shared_b[load_b_smem_k][load_b_smem_n]
        );
        __syncthreads();

        wmma::fragment<wmma::matrix_a, WmmaM, WmmaN, kWmmaK, half, wmma::row_major> a_fragments[WarpTileM];
        wmma::fragment<wmma::matrix_b, WmmaM, WmmaN, kWmmaK, half, wmma::row_major> b_fragments[WarpTileN];

#pragma unroll
        for(int i = 0; i < WarpTileM; ++i) {
            // A fragments step along M inside this warp's row region.
            const int shared_m = warp_m * (WmmaM * WarpTileM) + i * WmmaM;
            wmma::load_matrix_sync(a_fragments[i], &shared_a[shared_m][0], kWmmaK);
        }

#pragma unroll
        for(int j = 0; j < WarpTileN; ++j) {
            // B fragments step along N inside this warp's column region.
            const int shared_n = warp_n * (WmmaN * WarpTileN) + j * WmmaN;
            wmma::load_matrix_sync(b_fragments[j], &shared_b[0][shared_n], kBlockN);
        }

#pragma unroll
        for(int i = 0; i < WarpTileM; ++i) {
#pragma unroll
            for(int j = 0; j < WarpTileN; ++j) {
                wmma::mma_sync(c_fragments[i][j], a_fragments[i], b_fragments[j], c_fragments[i][j]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for(int i = 0; i < WarpTileM; ++i) {
#pragma unroll
        for(int j = 0; j < WarpTileN; ++j) {
            const int store_row = block_row + warp_m * (WmmaM * WarpTileM) + i * WmmaM;
            const int store_col = block_col + warp_n * (WmmaN * WarpTileN) + j * WmmaN;
            hgemm_wmma_store_float_fragment<WmmaM, WmmaN, kWmmaK>(
                c_fragments[i][j],
                shared_c[warp_id],
                c,
                store_row,
                store_col,
                m,
                n,
                lane
            );
        }
    }
}

template <
    int WmmaM,
    int WmmaN,
    int WmmaTileM,
    int WmmaTileN,
    int WarpTileM,
    int WarpTileN,
    int Offset>
__device__ void hgemm_wmma_warp2x4_dbuf_async_body(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    constexpr int kWmmaK = 16;
    constexpr int kBlockM = WmmaM * WmmaTileM * WarpTileM;
    constexpr int kBlockN = WmmaN * WmmaTileN * WarpTileN;
    constexpr int kWarpCount = WmmaTileM * WmmaTileN;
    static_assert(kBlockM == 128 && kBlockN == 128, "This WMMA body expects a 128x128 CTA tile.");
    static_assert(kWarpCount == 8, "This WMMA body expects 8 warps per block.");
    static_assert(Offset % 8 == 0, "WMMA shared-memory padding must preserve 16-byte alignment.");

    // Double-buffered cp.async WMMA body.  It uses the same warp/output mapping
    // as hgemm_wmma_warp2x4_body, but shared_a/shared_b contain two K buffers.
    // Each thread asynchronously copies one half8 from A and one half8 from B
    // for the next K tile.  Padding keeps rows 16-byte aligned after adding
    // Offset to the shared-memory leading dimension.
    const int block_row = static_cast<int>(blockIdx.y) * kBlockM;
    const int block_col = static_cast<int>(blockIdx.x) * kBlockN;
    if(block_row + kBlockM > m || block_col + kBlockN > n || (k % kWmmaK) != 0) {
        hgemm_wmma_scalar_fallback_tile<kBlockM, kBlockN>(a, b, c, m, n, k);
        return;
    }

    using namespace nvcuda;
    __shared__ half shared_a[2][kBlockM][kWmmaK + Offset];
    __shared__ half shared_b[2][kWmmaK][kBlockN + Offset];
    __shared__ float shared_c[kWarpCount][WmmaM * WmmaN];

    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane = tid & (kWarpSize - 1);
    const int warp_m = warp_id / WmmaTileN;
    const int warp_n = warp_id % WmmaTileN;

    const int load_a_smem_m = tid / 2;
    const int load_a_smem_k = (tid % 2) * 8;
    const int load_b_smem_k = tid / 16;
    const int load_b_smem_n = (tid % 16) * 8;
    const int k_tiles = k / kWmmaK;

    wmma::fragment<wmma::accumulator, WmmaM, WmmaN, kWmmaK, float> c_fragments[WarpTileM][WarpTileN];
#pragma unroll
    for(int i = 0; i < WarpTileM; ++i) {
#pragma unroll
        for(int j = 0; j < WarpTileN; ++j) {
            wmma::fill_fragment(c_fragments[i][j], 0.0f);
        }
    }

    // Prime buffer 0.  k_tiles is exact because the fallback above rejects
    // non-multiple-K cases.
    hgemm_wmma_load_half8_async(
        a + (block_row + load_a_smem_m) * k + load_a_smem_k,
        &shared_a[0][load_a_smem_m][load_a_smem_k]
    );
    hgemm_wmma_load_half8_async(
        b + load_b_smem_k * n + block_col + load_b_smem_n,
        &shared_b[0][load_b_smem_k][load_b_smem_n]
    );
    hgemm_cp_async_commit_group();
    hgemm_cp_async_wait_group<0>();
    __syncthreads();

    for(int tile = 1; tile < k_tiles; ++tile) {
        const int active_buffer = (tile - 1) & 1;
        const int next_buffer = tile & 1;
        const int k_base = tile * kWmmaK;

        // Start the next global->shared copies before consuming active_buffer.
        hgemm_wmma_load_half8_async(
            a + (block_row + load_a_smem_m) * k + k_base + load_a_smem_k,
            &shared_a[next_buffer][load_a_smem_m][load_a_smem_k]
        );
        hgemm_wmma_load_half8_async(
            b + (k_base + load_b_smem_k) * n + block_col + load_b_smem_n,
            &shared_b[next_buffer][load_b_smem_k][load_b_smem_n]
        );
        hgemm_cp_async_commit_group();

        wmma::fragment<wmma::matrix_a, WmmaM, WmmaN, kWmmaK, half, wmma::row_major> a_fragments[WarpTileM];
        wmma::fragment<wmma::matrix_b, WmmaM, WmmaN, kWmmaK, half, wmma::row_major> b_fragments[WarpTileN];

#pragma unroll
        for(int i = 0; i < WarpTileM; ++i) {
            const int shared_m = warp_m * (WmmaM * WarpTileM) + i * WmmaM;
            wmma::load_matrix_sync(a_fragments[i], &shared_a[active_buffer][shared_m][0], kWmmaK + Offset);
        }

#pragma unroll
        for(int j = 0; j < WarpTileN; ++j) {
            const int shared_n = warp_n * (WmmaN * WarpTileN) + j * WmmaN;
            wmma::load_matrix_sync(b_fragments[j], &shared_b[active_buffer][0][shared_n], kBlockN + Offset);
        }

#pragma unroll
        for(int i = 0; i < WarpTileM; ++i) {
#pragma unroll
            for(int j = 0; j < WarpTileN; ++j) {
                wmma::mma_sync(c_fragments[i][j], a_fragments[i], b_fragments[j], c_fragments[i][j]);
            }
        }

        hgemm_cp_async_wait_group<0>();
        __syncthreads();
    }

    const int final_buffer = (k_tiles - 1) & 1;
    // Drain the final prefetched K tile.
    wmma::fragment<wmma::matrix_a, WmmaM, WmmaN, kWmmaK, half, wmma::row_major> a_fragments[WarpTileM];
    wmma::fragment<wmma::matrix_b, WmmaM, WmmaN, kWmmaK, half, wmma::row_major> b_fragments[WarpTileN];

#pragma unroll
    for(int i = 0; i < WarpTileM; ++i) {
        const int shared_m = warp_m * (WmmaM * WarpTileM) + i * WmmaM;
        wmma::load_matrix_sync(a_fragments[i], &shared_a[final_buffer][shared_m][0], kWmmaK + Offset);
    }

#pragma unroll
    for(int j = 0; j < WarpTileN; ++j) {
        const int shared_n = warp_n * (WmmaN * WarpTileN) + j * WmmaN;
        wmma::load_matrix_sync(b_fragments[j], &shared_b[final_buffer][0][shared_n], kBlockN + Offset);
    }

#pragma unroll
    for(int i = 0; i < WarpTileM; ++i) {
#pragma unroll
        for(int j = 0; j < WarpTileN; ++j) {
            wmma::mma_sync(c_fragments[i][j], a_fragments[i], b_fragments[j], c_fragments[i][j]);
        }
    }

#pragma unroll
    for(int i = 0; i < WarpTileM; ++i) {
#pragma unroll
        for(int j = 0; j < WarpTileN; ++j) {
            const int store_row = block_row + warp_m * (WmmaM * WarpTileM) + i * WmmaM;
            const int store_col = block_col + warp_n * (WmmaN * WarpTileN) + j * WmmaN;
            hgemm_wmma_store_float_fragment<WmmaM, WmmaN, kWmmaK>(
                c_fragments[i][j],
                shared_c[warp_id],
                c,
                store_row,
                store_col,
                m,
                n,
                lane
            );
        }
    }
}

template <int BlockM, int BlockN, int BlockK, int APad, int BPad>
__device__ __forceinline__ void hgemm_wmma_stage_load_async(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ shared_a,
    half* __restrict__ shared_b,
    int n,
    int k,
    int block_row,
    int block_col,
    int k_base,
    int stage,
    int tid,
    int thread_count
) {
    constexpr int kAChunks = (BlockM * BlockK) / 8;
    constexpr int kBChunks = (BlockK * BlockN) / 8;
    constexpr int kAStageOffset = BlockM * (BlockK + APad);
    constexpr int kBStageOffset = BlockK * (BlockN + BPad);

    // Load one pipeline stage with cp.async half8 chunks.
    //
    // A stage layout: shared_a[stage][BlockM][BlockK + APad].
    // B stage layout: shared_b[stage][BlockK][BlockN + BPad].
    // One chunk is 8 half values = 16 bytes = one cp.async instruction.
    // Threads iterate chunk += thread_count, so per-thread load count is:
    //   ceil(kAChunks / thread_count) A chunks and
    //   ceil(kBChunks / thread_count) B chunks.
    //
    // Examples:
    //   128x128x16 with 256 threads: A=256 chunks, B=256 chunks,
    //   so each thread copies one half8 from A and one half8 from B.
    //   256x128x16 with 256 threads: A=512 chunks, B=256 chunks,
    //   so each thread copies two A half8 chunks and one B half8 chunk.
    //   256x256x16 with 512 threads: A=512 chunks, B=512 chunks,
    //   so each thread copies one half8 from each matrix.
    for(int chunk = tid; chunk < kAChunks; chunk += thread_count) {
        const int row = chunk / (BlockK / 8);
        const int col = (chunk % (BlockK / 8)) * 8;
        hgemm_wmma_load_half8_async(
            a + (block_row + row) * k + k_base + col,
            shared_a + stage * kAStageOffset + row * (BlockK + APad) + col
        );
    }

    for(int chunk = tid; chunk < kBChunks; chunk += thread_count) {
        const int row = chunk / (BlockN / 8);
        const int col = (chunk % (BlockN / 8)) * 8;
        hgemm_wmma_load_half8_async(
            b + (k_base + row) * n + block_col + col,
            shared_b + stage * kBStageOffset + row * (BlockN + BPad) + col
        );
    }
}

template <
    int WmmaM,
    int WmmaN,
    int WmmaK,
    int BlockN,
    int BlockK,
    int WarpTileM,
    int WarpTileN,
    int WarpTileK,
    int APad,
    int BPad>
__device__ __forceinline__ void hgemm_wmma_stage_compute(
    half* __restrict__ shared_a,
    half* __restrict__ shared_b,
    int warp_m,
    int warp_n,
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WmmaM, WmmaN, WmmaK, float> (&c_fragments)[WarpTileM][WarpTileN]
) {
    using namespace nvcuda;

    // Compute one staged shared-memory tile.  For each WarpTileK slice, a warp
    // loads WarpTileM A fragments and WarpTileN B fragments from shared memory,
    // then performs the full WarpTileM x WarpTileN set of mma_sync operations.
    //
    // warp_m/warp_n identify the warp's logical region in the CTA tile:
    //   rows start at warp_m * (WmmaM * WarpTileM)
    //   cols start at warp_n * (WmmaN * WarpTileN)
#pragma unroll
    for(int warp_k = 0; warp_k < WarpTileK; ++warp_k) {
        wmma::fragment<wmma::matrix_a, WmmaM, WmmaN, WmmaK, half, wmma::row_major> a_fragments[WarpTileM];
        wmma::fragment<wmma::matrix_b, WmmaM, WmmaN, WmmaK, half, wmma::row_major> b_fragments[WarpTileN];
        const int shared_k = warp_k * WmmaK;

#pragma unroll
        for(int i = 0; i < WarpTileM; ++i) {
            const int shared_m = warp_m * (WmmaM * WarpTileM) + i * WmmaM;
            const half* source = shared_a + shared_m * (BlockK + APad) + shared_k;
            wmma::load_matrix_sync(a_fragments[i], source, BlockK + APad);
        }

#pragma unroll
        for(int j = 0; j < WarpTileN; ++j) {
            const int shared_n = warp_n * (WmmaN * WarpTileN) + j * WmmaN;
            const half* source = shared_b + shared_k * (BlockN + BPad) + shared_n;
            wmma::load_matrix_sync(b_fragments[j], source, BlockN + BPad);
        }

#pragma unroll
        for(int i = 0; i < WarpTileM; ++i) {
#pragma unroll
            for(int j = 0; j < WarpTileN; ++j) {
                wmma::mma_sync(c_fragments[i][j], a_fragments[i], b_fragments[j], c_fragments[i][j]);
            }
        }
    }
}

template <
    int WmmaM,
    int WmmaN,
    int WmmaK,
    int WmmaTileM,
    int WmmaTileN,
    int WarpTileM,
    int WarpTileN,
    int WarpTileK,
    int APad,
    int BPad,
    int KStage,
    bool BlockSwizzle>
__device__ void hgemm_wmma_staged_pipeline_body(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k,
    half* __restrict__ shared_a,
    half* __restrict__ shared_b
) {
    constexpr int kBlockM = WmmaM * WmmaTileM * WarpTileM;
    constexpr int kBlockN = WmmaN * WmmaTileN * WarpTileN;
    constexpr int kBlockK = WmmaK * WarpTileK;
    constexpr int kWarpCount = WmmaTileM * WmmaTileN;
    constexpr int kThreadCount = kWarpCount * kWarpSize;
    constexpr int kAStageOffset = kBlockM * (kBlockK + APad);
    constexpr int kBStageOffset = kBlockK * (kBlockN + BPad);
    static_assert(KStage >= 2, "WMMA staged kernels require at least two stages.");
    static_assert((kBlockK % 8) == 0 && (kBlockN % 8) == 0, "WMMA staged cp.async copies use 8 half chunks.");
    static_assert((APad % 8) == 0 && (BPad % 8) == 0, "WMMA staged padding must preserve 16-byte alignment.");
    static_assert(
        KStage * kAStageOffset * static_cast<int>(sizeof(half)) >=
            kWarpCount * WmmaM * WmmaN * static_cast<int>(sizeof(float)),
        "WMMA staged A shared storage must be large enough to reuse for accumulator stores."
    );

    // Multistage WMMA pipeline.
    //
    // CTA/output tile:
    //   BlockM = WmmaM * WmmaTileM * WarpTileM.
    //   BlockN = WmmaN * WmmaTileN * WarpTileN.
    //   BlockK = WmmaK * WarpTileK.
    //   kWarpCount = WmmaTileM * WmmaTileN, threads = kWarpCount * 32.
    //
    // Grid coordinate:
    //   without swizzle: blockIdx.x is the N tile, blockIdx.y is the M tile.
    //   with swizzle: blockIdx.z and blockIdx.x are folded into bx as
    //       bx = blockIdx.z * gridDim.x + blockIdx.x.
    //   This changes CTA ordering along N while preserving the same C tile.
    //
    // Pipeline:
    //   prime KStage-1 stages with cp.async,
    //   in the steady state launch the next stage copy and compute one older
    //   stage, then wait only far enough to keep KStage-1 groups in flight,
    //   finally drain the remaining prefetched stages.
    const int bx = BlockSwizzle ? static_cast<int>(blockIdx.z) * static_cast<int>(gridDim.x) +
                                      static_cast<int>(blockIdx.x)
                                : static_cast<int>(blockIdx.x);
    const int by = static_cast<int>(blockIdx.y);
    const int block_row = by * kBlockM;
    const int block_col = bx * kBlockN;
    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane = tid & (kWarpSize - 1);
    const int warp_m = warp_id / WmmaTileN;
    const int warp_n = warp_id % WmmaTileN;
    const int k_tiles = k / kBlockK;

    if(block_row + kBlockM > m || block_col + kBlockN > n || (k % kBlockK) != 0 ||
       (n % 8) != 0 || k_tiles < (KStage - 1)) {
        hgemm_wmma_scalar_fallback_tile_at<kBlockM, kBlockN>(a, b, c, m, n, k, block_row, block_col);
        return;
    }

    using namespace nvcuda;
    wmma::fragment<wmma::accumulator, WmmaM, WmmaN, WmmaK, float> c_fragments[WarpTileM][WarpTileN];
#pragma unroll
    for(int i = 0; i < WarpTileM; ++i) {
#pragma unroll
        for(int j = 0; j < WarpTileN; ++j) {
            wmma::fill_fragment(c_fragments[i][j], 0.0f);
        }
    }

#pragma unroll
    for(int stage = 0; stage < KStage - 1; ++stage) {
        // Stage 0..KStage-2 are prefetched before the first compute.  The first
        // compute will consume stage 0 while the loop fetches stage KStage-1.
        hgemm_wmma_stage_load_async<kBlockM, kBlockN, kBlockK, APad, BPad>(
            a,
            b,
            shared_a,
            shared_b,
            n,
            k,
            block_row,
            block_col,
            stage * kBlockK,
            stage,
            tid,
            kThreadCount
        );
        hgemm_cp_async_commit_group();
    }

    hgemm_cp_async_wait_group<KStage - 2>();
    __syncthreads();

    for(int tile = KStage - 1; tile < k_tiles; ++tile) {
        const int active_stage = (tile + 1) % KStage;
        const int next_stage = tile % KStage;
        // active_stage is the oldest completed stage; next_stage is the slot
        // being recycled for this tile's global->shared copy.
        hgemm_wmma_stage_load_async<kBlockM, kBlockN, kBlockK, APad, BPad>(
            a,
            b,
            shared_a,
            shared_b,
            n,
            k,
            block_row,
            block_col,
            tile * kBlockK,
            next_stage,
            tid,
            kThreadCount
        );
        hgemm_cp_async_commit_group();

        hgemm_wmma_stage_compute<WmmaM, WmmaN, WmmaK, kBlockN, kBlockK, WarpTileM, WarpTileN, WarpTileK, APad, BPad>(
            shared_a + active_stage * kAStageOffset,
            shared_b + active_stage * kBStageOffset,
            warp_m,
            warp_n,
            c_fragments
        );

        hgemm_cp_async_wait_group<KStage - 2>();
        __syncthreads();
    }

    if constexpr(KStage > 2) {
        hgemm_cp_async_wait_group<0>();
        __syncthreads();
    }

#pragma unroll
    for(int item = 0; item < KStage - 1; ++item) {
        const int active_stage = (k_tiles - (KStage - 1) + item) % KStage;
        hgemm_wmma_stage_compute<WmmaM, WmmaN, WmmaK, kBlockN, kBlockK, WarpTileM, WarpTileN, WarpTileK, APad, BPad>(
            shared_a + active_stage * kAStageOffset,
            shared_b + active_stage * kBStageOffset,
            warp_m,
            warp_n,
            c_fragments
        );
    }

    __syncthreads();
    float* shared_c = reinterpret_cast<float*>(shared_a);
    // Reuse shared_a as a temporary row-major float buffer for WMMA fragment
    // stores.  The static_assert above guarantees the storage is large enough.
#pragma unroll
    for(int i = 0; i < WarpTileM; ++i) {
#pragma unroll
        for(int j = 0; j < WarpTileN; ++j) {
            const int store_row = block_row + warp_m * (WmmaM * WarpTileM) + i * WmmaM;
            const int store_col = block_col + warp_n * (WmmaN * WarpTileN) + j * WmmaN;
            hgemm_wmma_store_float_fragment<WmmaM, WmmaN, WmmaK>(
                c_fragments[i][j],
                shared_c + warp_id * WmmaM * WmmaN,
                c,
                store_row,
                store_col,
                m,
                n,
                lane
            );
        }
    }
}

__global__ void hgemm_wmma_m16n16k16_naive_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    // Thin kernel wrapper: one warp/block computes one 16x16 WMMA C tile.
    hgemm_wmma_m16n16k16_naive_body(a, b, c, m, n, k);
}

__global__ void hgemm_wmma_m16n16k16_mma4x2_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    // Thin kernel wrapper: 8 warps/block compute a 64x32 tile, one WMMA tile
    // per warp.
    hgemm_wmma_m16n16k16_mma4x2_body(a, b, c, m, n, k);
}

__global__ void hgemm_wmma_m16n16k16_mma4x2_warp2x4_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    // Thin kernel wrapper: 8 warps/block compute a 128x128 tile; each warp
    // owns a 2x4 group of 16x16 WMMA fragments.
    hgemm_wmma_warp2x4_body<16, 16, 4, 2, 2, 4>(a, b, c, m, n, k);
}

__global__ void hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    // Same 128x128 WMMA mapping as the non-dbuf body, but with cp.async
    // double buffering.
    hgemm_wmma_warp2x4_dbuf_async_body<16, 16, 4, 2, 2, 4, 8>(a, b, c, m, n, k);
}

__global__ void hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    // 128x128 WMMA mapping using m32n8k16 fragments.  Each warp computes a
    // 2x4 group of 32x8 fragments, with the same cp.async double buffering.
    hgemm_wmma_warp2x4_dbuf_async_body<32, 8, 2, 4, 2, 4, 8>(a, b, c, m, n, k);
}

template <
    int WmmaM,
    int WmmaN,
    int WmmaK,
    int WmmaTileM,
    int WmmaTileN,
    int WarpTileM,
    int WarpTileN,
    int APad,
    int BPad,
    int KStage,
    bool BlockSwizzle>
__global__ void __launch_bounds__(256) hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    constexpr int kWarpTileK = 1;
    constexpr int kBlockM = WmmaM * WmmaTileM * WarpTileM;
    constexpr int kBlockN = WmmaN * WmmaTileN * WarpTileN;
    constexpr int kBlockK = WmmaK * kWarpTileK;
    // Static shared-memory staged kernel.  This wrapper materializes the shared
    // arrays at compile time; the body owns all coordinate math and pipeline
    // scheduling.  The launch wrapper instantiates it as 128x128x16, 8 warps.
    __shared__ half shared_a[KStage * kBlockM * (kBlockK + APad)];
    __shared__ half shared_b[KStage * kBlockK * (kBlockN + BPad)];
    hgemm_wmma_staged_pipeline_body<
        WmmaM,
        WmmaN,
        WmmaK,
        WmmaTileM,
        WmmaTileN,
        WarpTileM,
        WarpTileN,
        kWarpTileK,
        APad,
        BPad,
        KStage,
        BlockSwizzle>(a, b, c, m, n, k, shared_a, shared_b);
}

template <
    int WmmaM,
    int WmmaN,
    int WmmaK,
    int WmmaTileM,
    int WmmaTileN,
    int WarpTileM,
    int WarpTileN,
    int APad,
    int BPad,
    int KStage,
    bool BlockSwizzle>
__global__ void __launch_bounds__(256) hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    constexpr int kWarpTileK = 1;
    constexpr int kBlockM = WmmaM * WmmaTileM * WarpTileM;
    constexpr int kBlockK = WmmaK * kWarpTileK;
    // Dynamic shared-memory version of the 128x128x16 staged kernel.  shared_a
    // occupies the first KStage A stages; shared_b follows immediately after.
    extern __shared__ half shared[];
    half* shared_a = shared;
    half* shared_b = shared_a + KStage * kBlockM * (kBlockK + APad);
    hgemm_wmma_staged_pipeline_body<
        WmmaM,
        WmmaN,
        WmmaK,
        WmmaTileM,
        WmmaTileN,
        WarpTileM,
        WarpTileN,
        kWarpTileK,
        APad,
        BPad,
        KStage,
        BlockSwizzle>(a, b, c, m, n, k, shared_a, shared_b);
}

template <
    int WmmaM,
    int WmmaN,
    int WmmaK,
    int WmmaTileM,
    int WmmaTileN,
    int WarpTileM,
    int WarpTileN,
    int WarpTileK,
    int APad,
    int BPad,
    int KStage,
    bool BlockSwizzle>
__global__ void __launch_bounds__(256) hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    constexpr int kBlockM = WmmaM * WmmaTileM * WarpTileM;
    constexpr int kBlockK = WmmaK * WarpTileK;
    // Dynamic shared-memory staged kernel used by the 256x128 variant.  It
    // keeps 8 warps but gives each warp a larger 4x4 WMMA-fragment output tile.
    extern __shared__ half shared[];
    half* shared_a = shared;
    half* shared_b = shared_a + KStage * kBlockM * (kBlockK + APad);
    hgemm_wmma_staged_pipeline_body<
        WmmaM,
        WmmaN,
        WmmaK,
        WmmaTileM,
        WmmaTileN,
        WarpTileM,
        WarpTileN,
        WarpTileK,
        APad,
        BPad,
        KStage,
        BlockSwizzle>(a, b, c, m, n, k, shared_a, shared_b);
}

template <
    int WmmaM,
    int WmmaN,
    int WmmaK,
    int WmmaTileM,
    int WmmaTileN,
    int WarpTileM,
    int WarpTileN,
    int APad,
    int BPad,
    int KStage,
    bool BlockSwizzle>
__global__ void __launch_bounds__(512) hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    constexpr int kWarpTileK = 1;
    constexpr int kBlockM = WmmaM * WmmaTileM * WarpTileM;
    constexpr int kBlockK = WmmaK * kWarpTileK;
    // Dynamic shared-memory staged kernel used by the 256x256 variant.  The
    // launch uses 16 warps (512 threads), so each thread copies one A half8 and
    // one B half8 per 16-wide K stage.
    extern __shared__ half shared[];
    half* shared_a = shared;
    half* shared_b = shared_a + KStage * kBlockM * (kBlockK + APad);
    hgemm_wmma_staged_pipeline_body<
        WmmaM,
        WmmaN,
        WmmaK,
        WmmaTileM,
        WmmaTileN,
        WarpTileM,
        WarpTileN,
        kWarpTileK,
        APad,
        BPad,
        KStage,
        BlockSwizzle>(a, b, c, m, n, k, shared_a, shared_b);
}


}  // namespace

bool hgemm_wmma_m16n16k16_naive(const half* a, const half* b, half* c, int m, int n, int k, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_wmma_m16n16k16_naive_kernel_launch");
    if(!validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    // One warp/block.  grid.x counts 16-column tiles, grid.y counts 16-row tiles.
    const dim3 block(kWarpSize);
    const dim3 grid(static_cast<unsigned int>((n + kWmmaTileN - 1) / kWmmaTileN), static_cast<unsigned int>((m + kWmmaTileM - 1) / kWmmaTileM));
    hgemm_wmma_m16n16k16_naive_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_wmma_m16n16k16_mma4x2(const half* a, const half* b, half* c, int m, int n, int k, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_wmma_m16n16k16_mma4x2_kernel_launch");
    if(!validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    enum : int { kBlockM = 64, kBlockN = 32, kThreads = 8 * kWarpSize };
    // 8 warps/block, one 64x32 output tile.  Each warp owns one 16x16 WMMA tile.
    const dim3 block(kThreads);
    const dim3 grid(static_cast<unsigned int>((n + kBlockN - 1) / kBlockN), static_cast<unsigned int>((m + kBlockM - 1) / kBlockM));
    hgemm_wmma_m16n16k16_mma4x2_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_wmma_m16n16k16_mma4x2_warp2x4(const half* a, const half* b, half* c, int m, int n, int k, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_wmma_m16n16k16_mma4x2_warp2x4_kernel_launch");
    if(!validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    enum : int { kBlockM = 128, kBlockN = 128, kThreads = 8 * kWarpSize };
    // 8 warps/block, one 128x128 output tile.  Each warp owns a 2x4 WMMA group.
    const dim3 block(kThreads);
    const dim3 grid(static_cast<unsigned int>((n + kBlockN - 1) / kBlockN), static_cast<unsigned int>((m + kBlockM - 1) / kBlockM));
    hgemm_wmma_m16n16k16_mma4x2_warp2x4_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async(const half* a, const half* b, half* c, int m, int n, int k, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_kernel_launch");
    if(!validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    enum : int { kBlockM = 128, kBlockN = 128, kThreads = 8 * kWarpSize };
    // Same grid/block as warp2x4; body adds double-buffered cp.async loading.
    const dim3 block(kThreads);
    const dim3 grid(static_cast<unsigned int>((n + kBlockN - 1) / kBlockN), static_cast<unsigned int>((m + kBlockM - 1) / kBlockM));
    hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async(const half* a, const half* b, half* c, int m, int n, int k, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async_kernel_launch");
    if(!validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    enum : int { kBlockM = 128, kBlockN = 128, kThreads = 8 * kWarpSize };
    // Same 128x128 grid/block as above, but with m32n8k16 WMMA fragments.
    const dim3 block(kThreads);
    const dim3 grid(static_cast<unsigned int>((n + kBlockN - 1) / kBlockN), static_cast<unsigned int>((m + kBlockM - 1) / kBlockM));
    hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

dim3 hgemm_wmma_stage_grid(int m, int n, int block_m, int block_n, bool swizzle, int swizzle_stride) {
    const auto grid_y = static_cast<unsigned int>((m + block_m - 1) / block_m);
    const auto n_tiles = static_cast<unsigned int>((n + block_n - 1) / block_n);
    if(!swizzle) {
        // Ordinary 2-D launch: x enumerates C tile columns, y enumerates rows.
        return dim3(n_tiles, grid_y);
    }

    // Swizzled launch: split the N-tile traversal across grid.z, then fold
    // (blockIdx.z, blockIdx.x) back into bx inside the kernel.  swizzle_stride
    // is expressed in columns, so n_swizzle is the number of N-column groups.
    const auto n_swizzle = static_cast<unsigned int>((n + swizzle_stride - 1) / swizzle_stride);
    return dim3((n_tiles + n_swizzle - 1) / n_swizzle, grid_y, n_swizzle);
}

template <int Stages, bool Swizzle>
bool hgemm_launch_wmma_mma4x2_warp2x4_stages_static(
    const half* a,
    const half* b,
    half* c,
    int m,
    int n,
    int k,
    int swizzle_stride,
    std::string& error
) {
    enum : int {
        kWmmaM = 16,
        kWmmaN = 16,
        kWmmaK = 16,
        kWmmaTileM = 4,
        kWmmaTileN = 2,
        kWarpTileM = 2,
        kWarpTileN = 4,
        kAPad = 8,
        kBPad = 8,
        kBlockM = kWmmaM * kWmmaTileM * kWarpTileM,
        kBlockN = kWmmaN * kWmmaTileN * kWarpTileN,
        kThreads = kWmmaTileM * kWmmaTileN * kWarpSize
    };
    // Instantiates the static-smem staged WMMA kernel:
    //   BlockM = 16*4*2 = 128
    //   BlockN = 16*2*4 = 128
    //   BlockK = 16
    //   threads = 4*2 warps = 256
    // Per stage, each thread copies one half8 from A and one half8 from B.
    const dim3 block(kThreads);
    const dim3 grid = hgemm_wmma_stage_grid(m, n, kBlockM, kBlockN, Swizzle, swizzle_stride);
    hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_kernel<
        kWmmaM,
        kWmmaN,
        kWmmaK,
        kWmmaTileM,
        kWmmaTileN,
        kWarpTileM,
        kWarpTileN,
        kAPad,
        kBPad,
        Stages,
        Swizzle><<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

template <int Stages, bool Swizzle>
bool hgemm_launch_wmma_mma4x2_warp2x4_stages_dsmem(
    const half* a,
    const half* b,
    half* c,
    int m,
    int n,
    int k,
    int swizzle_stride,
    std::string& error
) {
    enum : int {
        kWmmaM = 16,
        kWmmaN = 16,
        kWmmaK = 16,
        kWmmaTileM = 4,
        kWmmaTileN = 2,
        kWarpTileM = 2,
        kWarpTileN = 4,
        kAPad = 0,
        kBPad = 16,
        kBlockM = kWmmaM * kWmmaTileM * kWarpTileM,
        kBlockN = kWmmaN * kWmmaTileN * kWarpTileN,
        kBlockK = kWmmaK,
        kThreads = kWmmaTileM * kWmmaTileN * kWarpSize,
        kSharedBytes = Stages * kBlockM * (kBlockK + kAPad) * static_cast<int>(sizeof(half)) +
            Stages * kBlockK * (kBlockN + kBPad) * static_cast<int>(sizeof(half))
    };
    // Dynamic-smem version of the same 128x128x16, 8-warp staged kernel.
    // Shared bytes are split into:
    //   Stages * 128 * (16 + APad) half for A
    //   Stages * 16  * (128 + BPad) half for B
    // BPad=16 reduces shared bank conflicts and keeps half8 alignment.
    if(!ai_system::cuda_utils::check_status(
           cudaFuncSetAttribute(
               hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem_kernel<
                   kWmmaM,
                   kWmmaN,
                   kWmmaK,
                   kWmmaTileM,
                   kWmmaTileN,
                   kWarpTileM,
                   kWarpTileN,
                   kAPad,
                   kBPad,
                   Stages,
                   Swizzle>,
               cudaFuncAttributeMaxDynamicSharedMemorySize,
               98304
           ),
           "cudaFuncSetAttribute(hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem_kernel)",
           error
       )) {
        return false;
    }

    const dim3 block(kThreads);
    const dim3 grid = hgemm_wmma_stage_grid(m, n, kBlockM, kBlockN, Swizzle, swizzle_stride);
    hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem_kernel<
        kWmmaM,
        kWmmaN,
        kWmmaK,
        kWmmaTileM,
        kWmmaTileN,
        kWarpTileM,
        kWarpTileN,
        kAPad,
        kBPad,
        Stages,
        Swizzle><<<grid, block, kSharedBytes>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

template <int Stages, bool Swizzle>
bool hgemm_launch_wmma_mma4x2_warp4x4_stages_dsmem(
    const half* a,
    const half* b,
    half* c,
    int m,
    int n,
    int k,
    int swizzle_stride,
    std::string& error
) {
    enum : int {
        kWmmaM = 16,
        kWmmaN = 16,
        kWmmaK = 16,
        kWmmaTileM = 4,
        kWmmaTileN = 2,
        kWarpTileM = 4,
        kWarpTileN = 4,
        kWarpTileK = 1,
        kAPad = 0,
        kBPad = 16,
        kBlockM = kWmmaM * kWmmaTileM * kWarpTileM,
        kBlockN = kWmmaN * kWmmaTileN * kWarpTileN,
        kBlockK = kWmmaK * kWarpTileK,
        kThreads = kWmmaTileM * kWmmaTileN * kWarpSize,
        kSharedBytes = Stages * kBlockM * (kBlockK + kAPad) * static_cast<int>(sizeof(half)) +
            Stages * kBlockK * (kBlockN + kBPad) * static_cast<int>(sizeof(half))
    };
    // Dynamic-smem 256x128x16 variant.  It still launches 8 warps (256
    // threads), but each warp computes a 4x4 group of 16x16 fragments.  Per
    // stage, each thread copies two A half8 chunks and one B half8 chunk.
    if(!ai_system::cuda_utils::check_status(
           cudaFuncSetAttribute(
               hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem_kernel<
                   kWmmaM,
                   kWmmaN,
                   kWmmaK,
                   kWmmaTileM,
                   kWmmaTileN,
                   kWarpTileM,
                   kWarpTileN,
                   kWarpTileK,
                   kAPad,
                   kBPad,
                   Stages,
                   Swizzle>,
               cudaFuncAttributeMaxDynamicSharedMemorySize,
               98304
           ),
           "cudaFuncSetAttribute(hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem_kernel)",
           error
       )) {
        return false;
    }

    const dim3 block(kThreads);
    const dim3 grid = hgemm_wmma_stage_grid(m, n, kBlockM, kBlockN, Swizzle, swizzle_stride);
    hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem_kernel<
        kWmmaM,
        kWmmaN,
        kWmmaK,
        kWmmaTileM,
        kWmmaTileN,
        kWarpTileM,
        kWarpTileN,
        kWarpTileK,
        kAPad,
        kBPad,
        Stages,
        Swizzle><<<grid, block, kSharedBytes>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

template <int Stages, bool Swizzle>
bool hgemm_launch_wmma_mma4x4_warp4x4_stages_dsmem(
    const half* a,
    const half* b,
    half* c,
    int m,
    int n,
    int k,
    int swizzle_stride,
    std::string& error
) {
    enum : int {
        kWmmaM = 16,
        kWmmaN = 16,
        kWmmaK = 16,
        kWmmaTileM = 4,
        kWmmaTileN = 4,
        kWarpTileM = 4,
        kWarpTileN = 4,
        kAPad = 0,
        kBPad = 16,
        kBlockM = kWmmaM * kWmmaTileM * kWarpTileM,
        kBlockN = kWmmaN * kWmmaTileN * kWarpTileN,
        kBlockK = kWmmaK,
        kThreads = kWmmaTileM * kWmmaTileN * kWarpSize,
        kSharedBytes = Stages * kBlockM * (kBlockK + kAPad) * static_cast<int>(sizeof(half)) +
            Stages * kBlockK * (kBlockN + kBPad) * static_cast<int>(sizeof(half))
    };
    // Dynamic-smem 256x256x16 variant.  Launches 16 warps (512 threads);
    // per stage, each thread copies one A half8 chunk and one B half8 chunk.
    if(!ai_system::cuda_utils::check_status(
           cudaFuncSetAttribute(
               hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem_kernel<
                   kWmmaM,
                   kWmmaN,
                   kWmmaK,
                   kWmmaTileM,
                   kWmmaTileN,
                   kWarpTileM,
                   kWarpTileN,
                   kAPad,
                   kBPad,
                   Stages,
                   Swizzle>,
               cudaFuncAttributeMaxDynamicSharedMemorySize,
               98304
           ),
           "cudaFuncSetAttribute(hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem_kernel)",
           error
       )) {
        return false;
    }

    const dim3 block(kThreads);
    const dim3 grid = hgemm_wmma_stage_grid(m, n, kBlockM, kBlockN, Swizzle, swizzle_stride);
    hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem_kernel<
        kWmmaM,
        kWmmaN,
        kWmmaK,
        kWmmaTileM,
        kWmmaTileN,
        kWarpTileM,
        kWarpTileN,
        kAPad,
        kBPad,
        Stages,
        Swizzle><<<grid, block, kSharedBytes>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

template <bool Swizzle>
bool hgemm_dispatch_wmma_mma4x2_warp2x4_stages_static(
    const half* a,
    const half* b,
    half* c,
    int m,
    int n,
    int k,
    int stages,
    int swizzle_stride,
    std::string& error
) {
    switch(stages) {
        case 2:
            return hgemm_launch_wmma_mma4x2_warp2x4_stages_static<2, Swizzle>(a, b, c, m, n, k, swizzle_stride, error);
        case 3:
            return hgemm_launch_wmma_mma4x2_warp2x4_stages_static<3, Swizzle>(a, b, c, m, n, k, swizzle_stride, error);
        case 4:
            return hgemm_launch_wmma_mma4x2_warp2x4_stages_static<4, Swizzle>(a, b, c, m, n, k, swizzle_stride, error);
        default:
            return hgemm_launch_wmma_mma4x2_warp2x4_stages_static<2, Swizzle>(a, b, c, m, n, k, swizzle_stride, error);
    }
}

template <bool Swizzle>
bool hgemm_dispatch_wmma_mma4x2_warp2x4_stages_dsmem(
    const half* a,
    const half* b,
    half* c,
    int m,
    int n,
    int k,
    int stages,
    int swizzle_stride,
    std::string& error
) {
    switch(stages) {
        case 2:
            return hgemm_launch_wmma_mma4x2_warp2x4_stages_dsmem<2, Swizzle>(a, b, c, m, n, k, swizzle_stride, error);
        case 3:
            return hgemm_launch_wmma_mma4x2_warp2x4_stages_dsmem<3, Swizzle>(a, b, c, m, n, k, swizzle_stride, error);
        case 4:
            return hgemm_launch_wmma_mma4x2_warp2x4_stages_dsmem<4, Swizzle>(a, b, c, m, n, k, swizzle_stride, error);
        case 5:
            return hgemm_launch_wmma_mma4x2_warp2x4_stages_dsmem<5, Swizzle>(a, b, c, m, n, k, swizzle_stride, error);
        case 6:
            return hgemm_launch_wmma_mma4x2_warp2x4_stages_dsmem<6, Swizzle>(a, b, c, m, n, k, swizzle_stride, error);
        default:
            return hgemm_launch_wmma_mma4x2_warp2x4_stages_dsmem<2, Swizzle>(a, b, c, m, n, k, swizzle_stride, error);
    }
}

template <bool Swizzle>
bool hgemm_dispatch_wmma_mma4x2_warp4x4_stages_dsmem(
    const half* a,
    const half* b,
    half* c,
    int m,
    int n,
    int k,
    int stages,
    int swizzle_stride,
    std::string& error
) {
    switch(stages) {
        case 2:
            return hgemm_launch_wmma_mma4x2_warp4x4_stages_dsmem<2, Swizzle>(a, b, c, m, n, k, swizzle_stride, error);
        case 3:
            return hgemm_launch_wmma_mma4x2_warp4x4_stages_dsmem<3, Swizzle>(a, b, c, m, n, k, swizzle_stride, error);
        case 4:
            return hgemm_launch_wmma_mma4x2_warp4x4_stages_dsmem<4, Swizzle>(a, b, c, m, n, k, swizzle_stride, error);
        case 5:
            return hgemm_launch_wmma_mma4x2_warp4x4_stages_dsmem<5, Swizzle>(a, b, c, m, n, k, swizzle_stride, error);
        default:
            return hgemm_launch_wmma_mma4x2_warp4x4_stages_dsmem<2, Swizzle>(a, b, c, m, n, k, swizzle_stride, error);
    }
}

template <bool Swizzle>
bool hgemm_dispatch_wmma_mma4x4_warp4x4_stages_dsmem(
    const half* a,
    const half* b,
    half* c,
    int m,
    int n,
    int k,
    int stages,
    int swizzle_stride,
    std::string& error
) {
    switch(stages) {
        case 2:
            return hgemm_launch_wmma_mma4x4_warp4x4_stages_dsmem<2, Swizzle>(a, b, c, m, n, k, swizzle_stride, error);
        case 3:
            return hgemm_launch_wmma_mma4x4_warp4x4_stages_dsmem<3, Swizzle>(a, b, c, m, n, k, swizzle_stride, error);
        case 4:
            return hgemm_launch_wmma_mma4x4_warp4x4_stages_dsmem<4, Swizzle>(a, b, c, m, n, k, swizzle_stride, error);
        default:
            return hgemm_launch_wmma_mma4x4_warp4x4_stages_dsmem<2, Swizzle>(a, b, c, m, n, k, swizzle_stride, error);
    }
}

bool hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages(const half* a, const half* b, half* c, int m, int n, int k, int stages, bool swizzle, int swizzle_stride, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_kernel_launch");
    if(!validate_stage_options(stages, swizzle, swizzle_stride, error) || !validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    // Select a compile-time KStage/Swizzle instantiation for the static-smem
    // 128x128x16 staged pipeline.
    if(swizzle) {
        return hgemm_dispatch_wmma_mma4x2_warp2x4_stages_static<true>(
            a, b, c, m, n, k, stages, swizzle_stride, error
        );
    }
    return hgemm_dispatch_wmma_mma4x2_warp2x4_stages_static<false>(
        a, b, c, m, n, k, stages, swizzle_stride, error
    );
}

bool hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem(const half* a, const half* b, half* c, int m, int n, int k, int stages, bool swizzle, int swizzle_stride, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem_kernel_launch");
    if(!validate_stage_options(stages, swizzle, swizzle_stride, error) || !validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    // Dynamic-smem 128x128x16 staged pipeline; dispatch chooses KStage and
    // whether the grid folds blockIdx.z into the N tile coordinate.
    if(swizzle) {
        return hgemm_dispatch_wmma_mma4x2_warp2x4_stages_dsmem<true>(
            a, b, c, m, n, k, stages, swizzle_stride, error
        );
    }
    return hgemm_dispatch_wmma_mma4x2_warp2x4_stages_dsmem<false>(
        a, b, c, m, n, k, stages, swizzle_stride, error
    );
}

bool hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem(const half* a, const half* b, half* c, int m, int n, int k, int stages, bool swizzle, int swizzle_stride, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem_kernel_launch");
    if(!validate_stage_options(stages, swizzle, swizzle_stride, error) || !validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    // Dynamic-smem 256x128x16 staged pipeline.  The launch still uses 8 warps,
    // but each warp computes a 4x4 WMMA-fragment group.
    if(swizzle) {
        return hgemm_dispatch_wmma_mma4x2_warp4x4_stages_dsmem<true>(
            a, b, c, m, n, k, stages, swizzle_stride, error
        );
    }
    return hgemm_dispatch_wmma_mma4x2_warp4x4_stages_dsmem<false>(
        a, b, c, m, n, k, stages, swizzle_stride, error
    );
}

bool hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem(const half* a, const half* b, half* c, int m, int n, int k, int stages, bool swizzle, int swizzle_stride, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem_kernel_launch");
    if(!validate_stage_options(stages, swizzle, swizzle_stride, error) || !validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    // Dynamic-smem 256x256x16 staged pipeline.  This one launches 16 warps
    // because WmmaTileM x WmmaTileN = 4x4.
    if(swizzle) {
        return hgemm_dispatch_wmma_mma4x4_warp4x4_stages_dsmem<true>(
            a, b, c, m, n, k, stages, swizzle_stride, error
        );
    }
    return hgemm_dispatch_wmma_mma4x4_warp4x4_stages_dsmem<false>(
        a, b, c, m, n, k, stages, swizzle_stride, error
    );
}


}  // namespace ai_system::labs::hgemm
