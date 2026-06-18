#include "hgemm_lab.hpp"

#include <cuda_fp16.h>

#include <cstdint>

namespace ai_system::labs::hgemm {
namespace {

constexpr int kWarpSize = 32;

__device__ __forceinline__ half zero_half() {
    return __float2half_rn(0.0f);
}

__device__ __forceinline__ bool hgemm_is_aligned(const void* pointer, std::uintptr_t alignment) {
    return (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1u)) == 0u;
}

__device__ __forceinline__ half hgemm_load_or_zero(const half* values, int row, int col, int rows, int cols) {
    if(row < rows && col < cols) {
        return values[row * cols + col];
    }
    return zero_half();
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

__device__ __forceinline__ void hgemm_ldmatrix_x4(
    std::uint32_t& r0,
    std::uint32_t& r1,
    std::uint32_t& r2,
    std::uint32_t& r3,
    std::uint32_t shared_addr
) {
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(shared_addr)
    );
}

__device__ __forceinline__ void hgemm_ldmatrix_x2_trans(
    std::uint32_t& r0,
    std::uint32_t& r1,
    std::uint32_t shared_addr
) {
    asm volatile(
        "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(shared_addr)
    );
}

__device__ __forceinline__ void hgemm_mma_m16n8k16_f16(
    std::uint32_t& d0,
    std::uint32_t& d1,
    std::uint32_t a0,
    std::uint32_t a1,
    std::uint32_t a2,
    std::uint32_t a3,
    std::uint32_t b0,
    std::uint32_t b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};\n"
        : "+r"(d0), "+r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
}

__device__ __forceinline__ half hgemm_low_half(std::uint32_t packed) {
    return __ushort_as_half(static_cast<unsigned short>(packed & 0xffffu));
}

__device__ __forceinline__ half hgemm_high_half(std::uint32_t packed) {
    return __ushort_as_half(static_cast<unsigned short>((packed >> 16u) & 0xffffu));
}

__device__ __forceinline__ void hgemm_store_mma_m16n8k16_f16x2(
    half* values,
    int row,
    int col,
    int rows,
    int cols,
    std::uint32_t packed
) {
    if(row < rows && col < cols) {
        values[row * cols + col] = hgemm_low_half(packed);
    }
    if(row < rows && col + 1 < cols) {
        values[row * cols + col + 1] = hgemm_high_half(packed);
    }
}

__device__ __forceinline__ void hgemm_copy_half8_aligned(const half* source, half* destination) {
    *reinterpret_cast<float4*>(destination) = *reinterpret_cast<const float4*>(source);
}

__device__ __forceinline__ void hgemm_store_mma_m16n8k16_f16x2_aligned(
    half* values,
    int row,
    int col,
    int cols,
    std::uint32_t packed
) {
    *reinterpret_cast<std::uint32_t*>(values + row * cols + col) = packed;
}

template <int BlockM, int BlockN>
__device__ void hgemm_scalar_tile_body(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    const int block_row = static_cast<int>(blockIdx.y) * BlockM;
    const int block_col = static_cast<int>(blockIdx.x) * BlockN;
    const int tid = static_cast<int>(threadIdx.z) * blockDim.y * blockDim.x +
        static_cast<int>(threadIdx.y) * blockDim.x + threadIdx.x;
    const int thread_count = static_cast<int>(blockDim.x * blockDim.y * blockDim.z);

    for(int index = tid; index < BlockM * BlockN; index += thread_count) {
        const int row = block_row + index / BlockN;
        const int col = block_col + index % BlockN;
        if(row < m && col < n) {
            float accumulator = 0.0f;
            for(int inner = 0; inner < k; ++inner) {
                accumulator += __half2float(a[row * k + inner]) * __half2float(b[inner * n + col]);
            }
            c[row * n + col] = __float2half_rn(accumulator);
        }
    }
}

__device__ void hgemm_mma_m16n8k16_naive_body(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    constexpr int kMmaM = 16;
    constexpr int kMmaN = 8;
    constexpr int kMmaK = 16;

    const int tile_row = static_cast<int>(blockIdx.y) * kMmaM;
    const int tile_col = static_cast<int>(blockIdx.x) * kMmaN;
    const int lane = threadIdx.x & (kWarpSize - 1);

    // Edge tiles or non-16 K use a scalar teaching fallback.  This keeps the
    // inline PTX path simple: every active CTA reaches every __syncthreads(),
    // and ldmatrix only sees complete 16x8x16 fragments.
    if(tile_row + kMmaM > m || tile_col + kMmaN > n || (k % kMmaK) != 0) {
        hgemm_scalar_tile_body<kMmaM, kMmaN>(a, b, c, m, n, k);
        return;
    }

    // One warp computes one C fragment:
    //   C tile: 16 rows x 8 cols.
    //   A tile per K step: 16 x 16, loaded by all 32 lanes.
    //   B tile per K step: 16 x 8, loaded by lanes 0..15.
    // Each lane loads eight half values from A.  The two lane groups per row
    // cover A[row][0..7] and A[row][8..15].
    __shared__ half shared_a[kMmaM][kMmaK];
    __shared__ half shared_b[kMmaK][kMmaN];

    const int load_a_row = lane / 2;
    const int load_a_col = (lane & 1) * 8;
    const int load_b_row = lane;

    std::uint32_t rc0 = 0;
    std::uint32_t rc1 = 0;

    for(int k_base = 0; k_base < k; k_base += kMmaK) {
        hgemm_load_contiguous_half<8, 128>(
            a,
            tile_row + load_a_row,
            k_base + load_a_col,
            m,
            k,
            &shared_a[load_a_row][load_a_col]
        );

        if(load_b_row < kMmaK) {
            hgemm_load_contiguous_half<8, 128>(
                b,
                k_base + load_b_row,
                tile_col,
                k,
                n,
                &shared_b[load_b_row][0]
            );
        }

        __syncthreads();

        std::uint32_t ra0 = 0;
        std::uint32_t ra1 = 0;
        std::uint32_t ra2 = 0;
        std::uint32_t ra3 = 0;
        std::uint32_t rb0 = 0;
        std::uint32_t rb1 = 0;

        // For A, lanes 0..15 provide row addresses for the first 8 K columns
        // and lanes 16..31 provide row addresses for the second 8 K columns.
        // ldmatrix.x4 expands the four 8x8 row-major subtiles needed by
        // mma.m16n8k16.  B is consumed as row-major global data but the MMA
        // instruction expects the right operand in col-major fragment layout,
        // hence ldmatrix.x2.trans.
        const auto a_addr = static_cast<std::uint32_t>(
            __cvta_generic_to_shared(&shared_a[lane % 16][(lane / 16) * 8])
        );
        const auto b_addr = static_cast<std::uint32_t>(__cvta_generic_to_shared(&shared_b[lane % 16][0]));
        hgemm_ldmatrix_x4(ra0, ra1, ra2, ra3, a_addr);
        hgemm_ldmatrix_x2_trans(rb0, rb1, b_addr);
        hgemm_mma_m16n8k16_f16(rc0, rc1, ra0, ra1, ra2, ra3, rb0, rb1);

        __syncthreads();
    }

    const int store_col = tile_col + (lane % 4) * 2;
    const int store_row0 = tile_row + lane / 4;
    const int store_row1 = store_row0 + 8;
    hgemm_store_mma_m16n8k16_f16x2(c, store_row0, store_col, m, n, rc0);
    hgemm_store_mma_m16n8k16_f16x2(c, store_row1, store_col, m, n, rc1);
}

__device__ void hgemm_mma_m16n8k16_mma2x4_warp4x4_body(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    constexpr int kMmaM = 16;
    constexpr int kMmaN = 8;
    constexpr int kMmaK = 16;
    constexpr int kMmaTileM = 2;
    constexpr int kMmaTileN = 4;
    constexpr int kWarpTileM = 4;
    constexpr int kWarpTileN = 4;
    constexpr int kAPad = 8;
    constexpr int kBPad = 8;
    constexpr int kBlockM = kMmaM * kMmaTileM * kWarpTileM;
    constexpr int kBlockN = kMmaN * kMmaTileN * kWarpTileN;

    const int block_row = static_cast<int>(blockIdx.y) * kBlockM;
    const int block_col = static_cast<int>(blockIdx.x) * kBlockN;
    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane = tid & (kWarpSize - 1);
    const int warp_m = warp_id % kMmaTileM;
    const int warp_n = warp_id / kMmaTileM;

    // Uniform fallback for boundary CTAs or non-16 K.  The important bit is
    // that this branch is CTA-wide, so no subset of threads can skip a later
    // barrier while other threads enter the ldmatrix/MMA loop.
    if(block_row + kBlockM > m || block_col + kBlockN > n || (k % kMmaK) != 0 || (n % 8) != 0) {
        hgemm_scalar_tile_body<kBlockM, kBlockN>(a, b, c, m, n, k);
        return;
    }

    // CTA dataflow:
    //   256 threads = 8 warps.
    //   CTA output tile = 128x128.
    //   Warps are laid out as 2 along M and 4 along N.
    //   Each warp computes a 64x32 subtile = 4x4 m16n8k16 fragments.
    //
    // Per K step:
    //   A shared tile = 128x(16+8) half values.  Each thread loads one half8;
    //     tid/2 selects the A row, tid%2 selects K columns 0 or 8.
    //   B shared tile = 16x(128+8) half values.  Each thread loads one half8;
    //     tid/16 selects the K row, tid%16 selects the N half8 group.
    // The +8 padding matches the reference implementation and keeps both
    // ldmatrix source rows and global half8 copies 16-byte aligned while
    // reducing shared-memory bank conflicts.
    __shared__ half shared_a[kBlockM][kMmaK + kAPad];
    __shared__ half shared_b[kMmaK][kBlockN + kBPad];

    const int load_a_row = tid / 2;
    const int load_a_col = (tid & 1) * 8;
    const int load_b_row = tid / 16;
    const int load_b_col = (tid & 15) * 8;

    std::uint32_t rc[kWarpTileM][kWarpTileN][2];
#pragma unroll
    for(int mma_m = 0; mma_m < kWarpTileM; ++mma_m) {
#pragma unroll
        for(int mma_n = 0; mma_n < kWarpTileN; ++mma_n) {
            rc[mma_m][mma_n][0] = 0;
            rc[mma_m][mma_n][1] = 0;
        }
    }

    for(int k_base = 0; k_base < k; k_base += kMmaK) {
        hgemm_copy_half8_aligned(
            a + (block_row + load_a_row) * k + k_base + load_a_col,
            &shared_a[load_a_row][load_a_col]
        );
        hgemm_copy_half8_aligned(
            b + (k_base + load_b_row) * n + block_col + load_b_col,
            &shared_b[load_b_row][load_b_col]
        );

        __syncthreads();

        std::uint32_t ra[kWarpTileM][4];
        std::uint32_t rb[kWarpTileN][2];

#pragma unroll
        for(int mma_m = 0; mma_m < kWarpTileM; ++mma_m) {
            const int warp_smem_a_row = warp_m * (kMmaM * kWarpTileM) + mma_m * kMmaM;
            const int lane_smem_a_row = warp_smem_a_row + lane % 16;
            const int lane_smem_a_col = (lane / 16) * 8;
            const auto a_addr = static_cast<std::uint32_t>(
                __cvta_generic_to_shared(&shared_a[lane_smem_a_row][lane_smem_a_col])
            );
            hgemm_ldmatrix_x4(ra[mma_m][0], ra[mma_m][1], ra[mma_m][2], ra[mma_m][3], a_addr);
        }

#pragma unroll
        for(int mma_n = 0; mma_n < kWarpTileN; ++mma_n) {
            const int warp_smem_b_col = warp_n * (kMmaN * kWarpTileN) + mma_n * kMmaN;
            const int lane_smem_b_row = lane % 16;
            const auto b_addr = static_cast<std::uint32_t>(
                __cvta_generic_to_shared(&shared_b[lane_smem_b_row][warp_smem_b_col])
            );
            hgemm_ldmatrix_x2_trans(rb[mma_n][0], rb[mma_n][1], b_addr);
        }

#pragma unroll
        for(int mma_m = 0; mma_m < kWarpTileM; ++mma_m) {
#pragma unroll
            for(int mma_n = 0; mma_n < kWarpTileN; ++mma_n) {
                hgemm_mma_m16n8k16_f16(
                    rc[mma_m][mma_n][0],
                    rc[mma_m][mma_n][1],
                    ra[mma_m][0],
                    ra[mma_m][1],
                    ra[mma_m][2],
                    ra[mma_m][3],
                    rb[mma_n][0],
                    rb[mma_n][1]
                );
            }
        }

        __syncthreads();
    }

#pragma unroll
    for(int mma_m = 0; mma_m < kWarpTileM; ++mma_m) {
#pragma unroll
        for(int mma_n = 0; mma_n < kWarpTileN; ++mma_n) {
            const int fragment_row = block_row + warp_m * (kMmaM * kWarpTileM) + mma_m * kMmaM;
            const int fragment_col = block_col + warp_n * (kMmaN * kWarpTileN) + mma_n * kMmaN;
            const int store_row0 = fragment_row + lane / 4;
            const int store_row1 = store_row0 + 8;
            const int store_col = fragment_col + (lane % 4) * 2;
            hgemm_store_mma_m16n8k16_f16x2_aligned(c, store_row0, store_col, n, rc[mma_m][mma_n][0]);
            hgemm_store_mma_m16n8k16_f16x2_aligned(c, store_row1, store_col, n, rc[mma_m][mma_n][1]);
        }
    }
}

}  // namespace

__global__ void hgemm_mma_m16n8k16_naive_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_mma_m16n8k16_naive_body(a, b, c, m, n, k);
}

__global__ void __launch_bounds__(256) hgemm_mma_m16n8k16_mma2x4_warp4x4_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_mma_m16n8k16_mma2x4_warp4x4_body(a, b, c, m, n, k);
}

__global__ void hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    // Placeholder staged entry: keep a correct one-warp m16n8k16 body until
    // the staged PTX-MMA implementation is filled in.
    hgemm_mma_m16n8k16_naive_body(a, b, c, m, n, k);
}

__global__ void hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_mma_m16n8k16_naive_body(a, b, c, m, n, k);
}

__global__ void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_mma_m16n8k16_naive_body(a, b, c, m, n, k);
}

__global__ void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_mma_m16n8k16_naive_body(a, b, c, m, n, k);
}

__global__ void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_mma_m16n8k16_naive_body(a, b, c, m, n, k);
}

__global__ void hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_mma_m16n8k16_naive_body(a, b, c, m, n, k);
}

__global__ void hgemm_mma_stages_block_swizzle_tn_cute_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_mma_m16n8k16_naive_body(a, b, c, m, n, k);
}

__global__ void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_swizzle_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_mma_m16n8k16_naive_body(a, b, c, m, n, k);
}

__global__ void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x4_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_mma_m16n8k16_naive_body(a, b, c, m, n, k);
}

}  // namespace ai_system::labs::hgemm
