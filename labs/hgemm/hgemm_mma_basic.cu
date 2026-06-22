#include "hgemm_lab.hpp"

#include "ai_system/cuda/runtime.hpp"
#include "ai_system/profiling/nvtx.hpp"

#include <cuda_fp16.h>

#include <cstdint>
#include <limits>
#include <string>

namespace ai_system::labs::hgemm {
namespace {

constexpr int kWarpSize = 32;

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

__host__ __device__ inline int div_ceil(int lhs, int rhs) {
    return (lhs + rhs - 1) / rhs;
}

dim3 hgemm_mma_stage_grid(int m, int n, int block_m, int block_n, bool swizzle, int swizzle_stride) {
    const auto grid_y = static_cast<unsigned int>((m + block_m - 1) / block_m);
    const auto n_tiles = static_cast<unsigned int>((n + block_n - 1) / block_n);
    if(!swizzle) {
        return dim3(n_tiles, grid_y);
    }

    const auto n_swizzle = static_cast<unsigned int>((n + swizzle_stride - 1) / swizzle_stride);
    return dim3((n_tiles + n_swizzle - 1) / n_swizzle, grid_y, n_swizzle);
}

#define WARP_SIZE 32
#define LDST32BITS(value) (reinterpret_cast<half2*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
#define CP_ASYNC_CG(dst, src, bytes) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
#define LDMATRIX_X2(R0, R1, addr) \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))
#define LDMATRIX_X4(R0, R1, R2, R3, addr) \
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(addr))
#define LDMATRIX_X2_T(R0, R1, addr) \
    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))
#define LDMATRIX_X4_T(R0, R1, R2, R3, addr) \
    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(addr))
#define STMATRIX_X2(addr, R0, R1) \
    asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n" :: "r"(addr), "r"(R0), "r"(R1))
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1) \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" : "=r"(RD0), "=r"(RD1) : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))

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

template <int BlockM, int BlockN>
__device__ void hgemm_scalar_tile_body_at(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k,
    int block_row,
    int block_col
) {
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

template <int BlockM, int BlockN>
__device__ void hgemm_scalar_tile_tn_body_at(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k,
    int block_row,
    int block_col
) {
    const int tid = static_cast<int>(threadIdx.z) * blockDim.y * blockDim.x +
        static_cast<int>(threadIdx.y) * blockDim.x + threadIdx.x;
    const int thread_count = static_cast<int>(blockDim.x * blockDim.y * blockDim.z);

    for(int index = tid; index < BlockM * BlockN; index += thread_count) {
        const int row = block_row + index / BlockN;
        const int col = block_col + index % BlockN;
        if(row < m && col < n) {
            float accumulator = 0.0f;
            for(int inner = 0; inner < k; ++inner) {
                accumulator += __half2float(a[row * k + inner]) * __half2float(b[col * k + inner]);
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


// 128x128, mma2x4, warp4x4(64,32,16), stages, block swizzle
template<const int MMA_M=16,
         const int MMA_N=8,
         const int MMA_K=16,
         const int MMA_TILE_M=2,
         const int MMA_TILE_N=4,
         const int WARP_TILE_M=4,
         const int WARP_TILE_N=4,
         const int A_PAD=0,
         const int B_PAD=0,
         const int K_STAGE=2,
         const bool BLOCK_SWIZZLE=true>
__global__ void  __launch_bounds__(256)
hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_kernel(
  const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C, int M, int N, int K) {
  // BLOCK_SWIZZLE 0/1 control use block swizzle or not.
  const int bx = ((int) BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, MMA_K);
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M; // 16*2*4=128
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N; // 8*4*4=128
  constexpr int BK = MMA_K; // 16
  constexpr int kStageK = MMA_K;

  __shared__ half s_a[K_STAGE][BM][BK+A_PAD]; // 128*16*2=4KB
  __shared__ half s_b[K_STAGE][BK][BN+B_PAD]; // 16*128*2=4KB, 16*(128+16)*2=4.5KB
  constexpr int s_a_stage_offset = BM * (BK + A_PAD);
  constexpr int s_b_stage_offset = BK * (BN + B_PAD);

  const int tid = threadIdx.y * blockDim.x + threadIdx.x; // within block
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_m = warp_id % 2; // 0,1
  const int warp_n = warp_id / 2; // 0,1,2,3

  int load_smem_a_m = tid / 2; // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
  int load_smem_b_k = tid / 16; // row 0~15
  int load_smem_b_n = (tid % 16) * 8; // col 0,8,...,120
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
  if (by * BM + BM > M || bx * BN + BN > N || (K % kStageK) != 0 || (N % 8) != 0 || NUM_K_TILES < K_STAGE) {
    hgemm_scalar_tile_body_at<BM, BN>(A, B, C, M, N, K, by * BM, bx * BN);
    return;
  }

  uint32_t RC[WARP_TILE_M][WARP_TILE_N][2];
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      RC[i][j][0] = 0;
      RC[i][j][1] = 0;
    }
  }

  uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
  uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

  #pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) { // 0, 1
    // k * WMMA_K, WMMA_K=16 -> (k << 4)
    int load_gmem_a_k = k * BK + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * BK + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 

    uint32_t load_smem_a_ptr = (
      smem_a_base_ptr + (k * s_a_stage_offset + 
                         load_smem_a_m * (BK + A_PAD) + 
                         load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr = (
      smem_b_base_ptr + (k * s_b_stage_offset + 
                         load_smem_b_k * (BN + B_PAD) + 
                         load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(K_STAGE-2); // s2->0, s3->1, s4->2
  __syncthreads(); 

  #pragma unroll
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; ++k) {
    // gmem -> smem
    // s2/4 can use bitwise ops but s3 can not, so, we use mod
    // ops for all stages kernel. s2: (k + 1)&1, s4: (k + 1)&3
    // s3: (k + 1) % 3
    int smem_sel = (k + 1) % K_STAGE; // s3 k 2->0, k 3->1, k 4->2...
    int smem_sel_next = k % K_STAGE;  // s3 k 2->2, k 3->0, k 4->1...

    int load_gmem_a_k = k * BK + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * BK + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 

    uint32_t load_smem_a_ptr = (
      smem_a_base_ptr + (smem_sel_next * s_a_stage_offset + 
                         load_smem_a_m * (BK + A_PAD) + 
                         load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr = (
      smem_b_base_ptr + (smem_sel_next * s_b_stage_offset + 
                         load_smem_b_k * (BN + B_PAD) + 
                         load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    CP_ASYNC_COMMIT_GROUP();
    
    // ldmatrix for s_a, ldmatrix.trans for s_b.
    uint32_t RA[WARP_TILE_M][4];
    uint32_t RB[WARP_TILE_N][2];

    // smem -> reg
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
      int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
      uint32_t lane_smem_a_ptr = __cvta_generic_to_shared(
        &s_a[smem_sel][lane_smem_a_m][lane_smem_a_k]);
      LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], lane_smem_a_ptr);
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_k = lane_id % 16;  // 0~15
      int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
      uint32_t lane_smem_b_ptr = __cvta_generic_to_shared(
        &s_b[smem_sel][lane_smem_b_k][lane_smem_b_n]);
      LDMATRIX_X2_T(RB[j][0], RB[j][1], lane_smem_b_ptr);
    }
    
    // MMA compute
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        HMMA16816(RC[i][j][0], RC[i][j][1], 
                  RA[i][0], RA[i][1], RA[i][2], RA[i][3], 
                  RB[j][0], RB[j][1], 
                  RC[i][j][0], RC[i][j][1]);
      }
    }

    CP_ASYNC_WAIT_GROUP(K_STAGE-2);
    __syncthreads(); 
  }

  // make sure all memory issues ready.
  if constexpr ((K_STAGE - 2) > 0) {
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads(); 
  }

  // processing last (K_STAGE-1) k iters.
  {
    #pragma unroll
    for (int k = 0; k < (K_STAGE - 1); k++) {
      int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
      // ldmatrix for s_a, ldmatrix.trans for s_b.
      uint32_t RA[WARP_TILE_M][4];
      uint32_t RB[WARP_TILE_N][2];

      // smem -> reg
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
        int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
        int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
        uint32_t lane_smem_a_ptr = __cvta_generic_to_shared(
          &s_a[stage_sel][lane_smem_a_m][lane_smem_a_k]);
        LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], lane_smem_a_ptr);
      }

      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
        int lane_smem_b_k = lane_id % 16;  // 0~15
        int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
        uint32_t lane_smem_b_ptr = __cvta_generic_to_shared(
          &s_b[stage_sel][lane_smem_b_k][lane_smem_b_n]);
        LDMATRIX_X2_T(RB[j][0], RB[j][1], lane_smem_b_ptr);
      }

      // MMA compute
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          HMMA16816(RC[i][j][0], RC[i][j][1], 
                    RA[i][0], RA[i][1], RA[i][2], RA[i][3], 
                    RB[j][0], RB[j][1], 
                    RC[i][j][0], RC[i][j][1]);
        }
      }
    }
  }

  // reg -> gmem, MMA_MxMMA_N=16x8
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int store_warp_smem_c_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int store_warp_smem_c_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      // mapping lane smem index -> global index.
      // [16][8], https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
      // #matrix-fragments-for-mma-m16n8k16-with-floating-point-type
      // [0~7][0~3 u32 -> 0~7 f16], [8~15][0~3 u32 -> 0~7 f16]
      int store_lane_gmem_c_m = by * BM + store_warp_smem_c_m + lane_id / 4;
      int store_lane_gmem_c_n = bx * BN + store_warp_smem_c_n + (lane_id % 4) * 2;
      int store_gmem_c_addr_0 = store_lane_gmem_c_m * N + store_lane_gmem_c_n;
      int store_gmem_c_addr_1 = (store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n;
      LDST32BITS(C[store_gmem_c_addr_0]) = LDST32BITS(RC[i][j][0]); 
      LDST32BITS(C[store_gmem_c_addr_1]) = LDST32BITS(RC[i][j][1]); 
    }
  }
}

// 128x128, mma2x4, warp4x4(64,32,16), stages, block swizzle, dsmem
template<const int MMA_M=16, 
         const int MMA_N=8, 
         const int MMA_K=16,
         const int MMA_TILE_M=2,
         const int MMA_TILE_N=4,
         const int WARP_TILE_M=4,
         const int WARP_TILE_N=4,
         const int A_PAD=0, 
         const int B_PAD=0,
         const int K_STAGE=2, 
         const bool BLOCK_SWIZZLE=true,
         const bool COLLECTIVE_STORE=false>
__global__ void  __launch_bounds__(256) 
hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_kernel(
  const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C, int M, int N, int K) {
  // BLOCK_SWIZZLE 0/1 control use block swizzle or not.
  // COLLECTIVE_STORE true/false control use stmatrix or not.
  const int bx = ((int) BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, MMA_K);
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M; // 16*2*4=128
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N; // 8*4*4=128
  constexpr int BK = MMA_K; // 16
  constexpr int kStageK = MMA_K;

  extern __shared__ half smem[]; 
  half* s_a = smem;
  half* s_b = smem + K_STAGE * BM * (BK + A_PAD);
  constexpr int s_a_stage_offset = BM * (BK + A_PAD);
  constexpr int s_b_stage_offset = BK * (BN + B_PAD);

  const int tid = threadIdx.y * blockDim.x + threadIdx.x; // within block
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_m = warp_id % 2; // 0,1
  const int warp_n = warp_id / 2; // 0,1,2,3

  int load_smem_a_m = tid / 2; // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
  int load_smem_b_k = tid / 16; // row 0~15
  int load_smem_b_n = (tid % 16) * 8; // col 0,8,...,120
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
  if (by * BM + BM > M || bx * BN + BN > N || (K % kStageK) != 0 || (N % 8) != 0 || NUM_K_TILES < K_STAGE) {
    hgemm_scalar_tile_body_at<BM, BN>(A, B, C, M, N, K, by * BM, bx * BN);
    return;
  }

  uint32_t RC[WARP_TILE_M][WARP_TILE_N][2];
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      RC[i][j][0] = 0;
      RC[i][j][1] = 0;
    }
  }
  
  uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
  uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

  #pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) { // 0, 1
    // k * WMMA_K, WMMA_K=16 -> (k << 4)
    int load_gmem_a_k = k * BK + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * BK + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 

    uint32_t load_smem_a_ptr = (
      smem_a_base_ptr + (k * s_a_stage_offset + 
                         load_smem_a_m * (BK + A_PAD) + 
                         load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr = (
      smem_b_base_ptr + (k * s_b_stage_offset + 
                         load_smem_b_k * (BN + B_PAD) + 
                         load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(K_STAGE-2); // s2->0, s3->1, s4->2
  __syncthreads(); 

  #pragma unroll
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; ++k) {
    // gmem -> smem
    // s2/4 can use bitwise ops but s3 can not, so, we use mod
    // ops for all stages kernel. s2: (k + 1)&1, s4: (k + 1)&3
    // s3: (k + 1) % 3
    int smem_sel = (k + 1) % K_STAGE; // s3 k 2->0, k 3->1, k 4->2...
    int smem_sel_next = k % K_STAGE;  // s3 k 2->2, k 3->0, k 4->1...

    int load_gmem_a_k = k * BK + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * BK + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 

    uint32_t load_smem_a_ptr = (
      smem_a_base_ptr + (smem_sel_next * s_a_stage_offset + 
                         load_smem_a_m * (BK + A_PAD) + 
                         load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr = (
      smem_b_base_ptr + (smem_sel_next * s_b_stage_offset + 
                         load_smem_b_k * (BN + B_PAD) + 
                         load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    CP_ASYNC_COMMIT_GROUP();
    
    uint32_t RA[WARP_TILE_M][4];
    uint32_t RB[WARP_TILE_N][2];
    // ldmatrix for s_a, ldmatrix.trans for s_b.
    // smem -> reg
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
      int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
      uint32_t lane_smem_a_ptr = (
        smem_a_base_ptr + (smem_sel * s_a_stage_offset + 
                           lane_smem_a_m * (BK + A_PAD) + 
                           lane_smem_a_k) * sizeof(half)
      );
      LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], lane_smem_a_ptr);
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_k = lane_id % 16;  // 0~15
      int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
      uint32_t lane_smem_b_ptr = (
        smem_b_base_ptr + (smem_sel * s_b_stage_offset + 
                           lane_smem_b_k * (BN + B_PAD) + 
                           lane_smem_b_n) * sizeof(half)
      );
      LDMATRIX_X2_T(RB[j][0], RB[j][1], lane_smem_b_ptr);
    }
    
    // MMA compute
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        HMMA16816(RC[i][j][0], RC[i][j][1], 
                  RA[i][0], RA[i][1], RA[i][2], RA[i][3], 
                  RB[j][0], RB[j][1], 
                  RC[i][j][0], RC[i][j][1]);
      }
    }

    CP_ASYNC_WAIT_GROUP(K_STAGE-2);
    __syncthreads(); 
  }

  // make sure all memory issues ready.
  if constexpr ((K_STAGE - 2) > 0) {
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads(); 
  }
  
  // processing last (K_STAGE-1) k iters.
  {
    #pragma unroll
    for (int k = 0; k < (K_STAGE - 1); k++) {
      uint32_t RA[WARP_TILE_M][4];
      uint32_t RB[WARP_TILE_N][2];

      int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
      // ldmatrix for s_a, ldmatrix.trans for s_b.
      // smem -> reg
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
        int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
        int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
        uint32_t lane_smem_a_ptr = (
          smem_a_base_ptr + (stage_sel * s_a_stage_offset + 
                             lane_smem_a_m * (BK + A_PAD) + 
                             lane_smem_a_k) * sizeof(half)
        );
        LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], lane_smem_a_ptr);
      }

      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
        int lane_smem_b_k = lane_id % 16;  // 0~15
        int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
        uint32_t lane_smem_b_ptr = (
          smem_b_base_ptr + (stage_sel * s_b_stage_offset + 
                             lane_smem_b_k * (BN + B_PAD) + 
                             lane_smem_b_n) * sizeof(half)
        );
        LDMATRIX_X2_T(RB[j][0], RB[j][1], lane_smem_b_ptr);
      }

      // MMA compute
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          HMMA16816(RC[i][j][0], RC[i][j][1], 
                    RA[i][0], RA[i][1], RA[i][2], RA[i][3], 
                    RB[j][0], RB[j][1], 
                    RC[i][j][0], RC[i][j][1]);
        }
      }
    }
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 90)
  if (COLLECTIVE_STORE) {
    // The following code has not been tested because I do not have a GPU with sm>=90
    // reg -> smem(stmatrix) -> gmem(cp.async.bulk), MMA_MxMMA_N=16x8
    // NOTE: need [MMA_M][MMA_N] per warp to avoid overlap between warps.
    __shared__ half s_c[MMA_TILE_M][MMA_TILE_N][MMA_M][MMA_N]; // (2*4)*16*8*2=2KB
    uint32_t smem_c_base_ptr = __cvta_generic_to_shared(&s_c[warp_m][warp_n]);
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // store (i,j) warp tile -> smem c, 16x8
        uint32_t lane_smem_c_ptr = (
          smem_c_base_ptr + (lane_id % 16) * MMA_N * sizeof(half)); // (0~15)*8
        STMATRIX_X2(lane_smem_c_ptr, RC[i][j][0], RC[i][j][1]);
        // smem -> gmem, may use cp.async.bulk.global.share::cta?
        int store_warp_gmem_c_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
        int store_warp_gmem_c_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
        int store_lane_gmem_c_m = by * BM + store_warp_gmem_c_m;
        int store_lane_gmem_c_n = bx * BN + store_warp_gmem_c_n;
        // send 16 memory issues with 128 bits within lower half lanes.
        // TODO: use cp.async.bulk and wait outside the inner loop.
        if (lane_id < 16) {
          int store_gmem_c_addr = (store_lane_gmem_c_m + lane_id) * N + store_lane_gmem_c_n;
          LDST128BITS(C[store_gmem_c_addr]) = LDST128BITS(
            s_c[warp_m][warp_n][lane_id][0]);
        }
        __syncwarp();
      }
    }
  } else {
    // reg -> gmem, MMA_MxMMA_N=16x8
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        int store_warp_smem_c_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
        int store_warp_smem_c_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
        int store_lane_gmem_c_m = by * BM + store_warp_smem_c_m + lane_id / 4;
        int store_lane_gmem_c_n = bx * BN + store_warp_smem_c_n + (lane_id % 4) * 2;
        int store_gmem_c_addr_0 = store_lane_gmem_c_m * N + store_lane_gmem_c_n;
        int store_gmem_c_addr_1 = (store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n;
        LDST32BITS(C[store_gmem_c_addr_0]) = LDST32BITS(RC[i][j][0]); 
        LDST32BITS(C[store_gmem_c_addr_1]) = LDST32BITS(RC[i][j][1]); 
      }
    }
  }
#else 
// #warning "stmatrix need sm>=90, force use __shfl_sync for collective store!"
  {
    for (int i = 0; i < WARP_TILE_M; ++i) {
      // How to use LDST128BITS here? __shfl_sync -> lane 0 -> store 8 half.
      // thus, we only need 8 memory issues with 128 bits after shfl_sync.
      // may reuse RA[4][4] as RC0 ? only new RC1[4][4].
      uint32_t RC0[WARP_TILE_N][4];
      uint32_t RC1[WARP_TILE_N][4];
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // How to use LDST128BITS here? __shfl_sync -> lane 0 -> store 8 half.
        // thus, we only need 8 memory issues with 128 bits after shfl_sync.
        RC0[j][0] = RC[i][j][0];
        RC1[j][0] = RC[i][j][1];
        RC0[j][1] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 1);
        RC0[j][2] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 2);
        RC0[j][3] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 3);
        RC1[j][1] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 1);
        RC1[j][2] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 2);
        RC1[j][3] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 3);
      }

      if (lane_id % 4 == 0) {
        int store_warp_smem_c_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
        int store_lane_gmem_c_m = by * BM + store_warp_smem_c_m + lane_id / 4;
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          int store_warp_smem_c_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
          int store_lane_gmem_c_n = bx * BN + store_warp_smem_c_n;
          int store_gmem_c_addr_0 = store_lane_gmem_c_m * N + store_lane_gmem_c_n;
          int store_gmem_c_addr_1 = (store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n;
          LDST128BITS(C[store_gmem_c_addr_0]) = LDST128BITS(RC0[j][0]); 
          LDST128BITS(C[store_gmem_c_addr_1]) = LDST128BITS(RC1[j][0]); 
        }
      }
    }
  }
#endif
}

// TN variant from the reference implementation:
//   A is row-major [M][K].
//   B is physically row-major [N][K], i.e. the logical KxN matrix is already
//   transposed in global memory.
//   C is row-major [M][N].
// Keeping B as [BN][BK] in shared memory lets the warp use ldmatrix.x2 instead
// of ldmatrix.x2.trans for the right operand.
template<const int MMA_M=16,
         const int MMA_N=8,
         const int MMA_K=16,
         const int MMA_TILE_M=2,
         const int MMA_TILE_N=4,
         const int WARP_TILE_M=4,
         const int WARP_TILE_N=4,
         const int A_PAD=0,
         const int B_PAD=0,
         const int K_STAGE=2,
         const bool BLOCK_SWIZZLE=false>
__global__ void __launch_bounds__(256)
hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M,
    int N,
    int K
) {
    const int bx = static_cast<int>(BLOCK_SWIZZLE) * static_cast<int>(blockIdx.z) * static_cast<int>(gridDim.x) +
        static_cast<int>(blockIdx.x);
    const int by = static_cast<int>(blockIdx.y);
    const int num_k_tiles = div_ceil(K, MMA_K);
    constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M;
    constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N;
    constexpr int BK = MMA_K;
    constexpr int kStageK = MMA_K;

    extern __shared__ half smem[];
    half* s_a = smem;
    half* s_b = smem + K_STAGE * BM * (BK + A_PAD);
    constexpr int s_a_stage_offset = BM * (BK + A_PAD);
    constexpr int s_b_stage_offset = BN * (BK + B_PAD);

    const int tid = static_cast<int>(threadIdx.y) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warp_m = warp_id % 2;
    const int warp_n = warp_id / 2;

    // Per stage, each of the 256 threads moves:
    //   A: one contiguous half8 from A[block_m + tid/2][k + {0,8}]
    //   B: one contiguous half8 from B[block_n + tid/2][k + {0,8}]
    // This exactly covers A 128x16 and B 128x16.  B's row is an output-column
    // index because B is stored as [N][K] for the TN path.
    const int load_smem_a_m = tid / 2;
    const int load_smem_a_k = (tid & 1) * 8;
    const int load_smem_b_n = tid / 2;
    const int load_smem_b_k = (tid & 1) * 8;
    const int load_gmem_a_m = by * BM + load_smem_a_m;
    const int load_gmem_b_n = bx * BN + load_smem_b_n;

    if(by * BM + BM > M || bx * BN + BN > N || (K % kStageK) != 0 || num_k_tiles < K_STAGE) {
        hgemm_scalar_tile_tn_body_at<BM, BN>(A, B, C, M, N, K, by * BM, bx * BN);
        return;
    }

    uint32_t RC[WARP_TILE_M][WARP_TILE_N][2];
#pragma unroll
    for(int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for(int j = 0; j < WARP_TILE_N; ++j) {
            RC[i][j][0] = 0;
            RC[i][j][1] = 0;
        }
    }

    const auto smem_a_base_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(s_a));
    const auto smem_b_base_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(s_b));

#pragma unroll
    for(int k_tile = 0; k_tile < K_STAGE - 1; ++k_tile) {
        const int load_gmem_a_k = k_tile * BK + load_smem_a_k;
        const int load_gmem_b_k = k_tile * BK + load_smem_b_k;
        const uint32_t load_smem_a_ptr = smem_a_base_ptr +
            (k_tile * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) * sizeof(half);
        const uint32_t load_smem_b_ptr = smem_b_base_ptr +
            (k_tile * s_b_stage_offset + load_smem_b_n * (BK + B_PAD) + load_smem_b_k) * sizeof(half);
        CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_m * K + load_gmem_a_k], 16);
        CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_n * K + load_gmem_b_k], 16);
        CP_ASYNC_COMMIT_GROUP();
    }

    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();

#pragma unroll
    for(int k_tile = K_STAGE - 1; k_tile < num_k_tiles; ++k_tile) {
        const int smem_sel = (k_tile + 1) % K_STAGE;
        const int smem_sel_next = k_tile % K_STAGE;
        const int load_gmem_a_k = k_tile * BK + load_smem_a_k;
        const int load_gmem_b_k = k_tile * BK + load_smem_b_k;
        const uint32_t load_smem_a_ptr = smem_a_base_ptr +
            (smem_sel_next * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) *
                sizeof(half);
        const uint32_t load_smem_b_ptr = smem_b_base_ptr +
            (smem_sel_next * s_b_stage_offset + load_smem_b_n * (BK + B_PAD) + load_smem_b_k) *
                sizeof(half);
        CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_m * K + load_gmem_a_k], 16);
        CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_n * K + load_gmem_b_k], 16);
        CP_ASYNC_COMMIT_GROUP();

        uint32_t RA[WARP_TILE_M][4];
        uint32_t RB[WARP_TILE_N][2];

#pragma unroll
        for(int i = 0; i < WARP_TILE_M; ++i) {
            const int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
            const int lane_smem_a_m = warp_smem_a_m + lane_id % 16;
            const int lane_smem_a_k = (lane_id / 16) * 8;
            const uint32_t lane_smem_a_ptr = smem_a_base_ptr +
                (smem_sel * s_a_stage_offset + lane_smem_a_m * (BK + A_PAD) + lane_smem_a_k) *
                    sizeof(half);
            LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], lane_smem_a_ptr);
        }

#pragma unroll
        for(int j = 0; j < WARP_TILE_N; ++j) {
            const int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
            const int lane_smem_b_n = warp_smem_b_n + lane_id % 8;
            const int lane_smem_b_k = ((lane_id / 8) & 1) * 8;
            const uint32_t lane_smem_b_ptr = smem_b_base_ptr +
                (smem_sel * s_b_stage_offset + lane_smem_b_n * (BK + B_PAD) + lane_smem_b_k) *
                    sizeof(half);
            LDMATRIX_X2(RB[j][0], RB[j][1], lane_smem_b_ptr);
        }

#pragma unroll
        for(int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
            for(int j = 0; j < WARP_TILE_N; ++j) {
                HMMA16816(
                    RC[i][j][0],
                    RC[i][j][1],
                    RA[i][0],
                    RA[i][1],
                    RA[i][2],
                    RA[i][3],
                    RB[j][0],
                    RB[j][1],
                    RC[i][j][0],
                    RC[i][j][1]
                );
            }
        }

        CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
        __syncthreads();
    }

    if constexpr((K_STAGE - 2) > 0) {
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

#pragma unroll
    for(int k_tail = 0; k_tail < K_STAGE - 1; ++k_tail) {
        const int stage_sel = (num_k_tiles - (K_STAGE - 1) + k_tail) % K_STAGE;
        uint32_t RA[WARP_TILE_M][4];
        uint32_t RB[WARP_TILE_N][2];

#pragma unroll
        for(int i = 0; i < WARP_TILE_M; ++i) {
            const int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
            const int lane_smem_a_m = warp_smem_a_m + lane_id % 16;
            const int lane_smem_a_k = (lane_id / 16) * 8;
            const uint32_t lane_smem_a_ptr = smem_a_base_ptr +
                (stage_sel * s_a_stage_offset + lane_smem_a_m * (BK + A_PAD) + lane_smem_a_k) *
                    sizeof(half);
            LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], lane_smem_a_ptr);
        }

#pragma unroll
        for(int j = 0; j < WARP_TILE_N; ++j) {
            const int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
            const int lane_smem_b_n = warp_smem_b_n + lane_id % 8;
            const int lane_smem_b_k = ((lane_id / 8) & 1) * 8;
            const uint32_t lane_smem_b_ptr = smem_b_base_ptr +
                (stage_sel * s_b_stage_offset + lane_smem_b_n * (BK + B_PAD) + lane_smem_b_k) *
                    sizeof(half);
            LDMATRIX_X2(RB[j][0], RB[j][1], lane_smem_b_ptr);
        }

#pragma unroll
        for(int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
            for(int j = 0; j < WARP_TILE_N; ++j) {
                HMMA16816(
                    RC[i][j][0],
                    RC[i][j][1],
                    RA[i][0],
                    RA[i][1],
                    RA[i][2],
                    RA[i][3],
                    RB[j][0],
                    RB[j][1],
                    RC[i][j][0],
                    RC[i][j][1]
                );
            }
        }
    }

#pragma unroll
    for(int i = 0; i < WARP_TILE_M; ++i) {
        uint32_t RC0[WARP_TILE_N][4];
        uint32_t RC1[WARP_TILE_N][4];
#pragma unroll
        for(int j = 0; j < WARP_TILE_N; ++j) {
            RC0[j][0] = RC[i][j][0];
            RC1[j][0] = RC[i][j][1];
            RC0[j][1] = __shfl_sync(0xffffffff, RC[i][j][0], lane_id + 1);
            RC0[j][2] = __shfl_sync(0xffffffff, RC[i][j][0], lane_id + 2);
            RC0[j][3] = __shfl_sync(0xffffffff, RC[i][j][0], lane_id + 3);
            RC1[j][1] = __shfl_sync(0xffffffff, RC[i][j][1], lane_id + 1);
            RC1[j][2] = __shfl_sync(0xffffffff, RC[i][j][1], lane_id + 2);
            RC1[j][3] = __shfl_sync(0xffffffff, RC[i][j][1], lane_id + 3);
        }

        if((lane_id & 3) == 0) {
            const int store_warp_c_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
            const int store_c_m = by * BM + store_warp_c_m + lane_id / 4;
#pragma unroll
            for(int j = 0; j < WARP_TILE_N; ++j) {
                const int store_warp_c_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
                const int store_c_n = bx * BN + store_warp_c_n;
                LDST128BITS(C[store_c_m * N + store_c_n]) = LDST128BITS(RC0[j][0]);
                LDST128BITS(C[(store_c_m + 8) * N + store_c_n]) = LDST128BITS(RC1[j][0]);
            }
        }
    }
}

// In order to reduce bank conflicts, we will save the K(16x2=32) 
// dimension by half according to the stage dimension. For example, 
// stages=3, warp_tile_k=2, it will be saved as [3*2][BM][16].
// 128x128, mma2x4, warp4x4(64,32,32), stages, block swizzle, dsmem, 
// k32 with reg double buffers
template<const int MMA_M=16, 
         const int MMA_N=8, 
         const int MMA_K=16,
         const int MMA_TILE_M=2,
         const int MMA_TILE_N=4,
         const int WARP_TILE_M=4,
         const int WARP_TILE_N=4,
         const int WARP_TILE_K=2,
         const int A_PAD=0, 
         const int B_PAD=0,
         const int K_STAGE=2, 
         const bool BLOCK_SWIZZLE=true,
         const bool WARP_SWIZZLE=true>
__global__ void  __launch_bounds__(256) 
hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_kernel(
  const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C, 
  int M, int N, int K) {
  // BLOCK_SWIZZLE 0/1 control use block swizzle or not.
  const int bx = ((int) BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, MMA_K * WARP_TILE_K);
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M; // 16*2*4=128
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N; // 8*4*4=128
  constexpr int BK = MMA_K; // 16x2=32
  constexpr int kStageK = MMA_K * WARP_TILE_K;

  extern __shared__ half smem[]; 
  half* s_a = smem;
  half* s_b = smem + K_STAGE * BM * (BK + A_PAD) * WARP_TILE_K;
  constexpr int s_a_stage_offset = BM * (BK + A_PAD); // 128x16 
  constexpr int s_b_stage_offset = BK * (BN + B_PAD); // 16x128
  constexpr int s_a_mma_k_store_offset = K_STAGE * BM * (BK + A_PAD);
  constexpr int s_b_mma_k_store_offset = K_STAGE * BK * (BN + B_PAD);

  const int tid = threadIdx.y * blockDim.x + threadIdx.x; // within block
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_m = warp_id % 2; // 0,1
  const int warp_n = warp_id / 2; // 0,1,2,3

  int load_smem_a_m = tid / 2; // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
  int load_smem_b_k = tid / 16; // row 0~15
  int load_smem_b_n = (tid % 16) * 8; // col 0,8,16,...
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
  if (by * BM + BM > M || bx * BN + BN > N || (K % kStageK) != 0 || (N % 8) != 0 || NUM_K_TILES < K_STAGE) {
    hgemm_scalar_tile_body_at<BM, BN>(A, B, C, M, N, K, by * BM, bx * BN);
    return;
  }

  uint32_t RC[WARP_TILE_M][WARP_TILE_N][2];
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      RC[i][j][0] = 0;
      RC[i][j][1] = 0;
    }
  }

  uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
  uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

  #pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) { // 0, 1
    // k * WMMA_K, WMMA_K=16 -> (k << 4)
    int load_gmem_a_k = k * BK * WARP_TILE_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * BK * WARP_TILE_K + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 

    uint32_t load_smem_a_ptr = (
      smem_a_base_ptr + (k * s_a_stage_offset + 
                         load_smem_a_m * (BK + A_PAD) + 
                         load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16); // MMA_K 0
    uint32_t load_smem_a_mma_k_ptr = (
      smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) + 
      (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + 
      load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_mma_k_ptr, &A[load_gmem_a_addr + 16], 16); // MMA_K 1

    uint32_t load_smem_b_ptr = (
      smem_b_base_ptr + (k * s_b_stage_offset + 
                         load_smem_b_k * (BN + B_PAD) + 
                         load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    
    int load_gmem_b_k_mma_k = k * BK * WARP_TILE_K + MMA_K + load_smem_b_k;
    int load_gmem_b_addr_mma_k = load_gmem_b_k_mma_k * N + load_gmem_b_n; 
    uint32_t load_smem_b_mma_k_ptr = (
      smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) + 
      (k * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + 
      load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_mma_k_ptr, &B[load_gmem_b_addr_mma_k], 16);

    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(K_STAGE-2); // s2->0, s3->1, s4->2
  __syncthreads(); 

  uint32_t RA[2][WARP_TILE_M][4];
  uint32_t RB[2][WARP_TILE_N][2];

  int reg_store_idx = 0;
  int reg_load_idx = 1;

  { 
    // ldmatrix for s_a, ldmatrix.trans for s_b.
    // smem -> reg buffers 0, first MMA_K, 0~15
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
      int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
      uint32_t lane_smem_a_ptr = (
        smem_a_base_ptr + 
        (0 * s_a_stage_offset + lane_smem_a_m * (BK + A_PAD) + 
        lane_smem_a_k) * sizeof(half)
      );
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                  lane_smem_a_ptr);
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_k = lane_id % 16;  // 0~15, 0~15
      int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
      uint32_t lane_smem_b_ptr = (
        smem_b_base_ptr + 
        (0 * s_b_stage_offset + lane_smem_b_k * (BN + B_PAD) + 
        lane_smem_b_n) * sizeof(half)
      );
      // may use .x4.trans to load 4 matrix for reg double buffers at once?
      LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                    lane_smem_b_ptr);
    }
  }
  
  #pragma unroll
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; ++k) {
    reg_store_idx ^= 1; // 0->1
    reg_load_idx ^= 1; // 1->0
    int smem_sel = (k + 1) % K_STAGE; // s3 k 2->0, k 3->1, k 4->2...
    int smem_sel_next = k % K_STAGE;  // s3 k 2->2, k 3->0, k 4->1...

    // stage gmem -> smem
    int load_gmem_a_k = k * BK * WARP_TILE_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * BK * WARP_TILE_K + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 

    uint32_t load_smem_a_ptr = (
      smem_a_base_ptr + (smem_sel_next * s_a_stage_offset + 
                         load_smem_a_m * (BK + A_PAD) + 
                         load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16); // MMA_K 0
    uint32_t load_smem_a_mma_k_ptr = (
      smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) + 
      (smem_sel_next * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + 
      load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_mma_k_ptr, &A[load_gmem_a_addr + 16], 16); // MMA_K 1

    uint32_t load_smem_b_ptr = (
      smem_b_base_ptr + (smem_sel_next * s_b_stage_offset + 
                         load_smem_b_k * (BN + B_PAD) + 
                         load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    
    int load_gmem_b_k_mma_k = k * BK * WARP_TILE_K + MMA_K + load_smem_b_k;
    int load_gmem_b_addr_mma_k = load_gmem_b_k_mma_k * N + load_gmem_b_n; 
    uint32_t load_smem_b_mma_k_ptr = (
      smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) + 
      (smem_sel_next * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + 
      load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_mma_k_ptr, &B[load_gmem_b_addr_mma_k], 16);
    CP_ASYNC_COMMIT_GROUP();
    
    // ldmatrix for s_a, ldmatrix.trans for s_b.
    // smem -> reg buffers 1, second MMA_K, 16~31
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
      int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
      uint32_t lane_smem_a_ptr = (
        smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) + 
        (smem_sel * s_a_stage_offset + lane_smem_a_m * (BK + A_PAD) + 
        lane_smem_a_k) * sizeof(half)
      );
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                  lane_smem_a_ptr);
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_k = lane_id % 16; // 0~15
      int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
      uint32_t lane_smem_b_ptr = (
        smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) + 
        (smem_sel * s_b_stage_offset + lane_smem_b_k * (BN + B_PAD) + 
        lane_smem_b_n) * sizeof(half)
      );
      // may use .x4.trans to load 4 matrix for reg double buffers at once?
      LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                    lane_smem_b_ptr);
    }
    
    // MMA compute, first MMA_K
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // Warp swizzle: Right -> Left -> Right -> Left
        int j_s = ((i % 2) && WARP_SWIZZLE)? (WARP_TILE_N - j - 1) : j;
        HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
                  RA[reg_load_idx][i][0], RA[reg_load_idx][i][1], 
                  RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], 
                  RB[reg_load_idx][j_s][0], RB[reg_load_idx][j_s][1], 
                  RC[i][j_s][0], RC[i][j_s][1]);
      }
    }
    
    reg_store_idx ^= 1; // 1 -> 0
    reg_load_idx ^= 1; // 0 -> 1
    // MMA compute, second MMA_K 
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // Warp swizzle: Right -> Left -> Right -> Left
        int j_s = ((i % 2) && WARP_SWIZZLE)? (WARP_TILE_N - j - 1) : j;
        HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
                  RA[reg_load_idx][i][0], RA[reg_load_idx][i][1], 
                  RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], 
                  RB[reg_load_idx][j_s][0], RB[reg_load_idx][j_s][1], 
                  RC[i][j_s][0], RC[i][j_s][1]);
      }
    }

    CP_ASYNC_WAIT_GROUP(K_STAGE-2);
    __syncthreads(); 

    // load next k iters to reg buffers.
    // smem -> reg buffers 0, first MMA_K, 0~15
    // int smem_sel_reg = (k + 2) % K_STAGE; // vs smem_sel k=2->(0)1, k=3->(1)2
    int smem_sel_reg = (smem_sel + 1) % K_STAGE; // vs smem_sel k=2->(0)1, k=3->(1)2
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
      int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
      uint32_t lane_smem_a_ptr = (
        smem_a_base_ptr + (smem_sel_reg * s_a_stage_offset + 
                           lane_smem_a_m * (BK + A_PAD) + 
                           lane_smem_a_k) * sizeof(half)
      );
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                  lane_smem_a_ptr);
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_k = lane_id % 16;  // 0~15, 0~15
      int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
      uint32_t lane_smem_b_ptr = (
        smem_b_base_ptr + (smem_sel_reg * s_b_stage_offset + 
                           lane_smem_b_k * (BN + B_PAD) + 
                           lane_smem_b_n) * sizeof(half)
      );
      // may use .x4.trans to load 4 matrix for reg double buffers at once?
      LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                    lane_smem_b_ptr);
    }
  }

  // make sure all memory issues ready.
  if constexpr ((K_STAGE - 2) > 0) {
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads(); 
  }

  // processing last (K_STAGE-1) k iters.
  {
    #pragma unroll
    for (int k = 0; k < (K_STAGE - 1); k++) {
      reg_store_idx ^= 1; // 0->1
      reg_load_idx ^= 1; // 1->0

      int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
      // ldmatrix for s_a, ldmatrix.trans for s_b.
      // smem -> reg buffers 1, second MMA_K
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
        int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
        int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
        uint32_t lane_smem_a_ptr = (
          smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) +
          (stage_sel * s_a_stage_offset + lane_smem_a_m * (BK + A_PAD) + 
          lane_smem_a_k) * sizeof(half)
        );
        LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                    RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                    lane_smem_a_ptr);
      }

      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
        int lane_smem_b_k = lane_id % 16; // 0~15
        int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
        uint32_t lane_smem_b_ptr = (
          smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) + 
          (stage_sel * s_b_stage_offset + lane_smem_b_k * (BN + B_PAD) + 
          lane_smem_b_n) * sizeof(half)
        );
        LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                      lane_smem_b_ptr);
      }

      // MMA compute, first MMA_K
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          // Warp swizzle: Right -> Left -> Right -> Left
          int j_s = ((i % 2) && WARP_SWIZZLE)? (WARP_TILE_N - j - 1) : j;
          HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
                    RA[reg_load_idx][i][0], RA[reg_load_idx][i][1], 
                    RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], 
                    RB[reg_load_idx][j_s][0], RB[reg_load_idx][j_s][1], 
                    RC[i][j_s][0], RC[i][j_s][1]);
        }
      }

      reg_store_idx ^= 1; // 1 -> 0
      reg_load_idx ^= 1; // 0 -> 1

      // MMA compute, second MMA_K
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          // Warp swizzle: Right -> Left -> Right -> Left
          int j_s = ((i % 2) && WARP_SWIZZLE)? (WARP_TILE_N - j - 1) : j;
          HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
                    RA[reg_load_idx][i][0], RA[reg_load_idx][i][1], 
                    RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], 
                    RB[reg_load_idx][j_s][0], RB[reg_load_idx][j_s][1], 
                    RC[i][j_s][0], RC[i][j_s][1]);
        }
      }
      
      // load next k iters to reg buffers.
      // smem -> reg buffers 0, first MMA_K, 0~15
      // int stage_sel_reg = ((NUM_K_TILES - K_STAGE + k) % K_STAGE); 
      int stage_sel_reg = (stage_sel + 1) % K_STAGE; 
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
        int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
        int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
        uint32_t lane_smem_a_ptr = (
          smem_a_base_ptr + (stage_sel_reg * s_a_stage_offset + 
                             lane_smem_a_m * (BK + A_PAD) + 
                             lane_smem_a_k) * sizeof(half)
        );
        LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                    RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                    lane_smem_a_ptr);
      }

      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
        int lane_smem_b_k = lane_id % 16;  // 0~15, 0~15
        int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
        uint32_t lane_smem_b_ptr = (
          smem_b_base_ptr + (stage_sel_reg * s_b_stage_offset + 
                             lane_smem_b_k * (BN + B_PAD) + 
                             lane_smem_b_n) * sizeof(half)
        );
        LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                      lane_smem_b_ptr);
      }
    }
  }

  // collective store with reg reuse & warp shuffle
  for (int i = 0; i < WARP_TILE_M; ++i) {
    // reuse RA[2][4][4] reg here, this may boost 0.3~0.5 TFLOPS up.
    // may not put 'if' in N loop, it will crash the 'pragma unroll' hint ?
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // How to use LDST128BITS here? __shfl_sync -> lane 0 -> store 8 half.
      // thus, we only need 8 memory issues with 128 bits after shfl_sync.
      RA[0][j][0] = RC[i][j][0]; 
      RA[1][j][0] = RC[i][j][1];
      RA[0][j][1] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 1);
      RA[0][j][2] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 2);
      RA[0][j][3] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 3);
      RA[1][j][1] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 1);
      RA[1][j][2] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 2);
      RA[1][j][3] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 3);
    }

    if (lane_id % 4 == 0) {
      int store_warp_smem_c_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int store_lane_gmem_c_m = by * BM + store_warp_smem_c_m + lane_id / 4;
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        int store_warp_smem_c_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
        int store_lane_gmem_c_n = bx * BN + store_warp_smem_c_n;
        int store_gmem_c_addr_0 = store_lane_gmem_c_m * N + store_lane_gmem_c_n;
        int store_gmem_c_addr_1 = (store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n;
        LDST128BITS(C[store_gmem_c_addr_0]) = LDST128BITS(RA[0][j][0]); 
        LDST128BITS(C[store_gmem_c_addr_1]) = LDST128BITS(RA[1][j][0]); 
      }
    }
  }
}

// NOTE: use ldmatrix.x4.trans for matrix B smem -> reg
// In order to reduce bank conflicts, we will save the K(16x2=32) 
// dimension by half according to the stage dimension. For example, 
// stages=3, warp_tile_k=2, it will be saved as [3*2][BM][16].
// 128x128, mma2x4, warp4x4(64,32,32), stages, block swizzle, dsmem, 
// k32 with reg double buffers
template<const int MMA_M=16, 
         const int MMA_N=8, 
         const int MMA_K=16,
         const int MMA_TILE_M=2,
         const int MMA_TILE_N=4,
         const int WARP_TILE_M=4,
         const int WARP_TILE_N=4,
         const int WARP_TILE_K=2,
         const int A_PAD=0, 
         const int B_PAD=0,
         const int K_STAGE=2, 
         const bool BLOCK_SWIZZLE=true,
         const bool WARP_SWIZZLE=true>
__global__ void  __launch_bounds__(256) 
hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4_kernel(
  const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C, 
  int M, int N, int K) {
  // BLOCK_SWIZZLE 0/1 control use block swizzle or not.
  const int bx = ((int) BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, MMA_K * WARP_TILE_K);
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M; // 16*2*4=128
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N; // 8*4*4=128
  constexpr int BK = MMA_K; // 16x2=32
  constexpr int kStageK = MMA_K * WARP_TILE_K;

  extern __shared__ half smem[]; 
  half* s_a = smem;
  half* s_b = smem + K_STAGE * BM * (BK + A_PAD) * WARP_TILE_K;
  constexpr int s_a_stage_offset = BM * (BK + A_PAD); // 128x16 
  constexpr int s_b_stage_offset = BK * (BN + B_PAD); // 16x128
  constexpr int s_a_mma_k_store_offset = K_STAGE * BM * (BK + A_PAD);
  constexpr int s_b_mma_k_store_offset = K_STAGE * BK * (BN + B_PAD);

  const int tid = threadIdx.y * blockDim.x + threadIdx.x; // within block
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_m = warp_id % 2; // 0,1
  const int warp_n = warp_id / 2; // 0,1,2,3

  int load_smem_a_m = tid / 2; // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
  int load_smem_b_k = tid / 16; // row 0~15
  int load_smem_b_n = (tid % 16) * 8; // col 0,8,16,...
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
  if (by * BM + BM > M || bx * BN + BN > N || (K % kStageK) != 0 || (N % 8) != 0 || NUM_K_TILES < K_STAGE) {
    hgemm_scalar_tile_body_at<BM, BN>(A, B, C, M, N, K, by * BM, bx * BN);
    return;
  }

  uint32_t RC[WARP_TILE_M][WARP_TILE_N][2];
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      RC[i][j][0] = 0;
      RC[i][j][1] = 0;
    }
  }

  uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
  uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

  #pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) { // 0, 1
    // k * WMMA_K, WMMA_K=16 -> (k << 4)
    int load_gmem_a_k = k * BK * WARP_TILE_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * BK * WARP_TILE_K + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 

    uint32_t load_smem_a_ptr = (
      smem_a_base_ptr + (k * s_a_stage_offset + 
                         load_smem_a_m * (BK + A_PAD) + 
                         load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16); // MMA_K 0
    uint32_t load_smem_a_mma_k_ptr = (
      smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) + 
      (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + 
      load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_mma_k_ptr, &A[load_gmem_a_addr + 16], 16); // MMA_K 1

    uint32_t load_smem_b_ptr = (
      smem_b_base_ptr + (k * s_b_stage_offset + 
                         load_smem_b_k * (BN + B_PAD) + 
                         load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    
    int load_gmem_b_k_mma_k = k * BK * WARP_TILE_K + MMA_K + load_smem_b_k;
    int load_gmem_b_addr_mma_k = load_gmem_b_k_mma_k * N + load_gmem_b_n; 
    uint32_t load_smem_b_mma_k_ptr = (
      smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) + 
      (k * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + 
      load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_mma_k_ptr, &B[load_gmem_b_addr_mma_k], 16);

    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(K_STAGE-2); // s2->0, s3->1, s4->2
  __syncthreads(); 

  uint32_t RA[2][WARP_TILE_M][4];
  uint32_t RB[2][WARP_TILE_N][2];

  int reg_store_idx = 0;
  int reg_load_idx = 1;

  { 
    // ldmatrix for s_a, ldmatrix.trans for s_b.
    // smem -> reg buffers 0, first MMA_K, 0~15
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
      int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
      uint32_t lane_smem_a_ptr = (
        smem_a_base_ptr + 
        (0 * s_a_stage_offset + lane_smem_a_m * (BK + A_PAD) + 
        lane_smem_a_k) * sizeof(half)
      );
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                  lane_smem_a_ptr);
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_k = lane_id % 16;  // 0~15, 0~15
      int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
      uint32_t lane_smem_b_ptr = (
        smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) * (lane_id / 16) +
        (0 * s_b_stage_offset + lane_smem_b_k * (BN + B_PAD) + 
        lane_smem_b_n) * sizeof(half)
      );
      // TRICK: I use .x4.trans to load 4 matrix for reg double buffers at once.
      LDMATRIX_X4_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                    RB[reg_load_idx][j][0],  RB[reg_load_idx][j][1],
                    lane_smem_b_ptr);
    }
  }
  
  #pragma unroll
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; ++k) {
    reg_store_idx ^= 1; // 0->1
    reg_load_idx ^= 1; // 1->0
    int smem_sel = (k + 1) % K_STAGE; // s3 k 2->0, k 3->1, k 4->2...
    int smem_sel_next = k % K_STAGE;  // s3 k 2->2, k 3->0, k 4->1...

    // stage gmem -> smem
    int load_gmem_a_k = k * BK * WARP_TILE_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * BK * WARP_TILE_K + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 

    uint32_t load_smem_a_ptr = (
      smem_a_base_ptr + (smem_sel_next * s_a_stage_offset + 
                         load_smem_a_m * (BK + A_PAD) + 
                         load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16); // MMA_K 0
    uint32_t load_smem_a_mma_k_ptr = (
      smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) + 
      (smem_sel_next * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + 
      load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_mma_k_ptr, &A[load_gmem_a_addr + 16], 16); // MMA_K 1

    uint32_t load_smem_b_ptr = (
      smem_b_base_ptr + (smem_sel_next * s_b_stage_offset + 
                         load_smem_b_k * (BN + B_PAD) + 
                         load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    
    int load_gmem_b_k_mma_k = k * BK * WARP_TILE_K + MMA_K + load_smem_b_k;
    int load_gmem_b_addr_mma_k = load_gmem_b_k_mma_k * N + load_gmem_b_n; 
    uint32_t load_smem_b_mma_k_ptr = (
      smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) + 
      (smem_sel_next * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + 
      load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_mma_k_ptr, &B[load_gmem_b_addr_mma_k], 16);
    CP_ASYNC_COMMIT_GROUP();
    
    // ldmatrix for s_a, ldmatrix.trans for s_b.
    // smem -> reg buffers 1, second MMA_K, 16~31
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
      int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
      uint32_t lane_smem_a_ptr = (
        smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) + 
        (smem_sel * s_a_stage_offset + lane_smem_a_m * (BK + A_PAD) + 
        lane_smem_a_k) * sizeof(half)
      );
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                  lane_smem_a_ptr);
    }
    
    // MMA compute, first MMA_K
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // Warp swizzle: Right -> Left -> Right -> Left
        int j_s = ((i % 2) && WARP_SWIZZLE)? (WARP_TILE_N - j - 1) : j;
        HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
                  RA[reg_load_idx][i][0], RA[reg_load_idx][i][1], 
                  RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], 
                  RB[reg_load_idx][j_s][0], RB[reg_load_idx][j_s][1], 
                  RC[i][j_s][0], RC[i][j_s][1]);
      }
    }
    
    reg_store_idx ^= 1; // 1 -> 0
    reg_load_idx ^= 1; // 0 -> 1
    // MMA compute, second MMA_K 
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // Warp swizzle: Right -> Left -> Right -> Left
        int j_s = ((i % 2) && WARP_SWIZZLE)? (WARP_TILE_N - j - 1) : j;
        HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
                  RA[reg_load_idx][i][0], RA[reg_load_idx][i][1], 
                  RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], 
                  RB[reg_load_idx][j_s][0], RB[reg_load_idx][j_s][1], 
                  RC[i][j_s][0], RC[i][j_s][1]);
      }
    }

    CP_ASYNC_WAIT_GROUP(K_STAGE-2);
    __syncthreads(); 

    // load next k iters to reg buffers.
    // smem -> reg buffers 0, first MMA_K, 0~15
    // int smem_sel_reg = (k + 2) % K_STAGE; // vs smem_sel k=2->(0)1, k=3->(1)2
    int smem_sel_reg = (smem_sel + 1) % K_STAGE; // vs smem_sel k=2->(0)1, k=3->(1)2
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
      int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
      uint32_t lane_smem_a_ptr = (
        smem_a_base_ptr + (smem_sel_reg * s_a_stage_offset + 
                           lane_smem_a_m * (BK + A_PAD) + 
                           lane_smem_a_k) * sizeof(half)
      );
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                  lane_smem_a_ptr);
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_k = lane_id % 16;  // 0~15, 0~15
      int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
      uint32_t lane_smem_b_ptr = (
        smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) * (lane_id / 16) +
        (smem_sel_reg * s_b_stage_offset + lane_smem_b_k * (BN + B_PAD) + 
        lane_smem_b_n) * sizeof(half)
      );
      // may use .x4.trans to load 4 matrix for reg double buffers at once?
      LDMATRIX_X4_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                    RB[reg_load_idx][j][0],  RB[reg_load_idx][j][1],
                    lane_smem_b_ptr);
    }
  }

  // make sure all memory issues ready.
  if constexpr ((K_STAGE - 2) > 0) {
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads(); 
  }

  // processing last (K_STAGE-1) k iters.
  {
    #pragma unroll
    for (int k = 0; k < (K_STAGE - 1); k++) {
      reg_store_idx ^= 1; // 0->1
      reg_load_idx ^= 1; // 1->0

      int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
      // ldmatrix for s_a, ldmatrix.trans for s_b.
      // smem -> reg buffers 1, second MMA_K
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
        int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
        int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
        uint32_t lane_smem_a_ptr = (
          smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) +
          (stage_sel * s_a_stage_offset + lane_smem_a_m * (BK + A_PAD) + 
          lane_smem_a_k) * sizeof(half)
        );
        LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                    RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                    lane_smem_a_ptr);
      }

      // MMA compute, first MMA_K
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          // Warp swizzle: Right -> Left -> Right -> Left
          int j_s = ((i % 2) && WARP_SWIZZLE)? (WARP_TILE_N - j - 1) : j;
          HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
                    RA[reg_load_idx][i][0], RA[reg_load_idx][i][1], 
                    RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], 
                    RB[reg_load_idx][j_s][0], RB[reg_load_idx][j_s][1], 
                    RC[i][j_s][0], RC[i][j_s][1]);
        }
      }

      reg_store_idx ^= 1; // 1 -> 0
      reg_load_idx ^= 1; // 0 -> 1

      // MMA compute, second MMA_K
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          // Warp swizzle: Right -> Left -> Right -> Left
          int j_s = ((i % 2) && WARP_SWIZZLE)? (WARP_TILE_N - j - 1) : j;
          HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
                    RA[reg_load_idx][i][0], RA[reg_load_idx][i][1], 
                    RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], 
                    RB[reg_load_idx][j_s][0], RB[reg_load_idx][j_s][1], 
                    RC[i][j_s][0], RC[i][j_s][1]);
        }
      }
      
      // load next k iters to reg buffers.
      // smem -> reg buffers 0, first MMA_K, 0~15
      // int stage_sel_reg = ((NUM_K_TILES - K_STAGE + k) % K_STAGE); 
      int stage_sel_reg = (stage_sel + 1) % K_STAGE; 
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
        int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
        int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
        uint32_t lane_smem_a_ptr = (
          smem_a_base_ptr + (stage_sel_reg * s_a_stage_offset + 
                             lane_smem_a_m * (BK + A_PAD) + 
                             lane_smem_a_k) * sizeof(half)
        );
        LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                    RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                    lane_smem_a_ptr);
      }

      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
        int lane_smem_b_k = lane_id % 16;  // 0~15, 0~15
        int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
        uint32_t lane_smem_b_ptr = (
          smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) * (lane_id / 16) +
          (stage_sel_reg * s_b_stage_offset + lane_smem_b_k * (BN + B_PAD) + 
          lane_smem_b_n) * sizeof(half)
        );
        // may use .x4.trans to load 4 matrix for reg double buffers at once?
        LDMATRIX_X4_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                      RB[reg_load_idx][j][0],  RB[reg_load_idx][j][1],
                      lane_smem_b_ptr);
      }
    }
  }

  // collective store with reg reuse & warp shuffle
  for (int i = 0; i < WARP_TILE_M; ++i) {
    // reuse RA[2][4][4] reg here, this may boost 0.3~0.5 TFLOPS up.
    // may not put 'if' in N loop, it will crash the 'pragma unroll' hint ?
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // How to use LDST128BITS here? __shfl_sync -> lane 0 -> store 8 half.
      // thus, we only need 8 memory issues with 128 bits after shfl_sync.
      RA[0][j][0] = RC[i][j][0]; 
      RA[1][j][0] = RC[i][j][1];
      RA[0][j][1] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 1);
      RA[0][j][2] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 2);
      RA[0][j][3] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 3);
      RA[1][j][1] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 1);
      RA[1][j][2] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 2);
      RA[1][j][3] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 3);
    }

    if (lane_id % 4 == 0) {
      int store_warp_smem_c_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int store_lane_gmem_c_m = by * BM + store_warp_smem_c_m + lane_id / 4;
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        int store_warp_smem_c_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
        int store_lane_gmem_c_n = bx * BN + store_warp_smem_c_n;
        int store_gmem_c_addr_0 = store_lane_gmem_c_m * N + store_lane_gmem_c_n;
        int store_gmem_c_addr_1 = (store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n;
        LDST128BITS(C[store_gmem_c_addr_0]) = LDST128BITS(RA[0][j][0]); 
        LDST128BITS(C[store_gmem_c_addr_1]) = LDST128BITS(RA[1][j][0]); 
      }
    }
  }
}

// NOTE: reduce registers usage.
// In order to reduce bank conflicts, we will save the K(16x2=32) 
// dimension by half according to the stage dimension. For example, 
// stages=3, warp_tile_k=2, it will be saved as [3*2][BM][16].
// 128x128, mma2x4, warp4x4(64,32,32), stages, block swizzle, dsmem, 
// k32 with reg double buffers
template<const int MMA_M=16, 
         const int MMA_N=8, 
         const int MMA_K=16,
         const int MMA_TILE_M=2,
         const int MMA_TILE_N=4,
         const int WARP_TILE_M=4,
         const int WARP_TILE_N=4,
         const int WARP_TILE_K=2,
         const int A_PAD=0, 
         const int B_PAD=0,
         const int K_STAGE=2, 
         const bool BLOCK_SWIZZLE=true,
         const bool WARP_SWIZZLE=true>
__global__ void  __launch_bounds__(256) 
hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr_kernel(
  const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C, 
  int M, int N, int K) {
  // BLOCK_SWIZZLE 0/1 control use block swizzle or not.
  const int bx = ((int) BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, MMA_K * WARP_TILE_K);
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M; // 16*2*4=128
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N; // 8*4*4=128
  constexpr int BK = MMA_K; // 16x2=32
  constexpr int kStageK = MMA_K * WARP_TILE_K;

  extern __shared__ half smem[]; 
  half* s_a = smem;
  half* s_b = smem + K_STAGE * BM * (BK + A_PAD) * WARP_TILE_K;
  constexpr int s_a_stage_offset = BM * (BK + A_PAD); // 128x16 
  constexpr int s_b_stage_offset = BK * (BN + B_PAD); // 16x128
  constexpr int s_a_mma_k_store_offset = K_STAGE * BM * (BK + A_PAD);
  constexpr int s_b_mma_k_store_offset = K_STAGE * BK * (BN + B_PAD);

  const int tid = threadIdx.y * blockDim.x + threadIdx.x; // within block
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_m = warp_id % 2; // 0,1
  const int warp_n = warp_id / 2; // 0,1,2,3

  const int load_smem_a_m = tid / 2; // row 0~127
  const int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
  const int load_smem_b_k = tid / 16; // row 0~15
  const int load_smem_b_n = (tid % 16) * 8; // col 0,8,16,...
  const int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  const int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
  if (by * BM + BM > M || bx * BN + BN > N || (K % kStageK) != 0 || (N % 8) != 0 || NUM_K_TILES < K_STAGE) {
    hgemm_scalar_tile_body_at<BM, BN>(A, B, C, M, N, K, by * BM, bx * BN);
    return;
  }
  // 16 reg for pre-defined vars.

  uint32_t RC[WARP_TILE_M][WARP_TILE_N][2]; // 4*4*2=32 reg
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      RC[i][j][0] = 0;
      RC[i][j][1] = 0;
    }
  }

  uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
  uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

  #pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) { 
    // reduce 9 registers -> 4 registers.
    // a gmem -> a smem
    uint32_t load_smem_a_ptr = (
      smem_a_base_ptr + (k * s_a_stage_offset + 
                         load_smem_a_m * (BK + A_PAD) + 
                         load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(
      load_smem_a_ptr, 
      &A[load_gmem_a_m * K + k * BK * WARP_TILE_K + load_smem_a_k], 
      16
    ); // MMA_K 0
    uint32_t load_smem_a_mma_k_ptr = (
      smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) + 
      (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + 
      load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(
      load_smem_a_mma_k_ptr, 
      &A[load_gmem_a_m * K + k * BK * WARP_TILE_K + load_smem_a_k + 16], 
      16
    ); // MMA_K 1

    // b gmem -> b smem
    uint32_t load_smem_b_ptr = (
      smem_b_base_ptr + (k * s_b_stage_offset + 
                         load_smem_b_k * (BN + B_PAD) + 
                         load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(
      load_smem_b_ptr, 
      &B[(k * BK * WARP_TILE_K + load_smem_b_k) * N + load_gmem_b_n], 
      16
    ); // MMA_K 0
    uint32_t load_smem_b_mma_k_ptr = (
      smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) + 
      (k * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + 
      load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(
      load_smem_b_mma_k_ptr, 
      &B[(k * BK * WARP_TILE_K + MMA_K + load_smem_b_k) * N + load_gmem_b_n], 
      16
    ); // MMA_K 1

    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(K_STAGE-2); // s2->0, s3->1, s4->2
  __syncthreads(); 

  uint32_t RA[2][WARP_TILE_M][4]; // 2*4*4=32 reg
  uint32_t RB[2][WARP_TILE_N][2]; // 2*4*2=16 reg

  // 16+32+32+16=96 reg

  int reg_store_idx = 0;
  int reg_load_idx = 1;

  { 
    // ldmatrix for s_a, ldmatrix.trans for s_b.
    // smem -> reg buffers 0, first MMA_K, 0~15
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      // reduce 4 registers -> 1 registers.
      uint32_t lane_smem_a_ptr = (
        smem_a_base_ptr + (
                    (warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M + lane_id % 16) * 
                    (BK + A_PAD) + (lane_id / 16) * 8
                  ) * sizeof(half)
      );
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                  lane_smem_a_ptr
      );
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // reduce 4 registers -> 1 registers.
      uint32_t lane_smem_b_ptr = (
        smem_b_base_ptr + (
                      (lane_id % 16) * (BN + B_PAD) + 
                      (warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N)
                    ) * sizeof(half)
      );
      LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                    lane_smem_b_ptr
      );
    }
  }
  
  int smem_sel = 0; // s3 k 2->0, k 3->1, k 4->2...
  int smem_sel_next = (K_STAGE - 1) % K_STAGE;  // s3 k 2->2, k 3->0, k 4->1...
  #pragma unroll
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; ++k) {
    reg_store_idx ^= 1; // 0->1
    reg_load_idx ^= 1; // 1->0
    smem_sel = (k + 1) % K_STAGE; // s3 k 2->0, k 3->1, k 4->2...
    smem_sel_next = k % K_STAGE;  // s3 k 2->2, k 3->0, k 4->1...

    // reduce 9 registers -> 4 registers.
    // a gmem -> a smem
    uint32_t load_smem_a_ptr = (
      smem_a_base_ptr + (smem_sel_next * s_a_stage_offset + 
                         load_smem_a_m * (BK + A_PAD) + 
                         load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(
      load_smem_a_ptr, 
      &A[load_gmem_a_m * K + k * BK * WARP_TILE_K + load_smem_a_k], 
      16
    ); // MMA_K 0
    uint32_t load_smem_a_mma_k_ptr = (
      smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) + 
      (smem_sel_next * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + 
      load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(
      load_smem_a_mma_k_ptr, 
      &A[load_gmem_a_m * K + k * BK * WARP_TILE_K + load_smem_a_k + 16], 
      16
    ); // MMA_K 1

    // b gmem -> b smem
    uint32_t load_smem_b_ptr = (
      smem_b_base_ptr + (smem_sel_next * s_b_stage_offset + 
                         load_smem_b_k * (BN + B_PAD) + 
                         load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(
      load_smem_b_ptr, 
      &B[(k * BK * WARP_TILE_K + load_smem_b_k) * N + load_gmem_b_n], 
      16
    ); // MMA_K 0
    uint32_t load_smem_b_mma_k_ptr = (
      smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) + 
      (smem_sel_next * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + 
      load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(
      load_smem_b_mma_k_ptr, 
      &B[(k * BK * WARP_TILE_K + MMA_K + load_smem_b_k) * N + load_gmem_b_n], 
      16
    ); // MMA_K 1

    CP_ASYNC_COMMIT_GROUP();
    
    // ldmatrix for s_a, ldmatrix.trans for s_b.
    // smem -> reg buffers 1, second MMA_K, 16~31
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      // reduce 4 registers -> 1 registers.
      uint32_t lane_smem_a_ptr = (
        smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) + (
                    smem_sel * s_a_stage_offset + 
                    (warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M + lane_id % 16) * 
                    (BK + A_PAD) + (lane_id / 16) * 8
                  ) * sizeof(half)
      );
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                  lane_smem_a_ptr
      );
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // reduce 4 registers -> 1 registers.
      uint32_t lane_smem_b_ptr = (
        smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) + (
                      smem_sel * s_b_stage_offset + 
                      (lane_id % 16) * (BN + B_PAD) + 
                      (warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N)
                    ) * sizeof(half)
      );
      LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                    lane_smem_b_ptr
      );
    }
    
    // MMA compute, first MMA_K
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // Warp swizzle: Right -> Left -> Right -> Left
        int j_s = ((i % 2) && WARP_SWIZZLE)? (WARP_TILE_N - j - 1) : j;
        HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
                  RA[reg_load_idx][i][0], RA[reg_load_idx][i][1], 
                  RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], 
                  RB[reg_load_idx][j_s][0], RB[reg_load_idx][j_s][1], 
                  RC[i][j_s][0], RC[i][j_s][1]);
      }
    }
    
    reg_store_idx ^= 1; // 1 -> 0
    reg_load_idx ^= 1; // 0 -> 1
    // MMA compute, second MMA_K 
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // Warp swizzle: Right -> Left -> Right -> Left
        int j_s = ((i % 2) && WARP_SWIZZLE)? (WARP_TILE_N - j - 1) : j;
        HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
                  RA[reg_load_idx][i][0], RA[reg_load_idx][i][1], 
                  RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], 
                  RB[reg_load_idx][j_s][0], RB[reg_load_idx][j_s][1], 
                  RC[i][j_s][0], RC[i][j_s][1]);
      }
    }

    CP_ASYNC_WAIT_GROUP(K_STAGE-2);
    __syncthreads(); 

    // load next k iters to reg buffers.
    // smem -> reg buffers 0, first MMA_K, 0~15
    // int smem_sel_reg = (smem_sel + 1) % K_STAGE; // vs smem_sel k=2->(0)1, k=3->(1)2
    smem_sel = (smem_sel + 1) % K_STAGE;
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      // reduce 4 registers -> 1 registers.
      uint32_t lane_smem_a_ptr = (
        smem_a_base_ptr + (
                    smem_sel * s_a_stage_offset + 
                    (warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M + lane_id % 16) * 
                    (BK + A_PAD) + (lane_id / 16) * 8
                  ) * sizeof(half)
      );
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                  lane_smem_a_ptr
      );
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // reduce 4 registers -> 1 registers.
      uint32_t lane_smem_b_ptr = (
        smem_b_base_ptr + (
                      smem_sel * s_b_stage_offset + 
                      (lane_id % 16) * (BN + B_PAD) + 
                      (warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N)
                    ) * sizeof(half)
      );
      LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                    lane_smem_b_ptr
      );
    }
  }

  // make sure all memory issues ready.
  if constexpr ((K_STAGE - 2) > 0) {
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads(); 
  }

  // processing last (K_STAGE-1) k iters.
  {
    #pragma unroll
    for (int k = 0; k < (K_STAGE - 1); k++) {
      reg_store_idx ^= 1; // 0->1
      reg_load_idx ^= 1; // 1->0

      // int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
      smem_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
      // ldmatrix for s_a, ldmatrix.trans for s_b.
      // smem -> reg buffers 1, second MMA_K
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        // reduce 4 registers -> 1 registers.
        uint32_t lane_smem_a_ptr = (
          smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) + (
                      smem_sel * s_a_stage_offset + 
                      (warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M + lane_id % 16) * 
                      (BK + A_PAD) + (lane_id / 16) * 8
                    ) * sizeof(half)
        );
        LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                    RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                    lane_smem_a_ptr
        );
      }

      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // reduce 4 registers -> 1 registers.
        uint32_t lane_smem_b_ptr = (
          smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) + (
                        smem_sel * s_b_stage_offset + 
                        (lane_id % 16) * (BN + B_PAD) + 
                        (warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N)
                      ) * sizeof(half)
        );
        LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                      lane_smem_b_ptr
        );
      }

      // MMA compute, first MMA_K
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          // Warp swizzle: Right -> Left -> Right -> Left
          int j_s = ((i % 2) && WARP_SWIZZLE)? (WARP_TILE_N - j - 1) : j;
          HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
                    RA[reg_load_idx][i][0], RA[reg_load_idx][i][1], 
                    RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], 
                    RB[reg_load_idx][j_s][0], RB[reg_load_idx][j_s][1], 
                    RC[i][j_s][0], RC[i][j_s][1]);
        }
      }

      reg_store_idx ^= 1; // 1 -> 0
      reg_load_idx ^= 1; // 0 -> 1

      // MMA compute, second MMA_K
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          // Warp swizzle: Right -> Left -> Right -> Left
          int j_s = ((i % 2) && WARP_SWIZZLE)? (WARP_TILE_N - j - 1) : j;
          HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
                    RA[reg_load_idx][i][0], RA[reg_load_idx][i][1], 
                    RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], 
                    RB[reg_load_idx][j_s][0], RB[reg_load_idx][j_s][1], 
                    RC[i][j_s][0], RC[i][j_s][1]);
        }
      }
      
      // load next k iters to reg buffers.
      // smem -> reg buffers 0, first MMA_K, 0~15
      // int stage_sel_reg = (stage_sel + 1) % K_STAGE; 
      smem_sel = (smem_sel + 1) % K_STAGE; 
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        // reduce 4 registers -> 1 registers.
        uint32_t lane_smem_a_ptr = (
          smem_a_base_ptr + (
                      smem_sel * s_a_stage_offset + 
                      (warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M + lane_id % 16) * 
                      (BK + A_PAD) + (lane_id / 16) * 8
                    ) * sizeof(half)
        );
        LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                    RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                    lane_smem_a_ptr
        );
      }

      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // reduce 4 registers -> 1 registers.
        uint32_t lane_smem_b_ptr = (
          smem_b_base_ptr + (
                        smem_sel * s_b_stage_offset + 
                        (lane_id % 16) * (BN + B_PAD) + 
                        (warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N)
                      ) * sizeof(half)
        );
        LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                      lane_smem_b_ptr
        );
      }
    }
  }

  // collective store with reg reuse & warp shuffle
  for (int i = 0; i < WARP_TILE_M; ++i) {
    // reuse RA[2][4][4] reg here, this may boost 0.3~0.5 TFLOPS up.
    // may not put 'if' in N loop, it will crash the 'pragma unroll' hint ?
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // How to use LDST128BITS here? __shfl_sync -> lane 0 -> store 8 half.
      // thus, we only need 8 memory issues with 128 bits after shfl_sync.
      RA[0][j][0] = RC[i][j][0]; 
      RA[1][j][0] = RC[i][j][1];
      RA[0][j][1] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 1);
      RA[0][j][2] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 2);
      RA[0][j][3] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 3);
      RA[1][j][1] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 1);
      RA[1][j][2] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 2);
      RA[1][j][3] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 3);
    }

    if (lane_id % 4 == 0) {
      // reduce 6 registers -> 0 registers.
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        LDST128BITS(
          C[
            (by * BM + warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M + lane_id / 4) * N + 
            (bx * BN + warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N)
          ]
        ) = (LDST128BITS(RA[0][j][0])); 
        LDST128BITS(
          C[
            (by * BM + warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M + lane_id / 4 + 8) * N + 
            (bx * BN + warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N)
          ]
        ) = (LDST128BITS(RA[1][j][0])); 
      }
    }
  }
}

template <int Stages, bool Swizzle>
bool hgemm_launch_mma_m16n8k16_mma2x4_warp4x4_stages_static(
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
        kMmaM = 16,
        kMmaN = 8,
        kMmaK = 16,
        kMmaTileM = 2,
        kMmaTileN = 4,
        kWarpTileM = 4,
        kWarpTileN = 4,
        kAPad = 8,
        kBPad = 8,
        kBlockM = kMmaM * kMmaTileM * kWarpTileM,
        kBlockN = kMmaN * kMmaTileN * kWarpTileN,
        kThreads = kMmaTileM * kMmaTileN * kWarpSize
    };
    // Static shared-memory staged kernel:
    //   CTA tile = 128x128, K stage = 16.
    //   8 warps/CTA; every thread copies one A half8 and one B half8 per stage.
    const dim3 block(kThreads);
    const dim3 grid = hgemm_mma_stage_grid(m, n, kBlockM, kBlockN, Swizzle, swizzle_stride);
    hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_kernel<
        kMmaM,
        kMmaN,
        kMmaK,
        kMmaTileM,
        kMmaTileN,
        kWarpTileM,
        kWarpTileN,
        kAPad,
        kBPad,
        Stages,
        Swizzle><<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

template <int Stages, bool Swizzle>
bool hgemm_launch_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem(
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
        kMmaM = 16,
        kMmaN = 8,
        kMmaK = 16,
        kMmaTileM = 2,
        kMmaTileN = 4,
        kWarpTileM = 4,
        kWarpTileN = 4,
        kAPad = 8,
        kBPad = 8,
        kBlockM = kMmaM * kMmaTileM * kWarpTileM,
        kBlockN = kMmaN * kMmaTileN * kWarpTileN,
        kBlockK = kMmaK,
        kThreads = kMmaTileM * kMmaTileN * kWarpSize,
        kSharedBytes = Stages * kBlockM * (kBlockK + kAPad) * static_cast<int>(sizeof(half)) +
            Stages * kBlockK * (kBlockN + kBPad) * static_cast<int>(sizeof(half))
    };
    // Dynamic shared-memory layout:
    //   A: Stages * 128 * (16 + 8) half
    //   B: Stages * 16  * (128 + 8) half
    // Each thread issues two 16-byte cp.async copies per stage.
    if(!ai_system::cuda_utils::check_status(
           cudaFuncSetAttribute(
               hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_kernel<
                   kMmaM,
                   kMmaN,
                   kMmaK,
                   kMmaTileM,
                   kMmaTileN,
                   kWarpTileM,
                   kWarpTileN,
                   kAPad,
                   kBPad,
                   Stages,
                   Swizzle>,
               cudaFuncAttributeMaxDynamicSharedMemorySize,
               98304
           ),
           "cudaFuncSetAttribute(hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_kernel)",
           error
       )) {
        return false;
    }

    const dim3 block(kThreads);
    const dim3 grid = hgemm_mma_stage_grid(m, n, kBlockM, kBlockN, Swizzle, swizzle_stride);
    hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_kernel<
        kMmaM,
        kMmaN,
        kMmaK,
        kMmaTileM,
        kMmaTileN,
        kWarpTileM,
        kWarpTileN,
        kAPad,
        kBPad,
        Stages,
        Swizzle><<<grid, block, kSharedBytes>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

template <int Stages, bool Swizzle>
bool hgemm_launch_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn(
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
        kMmaM = 16,
        kMmaN = 8,
        kMmaK = 16,
        kMmaTileM = 2,
        kMmaTileN = 4,
        kWarpTileM = 4,
        kWarpTileN = 4,
        kAPad = 0,
        kBPad = 8,
        kBlockM = kMmaM * kMmaTileM * kWarpTileM,
        kBlockN = kMmaN * kMmaTileN * kWarpTileN,
        kBlockK = kMmaK,
        kThreads = kMmaTileM * kMmaTileN * kWarpSize,
        kSharedBytes = Stages * kBlockM * (kBlockK + kAPad) * static_cast<int>(sizeof(half)) +
            Stages * kBlockN * (kBlockK + kBPad) * static_cast<int>(sizeof(half))
    };
    // TN dynamic-smem layout:
    //   A: Stages * 128 * (16 + APad) half, row-major [M][K].
    //   B: Stages * 128 * (16 + BPad) half, physical [N][K].
    // Each thread copies one contiguous half8 from A and one contiguous half8
    // from B, so the fast path expects B to be pre-transposed in global memory.
    if(!ai_system::cuda_utils::check_status(
           cudaFuncSetAttribute(
               hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_kernel<
                   kMmaM,
                   kMmaN,
                   kMmaK,
                   kMmaTileM,
                   kMmaTileN,
                   kWarpTileM,
                   kWarpTileN,
                   kAPad,
                   kBPad,
                   Stages,
                   Swizzle>,
               cudaFuncAttributeMaxDynamicSharedMemorySize,
               98304
           ),
           "cudaFuncSetAttribute(hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_kernel)",
           error
       )) {
        return false;
    }

    const dim3 block(kThreads);
    const dim3 grid = hgemm_mma_stage_grid(m, n, kBlockM, kBlockN, Swizzle, swizzle_stride);
    hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_kernel<
        kMmaM,
        kMmaN,
        kMmaK,
        kMmaTileM,
        kMmaTileN,
        kWarpTileM,
        kWarpTileN,
        kAPad,
        kBPad,
        Stages,
        Swizzle><<<grid, block, kSharedBytes>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

enum class HgemmMmaStageDsmemKind {
    x2,
    x4,
    rr
};

template <HgemmMmaStageDsmemKind Kind, int Stages, bool Swizzle, int APad = 8, int BPad = 8>
bool hgemm_launch_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem(
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
        kMmaM = 16,
        kMmaN = 8,
        kMmaK = 16,
        kMmaTileM = 2,
        kMmaTileN = 4,
        kWarpTileM = 4,
        kWarpTileN = 4,
        kWarpTileK = 2,
        kAPad = APad,
        kBPad = BPad,
        kBlockM = kMmaM * kMmaTileM * kWarpTileM,
        kBlockN = kMmaN * kMmaTileN * kWarpTileN,
        kBlockK = kMmaK,
        kThreads = kMmaTileM * kMmaTileN * kWarpSize,
        kSharedBytes = Stages * kBlockM * (kBlockK + kAPad) * kWarpTileK * static_cast<int>(sizeof(half)) +
            Stages * kBlockK * (kBlockN + kBPad) * kWarpTileK * static_cast<int>(sizeof(half))
    };
    // WARP_TILE_K=2 stores the 32-wide K tile as two 16-wide panels:
    //   A: [KStage][128][16+pad] for k0, then the same region for k1
    //   B: [KStage][16][128+pad] for k0, then the same region for k1
    // The two panels let ldmatrix keep the native m16n8k16 fragment shape.
    const dim3 block(kThreads);
    const dim3 grid = hgemm_mma_stage_grid(m, n, kBlockM, kBlockN, Swizzle, swizzle_stride);

    if constexpr(Kind == HgemmMmaStageDsmemKind::x2) {
        if(!ai_system::cuda_utils::check_status(
               cudaFuncSetAttribute(
                   hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_kernel<
                       kMmaM,
                       kMmaN,
                       kMmaK,
                       kMmaTileM,
                       kMmaTileN,
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
               "cudaFuncSetAttribute(hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_kernel)",
               error
           )) {
            return false;
        }
        hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_kernel<
            kMmaM,
            kMmaN,
            kMmaK,
            kMmaTileM,
            kMmaTileN,
            kWarpTileM,
            kWarpTileN,
            kWarpTileK,
            kAPad,
            kBPad,
            Stages,
            Swizzle><<<grid, block, kSharedBytes>>>(a, b, c, m, n, k);
    } else if constexpr(Kind == HgemmMmaStageDsmemKind::x4) {
        if(!ai_system::cuda_utils::check_status(
               cudaFuncSetAttribute(
                   hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4_kernel<
                       kMmaM,
                       kMmaN,
                       kMmaK,
                       kMmaTileM,
                       kMmaTileN,
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
               "cudaFuncSetAttribute(hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4_kernel)",
               error
           )) {
            return false;
        }
        hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4_kernel<
            kMmaM,
            kMmaN,
            kMmaK,
            kMmaTileM,
            kMmaTileN,
            kWarpTileM,
            kWarpTileN,
            kWarpTileK,
            kAPad,
            kBPad,
            Stages,
            Swizzle><<<grid, block, kSharedBytes>>>(a, b, c, m, n, k);
    } else {
        if(!ai_system::cuda_utils::check_status(
               cudaFuncSetAttribute(
                   hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr_kernel<
                       kMmaM,
                       kMmaN,
                       kMmaK,
                       kMmaTileM,
                       kMmaTileN,
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
               "cudaFuncSetAttribute(hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr_kernel)",
               error
           )) {
            return false;
        }
        hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr_kernel<
            kMmaM,
            kMmaN,
            kMmaK,
            kMmaTileM,
            kMmaTileN,
            kWarpTileM,
            kWarpTileN,
            kWarpTileK,
            kAPad,
            kBPad,
            Stages,
            Swizzle><<<grid, block, kSharedBytes>>>(a, b, c, m, n, k);
    }

    return ai_system::cuda_utils::check_last_launch(error);
}

template <bool Swizzle>
bool hgemm_dispatch_mma_stage_static(
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
            return hgemm_launch_mma_m16n8k16_mma2x4_warp4x4_stages_static<2, Swizzle>(a, b, c, m, n, k, swizzle_stride, error);
        case 3:
            return hgemm_launch_mma_m16n8k16_mma2x4_warp4x4_stages_static<3, Swizzle>(a, b, c, m, n, k, swizzle_stride, error);
        case 4:
            return hgemm_launch_mma_m16n8k16_mma2x4_warp4x4_stages_static<4, Swizzle>(a, b, c, m, n, k, swizzle_stride, error);
        default:
            return hgemm_launch_mma_m16n8k16_mma2x4_warp4x4_stages_static<2, Swizzle>(a, b, c, m, n, k, swizzle_stride, error);
    }
}

template <bool Swizzle>
bool hgemm_dispatch_mma_stage_dsmem(
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
            return hgemm_launch_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem<2, Swizzle>(a, b, c, m, n, k, swizzle_stride, error);
        case 3:
            return hgemm_launch_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem<3, Swizzle>(a, b, c, m, n, k, swizzle_stride, error);
        case 4:
            return hgemm_launch_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem<4, Swizzle>(a, b, c, m, n, k, swizzle_stride, error);
        case 5:
            return hgemm_launch_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem<5, Swizzle>(a, b, c, m, n, k, swizzle_stride, error);
        default:
            return hgemm_launch_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem<2, Swizzle>(a, b, c, m, n, k, swizzle_stride, error);
    }
}

template <bool Swizzle>
bool hgemm_dispatch_mma_stage_dsmem_tn(
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
            return hgemm_launch_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn<2, Swizzle>(
                a, b, c, m, n, k, swizzle_stride, error
            );
        case 3:
            return hgemm_launch_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn<3, Swizzle>(
                a, b, c, m, n, k, swizzle_stride, error
            );
        case 4:
            return hgemm_launch_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn<4, Swizzle>(
                a, b, c, m, n, k, swizzle_stride, error
            );
        case 5:
            return hgemm_launch_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn<5, Swizzle>(
                a, b, c, m, n, k, swizzle_stride, error
            );
        default:
            return hgemm_launch_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn<2, Swizzle>(
                a, b, c, m, n, k, swizzle_stride, error
            );
    }
}

template <HgemmMmaStageDsmemKind Kind, bool Swizzle>
bool hgemm_dispatch_mma_stage_x2_dsmem(
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
            return hgemm_launch_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem<Kind, 2, Swizzle>(a, b, c, m, n, k, swizzle_stride, error);
        case 3:
            return hgemm_launch_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem<Kind, 3, Swizzle>(a, b, c, m, n, k, swizzle_stride, error);
        case 4:
            return hgemm_launch_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem<Kind, 4, Swizzle>(a, b, c, m, n, k, swizzle_stride, error);
        case 5:
            return hgemm_launch_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem<Kind, 5, Swizzle, 0, 16>(
                a, b, c, m, n, k, swizzle_stride, error
            );
        default:
            return hgemm_launch_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem<Kind, 2, Swizzle>(a, b, c, m, n, k, swizzle_stride, error);
    }
}

#undef HMMA16816
#undef STMATRIX_X2
#undef LDMATRIX_X4_T
#undef LDMATRIX_X2_T
#undef LDMATRIX_X4
#undef LDMATRIX_X2
#undef CP_ASYNC_CG
#undef CP_ASYNC_WAIT_GROUP
#undef CP_ASYNC_COMMIT_GROUP
#undef LDST128BITS
#undef LDST32BITS
#undef WARP_SIZE

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

bool hgemm_mma_m16n8k16_mma2x4_warp4x4_stages(
    const half* a,
    const half* b,
    half* c,
    int m,
    int n,
    int k,
    int stages,
    bool swizzle,
    int swizzle_stride,
    std::string& error
) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_kernel_launch");
    if(!validate_stage_options(stages, swizzle, swizzle_stride, error) || !validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    if(swizzle) {
        return hgemm_dispatch_mma_stage_static<true>(a, b, c, m, n, k, stages, swizzle_stride, error);
    }
    return hgemm_dispatch_mma_stage_static<false>(a, b, c, m, n, k, stages, swizzle_stride, error);
}

bool hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem(
    const half* a,
    const half* b,
    half* c,
    int m,
    int n,
    int k,
    int stages,
    bool swizzle,
    int swizzle_stride,
    std::string& error
) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_kernel_launch");
    if(!validate_stage_options(stages, swizzle, swizzle_stride, error) || !validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    if(swizzle) {
        return hgemm_dispatch_mma_stage_dsmem<true>(a, b, c, m, n, k, stages, swizzle_stride, error);
    }
    return hgemm_dispatch_mma_stage_dsmem<false>(a, b, c, m, n, k, stages, swizzle_stride, error);
}

bool hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn(
    const half* a,
    const half* b,
    half* c,
    int m,
    int n,
    int k,
    int stages,
    bool swizzle,
    int swizzle_stride,
    std::string& error
) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_kernel_launch");
    if(!validate_stage_options(stages, swizzle, swizzle_stride, error) || !validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    if(swizzle) {
        return hgemm_dispatch_mma_stage_dsmem_tn<true>(a, b, c, m, n, k, stages, swizzle_stride, error);
    }
    return hgemm_dispatch_mma_stage_dsmem_tn<false>(a, b, c, m, n, k, stages, swizzle_stride, error);
}

bool hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem(
    const half* a,
    const half* b,
    half* c,
    int m,
    int n,
    int k,
    int stages,
    bool swizzle,
    int swizzle_stride,
    std::string& error
) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_kernel_launch");
    if(!validate_stage_options(stages, swizzle, swizzle_stride, error) || !validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    if(swizzle) {
        return hgemm_dispatch_mma_stage_x2_dsmem<HgemmMmaStageDsmemKind::x2, true>(
            a, b, c, m, n, k, stages, swizzle_stride, error
        );
    }
    return hgemm_dispatch_mma_stage_x2_dsmem<HgemmMmaStageDsmemKind::x2, false>(
        a, b, c, m, n, k, stages, swizzle_stride, error
    );
}

bool hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4(
    const half* a,
    const half* b,
    half* c,
    int m,
    int n,
    int k,
    int stages,
    bool swizzle,
    int swizzle_stride,
    std::string& error
) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4_kernel_launch");
    if(!validate_stage_options(stages, swizzle, swizzle_stride, error) || !validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    if(swizzle) {
        return hgemm_dispatch_mma_stage_x2_dsmem<HgemmMmaStageDsmemKind::x4, true>(
            a, b, c, m, n, k, stages, swizzle_stride, error
        );
    }
    return hgemm_dispatch_mma_stage_x2_dsmem<HgemmMmaStageDsmemKind::x4, false>(
        a, b, c, m, n, k, stages, swizzle_stride, error
    );
}

bool hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr(
    const half* a,
    const half* b,
    half* c,
    int m,
    int n,
    int k,
    int stages,
    bool swizzle,
    int swizzle_stride,
    std::string& error
) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr_kernel_launch");
    if(!validate_stage_options(stages, swizzle, swizzle_stride, error) || !validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    if(swizzle) {
        return hgemm_dispatch_mma_stage_x2_dsmem<HgemmMmaStageDsmemKind::rr, true>(
            a, b, c, m, n, k, stages, swizzle_stride, error
        );
    }
    return hgemm_dispatch_mma_stage_x2_dsmem<HgemmMmaStageDsmemKind::rr, false>(
        a, b, c, m, n, k, stages, swizzle_stride, error
    );
}

}  // namespace ai_system::labs::hgemm
