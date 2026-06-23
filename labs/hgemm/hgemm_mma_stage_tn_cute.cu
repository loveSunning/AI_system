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

#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
#define CP_ASYNC_CG(dst, src, bytes) \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
#define LDMATRIX_X2(R0, R1, addr) \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))
#define LDMATRIX_X4(R0, R1, R2, R3, addr) \
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(addr))
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1) \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" : "=r"(RD0), "=r"(RD1) : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))

__host__ __device__ inline int div_ceil(int lhs, int rhs) {
    return (lhs + rhs - 1) / rhs;
}

__device__ __forceinline__ half zero_half() {
    return __float2half_rn(0.0f);
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

bool validate_cute_stage_options(int stages, bool swizzle, int swizzle_stride, std::string& error) {
    if(stages < 1 || stages > 8) {
        error = "CuTe-style HGEMM stages kernel expects stages in [1, 8].";
        return false;
    }
    if(swizzle && swizzle_stride <= 0) {
        error = "CuTe-style HGEMM stages kernel expects a positive swizzle_stride when swizzle is enabled.";
        return false;
    }
    return true;
}

dim3 hgemm_cute_stage_grid(int m, int n, int block_m, int block_n, bool swizzle, int swizzle_stride) {
    const auto grid_y = static_cast<unsigned int>((m + block_m - 1) / block_m);
    const auto n_tiles = static_cast<unsigned int>((n + block_n - 1) / block_n);
    if(!swizzle) {
        return dim3(n_tiles, grid_y);
    }

    const auto n_swizzle = static_cast<unsigned int>((n + swizzle_stride - 1) / swizzle_stride);
    return dim3((n_tiles + n_swizzle - 1) / n_swizzle, grid_y, n_swizzle);
}

template <int BlockM, int BlockN>
__device__ void hgemm_scalar_tile_tn_half_acc_body_at(
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
            half accumulator = zero_half();
            for(int inner = 0; inner < k; ++inner) {
                accumulator = __hfma(a[row * k + inner], b[col * k + inner], accumulator);
            }
            c[row * n + col] = accumulator;
        }
    }
}

template <
    int K_STAGE,
    bool BLOCK_SWIZZLE,
    int MMA_M = 16,
    int MMA_N = 8,
    int MMA_K = 16,
    int CTA_M = 128,
    int CTA_N = 256,
    int CTA_K = 32,
    int WARP_TILE_M = 4,
    int WARP_TILE_N = 4>
__global__ void __launch_bounds__(512) hgemm_mma_stages_block_swizzle_tn_cute_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M,
    int N,
    int K
) {
    static_assert(CTA_K == 2 * MMA_K);
    static_assert(CTA_M == 2 * MMA_M * WARP_TILE_M);
    static_assert(CTA_N == 8 * MMA_N * WARP_TILE_N);

    const int bx = static_cast<int>(BLOCK_SWIZZLE) * static_cast<int>(blockIdx.z) * static_cast<int>(gridDim.x) +
        static_cast<int>(blockIdx.x);
    const int by = static_cast<int>(blockIdx.y);
    const int block_m = by * CTA_M;
    const int block_n = bx * CTA_N;
    if(block_m >= M || block_n >= N) {
        return;
    }

    const int num_k_tiles = div_ceil(K, CTA_K);

    extern __shared__ half smem[];
    half* s_a = smem;
    half* s_b = smem + K_STAGE * CTA_M * CTA_K;
    constexpr int s_a_stage_offset = CTA_M * CTA_K;
    constexpr int s_b_stage_offset = CTA_N * CTA_K;

    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid / kWarpSize;
    const int lane_id = tid & (kWarpSize - 1);
    const int warp_m = warp_id & 1;
    const int warp_n = warp_id >> 1;

    // Edge CTAs and non-32 K sizes use a half-accumulate scalar fallback.
    // The fast path below assumes every cp.async and ldmatrix address is in
    // range and 16-byte aligned.
    if(block_m + CTA_M > M || block_n + CTA_N > N || (K % CTA_K) != 0 || num_k_tiles < K_STAGE) {
        hgemm_scalar_tile_tn_half_acc_body_at<CTA_M, CTA_N>(A, B, C, M, N, K, block_m, block_n);
        return;
    }

    std::uint32_t RC[WARP_TILE_M][WARP_TILE_N][2];
#pragma unroll
    for(int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for(int j = 0; j < WARP_TILE_N; ++j) {
            RC[i][j][0] = 0;
            RC[i][j][1] = 0;
        }
    }

    const auto smem_a_base_ptr = static_cast<std::uint32_t>(__cvta_generic_to_shared(s_a));
    const auto smem_b_base_ptr = static_cast<std::uint32_t>(__cvta_generic_to_shared(s_b));

    const int load_a_vec = tid;
    const int load_smem_a_m = load_a_vec / (CTA_K / 8);
    const int load_smem_a_k = (load_a_vec % (CTA_K / 8)) * 8;
    const int load_gmem_a_m = block_m + load_smem_a_m;

#pragma unroll
    for(int k_tile = 0; k_tile < K_STAGE - 1; ++k_tile) {
        const int load_gmem_a_k = k_tile * CTA_K + load_smem_a_k;
        const std::uint32_t load_smem_a_ptr = smem_a_base_ptr +
            (k_tile * s_a_stage_offset + load_smem_a_m * CTA_K + load_smem_a_k) * sizeof(half);
        CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_m * K + load_gmem_a_k], 16);

#pragma unroll
        for(int copy = 0; copy < 2; ++copy) {
            const int load_b_vec = tid + copy * 512;
            const int load_smem_b_n = load_b_vec / (CTA_K / 8);
            const int load_smem_b_k = (load_b_vec % (CTA_K / 8)) * 8;
            const int load_gmem_b_n = block_n + load_smem_b_n;
            const int load_gmem_b_k = k_tile * CTA_K + load_smem_b_k;
            const std::uint32_t load_smem_b_ptr = smem_b_base_ptr +
                (k_tile * s_b_stage_offset + load_smem_b_n * CTA_K + load_smem_b_k) * sizeof(half);
            CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_n * K + load_gmem_b_k], 16);
        }
        CP_ASYNC_COMMIT_GROUP();
    }

    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();

#pragma unroll 1
    for(int k_tile = K_STAGE - 1; k_tile < num_k_tiles; ++k_tile) {
        const int smem_sel = (k_tile + 1) % K_STAGE;
        const int smem_sel_next = k_tile % K_STAGE;
        const int load_gmem_a_k = k_tile * CTA_K + load_smem_a_k;
        const std::uint32_t load_smem_a_ptr = smem_a_base_ptr +
            (smem_sel_next * s_a_stage_offset + load_smem_a_m * CTA_K + load_smem_a_k) * sizeof(half);
        CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_m * K + load_gmem_a_k], 16);

#pragma unroll
        for(int copy = 0; copy < 2; ++copy) {
            const int load_b_vec = tid + copy * 512;
            const int load_smem_b_n = load_b_vec / (CTA_K / 8);
            const int load_smem_b_k = (load_b_vec % (CTA_K / 8)) * 8;
            const int load_gmem_b_n = block_n + load_smem_b_n;
            const int load_gmem_b_k = k_tile * CTA_K + load_smem_b_k;
            const std::uint32_t load_smem_b_ptr = smem_b_base_ptr +
                (smem_sel_next * s_b_stage_offset + load_smem_b_n * CTA_K + load_smem_b_k) * sizeof(half);
            CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_n * K + load_gmem_b_k], 16);
        }
        CP_ASYNC_COMMIT_GROUP();

#pragma unroll
        for(int k_inner = 0; k_inner < CTA_K; k_inner += MMA_K) {
            std::uint32_t RA[WARP_TILE_M][4];
            std::uint32_t RB[WARP_TILE_N][2];

#pragma unroll
            for(int i = 0; i < WARP_TILE_M; ++i) {
                const int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
                const int lane_smem_a_m = warp_smem_a_m + lane_id % 16;
                const int lane_smem_a_k = k_inner + (lane_id / 16) * 8;
                const std::uint32_t lane_smem_a_ptr = smem_a_base_ptr +
                    (smem_sel * s_a_stage_offset + lane_smem_a_m * CTA_K + lane_smem_a_k) * sizeof(half);
                LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], lane_smem_a_ptr);
            }

#pragma unroll
            for(int j = 0; j < WARP_TILE_N; ++j) {
                const int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
                const int lane_smem_b_n = warp_smem_b_n + lane_id % 8;
                const int lane_smem_b_k = k_inner + ((lane_id / 8) & 1) * 8;
                const std::uint32_t lane_smem_b_ptr = smem_b_base_ptr +
                    (smem_sel * s_b_stage_offset + lane_smem_b_n * CTA_K + lane_smem_b_k) * sizeof(half);
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

#pragma unroll
        for(int k_inner = 0; k_inner < CTA_K; k_inner += MMA_K) {
            std::uint32_t RA[WARP_TILE_M][4];
            std::uint32_t RB[WARP_TILE_N][2];

#pragma unroll
            for(int i = 0; i < WARP_TILE_M; ++i) {
                const int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
                const int lane_smem_a_m = warp_smem_a_m + lane_id % 16;
                const int lane_smem_a_k = k_inner + (lane_id / 16) * 8;
                const std::uint32_t lane_smem_a_ptr = smem_a_base_ptr +
                    (stage_sel * s_a_stage_offset + lane_smem_a_m * CTA_K + lane_smem_a_k) * sizeof(half);
                LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], lane_smem_a_ptr);
            }

#pragma unroll
            for(int j = 0; j < WARP_TILE_N; ++j) {
                const int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
                const int lane_smem_b_n = warp_smem_b_n + lane_id % 8;
                const int lane_smem_b_k = k_inner + ((lane_id / 8) & 1) * 8;
                const std::uint32_t lane_smem_b_ptr = smem_b_base_ptr +
                    (stage_sel * s_b_stage_offset + lane_smem_b_n * CTA_K + lane_smem_b_k) * sizeof(half);
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
    }

#pragma unroll
    for(int i = 0; i < WARP_TILE_M; ++i) {
        std::uint32_t RC0[WARP_TILE_N][4];
        std::uint32_t RC1[WARP_TILE_N][4];
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
            const int store_c_m = block_m + store_warp_c_m + lane_id / 4;
#pragma unroll
            for(int j = 0; j < WARP_TILE_N; ++j) {
                const int store_warp_c_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
                const int store_c_n = block_n + store_warp_c_n;
                LDST128BITS(C[store_c_m * N + store_c_n]) = LDST128BITS(RC0[j][0]);
                LDST128BITS(C[(store_c_m + 8) * N + store_c_n]) = LDST128BITS(RC1[j][0]);
            }
        }
    }
}

template <int Stages, bool Swizzle>
bool hgemm_launch_mma_stages_block_swizzle_tn_cute(
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
        kBlockM = 128,
        kBlockN = 256,
        kBlockK = 32,
        kThreads = 16 * kWarpSize,
        kSharedBytes = Stages * (kBlockM * kBlockK + kBlockN * kBlockK) * static_cast<int>(sizeof(half))
    };

    if(!ai_system::cuda_utils::check_status(
           cudaFuncSetAttribute(
               hgemm_mma_stages_block_swizzle_tn_cute_kernel<Stages, Swizzle>,
               cudaFuncAttributeMaxDynamicSharedMemorySize,
               kSharedBytes
           ),
           "cudaFuncSetAttribute(hgemm_mma_stages_block_swizzle_tn_cute_kernel)",
           error
       )) {
        return false;
    }

    const dim3 block(kThreads);
    const dim3 grid = hgemm_cute_stage_grid(m, n, kBlockM, kBlockN, Swizzle, swizzle_stride);
    hgemm_mma_stages_block_swizzle_tn_cute_kernel<Stages, Swizzle>
        <<<grid, block, kSharedBytes>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

template <bool Swizzle>
bool hgemm_dispatch_mma_stages_block_swizzle_tn_cute(
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
            return hgemm_launch_mma_stages_block_swizzle_tn_cute<2, Swizzle>(
                a, b, c, m, n, k, swizzle_stride, error
            );
        case 3:
            return hgemm_launch_mma_stages_block_swizzle_tn_cute<3, Swizzle>(
                a, b, c, m, n, k, swizzle_stride, error
            );
        case 4:
            return hgemm_launch_mma_stages_block_swizzle_tn_cute<4, Swizzle>(
                a, b, c, m, n, k, swizzle_stride, error
            );
        default:
            return hgemm_launch_mma_stages_block_swizzle_tn_cute<2, Swizzle>(
                a, b, c, m, n, k, swizzle_stride, error
            );
    }
}

}  // namespace

bool hgemm_mma_stages_block_swizzle_tn_cute(
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
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_mma_stages_block_swizzle_tn_cute_kernel_launch");
    if(!validate_cute_stage_options(stages, swizzle, swizzle_stride, error) ||
       !validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    if(swizzle) {
        return hgemm_dispatch_mma_stages_block_swizzle_tn_cute<true>(
            a, b, c, m, n, k, stages, swizzle_stride, error
        );
    }
    return hgemm_dispatch_mma_stages_block_swizzle_tn_cute<false>(
        a, b, c, m, n, k, stages, swizzle_stride, error
    );
}

}  // namespace ai_system::labs::hgemm
