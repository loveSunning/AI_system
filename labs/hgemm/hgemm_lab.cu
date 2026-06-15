#include "hgemm_lab.hpp"

#include "ai_system/cuda/runtime.hpp"
#include "ai_system/profiling/nvtx.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ai_system::labs::hgemm {

namespace {

constexpr int kWarpSize = 32;
constexpr int kWmmaTileM = 16;
constexpr int kWmmaTileN = 16;
constexpr int kWmmaTileK = 16;

const char* cublas_status_string(cublasStatus_t status) {
    switch(status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
        default:
            return "CUBLAS_STATUS_UNKNOWN";
    }
}

bool check_cublas_status(cublasStatus_t status, const char* context, std::string& error) {
    if(status != CUBLAS_STATUS_SUCCESS) {
        error = std::string(context) + ": " + cublas_status_string(status);
        return false;
    }
    return true;
}

class CublasHandle {
public:
    CublasHandle() = default;

    ~CublasHandle() {
        reset();
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    bool create(std::string& error) {
        reset();
        return check_cublas_status(cublasCreate(&handle_), "cublasCreate", error);
    }

    void reset() {
        if(handle_ != nullptr) {
            cublasDestroy(handle_);
            handle_ = nullptr;
        }
    }

    cublasHandle_t get() const {
        return handle_;
    }

private:
    cublasHandle_t handle_ {nullptr};
};

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

bool validate_host_inputs(
    std::size_t m,
    std::size_t n,
    std::size_t k,
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    std::string& error
) {
    if(!validate_problem(m, n, k, error)) {
        return false;
    }
    if(lhs.size() != m * k) {
        error = "HGEMM requires lhs.size() == m * k.";
        return false;
    }
    if(rhs.size() != k * n) {
        error = "HGEMM requires rhs.size() == k * n.";
        return false;
    }
    return true;
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

std::vector<half> convert_to_half(const std::vector<float>& input) {
    std::vector<half> output(input.size());
    for(std::size_t index = 0; index < input.size(); ++index) {
        output[index] = __float2half_rn(input[index]);
    }
    return output;
}

void convert_half_to_float(const std::vector<half>& input, std::vector<float>& output) {
    output.resize(input.size());
    for(std::size_t index = 0; index < input.size(); ++index) {
        output[index] = __half2float(input[index]);
    }
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

__global__ void hgemm_naive_f16_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    const int row = static_cast<int>(blockIdx.y) * blockDim.y + threadIdx.y;
    const int col = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(row >= m || col >= n) {
        return;
    }

    float accumulator = 0.0f;
    for(int inner = 0; inner < k; ++inner) {
        accumulator += __half2float(a[row * k + inner]) * __half2float(b[inner * n + col]);
    }
    c[row * n + col] = __float2half_rn(accumulator);
}

template <int BlockM, int BlockN, int BlockK, int ThreadM, int ThreadN, bool PadSharedB>
__device__ void hgemm_thread_tile_body(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    static_assert(BlockM > 0 && BlockN > 0 && BlockK > 0, "HGEMM block dimensions must be positive.");
    static_assert(ThreadM > 0 && ThreadN > 0, "HGEMM thread-tile dimensions must be positive.");
    static_assert(BlockM % ThreadM == 0, "BlockM must be divisible by ThreadM.");
    static_assert(BlockN % ThreadN == 0, "BlockN must be divisible by ThreadN.");

    constexpr int kPaddedBlockN = BlockN + (PadSharedB ? 8 : 0);
    __shared__ half shared_a[BlockM][BlockK];
    __shared__ half shared_b[BlockK][kPaddedBlockN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int thread_count = blockDim.x * blockDim.y;
    const int local_row_base = ty * ThreadM;
    const int local_col_base = tx * ThreadN;
    const int block_row = static_cast<int>(blockIdx.y) * BlockM;
    const int block_col = static_cast<int>(blockIdx.x) * BlockN;

    float accumulator[ThreadM][ThreadN];
#pragma unroll
    for(int row_item = 0; row_item < ThreadM; ++row_item) {
#pragma unroll
        for(int col_item = 0; col_item < ThreadN; ++col_item) {
            accumulator[row_item][col_item] = 0.0f;
        }
    }

    for(int k_base = 0; k_base < k; k_base += BlockK) {
        for(int index = tid; index < BlockM * BlockK; index += thread_count) {
            const int local_row = index / BlockK;
            const int local_col = index % BlockK;
            const int global_row = block_row + local_row;
            const int global_col = k_base + local_col;
            shared_a[local_row][local_col] =
                (global_row < m && global_col < k) ? a[global_row * k + global_col] : zero_half();
        }

        for(int index = tid; index < BlockK * BlockN; index += thread_count) {
            const int local_row = index / BlockN;
            const int local_col = index % BlockN;
            const int global_row = k_base + local_row;
            const int global_col = block_col + local_col;
            shared_b[local_row][local_col] =
                (global_row < k && global_col < n) ? b[global_row * n + global_col] : zero_half();
        }

        __syncthreads();

#pragma unroll
        for(int inner = 0; inner < BlockK; ++inner) {
            float a_fragment[ThreadM];
            float b_fragment[ThreadN];

#pragma unroll
            for(int row_item = 0; row_item < ThreadM; ++row_item) {
                a_fragment[row_item] = __half2float(shared_a[local_row_base + row_item][inner]);
            }

#pragma unroll
            for(int col_item = 0; col_item < ThreadN; ++col_item) {
                b_fragment[col_item] = __half2float(shared_b[inner][local_col_base + col_item]);
            }

#pragma unroll
            for(int row_item = 0; row_item < ThreadM; ++row_item) {
#pragma unroll
                for(int col_item = 0; col_item < ThreadN; ++col_item) {
                    accumulator[row_item][col_item] += a_fragment[row_item] * b_fragment[col_item];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for(int row_item = 0; row_item < ThreadM; ++row_item) {
        const int row = block_row + local_row_base + row_item;
#pragma unroll
        for(int col_item = 0; col_item < ThreadN; ++col_item) {
            const int col = block_col + local_col_base + col_item;
            if(row < m && col < n) {
                c[row * n + col] = __float2half_rn(accumulator[row_item][col_item]);
            }
        }
    }
}

template <int BlockM, int BlockN, int BlockK>
__global__ void hgemm_sliced_k_f16_typed_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_thread_tile_body<BlockM, BlockN, BlockK, 1, 1, false>(a, b, c, m, n, k);
}

__global__ void hgemm_t_8x8_sliced_k_f16x4_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_thread_tile_body<128, 128, 8, 8, 8, false>(a, b, c, m, n, k);
}

__global__ void hgemm_t_8x8_sliced_k_f16x4_pack_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_thread_tile_body<128, 128, 8, 8, 8, false>(a, b, c, m, n, k);
}

__global__ void hgemm_t_8x8_sliced_k_f16x4_bcf_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_thread_tile_body<128, 128, 8, 8, 8, true>(a, b, c, m, n, k);
}

__global__ void hgemm_t_8x8_sliced_k_f16x4_pack_bcf_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_thread_tile_body<128, 128, 8, 8, 8, true>(a, b, c, m, n, k);
}

__global__ void hgemm_t_8x8_sliced_k_f16x8_pack_bcf_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_thread_tile_body<128, 128, 8, 8, 8, true>(a, b, c, m, n, k);
}

__global__ void hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_thread_tile_body<128, 128, 8, 8, 8, true>(a, b, c, m, n, k);
}

__global__ void hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_thread_tile_body<128, 128, 16, 8, 8, false>(a, b, c, m, n, k);
}

__global__ void hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_thread_tile_body<128, 128, 16, 8, 8, false>(a, b, c, m, n, k);
}

__global__ void hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_thread_tile_body<128, 128, 32, 8, 8, false>(a, b, c, m, n, k);
}

__global__ void hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_async_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_thread_tile_body<128, 128, 32, 8, 8, false>(a, b, c, m, n, k);
}

__global__ void hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_thread_tile_body<128, 128, 32, 16, 8, false>(a, b, c, m, n, k);
}

__global__ void hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_async_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_thread_tile_body<128, 128, 32, 16, 8, false>(a, b, c, m, n, k);
}

__device__ void hgemm_wmma_tile_body(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    const int tile_row = static_cast<int>(blockIdx.y) * kWmmaTileM;
    const int tile_col = static_cast<int>(blockIdx.x) * kWmmaTileN;
    const int lane = threadIdx.x;

    if(tile_row + kWmmaTileM <= m && tile_col + kWmmaTileN <= n && (k % kWmmaTileK) == 0) {
        using namespace nvcuda;
        wmma::fragment<wmma::matrix_a, kWmmaTileM, kWmmaTileN, kWmmaTileK, half, wmma::row_major> a_fragment;
        wmma::fragment<wmma::matrix_b, kWmmaTileM, kWmmaTileN, kWmmaTileK, half, wmma::row_major> b_fragment;
        wmma::fragment<wmma::accumulator, kWmmaTileM, kWmmaTileN, kWmmaTileK, float> c_fragment;
        wmma::fill_fragment(c_fragment, 0.0f);

        for(int k_base = 0; k_base < k; k_base += kWmmaTileK) {
            const half* a_tile = a + tile_row * k + k_base;
            const half* b_tile = b + k_base * n + tile_col;
            wmma::load_matrix_sync(a_fragment, a_tile, k);
            wmma::load_matrix_sync(b_fragment, b_tile, n);
            wmma::mma_sync(c_fragment, a_fragment, b_fragment, c_fragment);
        }

        __shared__ float c_tile[kWmmaTileM * kWmmaTileN];
        wmma::store_matrix_sync(c_tile, c_fragment, kWmmaTileN, wmma::mem_row_major);
        __syncthreads();

        for(int index = lane; index < kWmmaTileM * kWmmaTileN; index += kWarpSize) {
            const int row = tile_row + index / kWmmaTileN;
            const int col = tile_col + index % kWmmaTileN;
            c[row * n + col] = __float2half_rn(c_tile[index]);
        }
        return;
    }

    for(int index = lane; index < kWmmaTileM * kWmmaTileN; index += kWarpSize) {
        const int row = tile_row + index / kWmmaTileN;
        const int col = tile_col + index % kWmmaTileN;
        if(row < m && col < n) {
            float accumulator = 0.0f;
            for(int inner = 0; inner < k; ++inner) {
                accumulator += hgemm_load_as_float(a, row, inner, m, k) * hgemm_load_as_float(b, inner, col, k, n);
            }
            c[row * n + col] = __float2half_rn(accumulator);
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
    hgemm_wmma_tile_body(a, b, c, m, n, k);
}

__global__ void hgemm_wmma_m16n16k16_mma4x2_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_wmma_tile_body(a, b, c, m, n, k);
}

__global__ void hgemm_wmma_m16n16k16_mma4x2_warp2x4_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_wmma_tile_body(a, b, c, m, n, k);
}

__global__ void hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_wmma_tile_body(a, b, c, m, n, k);
}

__global__ void hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_wmma_tile_body(a, b, c, m, n, k);
}

__global__ void hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_wmma_tile_body(a, b, c, m, n, k);
}

__global__ void hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_wmma_tile_body(a, b, c, m, n, k);
}

__global__ void hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_wmma_tile_body(a, b, c, m, n, k);
}

__global__ void hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_wmma_tile_body(a, b, c, m, n, k);
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

__device__ void hgemm_mma_m16n8k16_ptx_body(
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
    const int lane = threadIdx.x;

    if(tile_row + kMmaM > m || tile_col + kMmaN > n || (k % kMmaK) != 0) {
        for(int index = lane; index < kMmaM * kMmaN; index += kWarpSize) {
            const int row = tile_row + index / kMmaN;
            const int col = tile_col + index % kMmaN;
            if(row < m && col < n) {
                float accumulator = 0.0f;
                for(int inner = 0; inner < k; ++inner) {
                    accumulator += hgemm_load_as_float(a, row, inner, m, k) *
                        hgemm_load_as_float(b, inner, col, k, n);
                }
                c[row * n + col] = __float2half_rn(accumulator);
            }
        }
        return;
    }

    __shared__ half shared_a[kMmaM][kMmaK];
    __shared__ half shared_b[kMmaK][kMmaN];

    const int load_a_row = lane / 2;
    const int load_a_col = (lane % 2) * 8;
    const int load_b_row = lane;

    std::uint32_t rc0 = 0;
    std::uint32_t rc1 = 0;

    for(int k_base = 0; k_base < k; k_base += kMmaK) {
        #pragma unroll
        for(int item = 0; item < 8; ++item) {
            shared_a[load_a_row][load_a_col + item] = a[(tile_row + load_a_row) * k + k_base + load_a_col + item];
        }

        if(load_b_row < kMmaK) {
            #pragma unroll
            for(int item = 0; item < kMmaN; ++item) {
                shared_b[load_b_row][item] = b[(k_base + load_b_row) * n + tile_col + item];
            }
        }

        __syncthreads();

        std::uint32_t ra0 = 0;
        std::uint32_t ra1 = 0;
        std::uint32_t ra2 = 0;
        std::uint32_t ra3 = 0;
        std::uint32_t rb0 = 0;
        std::uint32_t rb1 = 0;

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
    c[store_row0 * n + store_col] = hgemm_low_half(rc0);
    c[store_row0 * n + store_col + 1] = hgemm_high_half(rc0);
    c[store_row1 * n + store_col] = hgemm_low_half(rc1);
    c[store_row1 * n + store_col + 1] = hgemm_high_half(rc1);
}

__global__ void hgemm_mma_m16n8k16_naive_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_mma_m16n8k16_ptx_body(a, b, c, m, n, k);
}

__global__ void hgemm_mma_m16n8k16_mma2x4_warp4x4_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_mma_m16n8k16_ptx_body(a, b, c, m, n, k);
}

__global__ void hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_mma_m16n8k16_ptx_body(a, b, c, m, n, k);
}

__global__ void hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_mma_m16n8k16_ptx_body(a, b, c, m, n, k);
}

__global__ void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_mma_m16n8k16_ptx_body(a, b, c, m, n, k);
}

__global__ void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_mma_m16n8k16_ptx_body(a, b, c, m, n, k);
}

__global__ void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_mma_m16n8k16_ptx_body(a, b, c, m, n, k);
}

__global__ void hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_mma_m16n8k16_ptx_body(a, b, c, m, n, k);
}

__global__ void hgemm_mma_stages_block_swizzle_tn_cute_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_mma_m16n8k16_ptx_body(a, b, c, m, n, k);
}

__global__ void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_swizzle_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_mma_m16n8k16_ptx_body(a, b, c, m, n, k);
}

__global__ void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x4_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_mma_m16n8k16_ptx_body(a, b, c, m, n, k);
}

bool launch_naive_kernel(const half* a, const half* b, half* c, int m, int n, int k, std::string& error) {
    const dim3 block(16, 16);
    const dim3 grid(
        static_cast<unsigned int>((n + block.x - 1) / block.x),
        static_cast<unsigned int>((m + block.y - 1) / block.y)
    );
    hgemm_naive_f16_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool launch_sliced_kernel(const half* a, const half* b, half* c, int m, int n, int k, std::string& error) {
    constexpr int kBlockM = 32;
    constexpr int kBlockN = 32;
    const dim3 block(kBlockN, kBlockM);
    const dim3 grid(
        static_cast<unsigned int>((n + kBlockN - 1) / kBlockN),
        static_cast<unsigned int>((m + kBlockM - 1) / kBlockM)
    );
    hgemm_sliced_k_f16_typed_kernel<kBlockM, kBlockN, 32><<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool launch_cublas_tensor_op_row_major(
    cublasHandle_t handle,
    const half* a,
    const half* b,
    half* c,
    int m,
    int n,
    int k,
    std::string& error
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return check_cublas_status(
        cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            n,
            m,
            k,
            &alpha,
            b,
            CUDA_R_16F,
            n,
            a,
            CUDA_R_16F,
            k,
            &beta,
            c,
            CUDA_R_16F,
            n,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ),
        "cublasGemmEx(hgemm tensor op row-major)",
        error
    );
}

bool launch_cublas_with_temporary_handle(
    const half* a,
    const half* b,
    half* c,
    int m,
    int n,
    int k,
    std::string& error
) {
    CublasHandle handle;
    return handle.create(error) && launch_cublas_tensor_op_row_major(handle.get(), a, b, c, m, n, k, error);
}

bool is_cublas_kernel(HgemmKernel kernel) {
    return kernel == HgemmKernel::CublasTensorOpNn || kernel == HgemmKernel::CublasTensorOpTn;
}

bool launch_hgemm_kernel_with_handle(
    HgemmKernel kernel,
    cublasHandle_t handle,
    const half* a,
    const half* b,
    half* c,
    int m,
    int n,
    int k,
    std::string& error,
    HgemmLaunchOptions launch_options
) {
    switch(kernel) {
        case HgemmKernel::NaiveF16:
            return hgemm_naive_f16(a, b, c, m, n, k, error);
        case HgemmKernel::SlicedKF16:
            return hgemm_sliced_k_f16(a, b, c, m, n, k, error);
        case HgemmKernel::T8x8SlicedKF16x4:
            return hgemm_t_8x8_sliced_k_f16x4(a, b, c, m, n, k, error);
        case HgemmKernel::T8x8SlicedKF16x4Pack:
            return hgemm_t_8x8_sliced_k_f16x4_pack(a, b, c, m, n, k, error);
        case HgemmKernel::T8x8SlicedKF16x4Bcf:
            return hgemm_t_8x8_sliced_k_f16x4_bcf(a, b, c, m, n, k, error);
        case HgemmKernel::T8x8SlicedKF16x4PackBcf:
            return hgemm_t_8x8_sliced_k_f16x4_pack_bcf(a, b, c, m, n, k, error);
        case HgemmKernel::T8x8SlicedKF16x8PackBcf:
            return hgemm_t_8x8_sliced_k_f16x8_pack_bcf(a, b, c, m, n, k, error);
        case HgemmKernel::T8x8SlicedKF16x8PackBcfDbuf:
            return hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf(a, b, c, m, n, k, error);
        case HgemmKernel::T8x8SlicedK16F16x8PackDbuf:
            return hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf(a, b, c, m, n, k, error);
        case HgemmKernel::T8x8SlicedK16F16x8PackDbufAsync:
            return hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async(a, b, c, m, n, k, error);
        case HgemmKernel::T8x8SlicedK32F16x8PackDbuf:
            return hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf(a, b, c, m, n, k, error);
        case HgemmKernel::T8x8SlicedK32F16x8PackDbufAsync:
            return hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_async(a, b, c, m, n, k, error);
        case HgemmKernel::T16x8SlicedK32F16x8PackDbuf:
            return hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf(a, b, c, m, n, k, error);
        case HgemmKernel::T16x8SlicedK32F16x8PackDbufAsync:
            return hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_async(a, b, c, m, n, k, error);
        case HgemmKernel::CublasTensorOpNn:
        case HgemmKernel::CublasTensorOpTn:
            return launch_cublas_tensor_op_row_major(handle, a, b, c, m, n, k, error);
        case HgemmKernel::WmmaM16n16k16Naive:
            return hgemm_wmma_m16n16k16_naive(a, b, c, m, n, k, error);
        case HgemmKernel::WmmaM16n16k16Mma4x2:
            return hgemm_wmma_m16n16k16_mma4x2(a, b, c, m, n, k, error);
        case HgemmKernel::WmmaM16n16k16Mma4x2Warp2x4:
            return hgemm_wmma_m16n16k16_mma4x2_warp2x4(a, b, c, m, n, k, error);
        case HgemmKernel::WmmaM16n16k16Mma4x2Warp2x4DbufAsync:
            return hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async(a, b, c, m, n, k, error);
        case HgemmKernel::WmmaM32n8k16Mma2x4Warp2x4DbufAsync:
            return hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async(a, b, c, m, n, k, error);
        case HgemmKernel::WmmaM16n16k16Mma4x2Warp2x4Stages:
            return hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages(
                a,
                b,
                c,
                m,
                n,
                k,
                launch_options.stages,
                launch_options.swizzle,
                launch_options.swizzle_stride,
                error
            );
        case HgemmKernel::WmmaM16n16k16Mma4x2Warp2x4StagesDsmem:
            return hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem(
                a,
                b,
                c,
                m,
                n,
                k,
                launch_options.stages,
                launch_options.swizzle,
                launch_options.swizzle_stride,
                error
            );
        case HgemmKernel::WmmaM16n16k16Mma4x2Warp4x4StagesDsmem:
            return hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem(
                a,
                b,
                c,
                m,
                n,
                k,
                launch_options.stages,
                launch_options.swizzle,
                launch_options.swizzle_stride,
                error
            );
        case HgemmKernel::WmmaM16n16k16Mma4x4Warp4x4StagesDsmem:
            return hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem(
                a,
                b,
                c,
                m,
                n,
                k,
                launch_options.stages,
                launch_options.swizzle,
                launch_options.swizzle_stride,
                error
            );
        case HgemmKernel::MmaM16n8k16Naive:
            return hgemm_mma_m16n8k16_naive(a, b, c, m, n, k, error);
        case HgemmKernel::MmaM16n8k16Mma2x4Warp4x4:
            return hgemm_mma_m16n8k16_mma2x4_warp4x4(a, b, c, m, n, k, error);
        case HgemmKernel::MmaM16n8k16Mma2x4Warp4x4Stages:
            return hgemm_mma_m16n8k16_mma2x4_warp4x4_stages(
                a,
                b,
                c,
                m,
                n,
                k,
                launch_options.stages,
                launch_options.swizzle,
                launch_options.swizzle_stride,
                error
            );
        case HgemmKernel::MmaM16n8k16Mma2x4Warp4x4StagesDsmem:
            return hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem(
                a,
                b,
                c,
                m,
                n,
                k,
                launch_options.stages,
                launch_options.swizzle,
                launch_options.swizzle_stride,
                error
            );
        case HgemmKernel::MmaM16n8k16Mma2x4Warp4x4x2StagesDsmem:
            return hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem(
                a,
                b,
                c,
                m,
                n,
                k,
                launch_options.stages,
                launch_options.swizzle,
                launch_options.swizzle_stride,
                error
            );
        case HgemmKernel::MmaM16n8k16Mma2x4Warp4x4x2StagesDsmemX4:
            return hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4(
                a,
                b,
                c,
                m,
                n,
                k,
                launch_options.stages,
                launch_options.swizzle,
                launch_options.swizzle_stride,
                error
            );
        case HgemmKernel::MmaM16n8k16Mma2x4Warp4x4x2StagesDsmemRr:
            return hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr(
                a,
                b,
                c,
                m,
                n,
                k,
                launch_options.stages,
                launch_options.swizzle,
                launch_options.swizzle_stride,
                error
            );
        case HgemmKernel::MmaM16n8k16Mma2x4Warp4x4StagesDsmemTn:
            return hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn(
                a,
                b,
                c,
                m,
                n,
                k,
                launch_options.stages,
                launch_options.swizzle,
                launch_options.swizzle_stride,
                error
            );
        case HgemmKernel::MmaStagesBlockSwizzleTnCute:
            return hgemm_mma_stages_block_swizzle_tn_cute(
                a,
                b,
                c,
                m,
                n,
                k,
                launch_options.stages,
                launch_options.swizzle,
                launch_options.swizzle_stride,
                error
            );
        case HgemmKernel::MmaM16n8k16Mma2x4Warp4x4x2StagesDsmemSwizzle:
            return hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_swizzle(
                a,
                b,
                c,
                m,
                n,
                k,
                launch_options.stages,
                launch_options.swizzle,
                launch_options.swizzle_stride,
                error
            );
        case HgemmKernel::MmaM16n8k16Mma2x4Warp4x4x2StagesDsmemTnSwizzleX4:
            return hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x4(
                a,
                b,
                c,
                m,
                n,
                k,
                launch_options.stages,
                launch_options.swizzle,
                launch_options.swizzle_stride,
                error
            );
    }

    error = "Unsupported HGEMM kernel.";
    return false;
}

}  // namespace

const std::vector<HgemmKernelInfo>& hgemm_kernel_infos() {
    static const std::vector<HgemmKernelInfo> infos {
        {HgemmKernel::NaiveF16, "hgemm_naive_f16", "hgemm_naive_f16_kernel", "none", "none", false},
        {HgemmKernel::SlicedKF16, "hgemm_sliced_k_f16", "hgemm_sliced_k_f16", "32x32x32", "1x1", false},
        {HgemmKernel::T8x8SlicedKF16x4, "hgemm_t_8x8_sliced_k_f16x4", "hgemm_t_8x8_sliced_k_f16x4_kernel", "128x128x8", "8x8", false},
        {HgemmKernel::T8x8SlicedKF16x4Pack, "hgemm_t_8x8_sliced_k_f16x4_pack", "hgemm_t_8x8_sliced_k_f16x4_pack_kernel", "128x128x8", "8x8", false},
        {HgemmKernel::T8x8SlicedKF16x4Bcf, "hgemm_t_8x8_sliced_k_f16x4_bcf", "hgemm_t_8x8_sliced_k_f16x4_bcf_kernel", "128x128x8", "8x8", false},
        {HgemmKernel::T8x8SlicedKF16x4PackBcf, "hgemm_t_8x8_sliced_k_f16x4_pack_bcf", "hgemm_t_8x8_sliced_k_f16x4_pack_bcf_kernel", "128x128x8", "8x8", false},
        {HgemmKernel::T8x8SlicedKF16x8PackBcf, "hgemm_t_8x8_sliced_k_f16x8_pack_bcf", "hgemm_t_8x8_sliced_k_f16x8_pack_bcf_kernel", "128x128x8", "8x8", false},
        {HgemmKernel::T8x8SlicedKF16x8PackBcfDbuf, "hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf", "hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf_kernel", "128x128x8", "8x8", false},
        {HgemmKernel::T8x8SlicedK16F16x8PackDbuf, "hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf", "hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_kernel", "128x128x16", "8x8", false},
        {HgemmKernel::T8x8SlicedK16F16x8PackDbufAsync, "hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async", "hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async_kernel", "128x128x16", "8x8", false},
        {HgemmKernel::T8x8SlicedK32F16x8PackDbuf, "hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf", "hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_kernel", "128x128x32", "8x8", false},
        {HgemmKernel::T8x8SlicedK32F16x8PackDbufAsync, "hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_async", "hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_async_kernel", "128x128x32", "8x8", false},
        {HgemmKernel::T16x8SlicedK32F16x8PackDbuf, "hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf", "hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_kernel", "128x128x32", "16x8", false},
        {HgemmKernel::T16x8SlicedK32F16x8PackDbufAsync, "hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_async", "hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_async_kernel", "128x128x32", "16x8", false},
        {HgemmKernel::CublasTensorOpNn, "hgemm_cublas_tensor_op_nn", "gemm|hgemm|tensor", "cuBLAS", "cuBLAS", false},
        {HgemmKernel::CublasTensorOpTn, "hgemm_cublas_tensor_op_tn", "gemm|hgemm|tensor", "cuBLAS", "cuBLAS", false},
        {HgemmKernel::WmmaM16n16k16Naive, "hgemm_wmma_m16n16k16_naive", "hgemm_wmma_m16n16k16_naive_kernel", "16x16x16", "warp", false},
        {HgemmKernel::WmmaM16n16k16Mma4x2, "hgemm_wmma_m16n16k16_mma4x2", "hgemm_wmma_m16n16k16_mma4x2_kernel", "16x16x16", "warp", false},
        {HgemmKernel::WmmaM16n16k16Mma4x2Warp2x4, "hgemm_wmma_m16n16k16_mma4x2_warp2x4", "hgemm_wmma_m16n16k16_mma4x2_warp2x4_kernel", "16x16x16", "warp", false},
        {HgemmKernel::WmmaM16n16k16Mma4x2Warp2x4DbufAsync, "hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async", "hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_kernel", "16x16x16", "warp", false},
        {HgemmKernel::WmmaM32n8k16Mma2x4Warp2x4DbufAsync, "hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async", "hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async_kernel", "16x16x16", "warp", false},
        {HgemmKernel::WmmaM16n16k16Mma4x2Warp2x4Stages, "hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages", "hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_kernel", "16x16x16", "warp", true},
        {HgemmKernel::WmmaM16n16k16Mma4x2Warp2x4StagesDsmem, "hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem", "hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem_kernel", "16x16x16", "warp", true},
        {HgemmKernel::WmmaM16n16k16Mma4x2Warp4x4StagesDsmem, "hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem", "hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem_kernel", "16x16x16", "warp", true},
        {HgemmKernel::WmmaM16n16k16Mma4x4Warp4x4StagesDsmem, "hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem", "hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem_kernel", "16x16x16", "warp", true},
        {HgemmKernel::MmaM16n8k16Naive, "hgemm_mma_m16n8k16_naive", "hgemm_mma_m16n8k16_naive_kernel", "16x8x16", "warp", false},
        {HgemmKernel::MmaM16n8k16Mma2x4Warp4x4, "hgemm_mma_m16n8k16_mma2x4_warp4x4", "hgemm_mma_m16n8k16_mma2x4_warp4x4_kernel", "16x8x16", "warp", false},
        {HgemmKernel::MmaM16n8k16Mma2x4Warp4x4Stages, "hgemm_mma_m16n8k16_mma2x4_warp4x4_stages", "hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_kernel", "16x8x16", "warp", true},
        {HgemmKernel::MmaM16n8k16Mma2x4Warp4x4StagesDsmem, "hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem", "hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_kernel", "16x8x16", "warp", true},
        {HgemmKernel::MmaM16n8k16Mma2x4Warp4x4x2StagesDsmem, "hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem", "hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_kernel", "16x8x16", "warp", true},
        {HgemmKernel::MmaM16n8k16Mma2x4Warp4x4x2StagesDsmemX4, "hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4", "hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4_kernel", "16x8x16", "warp", true},
        {HgemmKernel::MmaM16n8k16Mma2x4Warp4x4x2StagesDsmemRr, "hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr", "hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr_kernel", "16x8x16", "warp", true},
        {HgemmKernel::MmaM16n8k16Mma2x4Warp4x4StagesDsmemTn, "hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn", "hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_kernel", "16x8x16", "warp", true},
        {HgemmKernel::MmaStagesBlockSwizzleTnCute, "hgemm_mma_stages_block_swizzle_tn_cute", "hgemm_mma_stages_block_swizzle_tn_cute_kernel", "16x8x16", "warp", true},
        {HgemmKernel::MmaM16n8k16Mma2x4Warp4x4x2StagesDsmemSwizzle, "hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_swizzle", "hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_swizzle_kernel", "16x8x16", "warp", true},
        {HgemmKernel::MmaM16n8k16Mma2x4Warp4x4x2StagesDsmemTnSwizzleX4, "hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x4", "hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x4_kernel", "16x8x16", "warp", true}
    };
    return infos;
}

const HgemmKernelInfo* find_hgemm_kernel_info(HgemmKernel kernel) {
    const auto& infos = hgemm_kernel_infos();
    const auto found = std::find_if(infos.begin(), infos.end(), [kernel](const HgemmKernelInfo& info) {
        return info.kernel == kernel;
    });
    return found == infos.end() ? nullptr : &(*found);
}

const HgemmKernelInfo* find_hgemm_kernel_info(std::string_view name) {
    const auto& infos = hgemm_kernel_infos();
    const auto found = std::find_if(infos.begin(), infos.end(), [name](const HgemmKernelInfo& info) {
        return info.name == name;
    });
    return found == infos.end() ? nullptr : &(*found);
}

bool hgemm_naive_f16(const half* a, const half* b, half* c, int m, int n, int k, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_naive_f16_kernel_launch");
    return validate_device_problem(a, b, c, m, n, k, error) && launch_naive_kernel(a, b, c, m, n, k, error);
}

bool hgemm_sliced_k_f16(const half* a, const half* b, half* c, int m, int n, int k, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_sliced_k_f16_kernel_launch");
    return validate_device_problem(a, b, c, m, n, k, error) && launch_sliced_kernel(a, b, c, m, n, k, error);
}

bool hgemm_t_8x8_sliced_k_f16x4(const half* a, const half* b, half* c, int m, int n, int k, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_t_8x8_sliced_k_f16x4_kernel_launch");
    if(!validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    const dim3 block(16, 16);
    const dim3 grid(static_cast<unsigned int>((n + 127) / 128), static_cast<unsigned int>((m + 127) / 128));
    hgemm_t_8x8_sliced_k_f16x4_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_t_8x8_sliced_k_f16x4_pack(const half* a, const half* b, half* c, int m, int n, int k, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_t_8x8_sliced_k_f16x4_pack_kernel_launch");
    if(!validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    const dim3 block(16, 16);
    const dim3 grid(static_cast<unsigned int>((n + 127) / 128), static_cast<unsigned int>((m + 127) / 128));
    hgemm_t_8x8_sliced_k_f16x4_pack_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_t_8x8_sliced_k_f16x4_bcf(const half* a, const half* b, half* c, int m, int n, int k, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_t_8x8_sliced_k_f16x4_bcf_kernel_launch");
    if(!validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    const dim3 block(16, 16);
    const dim3 grid(static_cast<unsigned int>((n + 127) / 128), static_cast<unsigned int>((m + 127) / 128));
    hgemm_t_8x8_sliced_k_f16x4_bcf_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_t_8x8_sliced_k_f16x4_pack_bcf(const half* a, const half* b, half* c, int m, int n, int k, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_t_8x8_sliced_k_f16x4_pack_bcf_kernel_launch");
    if(!validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    const dim3 block(16, 16);
    const dim3 grid(static_cast<unsigned int>((n + 127) / 128), static_cast<unsigned int>((m + 127) / 128));
    hgemm_t_8x8_sliced_k_f16x4_pack_bcf_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_t_8x8_sliced_k_f16x8_pack_bcf(const half* a, const half* b, half* c, int m, int n, int k, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_t_8x8_sliced_k_f16x8_pack_bcf_kernel_launch");
    if(!validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    const dim3 block(16, 16);
    const dim3 grid(static_cast<unsigned int>((n + 127) / 128), static_cast<unsigned int>((m + 127) / 128));
    hgemm_t_8x8_sliced_k_f16x8_pack_bcf_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf(const half* a, const half* b, half* c, int m, int n, int k, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf_kernel_launch");
    if(!validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    const dim3 block(16, 16);
    const dim3 grid(static_cast<unsigned int>((n + 127) / 128), static_cast<unsigned int>((m + 127) / 128));
    hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf(const half* a, const half* b, half* c, int m, int n, int k, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_kernel_launch");
    if(!validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    const dim3 block(16, 16);
    const dim3 grid(static_cast<unsigned int>((n + 127) / 128), static_cast<unsigned int>((m + 127) / 128));
    hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async(const half* a, const half* b, half* c, int m, int n, int k, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async_kernel_launch");
    if(!validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    const dim3 block(16, 16);
    const dim3 grid(static_cast<unsigned int>((n + 127) / 128), static_cast<unsigned int>((m + 127) / 128));
    hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf(const half* a, const half* b, half* c, int m, int n, int k, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_kernel_launch");
    if(!validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    const dim3 block(16, 16);
    const dim3 grid(static_cast<unsigned int>((n + 127) / 128), static_cast<unsigned int>((m + 127) / 128));
    hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_async(const half* a, const half* b, half* c, int m, int n, int k, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_async_kernel_launch");
    if(!validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    const dim3 block(16, 16);
    const dim3 grid(static_cast<unsigned int>((n + 127) / 128), static_cast<unsigned int>((m + 127) / 128));
    hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_async_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf(const half* a, const half* b, half* c, int m, int n, int k, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_kernel_launch");
    if(!validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    const dim3 block(16, 8);
    const dim3 grid(static_cast<unsigned int>((n + 127) / 128), static_cast<unsigned int>((m + 127) / 128));
    hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_async(const half* a, const half* b, half* c, int m, int n, int k, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_async_kernel_launch");
    if(!validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    const dim3 block(16, 8);
    const dim3 grid(static_cast<unsigned int>((n + 127) / 128), static_cast<unsigned int>((m + 127) / 128));
    hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_async_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_cublas_tensor_op_nn(const half* a, const half* b, half* c, int m, int n, int k, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_cublas_tensor_op_nn_launch");
    return validate_device_problem(a, b, c, m, n, k, error) && launch_cublas_with_temporary_handle(a, b, c, m, n, k, error);
}

bool hgemm_cublas_tensor_op_tn(const half* a, const half* b, half* c, int m, int n, int k, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_cublas_tensor_op_tn_launch");
    return validate_device_problem(a, b, c, m, n, k, error) && launch_cublas_with_temporary_handle(a, b, c, m, n, k, error);
}

bool hgemm_wmma_m16n16k16_naive(const half* a, const half* b, half* c, int m, int n, int k, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_wmma_m16n16k16_naive_kernel_launch");
    if(!validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
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
    const dim3 block(kWarpSize);
    const dim3 grid(static_cast<unsigned int>((n + kWmmaTileN - 1) / kWmmaTileN), static_cast<unsigned int>((m + kWmmaTileM - 1) / kWmmaTileM));
    hgemm_wmma_m16n16k16_mma4x2_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_wmma_m16n16k16_mma4x2_warp2x4(const half* a, const half* b, half* c, int m, int n, int k, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_wmma_m16n16k16_mma4x2_warp2x4_kernel_launch");
    if(!validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    const dim3 block(kWarpSize);
    const dim3 grid(static_cast<unsigned int>((n + kWmmaTileN - 1) / kWmmaTileN), static_cast<unsigned int>((m + kWmmaTileM - 1) / kWmmaTileM));
    hgemm_wmma_m16n16k16_mma4x2_warp2x4_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async(const half* a, const half* b, half* c, int m, int n, int k, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_kernel_launch");
    if(!validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    const dim3 block(kWarpSize);
    const dim3 grid(static_cast<unsigned int>((n + kWmmaTileN - 1) / kWmmaTileN), static_cast<unsigned int>((m + kWmmaTileM - 1) / kWmmaTileM));
    hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async(const half* a, const half* b, half* c, int m, int n, int k, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async_kernel_launch");
    if(!validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    const dim3 block(kWarpSize);
    const dim3 grid(static_cast<unsigned int>((n + kWmmaTileN - 1) / kWmmaTileN), static_cast<unsigned int>((m + kWmmaTileM - 1) / kWmmaTileM));
    hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_mma_m16n8k16_naive(const half* a, const half* b, half* c, int m, int n, int k, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_mma_m16n8k16_naive_kernel_launch");
    if(!validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    const dim3 block(kWarpSize);
    const dim3 grid(static_cast<unsigned int>((n + 7) / 8), static_cast<unsigned int>((m + 15) / 16));
    hgemm_mma_m16n8k16_naive_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_mma_m16n8k16_mma2x4_warp4x4(const half* a, const half* b, half* c, int m, int n, int k, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_mma_m16n8k16_mma2x4_warp4x4_kernel_launch");
    if(!validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    const dim3 block(kWarpSize);
    const dim3 grid(static_cast<unsigned int>((n + 7) / 8), static_cast<unsigned int>((m + 15) / 16));
    hgemm_mma_m16n8k16_mma2x4_warp4x4_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages(const half* a, const half* b, half* c, int m, int n, int k, int stages, bool swizzle, int swizzle_stride, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_kernel_launch");
    if(!validate_stage_options(stages, swizzle, swizzle_stride, error) || !validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    const dim3 block(kWarpSize);
    const dim3 grid(static_cast<unsigned int>((n + kWmmaTileN - 1) / kWmmaTileN), static_cast<unsigned int>((m + kWmmaTileM - 1) / kWmmaTileM));
    hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem(const half* a, const half* b, half* c, int m, int n, int k, int stages, bool swizzle, int swizzle_stride, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem_kernel_launch");
    if(!validate_stage_options(stages, swizzle, swizzle_stride, error) || !validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    const dim3 block(kWarpSize);
    const dim3 grid(static_cast<unsigned int>((n + kWmmaTileN - 1) / kWmmaTileN), static_cast<unsigned int>((m + kWmmaTileM - 1) / kWmmaTileM));
    hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem(const half* a, const half* b, half* c, int m, int n, int k, int stages, bool swizzle, int swizzle_stride, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem_kernel_launch");
    if(!validate_stage_options(stages, swizzle, swizzle_stride, error) || !validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    const dim3 block(kWarpSize);
    const dim3 grid(static_cast<unsigned int>((n + kWmmaTileN - 1) / kWmmaTileN), static_cast<unsigned int>((m + kWmmaTileM - 1) / kWmmaTileM));
    hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem(const half* a, const half* b, half* c, int m, int n, int k, int stages, bool swizzle, int swizzle_stride, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem_kernel_launch");
    if(!validate_stage_options(stages, swizzle, swizzle_stride, error) || !validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    const dim3 block(kWarpSize);
    const dim3 grid(static_cast<unsigned int>((n + kWmmaTileN - 1) / kWmmaTileN), static_cast<unsigned int>((m + kWmmaTileM - 1) / kWmmaTileM));
    hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_mma_m16n8k16_mma2x4_warp4x4_stages(const half* a, const half* b, half* c, int m, int n, int k, int stages, bool swizzle, int swizzle_stride, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_kernel_launch");
    if(!validate_stage_options(stages, swizzle, swizzle_stride, error) || !validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    const dim3 block(kWarpSize);
    const dim3 grid(static_cast<unsigned int>((n + 7) / 8), static_cast<unsigned int>((m + 15) / 16));
    hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem(const half* a, const half* b, half* c, int m, int n, int k, int stages, bool swizzle, int swizzle_stride, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_kernel_launch");
    if(!validate_stage_options(stages, swizzle, swizzle_stride, error) || !validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    const dim3 block(kWarpSize);
    const dim3 grid(static_cast<unsigned int>((n + 7) / 8), static_cast<unsigned int>((m + 15) / 16));
    hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem(const half* a, const half* b, half* c, int m, int n, int k, int stages, bool swizzle, int swizzle_stride, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_kernel_launch");
    if(!validate_stage_options(stages, swizzle, swizzle_stride, error) || !validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    const dim3 block(kWarpSize);
    const dim3 grid(static_cast<unsigned int>((n + 7) / 8), static_cast<unsigned int>((m + 15) / 16));
    hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4(const half* a, const half* b, half* c, int m, int n, int k, int stages, bool swizzle, int swizzle_stride, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4_kernel_launch");
    if(!validate_stage_options(stages, swizzle, swizzle_stride, error) || !validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    const dim3 block(kWarpSize);
    const dim3 grid(static_cast<unsigned int>((n + 7) / 8), static_cast<unsigned int>((m + 15) / 16));
    hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr(const half* a, const half* b, half* c, int m, int n, int k, int stages, bool swizzle, int swizzle_stride, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr_kernel_launch");
    if(!validate_stage_options(stages, swizzle, swizzle_stride, error) || !validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    const dim3 block(kWarpSize);
    const dim3 grid(static_cast<unsigned int>((n + 7) / 8), static_cast<unsigned int>((m + 15) / 16));
    hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn(const half* a, const half* b, half* c, int m, int n, int k, int stages, bool swizzle, int swizzle_stride, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_kernel_launch");
    if(!validate_stage_options(stages, swizzle, swizzle_stride, error) || !validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    const dim3 block(kWarpSize);
    const dim3 grid(static_cast<unsigned int>((n + 7) / 8), static_cast<unsigned int>((m + 15) / 16));
    hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_mma_stages_block_swizzle_tn_cute(const half* a, const half* b, half* c, int m, int n, int k, int stages, bool swizzle, int swizzle_stride, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_mma_stages_block_swizzle_tn_cute_kernel_launch");
    if(!validate_stage_options(stages, swizzle, swizzle_stride, error) || !validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    const dim3 block(kWarpSize);
    const dim3 grid(static_cast<unsigned int>((n + 7) / 8), static_cast<unsigned int>((m + 15) / 16));
    hgemm_mma_stages_block_swizzle_tn_cute_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_swizzle(const half* a, const half* b, half* c, int m, int n, int k, int stages, bool swizzle, int swizzle_stride, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_swizzle_kernel_launch");
    if(!validate_stage_options(stages, swizzle, swizzle_stride, error) || !validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    const dim3 block(kWarpSize);
    const dim3 grid(static_cast<unsigned int>((n + 7) / 8), static_cast<unsigned int>((m + 15) / 16));
    hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_swizzle_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x4(const half* a, const half* b, half* c, int m, int n, int k, int stages, bool swizzle, int swizzle_stride, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x4_kernel_launch");
    if(!validate_stage_options(stages, swizzle, swizzle_stride, error) || !validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    const dim3 block(kWarpSize);
    const dim3 grid(static_cast<unsigned int>((n + 7) / 8), static_cast<unsigned int>((m + 15) / 16));
    hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x4_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool launch_hgemm_kernel(
    HgemmKernel kernel,
    const half* a,
    const half* b,
    half* c,
    int m,
    int n,
    int k,
    std::string& error,
    HgemmLaunchOptions launch_options
) {
    if(is_cublas_kernel(kernel)) {
        CublasHandle handle;
        if(!handle.create(error)) {
            return false;
        }
        return launch_hgemm_kernel_with_handle(kernel, handle.get(), a, b, c, m, n, k, error, launch_options);
    }
    return launch_hgemm_kernel_with_handle(kernel, nullptr, a, b, c, m, n, k, error, launch_options);
}

struct PreparedHgemmLabRunner::Impl {
    HgemmKernel kernel {HgemmKernel::NaiveF16};
    HgemmLaunchOptions launch_options;
    int m {0};
    int n {0};
    int k {0};
    bool prepared {false};
    ai_system::cuda_utils::DeviceBuffer<half> lhs_device;
    ai_system::cuda_utils::DeviceBuffer<half> rhs_device;
    ai_system::cuda_utils::DeviceBuffer<half> out_device;
    ai_system::cuda_utils::EventPair events;
    CublasHandle handle;

    bool prepare(
        HgemmKernel requested_kernel,
        std::size_t requested_m,
        std::size_t requested_n,
        std::size_t requested_k,
        const std::vector<float>& lhs,
        const std::vector<float>& rhs,
        std::string& error,
        HgemmLaunchOptions requested_launch_options
    ) {
        const ai_system::profiling::ScopedNvtxRange phase_range("hgemm_lab_prepare_h2d");
        if(!validate_host_inputs(requested_m, requested_n, requested_k, lhs, rhs, error)) {
            return false;
        }
        if(!validate_stage_options(
               requested_launch_options.stages,
               requested_launch_options.swizzle,
               requested_launch_options.swizzle_stride,
               error
           )) {
            return false;
        }
        if(!events.ensure(error)) {
            return false;
        }

        kernel = requested_kernel;
        launch_options = requested_launch_options;
        m = static_cast<int>(requested_m);
        n = static_cast<int>(requested_n);
        k = static_cast<int>(requested_k);
        prepared = false;

        lhs_device.reset();
        rhs_device.reset();
        out_device.reset();
        handle.reset();

        const std::vector<half> lhs_half = convert_to_half(lhs);
        const std::vector<half> rhs_half = convert_to_half(rhs);
        if(!lhs_device.allocate(lhs_half.size(), error) || !rhs_device.allocate(rhs_half.size(), error) ||
           !out_device.allocate(requested_m * requested_n, error)) {
            return false;
        }
        if(!ai_system::cuda_utils::copy_to_device(lhs_device.get(), lhs_half, error) ||
           !ai_system::cuda_utils::copy_to_device(rhs_device.get(), rhs_half, error)) {
            return false;
        }
        if(is_cublas_kernel(kernel) && !handle.create(error)) {
            return false;
        }

        prepared = true;
        return true;
    }

    bool run(std::string& error) {
        if(!launch(error)) {
            return false;
        }
        return ai_system::cuda_utils::synchronize(error);
    }

    bool run_timed(double& elapsed_ms, std::string& error) {
        elapsed_ms = 0.0;
        if(!prepared) {
            error = "PreparedHgemmLabRunner::prepare must succeed before run_timed.";
            return false;
        }
        if(!events.record_start(error)) {
            return false;
        }
        if(!launch(error)) {
            return false;
        }
        if(!events.record_stop(error)) {
            return false;
        }
        if(!events.synchronize_stop(error)) {
            return false;
        }
        float event_ms = 0.0f;
        if(!events.elapsed_ms(event_ms, error)) {
            return false;
        }
        elapsed_ms = static_cast<double>(event_ms);
        return true;
    }

    bool copy_output(std::vector<float>& out, std::string& error) const {
        if(!prepared) {
            error = "PreparedHgemmLabRunner::prepare must succeed before copy_output.";
            return false;
        }
        std::vector<half> out_half(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
        if(!ai_system::cuda_utils::copy_to_host(out_half, out_device.get(), error)) {
            return false;
        }
        convert_half_to_float(out_half, out);
        return true;
    }

private:
    bool launch(std::string& error) {
        if(!prepared) {
            error = "PreparedHgemmLabRunner::prepare must succeed before run.";
            return false;
        }
        return launch_hgemm_kernel_with_handle(
            kernel,
            handle.get(),
            lhs_device.get(),
            rhs_device.get(),
            out_device.get(),
            m,
            n,
            k,
            error,
            launch_options
        );
    }
};

PreparedHgemmLabRunner::PreparedHgemmLabRunner() : impl_(std::make_unique<Impl>()) {}
PreparedHgemmLabRunner::~PreparedHgemmLabRunner() = default;
PreparedHgemmLabRunner::PreparedHgemmLabRunner(PreparedHgemmLabRunner&& other) noexcept = default;
PreparedHgemmLabRunner& PreparedHgemmLabRunner::operator=(PreparedHgemmLabRunner&& other) noexcept = default;

bool PreparedHgemmLabRunner::prepare(
    HgemmKernel kernel,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    std::string& error,
    HgemmLaunchOptions launch_options
) {
    return impl_->prepare(kernel, m, n, k, lhs, rhs, error, launch_options);
}

bool PreparedHgemmLabRunner::run(std::string& error) {
    return impl_->run(error);
}

bool PreparedHgemmLabRunner::run_timed(double& elapsed_ms, std::string& error) {
    return impl_->run_timed(elapsed_ms, error);
}

bool PreparedHgemmLabRunner::copy_output(std::vector<float>& out, std::string& error) const {
    return impl_->copy_output(out, error);
}

}  // namespace ai_system::labs::hgemm
