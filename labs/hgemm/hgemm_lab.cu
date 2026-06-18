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

__global__ void hgemm_mma_m16n8k16_naive_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
);

__global__ void hgemm_mma_m16n8k16_mma2x4_warp4x4_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
);

__global__ void hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
);

__global__ void hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
);

__global__ void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
);

__global__ void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
);

__global__ void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
);

__global__ void hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
);

__global__ void hgemm_mma_stages_block_swizzle_tn_cute_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
);

__global__ void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_swizzle_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
);

__global__ void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x4_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
);

namespace {

constexpr int kWarpSize = 32;
constexpr int kWmmaTileM = 16;
constexpr int kWmmaTileN = 16;

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

// Load/store helpers used by the teaching kernels below.  Count is the
// number of half values moved by the caller, and PackBits is the preferred
// vector width:
//   PackBits >= 128: move groups of 8 half values with float4 when aligned.
//   PackBits >= 64 : move groups of 4 half values with float2 when aligned.
//   PackBits >= 32 : move groups of 2 half values with half2 when aligned.
// Boundary tiles fall back to scalar masked loads/stores so the kernels can
// run on arbitrary M/N/K, even when the fast path wants vector alignment.
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

template <int Count, int PackBits>
__device__ __forceinline__ void hgemm_store_contiguous_half(
    half* values,
    int row,
    int col,
    int rows,
    int cols,
    const half* source
) {
    if(row < rows && col + Count <= cols) {
        hgemm_copy_contiguous_half<Count, PackBits>(source, values + row * cols + col);
        return;
    }

#pragma unroll
    for(int index = 0; index < Count; ++index) {
        if(row < rows && col + index < cols) {
            values[row * cols + col + index] = source[index];
        }
    }
}

// One thread computes exactly one C(row, col) element.
// Grid mapping:
//   blockIdx.x covers columns N in groups of blockDim.x.
//   blockIdx.y covers rows M in groups of blockDim.y.
//   threadIdx.(x,y) selects the element inside that 2-D block tile.
// Work per thread:
//   reads K half values from A(row, 0:K) and K half values from B(0:K, col),
//   performs K half FMA operations, and writes one half value to C.
__global__ void hgemm_naive_f16_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    // Global output coordinate owned by this thread.
    const int row = static_cast<int>(blockIdx.y) * blockDim.y + threadIdx.y;
    const int col = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(row >= m || col >= n) {
        return;
    }

    half accumulator = zero_half();
    for(int inner = 0; inner < k; ++inner) {
        accumulator = __hfma(a[row * k + inner], b[inner * n + col], accumulator);
    }
    c[row * n + col] = accumulator;
}

__device__ __forceinline__ void hgemm_cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int Groups>
__device__ __forceinline__ void hgemm_cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(Groups));
}

template <int Bytes>
__device__ __forceinline__ void hgemm_cp_async_ca(std::uint32_t shared_destination, const void* global_source) {
    asm volatile(
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
        ::"r"(shared_destination), "l"(global_source), "n"(Bytes)
    );
}

template <int Bytes>
__device__ __forceinline__ void hgemm_cp_async_cg(std::uint32_t shared_destination, const void* global_source) {
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
        ::"r"(shared_destination), "l"(global_source), "n"(Bytes)
    );
}

template <int BlockM, int BlockN, int BlockK>
__global__ void hgemm_sliced_k_f16_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    static_assert(BlockM > 0 && BlockN > 0 && BlockK > 0, "HGEMM sliced tile dimensions must be positive.");
    static_assert(BlockM * BlockN <= 1024, "HGEMM sliced output tile exceeds CUDA's thread block limit.");

    // CTA tile shape is BlockM x BlockN output elements, with the K dimension
    // processed in BlockK slices.  The launch uses block(BlockN, BlockM), so
    // there is one thread per output element in the tile.
    //
    // For the instantiated 32x32x32 kernel:
    //   threads/CTA     = 32 * 32 = 1024
    //   shared A slice  = 32 * 32 half values
    //   shared B slice  = 32 * 32 half values
    //   per thread load = one A half and one B half per K slice
    //   per thread math = 32 half FMA operations per slice, one C output
    __shared__ half shared_a[BlockM][BlockK];
    __shared__ half shared_b[BlockK][BlockN];

    const int local_col = threadIdx.x;
    const int local_row = threadIdx.y;
    const int tid = local_row * blockDim.x + local_col;
    const int thread_count = blockDim.x * blockDim.y;
    const int block_row = static_cast<int>(blockIdx.y) * BlockM;
    const int block_col = static_cast<int>(blockIdx.x) * BlockN;
    const int row = block_row + local_row;
    const int col = block_col + local_col;
    half accumulator = zero_half();

    for(int k_base = 0; k_base < k; k_base += BlockK) {
        // Flatten the CTA's 2-D threads to cooperatively cover the A tile.
        // index -> (tile_row, tile_col) in shared_a[BlockM][BlockK].
        for(int index = tid; index < BlockM * BlockK; index += thread_count) {
            const int tile_row = index / BlockK;
            const int tile_col = index % BlockK;
            shared_a[tile_row][tile_col] = hgemm_load_or_zero(a, block_row + tile_row, k_base + tile_col, m, k);
        }

        // Same flattened mapping for B, but the tile shape is BlockK x BlockN.
        for(int index = tid; index < BlockK * BlockN; index += thread_count) {
            const int tile_row = index / BlockN;
            const int tile_col = index % BlockN;
            shared_b[tile_row][tile_col] = hgemm_load_or_zero(b, k_base + tile_row, block_col + tile_col, k, n);
        }

        __syncthreads();

#pragma unroll
        for(int inner = 0; inner < BlockK; ++inner) {
            // Each thread reuses its row from shared_a and column from shared_b
            // to accumulate the single C(row, col) value it owns.
            accumulator = __hfma(shared_a[local_row][inner], shared_b[inner][local_col], accumulator);
        }
        __syncthreads();
    }

    if(row < m && col < n) {
        c[row * n + col] = accumulator;
    }
}

template <int LoadPackBits, int StorePackBits>
__device__ void hgemm_thread_tile_8x8_sliced_k_plain_body(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    constexpr int kBlockM = 128;
    constexpr int kBlockN = 128;
    constexpr int kBlockK = 8;
    constexpr int kThreadM = 8;
    constexpr int kThreadN = 8;
    // Plain 8x8 thread tile SIMT kernel.
    //
    // CTA shape:
    //   blockDim = (16, 16) = 256 threads.
    //   one CTA computes a 128x128 C tile.
    //   each thread computes an 8x8 C micro-tile:
    //       output rows = block_row + threadIdx.y * 8 + [0, 7]
    //       output cols = block_col + threadIdx.x * 8 + [0, 7]
    //
    // Per K slice (BlockK = 8):
    //   A tile = 128x8  half values = 1024 half.
    //   B tile = 8x128  half values = 1024 half.
    //   each of 256 threads loads 4 contiguous half from A and 4 from B.
    //   compute reads 8 A half + 8 B half from shared memory for each inner k,
    //   then performs 8x8 half FMAs into registers.
    __shared__ half shared_a[kBlockM][kBlockK];
    __shared__ half shared_b[kBlockK][kBlockN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int block_row = static_cast<int>(blockIdx.y) * kBlockM;
    const int block_col = static_cast<int>(blockIdx.x) * kBlockN;
    // A is 128x8, so two vector-4 loads cover each row.  tid / 2 picks the
    // row, and (tid & 1) chooses columns 0..3 or 4..7 inside the K slice.
    const int load_a_row = tid / 2;
    const int load_a_col = (tid & 1) << 2;
    // B is 8x128, so thirty-two vector-4 loads cover each row.
    // tid / 32 picks the K row and (tid & 31) selects the N vector.
    const int load_b_row = tid / 32;
    const int load_b_col = (tid & 31) << 2;

    alignas(16) half load_a[4];
    alignas(16) half load_b[4];
    alignas(16) half fragment_a[kThreadM];
    alignas(16) half fragment_b[kThreadN];
    alignas(16) half accumulator[kThreadM][kThreadN];

#pragma unroll
    for(int row_item = 0; row_item < kThreadM; ++row_item) {
#pragma unroll
        for(int col_item = 0; col_item < kThreadN; ++col_item) {
            accumulator[row_item][col_item] = zero_half();
        }
    }

    for(int k_base = 0; k_base < k; k_base += kBlockK) {
        // Global -> register -> shared.  The templated helper chooses scalar,
        // half2, float2, or float4 copies according to LoadPackBits/alignment.
        hgemm_load_contiguous_half<4, LoadPackBits>(a, block_row + load_a_row, k_base + load_a_col, m, k, load_a);
        hgemm_load_contiguous_half<4, LoadPackBits>(b, k_base + load_b_row, block_col + load_b_col, k, n, load_b);
        hgemm_copy_contiguous_half<4, LoadPackBits>(load_a, &shared_a[load_a_row][load_a_col]);
        hgemm_copy_contiguous_half<4, LoadPackBits>(load_b, &shared_b[load_b_row][load_b_col]);
        __syncthreads();

#pragma unroll
        for(int inner = 0; inner < kBlockK; ++inner) {
#pragma unroll
            for(int row_item = 0; row_item < kThreadM; ++row_item) {
                // Thread row lane ty owns eight rows spaced contiguously in M.
                fragment_a[row_item] = shared_a[ty * kThreadM + row_item][inner];
            }
#pragma unroll
            for(int col_item = 0; col_item < kThreadN; ++col_item) {
                // Thread column lane tx owns eight contiguous columns in N.
                fragment_b[col_item] = shared_b[inner][tx * kThreadN + col_item];
            }
#pragma unroll
            for(int row_item = 0; row_item < kThreadM; ++row_item) {
#pragma unroll
                for(int col_item = 0; col_item < kThreadN; ++col_item) {
                    accumulator[row_item][col_item] =
                        __hfma(fragment_a[row_item], fragment_b[col_item], accumulator[row_item][col_item]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for(int row_item = 0; row_item < kThreadM; ++row_item) {
        const int row = block_row + ty * kThreadM + row_item;
#pragma unroll
        for(int col_item = 0; col_item < kThreadN; col_item += (StorePackBits >= 64 ? 4 : 2)) {
            const int col = block_col + tx * kThreadN + col_item;
            // Store the 8 columns of each row as half4/half2 chunks when the
            // chosen kernel variant allows packed stores.
            if constexpr(StorePackBits >= 64) {
                hgemm_store_contiguous_half<4, StorePackBits>(c, row, col, m, n, &accumulator[row_item][col_item]);
            } else {
                hgemm_store_contiguous_half<2, StorePackBits>(c, row, col, m, n, &accumulator[row_item][col_item]);
            }
        }
    }
}

__device__ void hgemm_thread_tile_8x8_sliced_k_f16x4_body(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_thread_tile_8x8_sliced_k_plain_body<32, 32>(a, b, c, m, n, k);
}

__device__ void hgemm_thread_tile_8x8_sliced_k_f16x4_pack_body(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_thread_tile_8x8_sliced_k_plain_body<64, 64>(a, b, c, m, n, k);
}

template <int LoadPackBits, int SharedPackBits, int StorePackBits, int Offset>
__device__ void hgemm_thread_tile_8x8_sliced_k_bcf_split_body(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    constexpr int kBlockM = 128;
    constexpr int kBlockN = 128;
    constexpr int kBlockK = 8;
    constexpr int kThreadM = 8;
    constexpr int kThreadN = 8;
    // Bank-conflict-friendly split layout for the 8x8 thread tile kernels.
    //
    // The CTA/output shape is still 128x128 and each thread still owns 8x8 C.
    // The shared A tile is stored transposed as [K][M + Offset].  This makes
    // the per-inner-k reads contiguous along M when a thread gathers A values,
    // reducing shared-memory bank conflicts.  B remains [K][N + Offset].
    //
    // The x4 variants gather the per-thread 8x8 tile as four 4x4 quadrants:
    //   rows ty*4+[0,3] and 64+ty*4+[0,3]
    //   cols tx*4+[0,3] and 64+tx*4+[0,3]
    // This keeps the load/store vectors contiguous while preserving 128x128
    // coverage with 16x16 threads.
    __shared__ half shared_a[kBlockK][kBlockM + Offset];
    __shared__ half shared_b[kBlockK][kBlockN + Offset];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int block_row = static_cast<int>(blockIdx.y) * kBlockM;
    const int block_col = static_cast<int>(blockIdx.x) * kBlockN;
    // Same 4-half-per-thread global load assignment as the plain body.
    const int load_a_smem_m = tid / 2;
    const int load_a_smem_k = (tid & 1) << 2;
    const int load_b_smem_k = tid / 32;
    const int load_b_smem_n = (tid & 31) << 2;

    alignas(16) half load_a[4];
    alignas(16) half load_b[4];
    alignas(16) half fragment_a[kThreadM];
    alignas(16) half fragment_b[kThreadN];
    alignas(16) half accumulator[kThreadM][kThreadN];

#pragma unroll
    for(int row_item = 0; row_item < kThreadM; ++row_item) {
#pragma unroll
        for(int col_item = 0; col_item < kThreadN; ++col_item) {
            accumulator[row_item][col_item] = zero_half();
        }
    }

    for(int k_base = 0; k_base < k; k_base += kBlockK) {
        hgemm_load_contiguous_half<4, LoadPackBits>(
            a,
            block_row + load_a_smem_m,
            k_base + load_a_smem_k,
            m,
            k,
            load_a
        );
        hgemm_load_contiguous_half<4, LoadPackBits>(
            b,
            k_base + load_b_smem_k,
            block_col + load_b_smem_n,
            k,
            n,
            load_b
        );
#pragma unroll
        for(int item = 0; item < 4; ++item) {
            // Transpose A from global row-major A[m][k] into shared_a[k][m].
            shared_a[load_a_smem_k + item][load_a_smem_m] = load_a[item];
        }
        hgemm_copy_contiguous_half<4, SharedPackBits>(load_b, &shared_b[load_b_smem_k][load_b_smem_n]);
        __syncthreads();

#pragma unroll
        for(int inner = 0; inner < kBlockK; ++inner) {
            // Gather four rows from the top half and four from the bottom half
            // of the 128-row tile.  The +kBlockM/2 address is why the thread's
            // logical 8 rows are split into two 4-row groups.
            hgemm_copy_contiguous_half<4, SharedPackBits>(&shared_a[inner][ty * kThreadM / 2], fragment_a);
            hgemm_copy_contiguous_half<4, SharedPackBits>(
                &shared_a[inner][ty * kThreadM / 2 + kBlockM / 2],
                fragment_a + 4
            );
            // B is split the same way across the 128 columns.
            hgemm_copy_contiguous_half<4, SharedPackBits>(&shared_b[inner][tx * kThreadN / 2], fragment_b);
            hgemm_copy_contiguous_half<4, SharedPackBits>(
                &shared_b[inner][tx * kThreadN / 2 + kBlockN / 2],
                fragment_b + 4
            );

#pragma unroll
            for(int row_item = 0; row_item < kThreadM; ++row_item) {
#pragma unroll
                for(int col_item = 0; col_item < kThreadN; ++col_item) {
                    accumulator[row_item][col_item] =
                        __hfma(fragment_a[row_item], fragment_b[col_item], accumulator[row_item][col_item]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for(int row_item = 0; row_item < kThreadM / 2; ++row_item) {
        const int row0 = block_row + ty * kThreadM / 2 + row_item;
        const int row1 = block_row + kBlockM / 2 + ty * kThreadM / 2 + row_item;
        const int col0 = block_col + tx * kThreadN / 2;
        const int col1 = col0 + kBlockN / 2;
        // Store the same four quadrants used during register accumulation.
        hgemm_store_contiguous_half<4, StorePackBits>(c, row0, col0, m, n, &accumulator[row_item][0]);
        hgemm_store_contiguous_half<4, StorePackBits>(c, row0, col1, m, n, &accumulator[row_item][4]);
        hgemm_store_contiguous_half<4, StorePackBits>(c, row1, col0, m, n, &accumulator[row_item + 4][0]);
        hgemm_store_contiguous_half<4, StorePackBits>(c, row1, col1, m, n, &accumulator[row_item + 4][4]);
    }
}

__device__ void hgemm_thread_tile_8x8_sliced_k_f16x4_bcf_body(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_thread_tile_8x8_sliced_k_bcf_split_body<32, 32, 32, 0>(a, b, c, m, n, k);
}

__device__ void hgemm_thread_tile_8x8_sliced_k_f16x4_pack_bcf_body(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_thread_tile_8x8_sliced_k_bcf_split_body<64, 64, 64, 4>(a, b, c, m, n, k);
}

template <int Offset>
__device__ void hgemm_thread_tile_8x8_sliced_k_f16x8_pack_bcf_body_impl(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    constexpr int kBlockM = 128;
    constexpr int kBlockN = 128;
    constexpr int kBlockK = 8;
    constexpr int kThreadM = 8;
    constexpr int kThreadN = 8;
    // x8 packed BCF variant.  The global load assignment is still 4 half from
    // A and 4 half from B per thread per K slice, but the shared layout/padding
    // lets each compute step read the thread's 8 A values and 8 B values as
    // 128-bit contiguous chunks from shared memory.
    __shared__ half shared_a[kBlockK][kBlockM + Offset];
    __shared__ half shared_b[kBlockK][kBlockN + Offset];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int block_row = static_cast<int>(blockIdx.y) * kBlockM;
    const int block_col = static_cast<int>(blockIdx.x) * kBlockN;
    const int load_a_smem_m = tid / 2;
    const int load_a_smem_k = (tid & 1) << 2;
    const int load_b_smem_k = tid / 32;
    const int load_b_smem_n = (tid & 31) << 2;

    alignas(16) half load_a[4];
    alignas(16) half load_b[4];
    alignas(16) half fragment_a[kThreadM];
    alignas(16) half fragment_b[kThreadN];
    alignas(16) half accumulator[kThreadM][kThreadN];

#pragma unroll
    for(int row_item = 0; row_item < kThreadM; ++row_item) {
#pragma unroll
        for(int col_item = 0; col_item < kThreadN; ++col_item) {
            accumulator[row_item][col_item] = zero_half();
        }
    }

    for(int k_base = 0; k_base < k; k_base += kBlockK) {
        hgemm_load_contiguous_half<4, 64>(a, block_row + load_a_smem_m, k_base + load_a_smem_k, m, k, load_a);
        hgemm_load_contiguous_half<4, 64>(b, k_base + load_b_smem_k, block_col + load_b_smem_n, k, n, load_b);
#pragma unroll
        for(int item = 0; item < 4; ++item) {
            shared_a[load_a_smem_k + item][load_a_smem_m] = load_a[item];
        }
        hgemm_copy_contiguous_half<4, 64>(load_b, &shared_b[load_b_smem_k][load_b_smem_n]);
        __syncthreads();

#pragma unroll
        for(int inner = 0; inner < kBlockK; ++inner) {
            // Each thread now maps to one contiguous 8-row and one contiguous
            // 8-column stripe in the transposed/padded shared tiles.
            hgemm_copy_contiguous_half<8, 128>(&shared_a[inner][ty * kThreadM], fragment_a);
            hgemm_copy_contiguous_half<8, 128>(&shared_b[inner][tx * kThreadN], fragment_b);

#pragma unroll
            for(int row_item = 0; row_item < kThreadM; ++row_item) {
#pragma unroll
                for(int col_item = 0; col_item < kThreadN; ++col_item) {
                    accumulator[row_item][col_item] =
                        __hfma(fragment_a[row_item], fragment_b[col_item], accumulator[row_item][col_item]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for(int row_item = 0; row_item < kThreadM; ++row_item) {
        const int row = block_row + ty * kThreadM + row_item;
        const int col = block_col + tx * kThreadN;
        hgemm_store_contiguous_half<8, 128>(c, row, col, m, n, accumulator[row_item]);
    }
}

__device__ void hgemm_thread_tile_8x8_sliced_k_f16x8_pack_bcf_body(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_thread_tile_8x8_sliced_k_f16x8_pack_bcf_body_impl<8>(a, b, c, m, n, k);
}

template <int Offset>
__device__ void hgemm_thread_tile_8x8_sliced_k_f16x8_pack_bcf_dbuf_body_impl(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    constexpr int kBlockM = 128;
    constexpr int kBlockN = 128;
    constexpr int kBlockK = 8;
    constexpr int kThreadM = 8;
    constexpr int kThreadN = 8;
    // Double-buffered version of the x8 packed BCF kernel.  shared_a/shared_b
    // have two K-slice buffers.  While the CTA computes from active_buffer,
    // each thread prepares the next 4-half A and 4-half B vectors in registers
    // and then publishes them to next_buffer before the next __syncthreads().
    __shared__ half shared_a[2][kBlockK][kBlockM + Offset];
    __shared__ half shared_b[2][kBlockK][kBlockN + Offset];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int block_row = static_cast<int>(blockIdx.y) * kBlockM;
    const int block_col = static_cast<int>(blockIdx.x) * kBlockN;
    const int load_a_smem_m = tid / 2;
    const int load_a_smem_k = (tid & 1) << 2;
    const int load_b_smem_k = tid / 32;
    const int load_b_smem_n = (tid & 31) << 2;
    const int k_tiles = (k + kBlockK - 1) / kBlockK;

    alignas(16) half load_a[4];
    alignas(16) half load_b[4];
    alignas(16) half fragment_a[kThreadM];
    alignas(16) half fragment_b[kThreadN];
    alignas(16) half accumulator[kThreadM][kThreadN];

#pragma unroll
    for(int row_item = 0; row_item < kThreadM; ++row_item) {
#pragma unroll
        for(int col_item = 0; col_item < kThreadN; ++col_item) {
            accumulator[row_item][col_item] = zero_half();
        }
    }

    // Prime buffer 0 with K tile 0 before entering the pipelined loop.
    hgemm_load_contiguous_half<4, 64>(a, block_row + load_a_smem_m, load_a_smem_k, m, k, load_a);
    hgemm_load_contiguous_half<4, 64>(b, load_b_smem_k, block_col + load_b_smem_n, k, n, load_b);
#pragma unroll
    for(int item = 0; item < 4; ++item) {
        shared_a[0][load_a_smem_k + item][load_a_smem_m] = load_a[item];
    }
    hgemm_copy_contiguous_half<4, 64>(load_b, &shared_b[0][load_b_smem_k][load_b_smem_n]);
    __syncthreads();

    for(int tile = 1; tile < k_tiles; ++tile) {
        const int active_buffer = (tile - 1) & 1;
        const int next_buffer = tile & 1;
        // Load the next K tile to registers first.  The CTA then computes from
        // active_buffer and finally writes those register values to next_buffer.
        hgemm_load_contiguous_half<4, 64>(
            a,
            block_row + load_a_smem_m,
            tile * kBlockK + load_a_smem_k,
            m,
            k,
            load_a
        );
        hgemm_load_contiguous_half<4, 64>(
            b,
            tile * kBlockK + load_b_smem_k,
            block_col + load_b_smem_n,
            k,
            n,
            load_b
        );

#pragma unroll
        for(int inner = 0; inner < kBlockK; ++inner) {
            hgemm_copy_contiguous_half<8, 128>(&shared_a[active_buffer][inner][ty * kThreadM], fragment_a);
            hgemm_copy_contiguous_half<8, 128>(&shared_b[active_buffer][inner][tx * kThreadN], fragment_b);

#pragma unroll
            for(int row_item = 0; row_item < kThreadM; ++row_item) {
#pragma unroll
                for(int col_item = 0; col_item < kThreadN; ++col_item) {
                    accumulator[row_item][col_item] =
                        __hfma(fragment_a[row_item], fragment_b[col_item], accumulator[row_item][col_item]);
                }
            }
        }

#pragma unroll
        for(int item = 0; item < 4; ++item) {
            shared_a[next_buffer][load_a_smem_k + item][load_a_smem_m] = load_a[item];
        }
        hgemm_copy_contiguous_half<4, 64>(load_b, &shared_b[next_buffer][load_b_smem_k][load_b_smem_n]);
        __syncthreads();
    }

    const int final_buffer = (k_tiles - 1) & 1;
    // Drain the last prefetched buffer after the loop.
#pragma unroll
    for(int inner = 0; inner < kBlockK; ++inner) {
        hgemm_copy_contiguous_half<8, 128>(&shared_a[final_buffer][inner][ty * kThreadM], fragment_a);
        hgemm_copy_contiguous_half<8, 128>(&shared_b[final_buffer][inner][tx * kThreadN], fragment_b);

#pragma unroll
        for(int row_item = 0; row_item < kThreadM; ++row_item) {
#pragma unroll
            for(int col_item = 0; col_item < kThreadN; ++col_item) {
                accumulator[row_item][col_item] =
                    __hfma(fragment_a[row_item], fragment_b[col_item], accumulator[row_item][col_item]);
            }
        }
    }

#pragma unroll
    for(int row_item = 0; row_item < kThreadM; ++row_item) {
        const int row = block_row + ty * kThreadM + row_item;
        const int col = block_col + tx * kThreadN;
        hgemm_store_contiguous_half<8, 128>(c, row, col, m, n, accumulator[row_item]);
    }
}

__device__ void hgemm_thread_tile_8x8_sliced_k_f16x8_pack_bcf_dbuf_body(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    hgemm_thread_tile_8x8_sliced_k_f16x8_pack_bcf_dbuf_body_impl<8>(a, b, c, m, n, k);
}

template <int BlockK, int BlockItems, int Offset>
__device__ __forceinline__ int hgemm_dbuf_shared_offset(int buffer, int row, int col) {
    // Linearized address for shared[2][BlockK][BlockItems + Offset].
    // row is the K coordinate inside the CTA slice; col is M for A's
    // transposed tile or N for B's row-major tile.
    return (buffer * BlockK + row) * (BlockItems + Offset) + col;
}

template <int Count, bool AsyncB>
__device__ __forceinline__ void hgemm_load_b_dbuf_to_shared(
    const half* __restrict__ b,
    int row,
    int col,
    int rows,
    int cols,
    half* shared_destination
) {
    static_assert(Count == 8 || Count == 16 || Count == 32, "HGEMM dbuf B loads are 128-bit chunks.");

    // Async variants use cp.async for B when the full vector is in bounds and
    // both addresses are 16-byte aligned.  Boundary tiles or unaligned pointers
    // use the normal masked vector/scalar path.
    if constexpr(AsyncB) {
        if(row < rows && col + Count <= cols) {
            const half* source = b + row * cols + col;
            if(hgemm_is_aligned(source, 16u) && hgemm_is_aligned(shared_destination, 16u)) {
                const auto shared_base =
                    static_cast<std::uint32_t>(__cvta_generic_to_shared(shared_destination));
#pragma unroll
                for(int offset = 0; offset < Count; offset += 8) {
                    const auto shared_address =
                        shared_base + static_cast<std::uint32_t>(offset * static_cast<int>(sizeof(half)));
                    if constexpr(Count == 8) {
                        hgemm_cp_async_cg<16>(shared_address, source + offset);
                    } else {
                        hgemm_cp_async_ca<16>(shared_address, source + offset);
                    }
                }
                return;
            }
        }
    }

    hgemm_load_contiguous_half<Count, 128>(b, row, col, rows, cols, shared_destination);
}

template <int BlockM, int BlockN, int BlockK, int ThreadM, int ThreadN, int Offset, bool AsyncB>
__device__ void hgemm_thread_tile_sliced_k_f16x8_pack_dbuf_body(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    static_assert(BlockM == 128 && BlockN == 128, "HGEMM dbuf pack kernels are specialized for 128x128 CTA tiles.");
    static_assert(BlockK == 16 || BlockK == 32, "HGEMM dbuf pack kernels support block_k 16 or 32.");
    static_assert(ThreadM == 8 || ThreadM == 16, "HGEMM dbuf pack kernels support thread_m 8 or 16.");
    static_assert(ThreadN == 8, "HGEMM dbuf pack kernels expect thread_n 8.");
    static_assert(BlockM % ThreadM == 0 && BlockN % ThreadN == 0, "HGEMM dbuf thread tile must divide CTA tile.");

    constexpr int kBlockDimX = BlockN / ThreadN;
    constexpr int kBlockDimY = BlockM / ThreadM;
    constexpr int kThreadCount = kBlockDimX * kBlockDimY;
    static_assert((BlockM * BlockK) % kThreadCount == 0, "HGEMM dbuf A tile load must divide threads.");
    static_assert((BlockK * BlockN) % kThreadCount == 0, "HGEMM dbuf B tile load must divide threads.");

    constexpr int kLoadAItems = (BlockM * BlockK) / kThreadCount;
    constexpr int kLoadBItems = (BlockK * BlockN) / kThreadCount;
    static_assert(kLoadAItems == 8 || kLoadAItems == 16 || kLoadAItems == 32, "Unsupported A load width.");
    static_assert(kLoadBItems == 8 || kLoadBItems == 16 || kLoadBItems == 32, "Unsupported B load width.");
    static_assert(BlockK % kLoadAItems == 0, "A load width must divide block_k.");
    static_assert(BlockN % kLoadBItems == 0, "B load width must divide block_n.");

    // Generic double-buffered SIMT kernel used by these launch wrappers:
    //   hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf(_async)
    //   hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf(_async)
    //   hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf(_async)
    //
    // CTA/output mapping:
    //   BlockM x BlockN is always 128x128.
    //   blockDim.x = 128 / ThreadN, blockDim.y = 128 / ThreadM.
    //   thread (tx, ty) owns C rows ty*ThreadM+[0,ThreadM) and
    //   C cols tx*ThreadN+[0,ThreadN) inside the CTA tile.
    //
    // Per K tile cooperative loads:
    //   A has BlockM*BlockK half values, stored transposed as [K][M+Offset].
    //   B has BlockK*BlockN half values, stored as [K][N+Offset].
    //   kLoadAItems = (BlockM*BlockK)/threads per thread.
    //   kLoadBItems = (BlockK*BlockN)/threads per thread.
    //
    // Concrete instantiations:
    //   8x8,  K16, 256 threads: each thread loads  8 A half +  8 B half.
    //   8x8,  K32, 256 threads: each thread loads 16 A half + 16 B half.
    //   16x8, K32, 128 threads: each thread loads 32 A half + 32 B half.
    //
    // AsyncB overlaps only B's global->shared cp.async transfer with compute.
    // A is loaded by normal vector loads because it is immediately transposed
    // through registers into the shared A layout.
    __shared__ half shared_a[2 * BlockK * (BlockM + Offset)];
    __shared__ half shared_b[2 * BlockK * (BlockN + Offset)];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * kBlockDimX + tx;
    const int block_row = static_cast<int>(blockIdx.y) * BlockM;
    const int block_col = static_cast<int>(blockIdx.x) * BlockN;
    const int k_tiles = (k + BlockK - 1) / BlockK;

    constexpr int kLoadAVectorsPerRow = BlockK / kLoadAItems;
    constexpr int kLoadBVectorsPerRow = BlockN / kLoadBItems;
    // A loader: each thread takes one contiguous vector along K for one M row.
    const int load_a_smem_m = tid / kLoadAVectorsPerRow;
    const int load_a_smem_k = (tid % kLoadAVectorsPerRow) * kLoadAItems;
    // B loader: each thread takes one contiguous vector along N for one K row.
    const int load_b_smem_k = tid / kLoadBVectorsPerRow;
    const int load_b_smem_n = (tid % kLoadBVectorsPerRow) * kLoadBItems;

    alignas(16) half load_a[kLoadAItems];
    alignas(16) half fragment_a[ThreadM];
    alignas(16) half fragment_b[ThreadN];
    alignas(16) half accumulator[ThreadM][ThreadN];

#pragma unroll
    for(int row_item = 0; row_item < ThreadM; ++row_item) {
#pragma unroll
        for(int col_item = 0; col_item < ThreadN; ++col_item) {
            accumulator[row_item][col_item] = zero_half();
        }
    }

    // Prime buffer 0.  A is written as shared_a[buffer][k][m] and B as
    // shared_b[buffer][k][n], both through the same linear offset helper.
    hgemm_load_contiguous_half<kLoadAItems, 128>(
        a,
        block_row + load_a_smem_m,
        load_a_smem_k,
        m,
        k,
        load_a
    );
#pragma unroll
    for(int item = 0; item < kLoadAItems; ++item) {
        shared_a[hgemm_dbuf_shared_offset<BlockK, BlockM, Offset>(
            0,
            load_a_smem_k + item,
            load_a_smem_m
        )] = load_a[item];
    }
    hgemm_load_b_dbuf_to_shared<kLoadBItems, AsyncB>(
        b,
        load_b_smem_k,
        block_col + load_b_smem_n,
        k,
        n,
        &shared_b[hgemm_dbuf_shared_offset<BlockK, BlockN, Offset>(0, load_b_smem_k, load_b_smem_n)]
    );
    if constexpr(AsyncB) {
        hgemm_cp_async_commit_group();
        hgemm_cp_async_wait_group<0>();
    }
    __syncthreads();

    for(int tile = 1; tile < k_tiles; ++tile) {
        const int active_buffer = (tile - 1) & 1;
        const int next_buffer = tile & 1;
        const int k_base = tile * BlockK;

        if constexpr(AsyncB) {
            // Start B's next-buffer cp.async before computing active_buffer.
            hgemm_load_b_dbuf_to_shared<kLoadBItems, true>(
                b,
                k_base + load_b_smem_k,
                block_col + load_b_smem_n,
                k,
                n,
                &shared_b[hgemm_dbuf_shared_offset<BlockK, BlockN, Offset>(
                    next_buffer,
                    load_b_smem_k,
                    load_b_smem_n
                )]
            );
            hgemm_cp_async_commit_group();
        } else {
            // Synchronous variant loads both A and B for next_buffer first.
            hgemm_load_contiguous_half<kLoadAItems, 128>(
                a,
                block_row + load_a_smem_m,
                k_base + load_a_smem_k,
                m,
                k,
                load_a
            );
#pragma unroll
            for(int item = 0; item < kLoadAItems; ++item) {
                shared_a[hgemm_dbuf_shared_offset<BlockK, BlockM, Offset>(
                    next_buffer,
                    load_a_smem_k + item,
                    load_a_smem_m
                )] = load_a[item];
            }
            hgemm_load_b_dbuf_to_shared<kLoadBItems, false>(
                b,
                k_base + load_b_smem_k,
                block_col + load_b_smem_n,
                k,
                n,
                &shared_b[hgemm_dbuf_shared_offset<BlockK, BlockN, Offset>(
                    next_buffer,
                    load_b_smem_k,
                    load_b_smem_n
                )]
            );
        }

#pragma unroll
        for(int inner = 0; inner < BlockK; ++inner) {
            // Compute phase: each thread gathers ThreadM A half values and
            // ThreadN B half values for the current K coordinate from shared.
            hgemm_copy_contiguous_half<ThreadM, 128>(
                &shared_a[hgemm_dbuf_shared_offset<BlockK, BlockM, Offset>(
                    active_buffer,
                    inner,
                    ty * ThreadM
                )],
                fragment_a
            );
            hgemm_copy_contiguous_half<ThreadN, 128>(
                &shared_b[hgemm_dbuf_shared_offset<BlockK, BlockN, Offset>(
                    active_buffer,
                    inner,
                    tx * ThreadN
                )],
                fragment_b
            );

#pragma unroll
            for(int row_item = 0; row_item < ThreadM; ++row_item) {
#pragma unroll
                for(int col_item = 0; col_item < ThreadN; ++col_item) {
                    accumulator[row_item][col_item] =
                        __hfma(fragment_a[row_item], fragment_b[col_item], accumulator[row_item][col_item]);
                }
            }
        }

        if constexpr(AsyncB) {
            // After compute has hidden B's copy latency, load/transposed-store
            // A for the same next_buffer, then wait for B's cp.async group.
            hgemm_load_contiguous_half<kLoadAItems, 128>(
                a,
                block_row + load_a_smem_m,
                k_base + load_a_smem_k,
                m,
                k,
                load_a
            );
#pragma unroll
            for(int item = 0; item < kLoadAItems; ++item) {
                shared_a[hgemm_dbuf_shared_offset<BlockK, BlockM, Offset>(
                    next_buffer,
                    load_a_smem_k + item,
                    load_a_smem_m
                )] = load_a[item];
            }
            hgemm_cp_async_wait_group<0>();
        }

        __syncthreads();
    }

    const int final_buffer = (k_tiles - 1) & 1;
    // Drain the final prefetched K tile.
#pragma unroll
    for(int inner = 0; inner < BlockK; ++inner) {
        hgemm_copy_contiguous_half<ThreadM, 128>(
            &shared_a[hgemm_dbuf_shared_offset<BlockK, BlockM, Offset>(final_buffer, inner, ty * ThreadM)],
            fragment_a
        );
        hgemm_copy_contiguous_half<ThreadN, 128>(
            &shared_b[hgemm_dbuf_shared_offset<BlockK, BlockN, Offset>(final_buffer, inner, tx * ThreadN)],
            fragment_b
        );

#pragma unroll
        for(int row_item = 0; row_item < ThreadM; ++row_item) {
#pragma unroll
            for(int col_item = 0; col_item < ThreadN; ++col_item) {
                accumulator[row_item][col_item] =
                    __hfma(fragment_a[row_item], fragment_b[col_item], accumulator[row_item][col_item]);
            }
        }
    }

#pragma unroll
    for(int row_item = 0; row_item < ThreadM; ++row_item) {
        const int row = block_row + ty * ThreadM + row_item;
        const int col = block_col + tx * ThreadN;
        hgemm_store_contiguous_half<ThreadN, 128>(c, row, col, m, n, accumulator[row_item]);
    }
}

__global__ void hgemm_t_8x8_sliced_k_f16x4_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    // 128x128x8 CTA, 16x16 threads, one 8x8 output tile per thread.
    // Uses 4-half loads/stores with scalar/half2 fallback (PackBits=32).
    hgemm_thread_tile_8x8_sliced_k_f16x4_body(a, b, c, m, n, k);
}

__global__ void hgemm_t_8x8_sliced_k_f16x4_pack_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    // Same coordinates as f16x4, but prefers 64-bit vectorized loads/stores.
    hgemm_thread_tile_8x8_sliced_k_f16x4_pack_body(a, b, c, m, n, k);
}

__global__ void hgemm_t_8x8_sliced_k_f16x4_bcf_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    // Same 128x128x8 CTA, with transposed/padded shared A layout to reduce
    // bank conflicts while gathering per-thread A fragments.
    hgemm_thread_tile_8x8_sliced_k_f16x4_bcf_body(a, b, c, m, n, k);
}

__global__ void hgemm_t_8x8_sliced_k_f16x4_pack_bcf_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    // BCF shared layout plus 64-bit packed load/store preference.
    hgemm_thread_tile_8x8_sliced_k_f16x4_pack_bcf_body(a, b, c, m, n, k);
}

__global__ void hgemm_t_8x8_sliced_k_f16x8_pack_bcf_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    // BCF shared layout, 8-half fragment reads from shared, and 128-bit stores.
    hgemm_thread_tile_8x8_sliced_k_f16x8_pack_bcf_body(a, b, c, m, n, k);
}

__global__ void hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    // Adds two shared-memory K-slice buffers to the x8 packed BCF variant.
    hgemm_thread_tile_8x8_sliced_k_f16x8_pack_bcf_dbuf_body(a, b, c, m, n, k);
}

__global__ void hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    // Generic dbuf body: BlockK=16, ThreadM x ThreadN = 8x8, sync B loads.
    hgemm_thread_tile_sliced_k_f16x8_pack_dbuf_body<128, 128, 16, 8, 8, 8, false>(a, b, c, m, n, k);
}

__global__ void hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    // Generic dbuf body: BlockK=16, ThreadM x ThreadN = 8x8, async B loads.
    hgemm_thread_tile_sliced_k_f16x8_pack_dbuf_body<128, 128, 16, 8, 8, 8, true>(a, b, c, m, n, k);
}

__global__ void hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    // Generic dbuf body: BlockK=32, ThreadM x ThreadN = 8x8, sync B loads.
    hgemm_thread_tile_sliced_k_f16x8_pack_dbuf_body<128, 128, 32, 8, 8, 8, false>(a, b, c, m, n, k);
}

__global__ void hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_async_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    // Generic dbuf body: BlockK=32, ThreadM x ThreadN = 8x8, async B loads.
    hgemm_thread_tile_sliced_k_f16x8_pack_dbuf_body<128, 128, 32, 8, 8, 8, true>(a, b, c, m, n, k);
}

__global__ void hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    // Generic dbuf body: BlockK=32, ThreadM x ThreadN = 16x8, sync B loads.
    hgemm_thread_tile_sliced_k_f16x8_pack_dbuf_body<128, 128, 32, 16, 8, 8, false>(a, b, c, m, n, k);
}

__global__ void hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_async_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ c,
    int m,
    int n,
    int k
) {
    // Generic dbuf body: BlockK=32, ThreadM x ThreadN = 16x8, async B loads.
    hgemm_thread_tile_sliced_k_f16x8_pack_dbuf_body<128, 128, 32, 16, 8, 8, true>(a, b, c, m, n, k);
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

bool launch_naive_kernel(const half* a, const half* b, half* c, int m, int n, int k, std::string& error) {
    // 16x16 CUDA-thread tile.  grid.x covers N columns, grid.y covers M rows.
    // The kernel itself maps each thread to one C element.
    const dim3 block(16, 16);
    const dim3 grid(
        static_cast<unsigned int>((n + block.x - 1) / block.x),
        static_cast<unsigned int>((m + block.y - 1) / block.y)
    );
    hgemm_naive_f16_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool launch_sliced_kernel(const half* a, const half* b, half* c, int m, int n, int k, std::string& error) {
    enum : int { kBlockM = 32, kBlockN = 32, kBlockK = 32 };
    // 32x32 output tile with one CUDA thread per C element and a 32-wide K
    // shared-memory slice.
    const dim3 block(kBlockN, kBlockM);
    const dim3 grid(
        static_cast<unsigned int>((n + kBlockN - 1) / kBlockN),
        static_cast<unsigned int>((m + kBlockM - 1) / kBlockM)
    );
    hgemm_sliced_k_f16_kernel<kBlockM, kBlockN, kBlockK><<<grid, block>>>(a, b, c, m, n, k);
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
    // Matrices in this lab are row-major.  cuBLAS is column-major, so the call
    // computes C^T = B^T * A^T by passing dimensions as (n, m, k) and operands
    // in B, A order with C leading dimension n.  Both NN/TN benchmark entries
    // currently share this row-major tensor-op implementation.
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
        {HgemmKernel::SlicedKF16, "hgemm_sliced_k_f16", "hgemm_sliced_k_f16_kernel", "32x32x32", "1x1", false},
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
        {HgemmKernel::WmmaM16n16k16Naive, "hgemm_wmma_m16n16k16_naive", "hgemm_wmma_m16n16k16_naive_kernel", "16x16x16", "1wmma/warp", false},
        {HgemmKernel::WmmaM16n16k16Mma4x2, "hgemm_wmma_m16n16k16_mma4x2", "hgemm_wmma_m16n16k16_mma4x2_kernel", "64x32x16", "1wmma/warp", false},
        {HgemmKernel::WmmaM16n16k16Mma4x2Warp2x4, "hgemm_wmma_m16n16k16_mma4x2_warp2x4", "hgemm_wmma_m16n16k16_mma4x2_warp2x4_kernel", "128x128x16", "2x4wmma/warp", false},
        {HgemmKernel::WmmaM16n16k16Mma4x2Warp2x4DbufAsync, "hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async", "hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_kernel", "128x128x16", "2x4wmma/warp", false},
        {HgemmKernel::WmmaM32n8k16Mma2x4Warp2x4DbufAsync, "hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async", "hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async_kernel", "128x128x16", "2x4wmma/warp", false},
        {HgemmKernel::WmmaM16n16k16Mma4x2Warp2x4Stages, "hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages", "hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_kernel", "128x128x16", "2x4wmma/warp", true},
        {HgemmKernel::WmmaM16n16k16Mma4x2Warp2x4StagesDsmem, "hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem", "hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem_kernel", "128x128x16", "2x4wmma/warp", true},
        {HgemmKernel::WmmaM16n16k16Mma4x2Warp4x4StagesDsmem, "hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem", "hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem_kernel", "256x128x16", "4x4wmma/warp", true},
        {HgemmKernel::WmmaM16n16k16Mma4x4Warp4x4StagesDsmem, "hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem", "hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem_kernel", "256x256x16", "4x4wmma/warp", true},
        {HgemmKernel::MmaM16n8k16Naive, "hgemm_mma_m16n8k16_naive", "hgemm_mma_m16n8k16_naive_kernel", "16x8x16", "warp", false},
        {HgemmKernel::MmaM16n8k16Mma2x4Warp4x4, "hgemm_mma_m16n8k16_mma2x4_warp4x4", "hgemm_mma_m16n8k16_mma2x4_warp4x4_kernel", "128x128x16", "4x4mma/warp", false},
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
    // 16x16 threads * 8x8 outputs/thread -> one 128x128 output tile per CTA.
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
    // Same launch shape as f16x4; only the body's vectorized load/store width changes.
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
    // Same 128x128 CTA tile; the kernel body changes shared-memory layout to BCF.
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
    // Same 128x128 CTA tile; BCF shared layout plus packed load/store variant.
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
    // Same 128x128 CTA tile; body reads/writes 8-half chunks where possible.
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
    // Same 128x128 CTA tile with two shared-memory K-slice buffers.
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
    // 16x16 threads, 8x8 outputs/thread, BlockK=16 in the templated body.
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
    // Same as k16 dbuf, but B's next K tile is copied with cp.async.
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
    // 16x16 threads, 8x8 outputs/thread, BlockK=32 in the templated body.
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
    // Same as k32 dbuf, but B's next K tile is copied with cp.async.
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
    // 16x8 threads * 16x8 outputs/thread -> one 128x128 output tile per CTA.
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
    // Same 16x8 thread tile as above, with cp.async for B's next K tile.
    const dim3 block(16, 8);
    const dim3 grid(static_cast<unsigned int>((n + 127) / 128), static_cast<unsigned int>((m + 127) / 128));
    hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_async_kernel<<<grid, block>>>(a, b, c, m, n, k);
    return ai_system::cuda_utils::check_last_launch(error);
}

bool hgemm_cublas_tensor_op_nn(const half* a, const half* b, half* c, int m, int n, int k, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_cublas_tensor_op_nn_launch");
    // cuBLAS path uses tensor cores through CUBLAS_GEMM_DEFAULT_TENSOR_OP and
    // the row-major operand swap described in launch_cublas_tensor_op_row_major.
    return validate_device_problem(a, b, c, m, n, k, error) && launch_cublas_with_temporary_handle(a, b, c, m, n, k, error);
}

bool hgemm_cublas_tensor_op_tn(const half* a, const half* b, half* c, int m, int n, int k, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_cublas_tensor_op_tn_launch");
    // Benchmark alias for the same row-major cuBLAS tensor-op implementation.
    return validate_device_problem(a, b, c, m, n, k, error) && launch_cublas_with_temporary_handle(a, b, c, m, n, k, error);
}

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

bool hgemm_mma_m16n8k16_naive(const half* a, const half* b, half* c, int m, int n, int k, std::string& error) {
    const ai_system::profiling::ScopedNvtxRange launch_range("hgemm_mma_m16n8k16_naive_kernel_launch");
    if(!validate_device_problem(a, b, c, m, n, k, error)) {
        return false;
    }
    // One warp per CTA, one 16x8 m16n8k16 MMA output tile per CTA.
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
    enum : int { kBlockM = 128, kBlockN = 128, kThreads = 8 * kWarpSize };
    // Eight warps per CTA, one 128x128 output tile per CTA.  Each warp computes
    // a 4x4 group of m16n8k16 fragments, i.e. a 64x32 C sub-tile.
    const dim3 block(kThreads);
    const dim3 grid(static_cast<unsigned int>((n + kBlockN - 1) / kBlockN), static_cast<unsigned int>((m + kBlockM - 1) / kBlockM));
    hgemm_mma_m16n8k16_mma2x4_warp4x4_kernel<<<grid, block>>>(a, b, c, m, n, k);
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
