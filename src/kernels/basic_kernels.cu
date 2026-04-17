#include "ai_system/kernels/basic_kernels.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace ai_system::kernels {

namespace {

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    ~DeviceBuffer() {
        reset();
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if(this != &other) {
            reset();
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    bool allocate(std::size_t count, std::string& error) {
        reset();
        const auto status = cudaMalloc(&ptr_, sizeof(T) * count);
        if(status != cudaSuccess) {
            error = cudaGetErrorString(status);
            ptr_ = nullptr;
            return false;
        }
        return true;
    }

    void reset() {
        if(ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
    }

    T* get() {
        return ptr_;
    }

    const T* get() const {
        return ptr_;
    }

private:
    T* ptr_ {nullptr};
};

template <typename T>
bool copy_buffer_to_device(T* dst, const T* src, std::size_t count, std::string& error) {
    const auto status = cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyHostToDevice);
    if(status != cudaSuccess) {
        error = cudaGetErrorString(status);
        return false;
    }
    return true;
}

template <typename T>
bool copy_buffer_to_host(T* dst, const T* src, std::size_t count, std::string& error) {
    const auto status = cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyDeviceToHost);
    if(status != cudaSuccess) {
        error = cudaGetErrorString(status);
        return false;
    }
    return true;
}

bool copy_to_device(float* dst, const std::vector<float>& src, std::string& error) {
    return copy_buffer_to_device(dst, src.data(), src.size(), error);
}

bool copy_to_device(__half* dst, const std::vector<__half>& src, std::string& error) {
    return copy_buffer_to_device(dst, src.data(), src.size(), error);
}

bool copy_to_host(std::vector<float>& dst, const float* src, std::string& error) {
    return copy_buffer_to_host(dst.data(), src, dst.size(), error);
}

bool copy_to_host(std::vector<__half>& dst, const __half* src, std::string& error) {
    return copy_buffer_to_host(dst.data(), src, dst.size(), error);
}

bool copy_scalar_to_host(float& dst, const float* src, std::string& error) {
    const auto status = cudaMemcpy(&dst, src, sizeof(float), cudaMemcpyDeviceToHost);
    if(status != cudaSuccess) {
        error = cudaGetErrorString(status);
        return false;
    }
    return true;
}

bool check_cuda_status(cudaError_t status, const char* context, std::string& error) {
    if(status != cudaSuccess) {
        error = std::string(context) + ": " + cudaGetErrorString(status);
        return false;
    }
    return true;
}

bool check_last_launch(std::string& error) {
    return check_cuda_status(cudaGetLastError(), "cudaGetLastError", error);
}

bool synchronize(std::string& error) {
    return check_cuda_status(cudaDeviceSynchronize(), "cudaDeviceSynchronize", error);
}

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

bool validate_gemm_inputs(
    std::size_t m,
    std::size_t n,
    std::size_t k,
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    std::string& error
) {
    if(lhs.size() != m * k) {
        error = "GEMM requires lhs.size() == m * k.";
        return false;
    }
    if(rhs.size() != k * n) {
        error = "GEMM requires rhs.size() == k * n.";
        return false;
    }
    if(m > static_cast<std::size_t>((std::numeric_limits<int>::max)()) ||
       n > static_cast<std::size_t>((std::numeric_limits<int>::max)()) ||
       k > static_cast<std::size_t>((std::numeric_limits<int>::max)())) {
        error = "GEMM dimensions exceed cuBLAS int limits.";
        return false;
    }
    return true;
}

std::vector<__half> convert_to_half(const std::vector<float>& input) {
    std::vector<__half> output(input.size());
    for(std::size_t index = 0; index < input.size(); ++index) {
        output[index] = __float2half_rn(input[index]);
    }
    return output;
}

void convert_half_to_float(const std::vector<__half>& input, std::vector<float>& output) {
    output.resize(input.size());
    for(std::size_t index = 0; index < input.size(); ++index) {
        output[index] = __half2float(input[index]);
    }
}

__global__ void vector_add_kernel(const float* lhs, const float* rhs, float* out, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(index < count) {
        out[index] = lhs[index] + rhs[index];
    }
}

__global__ void reduce_sum_kernel(const float* input, float* partial, std::size_t count) {
    extern __shared__ float shared[];
    const unsigned int tid = threadIdx.x;
    const std::size_t global_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x * 2 + tid;

    float sum = 0.0f;
    if(global_index < count) {
        sum += input[global_index];
    }
    if(global_index + blockDim.x < count) {
        sum += input[global_index + blockDim.x];
    }

    shared[tid] = sum;
    __syncthreads();

    for(unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if(tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if(tid == 0) {
        partial[blockIdx.x] = shared[0];
    }
}

__global__ void naive_gemm_kernel(
    const float* lhs,
    const float* rhs,
    float* out,
    std::size_t m,
    std::size_t n,
    std::size_t k
) {
    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const std::size_t column = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if(row >= m || column >= n) {
        return;
    }

    float accumulator = 0.0f;
    for(std::size_t inner = 0; inner < k; ++inner) {
        accumulator += lhs[row * k + inner] * rhs[inner * n + column];
    }

    out[row * n + column] = accumulator;
}

bool launch_naive_gemm(
    const float* lhs,
    const float* rhs,
    float* out,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    std::string& error
) {
    const dim3 block(16, 16);
    const dim3 grid(
        static_cast<unsigned int>((n + block.x - 1) / block.x),
        static_cast<unsigned int>((m + block.y - 1) / block.y)
    );

    naive_gemm_kernel<<<grid, block>>>(lhs, rhs, out, m, n, k);
    return check_last_launch(error);
}

bool launch_cublas_sgemm(
    cublasHandle_t handle,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    const float* lhs,
    const float* rhs,
    float* out,
    std::string& error
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return check_cublas_status(
        cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            static_cast<int>(n),
            static_cast<int>(m),
            static_cast<int>(k),
            &alpha,
            rhs,
            static_cast<int>(n),
            lhs,
            static_cast<int>(k),
            &beta,
            out,
            static_cast<int>(n)
        ),
        "cublasSgemm",
        error
    );
}

bool launch_cublas_hgemm(
    cublasHandle_t handle,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    const __half* lhs,
    const __half* rhs,
    __half* out,
    std::string& error
) {
    const __half alpha = __float2half_rn(1.0f);
    const __half beta = __float2half_rn(0.0f);
    return check_cublas_status(
        cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            static_cast<int>(n),
            static_cast<int>(m),
            static_cast<int>(k),
            &alpha,
            rhs,
            CUDA_R_16F,
            static_cast<int>(n),
            lhs,
            CUDA_R_16F,
            static_cast<int>(k),
            &beta,
            out,
            CUDA_R_16F,
            static_cast<int>(n),
            CUBLAS_COMPUTE_16F,
            CUBLAS_GEMM_DEFAULT
        ),
        "cublasGemmEx(hgemm)",
        error
    );
}

bool launch_cublas_tensor_core_gemm(
    cublasHandle_t handle,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    const __half* lhs,
    const __half* rhs,
    float* out,
    std::string& error
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return check_cublas_status(
        cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            static_cast<int>(n),
            static_cast<int>(m),
            static_cast<int>(k),
            &alpha,
            rhs,
            CUDA_R_16F,
            static_cast<int>(n),
            lhs,
            CUDA_R_16F,
            static_cast<int>(k),
            &beta,
            out,
            CUDA_R_32F,
            static_cast<int>(n),
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ),
        "cublasGemmEx(tensor_core)",
        error
    );
}

}  // namespace

struct PreparedGemmKernelRunner::Impl {
    GemmBackend backend {GemmBackend::CudaNaive};
    std::size_t m {0};
    std::size_t n {0};
    std::size_t k {0};
    bool prepared {false};
    DeviceBuffer<float> lhs_float;
    DeviceBuffer<float> rhs_float;
    DeviceBuffer<float> out_float;
    DeviceBuffer<__half> lhs_half;
    DeviceBuffer<__half> rhs_half;
    DeviceBuffer<__half> out_half;
    CublasHandle handle;
    cudaEvent_t start_event {nullptr};
    cudaEvent_t stop_event {nullptr};

    ~Impl() {
        reset_events();
    }

    bool prepare(
        GemmBackend requested_backend,
        std::size_t requested_m,
        std::size_t requested_n,
        std::size_t requested_k,
        const std::vector<float>& lhs,
        const std::vector<float>& rhs,
        std::string& error
    ) {
        if(!validate_gemm_inputs(requested_m, requested_n, requested_k, lhs, rhs, error)) {
            return false;
        }

        if(!ensure_events(error)) {
            return false;
        }

        backend = requested_backend;
        m = requested_m;
        n = requested_n;
        k = requested_k;
        prepared = false;
        handle.reset();

        lhs_float.reset();
        rhs_float.reset();
        out_float.reset();
        lhs_half.reset();
        rhs_half.reset();
        out_half.reset();

        switch(backend) {
            case GemmBackend::CudaNaive:
                if(!lhs_float.allocate(lhs.size(), error) ||
                   !rhs_float.allocate(rhs.size(), error) ||
                   !out_float.allocate(m * n, error)) {
                    return false;
                }
                if(!copy_to_device(lhs_float.get(), lhs, error) || !copy_to_device(rhs_float.get(), rhs, error)) {
                    return false;
                }
                break;
            case GemmBackend::CublasSgemm:
                if(!handle.create(error) ||
                   !lhs_float.allocate(lhs.size(), error) ||
                   !rhs_float.allocate(rhs.size(), error) ||
                   !out_float.allocate(m * n, error)) {
                    return false;
                }
                if(!copy_to_device(lhs_float.get(), lhs, error) || !copy_to_device(rhs_float.get(), rhs, error)) {
                    return false;
                }
                if(!check_cublas_status(cublasSetMathMode(handle.get(), CUBLAS_DEFAULT_MATH), "cublasSetMathMode", error)) {
                    return false;
                }
                break;
            case GemmBackend::CublasHgemm: {
                const std::vector<__half> lhs_converted = convert_to_half(lhs);
                const std::vector<__half> rhs_converted = convert_to_half(rhs);
                if(!handle.create(error) ||
                   !lhs_half.allocate(lhs_converted.size(), error) ||
                   !rhs_half.allocate(rhs_converted.size(), error) ||
                   !out_half.allocate(m * n, error)) {
                    return false;
                }
                if(!copy_to_device(lhs_half.get(), lhs_converted, error) ||
                   !copy_to_device(rhs_half.get(), rhs_converted, error)) {
                    return false;
                }
                if(!check_cublas_status(cublasSetMathMode(handle.get(), CUBLAS_DEFAULT_MATH), "cublasSetMathMode", error)) {
                    return false;
                }
                break;
            }
            case GemmBackend::CublasTensorCore: {
                const std::vector<__half> lhs_converted = convert_to_half(lhs);
                const std::vector<__half> rhs_converted = convert_to_half(rhs);
                if(!handle.create(error) ||
                   !lhs_half.allocate(lhs_converted.size(), error) ||
                   !rhs_half.allocate(rhs_converted.size(), error) ||
                   !out_float.allocate(m * n, error)) {
                    return false;
                }
                if(!copy_to_device(lhs_half.get(), lhs_converted, error) ||
                   !copy_to_device(rhs_half.get(), rhs_converted, error)) {
                    return false;
                }
                if(!check_cublas_status(cublasSetMathMode(handle.get(), CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode", error)) {
                    return false;
                }
                break;
            }
        }

        prepared = true;
        return true;
    }

    bool run(std::string& error) {
        if(!launch(error)) {
            return false;
        }
        return synchronize(error);
    }

    bool run_timed(double& elapsed_ms, std::string& error) {
        elapsed_ms = 0.0;
        if(!prepared) {
            error = "PreparedGemmKernelRunner::prepare must succeed before run_timed.";
            return false;
        }

        if(!check_cuda_status(cudaEventRecord(start_event), "cudaEventRecord(start)", error)) {
            return false;
        }
        if(!launch(error)) {
            return false;
        }
        if(!check_cuda_status(cudaEventRecord(stop_event), "cudaEventRecord(stop)", error)) {
            return false;
        }
        if(!check_cuda_status(cudaEventSynchronize(stop_event), "cudaEventSynchronize(stop)", error)) {
            return false;
        }

        float event_ms = 0.0f;
        if(!check_cuda_status(cudaEventElapsedTime(&event_ms, start_event, stop_event), "cudaEventElapsedTime", error)) {
            return false;
        }

        elapsed_ms = static_cast<double>(event_ms);
        return true;
    }

    bool copy_output(std::vector<float>& out, std::string& error) const {
        if(!prepared) {
            error = "PreparedGemmKernelRunner::prepare must succeed before copy_output.";
            return false;
        }

        switch(backend) {
            case GemmBackend::CudaNaive:
            case GemmBackend::CublasSgemm:
            case GemmBackend::CublasTensorCore:
                out.assign(m * n, 0.0f);
                return copy_to_host(out, out_float.get(), error);
            case GemmBackend::CublasHgemm: {
                std::vector<__half> out_converted(m * n);
                if(!copy_to_host(out_converted, out_half.get(), error)) {
                    return false;
                }
                convert_half_to_float(out_converted, out);
                return true;
            }
        }

        error = "Unsupported GEMM backend.";
        return false;
    }

private:
    bool ensure_events(std::string& error) {
        if(start_event == nullptr) {
            if(!check_cuda_status(cudaEventCreate(&start_event), "cudaEventCreate(start)", error)) {
                start_event = nullptr;
                return false;
            }
        }

        if(stop_event == nullptr) {
            if(!check_cuda_status(cudaEventCreate(&stop_event), "cudaEventCreate(stop)", error)) {
                return false;
            }
        }

        return true;
    }

    void reset_events() {
        if(start_event != nullptr) {
            cudaEventDestroy(start_event);
            start_event = nullptr;
        }
        if(stop_event != nullptr) {
            cudaEventDestroy(stop_event);
            stop_event = nullptr;
        }
    }

    bool launch(std::string& error) {
        if(!prepared) {
            error = "PreparedGemmKernelRunner::prepare must succeed before run.";
            return false;
        }

        switch(backend) {
            case GemmBackend::CudaNaive:
                return launch_naive_gemm(lhs_float.get(), rhs_float.get(), out_float.get(), m, n, k, error);
            case GemmBackend::CublasSgemm:
                return launch_cublas_sgemm(handle.get(), m, n, k, lhs_float.get(), rhs_float.get(), out_float.get(), error);
            case GemmBackend::CublasHgemm:
                return launch_cublas_hgemm(handle.get(), m, n, k, lhs_half.get(), rhs_half.get(), out_half.get(), error);
            case GemmBackend::CublasTensorCore:
                return launch_cublas_tensor_core_gemm(
                    handle.get(),
                    m,
                    n,
                    k,
                    lhs_half.get(),
                    rhs_half.get(),
                    out_float.get(),
                    error
                );
        }

        error = "Unsupported GEMM backend.";
        return false;
    }
};

PreparedGemmKernelRunner::PreparedGemmKernelRunner() : impl_(std::make_unique<Impl>()) {}
PreparedGemmKernelRunner::~PreparedGemmKernelRunner() = default;
PreparedGemmKernelRunner::PreparedGemmKernelRunner(PreparedGemmKernelRunner&& other) noexcept = default;
PreparedGemmKernelRunner& PreparedGemmKernelRunner::operator=(PreparedGemmKernelRunner&& other) noexcept = default;

bool PreparedGemmKernelRunner::prepare(
    GemmBackend backend,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    std::string& error
) {
    return impl_->prepare(backend, m, n, k, lhs, rhs, error);
}

bool PreparedGemmKernelRunner::run(std::string& error) {
    return impl_->run(error);
}

bool PreparedGemmKernelRunner::run_timed(double& elapsed_ms, std::string& error) {
    return impl_->run_timed(elapsed_ms, error);
}

bool PreparedGemmKernelRunner::copy_output(std::vector<float>& out, std::string& error) const {
    return impl_->copy_output(out, error);
}

bool vector_add_cuda(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    std::vector<float>& out,
    std::string& error
) {
    if(lhs.size() != rhs.size()) {
        error = "vector_add_cuda requires lhs.size() == rhs.size().";
        return false;
    }

    out.resize(lhs.size());

    DeviceBuffer<float> lhs_device;
    DeviceBuffer<float> rhs_device;
    DeviceBuffer<float> out_device;

    if(!lhs_device.allocate(lhs.size(), error) ||
       !rhs_device.allocate(rhs.size(), error) ||
       !out_device.allocate(out.size(), error)) {
        return false;
    }

    if(!copy_to_device(lhs_device.get(), lhs, error) || !copy_to_device(rhs_device.get(), rhs, error)) {
        return false;
    }

    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((lhs.size() + threads_per_block - 1) / threads_per_block);
    vector_add_kernel<<<blocks, threads_per_block>>>(lhs_device.get(), rhs_device.get(), out_device.get(), lhs.size());

    if(!check_last_launch(error) || !synchronize(error)) {
        return false;
    }

    return copy_to_host(out, out_device.get(), error);
}

bool reduction_sum_cuda(
    const std::vector<float>& values,
    float& result,
    std::string& error
) {
    if(values.empty()) {
        result = 0.0f;
        return true;
    }

    constexpr int threads_per_block = 256;

    DeviceBuffer<float> current_input;
    if(!current_input.allocate(values.size(), error)) {
        return false;
    }

    if(!copy_to_device(current_input.get(), values, error)) {
        return false;
    }

    std::size_t current_count = values.size();
    while(current_count > 1) {
        const std::size_t blocks = (current_count + threads_per_block * 2 - 1) / (threads_per_block * 2);
        DeviceBuffer<float> next_output;
        if(!next_output.allocate(blocks, error)) {
            return false;
        }

        reduce_sum_kernel<<<static_cast<unsigned int>(blocks), threads_per_block, threads_per_block * sizeof(float)>>>(
            current_input.get(),
            next_output.get(),
            current_count
        );

        if(!check_last_launch(error) || !synchronize(error)) {
            return false;
        }

        current_input = std::move(next_output);
        current_count = blocks;
    }

    return copy_scalar_to_host(result, current_input.get(), error);
}

bool naive_gemm_cuda(
    std::size_t m,
    std::size_t n,
    std::size_t k,
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    std::vector<float>& out,
    std::string& error
) {
    PreparedGemmKernelRunner runner;
    return runner.prepare(GemmBackend::CudaNaive, m, n, k, lhs, rhs, error) && runner.run(error) &&
           runner.copy_output(out, error);
}

bool cublas_sgemm_cuda(
    std::size_t m,
    std::size_t n,
    std::size_t k,
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    std::vector<float>& out,
    std::string& error
) {
    PreparedGemmKernelRunner runner;
    return runner.prepare(GemmBackend::CublasSgemm, m, n, k, lhs, rhs, error) && runner.run(error) &&
           runner.copy_output(out, error);
}

bool cublas_hgemm_cuda(
    std::size_t m,
    std::size_t n,
    std::size_t k,
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    std::vector<float>& out,
    std::string& error
) {
    PreparedGemmKernelRunner runner;
    return runner.prepare(GemmBackend::CublasHgemm, m, n, k, lhs, rhs, error) && runner.run(error) &&
           runner.copy_output(out, error);
}

bool cublas_tensor_core_gemm_cuda(
    std::size_t m,
    std::size_t n,
    std::size_t k,
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    std::vector<float>& out,
    std::string& error
) {
    PreparedGemmKernelRunner runner;
    return runner.prepare(GemmBackend::CublasTensorCore, m, n, k, lhs, rhs, error) && runner.run(error) &&
           runner.copy_output(out, error);
}

}  // namespace ai_system::kernels
