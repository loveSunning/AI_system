#include "ai_system/kernels/basic_kernels.hpp"

#include <cuda_runtime.h>

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

bool copy_to_device(float* dst, const std::vector<float>& src, std::string& error) {
    const auto status = cudaMemcpy(dst, src.data(), sizeof(float) * src.size(), cudaMemcpyHostToDevice);
    if(status != cudaSuccess) {
        error = cudaGetErrorString(status);
        return false;
    }
    return true;
}

bool copy_to_host(std::vector<float>& dst, const float* src, std::string& error) {
    const auto status = cudaMemcpy(dst.data(), src, sizeof(float) * dst.size(), cudaMemcpyDeviceToHost);
    if(status != cudaSuccess) {
        error = cudaGetErrorString(status);
        return false;
    }
    return true;
}

bool copy_scalar_to_host(float& dst, const float* src, std::string& error) {
    const auto status = cudaMemcpy(&dst, src, sizeof(float), cudaMemcpyDeviceToHost);
    if(status != cudaSuccess) {
        error = cudaGetErrorString(status);
        return false;
    }
    return true;
}

bool check_last_launch(std::string& error) {
    const auto status = cudaGetLastError();
    if(status != cudaSuccess) {
        error = cudaGetErrorString(status);
        return false;
    }
    return true;
}

bool synchronize(std::string& error) {
    const auto status = cudaDeviceSynchronize();
    if(status != cudaSuccess) {
        error = cudaGetErrorString(status);
        return false;
    }
    return true;
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

}  // namespace

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
    if(lhs.size() != m * k) {
        error = "naive_gemm_cuda requires lhs.size() == m * k.";
        return false;
    }
    if(rhs.size() != k * n) {
        error = "naive_gemm_cuda requires rhs.size() == k * n.";
        return false;
    }

    out.assign(m * n, 0.0f);

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

    const dim3 block(16, 16);
    const dim3 grid(
        static_cast<unsigned int>((n + block.x - 1) / block.x),
        static_cast<unsigned int>((m + block.y - 1) / block.y)
    );

    naive_gemm_kernel<<<grid, block>>>(lhs_device.get(), rhs_device.get(), out_device.get(), m, n, k);

    if(!check_last_launch(error) || !synchronize(error)) {
        return false;
    }

    return copy_to_host(out, out_device.get(), error);
}

}  // namespace ai_system::kernels
