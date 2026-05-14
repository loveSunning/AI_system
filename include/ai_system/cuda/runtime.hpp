#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <string>
#include <vector>

namespace ai_system::cuda_utils {

inline bool check_status(cudaError_t status, const char* context, std::string& error) {
    if(status != cudaSuccess) {
        error = std::string(context) + ": " + cudaGetErrorString(status);
        return false;
    }
    return true;
}

inline bool check_last_launch(std::string& error) {
    return check_status(cudaGetLastError(), "cudaGetLastError", error);
}

inline bool synchronize(std::string& error) {
    return check_status(cudaDeviceSynchronize(), "cudaDeviceSynchronize", error);
}

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
            error = std::string("cudaMalloc: ") + cudaGetErrorString(status);
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
bool copy_to_device(T* dst, const T* src, std::size_t count, std::string& error) {
    return check_status(cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyHostToDevice), "cudaMemcpy(H2D)", error);
}

template <typename T>
bool copy_to_device(T* dst, const std::vector<T>& src, std::string& error) {
    return copy_to_device(dst, src.data(), src.size(), error);
}

template <typename T>
bool copy_to_host(T* dst, const T* src, std::size_t count, std::string& error) {
    return check_status(cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyDeviceToHost), "cudaMemcpy(D2H)", error);
}

template <typename T>
bool copy_to_host(std::vector<T>& dst, const T* src, std::string& error) {
    return copy_to_host(dst.data(), src, dst.size(), error);
}

template <typename T>
bool copy_scalar_to_host(T& dst, const T* src, std::string& error) {
    return check_status(cudaMemcpy(&dst, src, sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy(D2H scalar)", error);
}

class EventPair {
public:
    EventPair() = default;

    ~EventPair() {
        reset();
    }

    EventPair(const EventPair&) = delete;
    EventPair& operator=(const EventPair&) = delete;

    bool ensure(std::string& error) {
        if(start_event_ == nullptr) {
            if(!check_status(cudaEventCreate(&start_event_), "cudaEventCreate(start)", error)) {
                start_event_ = nullptr;
                return false;
            }
        }

        if(stop_event_ == nullptr) {
            if(!check_status(cudaEventCreate(&stop_event_), "cudaEventCreate(stop)", error)) {
                return false;
            }
        }

        return true;
    }

    bool record_start(std::string& error) {
        return ensure(error) && check_status(cudaEventRecord(start_event_), "cudaEventRecord(start)", error);
    }

    bool record_stop(std::string& error) {
        return ensure(error) && check_status(cudaEventRecord(stop_event_), "cudaEventRecord(stop)", error);
    }

    bool synchronize_stop(std::string& error) {
        return ensure(error) && check_status(cudaEventSynchronize(stop_event_), "cudaEventSynchronize(stop)", error);
    }

    bool elapsed_ms(float& elapsed, std::string& error) {
        return ensure(error) && check_status(cudaEventElapsedTime(&elapsed, start_event_, stop_event_), "cudaEventElapsedTime", error);
    }

    void reset() {
        if(start_event_ != nullptr) {
            cudaEventDestroy(start_event_);
            start_event_ = nullptr;
        }
        if(stop_event_ != nullptr) {
            cudaEventDestroy(stop_event_);
            stop_event_ = nullptr;
        }
    }

private:
    cudaEvent_t start_event_ {nullptr};
    cudaEvent_t stop_event_ {nullptr};
};

}  // namespace ai_system::cuda_utils
