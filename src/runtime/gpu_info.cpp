#include "ai_system/config.hpp"
#include "ai_system/runtime/gpu_info.hpp"

#if AI_SYSTEM_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

#include <sstream>

namespace ai_system::runtime {

std::vector<GpuInfo> query_gpus() {
    std::vector<GpuInfo> gpus;

#if AI_SYSTEM_HAS_CUDA
    int device_count = 0;
    if(cudaGetDeviceCount(&device_count) != cudaSuccess) {
        return gpus;
    }

    for(int device_index = 0; device_index < device_count; ++device_index) {
        cudaDeviceProp properties {};
        if(cudaGetDeviceProperties(&properties, device_index) != cudaSuccess) {
            continue;
        }

        GpuInfo info;
        info.device_index = device_index;
        info.name = properties.name;
        info.major = properties.major;
        info.minor = properties.minor;
        info.total_memory_mib = static_cast<std::size_t>(properties.totalGlobalMem / (1024ULL * 1024ULL));
        gpus.push_back(info);
    }
#endif

    return gpus;
}

std::string summarize_gpus(const std::vector<GpuInfo>& gpus) {
    if(gpus.empty()) {
        return "No CUDA GPU detected or CUDA support is disabled.\n";
    }

    std::ostringstream output;
    for(const auto& gpu : gpus) {
        output << "GPU[" << gpu.device_index << "] " << gpu.name
               << " | compute capability " << gpu.major << "." << gpu.minor
               << " | memory " << gpu.total_memory_mib << " MiB\n";
    }
    return output.str();
}

}  // namespace ai_system::runtime
