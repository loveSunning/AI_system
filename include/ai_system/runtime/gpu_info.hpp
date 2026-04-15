#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace ai_system::runtime {

struct GpuInfo {
    int device_index {0};
    std::string name;
    int major {0};
    int minor {0};
    std::size_t total_memory_mib {0};
};

std::vector<GpuInfo> query_gpus();
std::string summarize_gpus(const std::vector<GpuInfo>& gpus);

}  // namespace ai_system::runtime
