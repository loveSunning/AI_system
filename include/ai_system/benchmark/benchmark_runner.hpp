#pragma once

#include <cstddef>
#include <functional>
#include <string>

namespace ai_system::benchmark {

struct BenchmarkConfig {
    std::size_t warmup_iterations {3};
    std::size_t measured_iterations {10};
};

struct BenchmarkResult {
    std::string name;
    double average_ms {0.0};
    double min_ms {0.0};
    double max_ms {0.0};
};

BenchmarkResult run_benchmark(
    const std::string& name,
    const BenchmarkConfig& config,
    const std::function<void()>& fn
);

}  // namespace ai_system::benchmark
