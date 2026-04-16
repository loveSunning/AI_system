#include "ai_system/benchmark/benchmark_runner.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <vector>

namespace ai_system::benchmark {

BenchmarkResult run_benchmark(
    const std::string& name,
    const BenchmarkConfig& config,
    const std::function<void()>& fn
) {
    for(std::size_t iteration = 0; iteration < config.warmup_iterations; ++iteration) {
        fn();
    }

    std::vector<double> measurements;
    measurements.reserve(config.measured_iterations);

    for(std::size_t iteration = 0; iteration < config.measured_iterations; ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        fn();
        const auto stop = std::chrono::steady_clock::now();
        measurements.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }

    BenchmarkResult result;
    result.name = name;
    result.warmup_iterations = config.warmup_iterations;
    result.measured_iterations = config.measured_iterations;

    if(measurements.empty()) {
        return result;
    }

    double total = 0.0;
    result.min_ms = std::numeric_limits<double>::max();
    result.max_ms = 0.0;

    for(const double measurement : measurements) {
        total += measurement;
        result.min_ms = std::min(result.min_ms, measurement);
        result.max_ms = std::max(result.max_ms, measurement);
    }

    result.average_ms = total / static_cast<double>(measurements.size());
    return result;
}

}  // namespace ai_system::benchmark
