#include "ai_system/benchmark/benchmark_runner.hpp"
#include "ai_system/config.hpp"
#include "ai_system/kernels/basic_kernels.hpp"
#include "ai_system/runtime/gpu_info.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void print_benchmark(const ai_system::benchmark::BenchmarkResult& result) {
    std::cout << std::left << std::setw(24) << result.name
              << " avg=" << std::setw(10) << std::fixed << std::setprecision(3) << result.average_ms
              << " ms min=" << std::setw(10) << result.min_ms
              << " ms max=" << result.max_ms << " ms\n";
}

void print_case_status(const std::string& label, bool success, const std::string& detail = {}) {
    std::cout << "[" << (success ? "PASS" : "FAIL") << "] " << label;
    if(!detail.empty()) {
        std::cout << " | " << detail;
    }
    std::cout << "\n";
}

int run_vector_add_case() {
    constexpr std::size_t element_count = 1U << 20U;

    std::vector<float> lhs(element_count);
    std::vector<float> rhs(element_count);
    std::vector<float> cpu_out;
    std::vector<float> cuda_out;

    ai_system::kernels::fill_random(lhs, -1.0f, 1.0f, 7u);
    ai_system::kernels::fill_random(rhs, -1.0f, 1.0f, 13u);
    ai_system::kernels::vector_add_cpu(lhs, rhs, cpu_out);

    const ai_system::benchmark::BenchmarkConfig config {2, 6};
    print_benchmark(ai_system::benchmark::run_benchmark("vector_add/cpu", config, [&]() {
        ai_system::kernels::vector_add_cpu(lhs, rhs, cpu_out);
    }));

    std::string error;
    if(ai_system::kernels::vector_add_cuda(lhs, rhs, cuda_out, error)) {
        const bool matches = ai_system::kernels::allclose(cpu_out, cuda_out);
        print_case_status("vector_add correctness", matches);
        print_benchmark(ai_system::benchmark::run_benchmark("vector_add/cuda", config, [&]() {
            std::string local_error;
            if(!ai_system::kernels::vector_add_cuda(lhs, rhs, cuda_out, local_error)) {
                throw std::runtime_error(local_error);
            }
        }));
        return matches ? 0 : 1;
    }

    print_case_status("vector_add CUDA path", false, error);
    return AI_SYSTEM_HAS_CUDA ? 1 : 0;
}

int run_reduction_case() {
    constexpr std::size_t element_count = 1U << 20U;

    std::vector<float> values(element_count);
    ai_system::kernels::fill_random(values, -0.5f, 0.5f, 29u);

    float cpu_sum = 0.0f;
    float cuda_sum = 0.0f;

    const ai_system::benchmark::BenchmarkConfig config {2, 6};
    print_benchmark(ai_system::benchmark::run_benchmark("reduction/cpu", config, [&]() {
        cpu_sum = ai_system::kernels::reduction_sum_cpu(values);
    }));

    std::string error;
    if(ai_system::kernels::reduction_sum_cuda(values, cuda_sum, error)) {
        const bool matches = std::fabs(cpu_sum - cuda_sum) < 1.0e-2f;
        print_case_status(
            "reduction correctness",
            matches,
            "cpu=" + std::to_string(cpu_sum) + ", gpu=" + std::to_string(cuda_sum)
        );
        print_benchmark(ai_system::benchmark::run_benchmark("reduction/cuda", config, [&]() {
            std::string local_error;
            if(!ai_system::kernels::reduction_sum_cuda(values, cuda_sum, local_error)) {
                throw std::runtime_error(local_error);
            }
        }));
        return matches ? 0 : 1;
    }

    print_case_status("reduction CUDA path", false, error);
    return AI_SYSTEM_HAS_CUDA ? 1 : 0;
}

int run_gemm_case() {
    constexpr std::size_t m = 128;
    constexpr std::size_t n = 128;
    constexpr std::size_t k = 128;

    std::vector<float> lhs(m * k);
    std::vector<float> rhs(k * n);
    std::vector<float> cpu_out;
    std::vector<float> cuda_out;

    ai_system::kernels::fill_random(lhs, -1.0f, 1.0f, 41u);
    ai_system::kernels::fill_random(rhs, -1.0f, 1.0f, 53u);
    ai_system::kernels::naive_gemm_cpu(m, n, k, lhs, rhs, cpu_out);

    const ai_system::benchmark::BenchmarkConfig config {1, 4};
    print_benchmark(ai_system::benchmark::run_benchmark("naive_gemm/cpu", config, [&]() {
        ai_system::kernels::naive_gemm_cpu(m, n, k, lhs, rhs, cpu_out);
    }));

    std::string error;
    if(ai_system::kernels::naive_gemm_cuda(m, n, k, lhs, rhs, cuda_out, error)) {
        const bool matches = ai_system::kernels::allclose(cpu_out, cuda_out, 1.0e-3f, 1.0e-3f);
        print_case_status("naive_gemm correctness", matches);
        print_benchmark(ai_system::benchmark::run_benchmark("naive_gemm/cuda", config, [&]() {
            std::string local_error;
            if(!ai_system::kernels::naive_gemm_cuda(m, n, k, lhs, rhs, cuda_out, local_error)) {
                throw std::runtime_error(local_error);
            }
        }));
        return matches ? 0 : 1;
    }

    print_case_status("naive_gemm CUDA path", false, error);
    return AI_SYSTEM_HAS_CUDA ? 1 : 0;
}

}  // namespace

int main() {
    std::cout << "AI_system perf engineering lab\n";
    std::cout << "Configured architectures: " << AI_SYSTEM_CUDA_ARCHITECTURES << "\n";
    std::cout << ai_system::runtime::summarize_gpus(ai_system::runtime::query_gpus());

    int failures = 0;
    failures += run_vector_add_case();
    failures += run_reduction_case();
    failures += run_gemm_case();

    return failures == 0 ? 0 : 1;
}
