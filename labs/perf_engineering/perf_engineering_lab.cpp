#include "ai_system/benchmark/benchmark_runner.hpp"
#include "ai_system/config.hpp"
#include "ai_system/kernels/basic_kernels.hpp"
#include "ai_system/runtime/gpu_info.hpp"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct LabOptions {
    std::size_t vector_size {1U << 20U};
    std::size_t reduction_size {1U << 20U};
    std::size_t gemm_m {128};
    std::size_t gemm_n {128};
    std::size_t gemm_k {128};
    std::size_t warmup_iterations {2};
    std::size_t measured_iterations {6};
};

void print_usage() {
    std::cout << "AI_system perf engineering lab\n"
              << "  --vector-size N      Vector add input length\n"
              << "  --reduction-size N   Reduction input length\n"
              << "  --gemm-m M           GEMM rows of lhs/out\n"
              << "  --gemm-n N           GEMM columns of rhs/out\n"
              << "  --gemm-k K           GEMM shared dimension\n"
              << "  --warmup I           Warmup iterations for each benchmark\n"
              << "  --iters I            Measured iterations for each benchmark\n"
              << "  --help               Show this help message\n";
}

bool parse_size_argument(const char* raw_value, const char* option_name, std::size_t& output) {
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(raw_value, &end, 10);
    if(raw_value == nullptr || *raw_value == '\0' || end == nullptr || *end != '\0' || parsed == 0ULL) {
        std::cerr << "Invalid value for " << option_name << ": " << (raw_value == nullptr ? "<null>" : raw_value) << "\n";
        return false;
    }

    output = static_cast<std::size_t>(parsed);
    return true;
}

bool parse_options(int argc, char** argv, LabOptions& options) {
    for(int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        auto require_value = [&](std::size_t& destination) -> bool {
            if(index + 1 >= argc) {
                std::cerr << "Missing value for " << argument << "\n";
                return false;
            }
            ++index;
            return parse_size_argument(argv[index], argv[index - 1], destination);
        };

        if(argument == "--help") {
            print_usage();
            return false;
        }
        if(argument == "--vector-size") {
            if(!require_value(options.vector_size)) {
                return false;
            }
            continue;
        }
        if(argument == "--reduction-size") {
            if(!require_value(options.reduction_size)) {
                return false;
            }
            continue;
        }
        if(argument == "--gemm-m") {
            if(!require_value(options.gemm_m)) {
                return false;
            }
            continue;
        }
        if(argument == "--gemm-n") {
            if(!require_value(options.gemm_n)) {
                return false;
            }
            continue;
        }
        if(argument == "--gemm-k") {
            if(!require_value(options.gemm_k)) {
                return false;
            }
            continue;
        }
        if(argument == "--warmup") {
            if(!require_value(options.warmup_iterations)) {
                return false;
            }
            continue;
        }
        if(argument == "--iters") {
            if(!require_value(options.measured_iterations)) {
                return false;
            }
            continue;
        }

        std::cerr << "Unknown argument: " << argument << "\n";
        print_usage();
        return false;
    }

    return true;
}

ai_system::benchmark::BenchmarkConfig make_benchmark_config(const LabOptions& options) {
    return ai_system::benchmark::BenchmarkConfig {options.warmup_iterations, options.measured_iterations};
}

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

int run_vector_add_case(const LabOptions& options) {
    const std::size_t element_count = options.vector_size;

    std::vector<float> lhs(element_count);
    std::vector<float> rhs(element_count);
    std::vector<float> cpu_out;
    std::vector<float> cuda_out;

    ai_system::kernels::fill_random(lhs, -1.0f, 1.0f, 7u);
    ai_system::kernels::fill_random(rhs, -1.0f, 1.0f, 13u);
    ai_system::kernels::vector_add_cpu(lhs, rhs, cpu_out);

    const auto config = make_benchmark_config(options);
    std::cout << "vector_add size=" << element_count << "\n";
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

int run_reduction_case(const LabOptions& options) {
    const std::size_t element_count = options.reduction_size;

    std::vector<float> values(element_count);
    ai_system::kernels::fill_random(values, -0.5f, 0.5f, 29u);

    float cpu_sum = 0.0f;
    float cuda_sum = 0.0f;

    const auto config = make_benchmark_config(options);
    std::cout << "reduction size=" << element_count << "\n";
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

int run_gemm_case(const LabOptions& options) {
    const std::size_t m = options.gemm_m;
    const std::size_t n = options.gemm_n;
    const std::size_t k = options.gemm_k;

    std::vector<float> lhs(m * k);
    std::vector<float> rhs(k * n);
    std::vector<float> cpu_out;
    std::vector<float> cuda_out;

    ai_system::kernels::fill_random(lhs, -1.0f, 1.0f, 41u);
    ai_system::kernels::fill_random(rhs, -1.0f, 1.0f, 53u);
    ai_system::kernels::naive_gemm_cpu(m, n, k, lhs, rhs, cpu_out);

    const auto config = make_benchmark_config(options);
    std::cout << "naive_gemm shape=" << m << "x" << n << "x" << k << "\n";
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

int main(int argc, char** argv) {
    LabOptions options;
    if(!parse_options(argc, argv, options)) {
        return argc > 1 && std::string_view(argv[1]) == "--help" ? 0 : 1;
    }

    std::cout << "AI_system perf engineering lab\n";
    std::cout << "Configured architectures: " << AI_SYSTEM_CUDA_ARCHITECTURES << "\n";
    std::cout << "Warmup iterations: " << options.warmup_iterations << "\n";
    std::cout << "Measured iterations: " << options.measured_iterations << "\n";
    std::cout << ai_system::runtime::summarize_gpus(ai_system::runtime::query_gpus());

    int failures = 0;
    failures += run_vector_add_case(options);
    failures += run_reduction_case(options);
    failures += run_gemm_case(options);

    return failures == 0 ? 0 : 1;
}
