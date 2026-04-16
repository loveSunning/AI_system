#include "ai_system/benchmark/benchmark_report.hpp"
#include "ai_system/benchmark/benchmark_runner.hpp"
#include "ai_system/config.hpp"
#include "ai_system/kernels/basic_kernels.hpp"
#include "ai_system/profiling/nvtx.hpp"
#include "ai_system/runtime/gpu_info.hpp"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct LabOptions {
    std::size_t vector_size {1U << 20U};
    std::size_t reduction_size {1U << 20U};
    std::size_t gemm_m {1024};
    std::size_t gemm_n {1024};
    std::size_t gemm_k {1024};
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

std::string format_count_shape(std::size_t count) {
    return "N=" + std::to_string(count);
}

std::string format_gemm_shape(std::size_t m, std::size_t n, std::size_t k) {
    return std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k);
}

double throughput_in_gigabytes(double bytes, double average_ms) {
    if(average_ms <= 0.0) {
        return 0.0;
    }
    return bytes / 1.0e9 / (average_ms / 1000.0);
}

double compute_vector_add_gbps(std::size_t element_count, double average_ms) {
    const double bytes = static_cast<double>(element_count) * static_cast<double>(sizeof(float)) * 3.0;
    return throughput_in_gigabytes(bytes, average_ms);
}

double compute_reduction_gbps(std::size_t element_count, double average_ms) {
    const double bytes = static_cast<double>(element_count) * static_cast<double>(sizeof(float));
    return throughput_in_gigabytes(bytes, average_ms);
}

double compute_gemm_gflops(std::size_t m, std::size_t n, std::size_t k, double average_ms) {
    if(average_ms <= 0.0) {
        return 0.0;
    }

    const double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    return flops / 1.0e9 / (average_ms / 1000.0);
}

int run_vector_add_case(const LabOptions& options, ai_system::benchmark::BenchmarkReport& report) {
    const ai_system::profiling::ScopedNvtxRange case_range("vector_add");

    const std::size_t element_count = options.vector_size;
    const std::string shape = format_count_shape(element_count);

    std::vector<float> lhs(element_count);
    std::vector<float> rhs(element_count);
    std::vector<float> cpu_out;
    std::vector<float> cuda_out;

    {
        const ai_system::profiling::ScopedNvtxRange phase_range("prepare_inputs");
        ai_system::kernels::fill_random(lhs, -1.0f, 1.0f, 7u);
        ai_system::kernels::fill_random(rhs, -1.0f, 1.0f, 13u);
        ai_system::kernels::vector_add_cpu(lhs, rhs, cpu_out);
    }

    const auto config = make_benchmark_config(options);
    const auto cpu_result = [&]() {
        const ai_system::profiling::ScopedNvtxRange phase_range("cpu_benchmark");
        return ai_system::benchmark::run_benchmark("vector_add/cpu", config, [&]() {
            ai_system::kernels::vector_add_cpu(lhs, rhs, cpu_out);
        });
    }();
    ai_system::benchmark::add_benchmark_row(
        report,
        "vector_add",
        "cpu",
        shape,
        cpu_result,
        compute_vector_add_gbps(element_count, cpu_result.average_ms),
        "GB/s"
    );

    std::string error;
    if(ai_system::kernels::vector_add_cuda(lhs, rhs, cuda_out, error)) {
        const bool matches = [&]() {
            const ai_system::profiling::ScopedNvtxRange phase_range("cuda_correctness");
            return ai_system::kernels::allclose(cpu_out, cuda_out);
        }();
        ai_system::benchmark::add_validation_row(
            report,
            "vector_add",
            "cuda",
            "correctness",
            matches ? "PASS" : "FAIL",
            matches ? "CPU/GPU outputs match." : "CPU/GPU outputs differ."
        );

        const auto cuda_result = [&]() {
            const ai_system::profiling::ScopedNvtxRange phase_range("cuda_benchmark");
            return ai_system::benchmark::run_benchmark("vector_add/cuda", config, [&]() {
                std::string local_error;
                if(!ai_system::kernels::vector_add_cuda(lhs, rhs, cuda_out, local_error)) {
                    throw std::runtime_error(local_error);
                }
            });
        }();
        ai_system::benchmark::add_benchmark_row(
            report,
            "vector_add",
            "cuda",
            shape,
            cuda_result,
            compute_vector_add_gbps(element_count, cuda_result.average_ms),
            "GB/s"
        );

        return matches ? 0 : 1;
    }

    ai_system::benchmark::add_validation_row(
        report,
        "vector_add",
        "cuda",
        "runtime",
        AI_SYSTEM_HAS_CUDA ? "FAIL" : "SKIP",
        error
    );
    return AI_SYSTEM_HAS_CUDA ? 1 : 0;
}

int run_reduction_case(const LabOptions& options, ai_system::benchmark::BenchmarkReport& report) {
    const ai_system::profiling::ScopedNvtxRange case_range("reduction");

    const std::size_t element_count = options.reduction_size;
    const std::string shape = format_count_shape(element_count);

    std::vector<float> values(element_count);
    {
        const ai_system::profiling::ScopedNvtxRange phase_range("prepare_inputs");
        ai_system::kernels::fill_random(values, -0.5f, 0.5f, 29u);
    }

    float cpu_sum = 0.0f;
    float cuda_sum = 0.0f;

    const auto config = make_benchmark_config(options);
    const auto cpu_result = [&]() {
        const ai_system::profiling::ScopedNvtxRange phase_range("cpu_benchmark");
        return ai_system::benchmark::run_benchmark("reduction/cpu", config, [&]() {
            cpu_sum = ai_system::kernels::reduction_sum_cpu(values);
        });
    }();
    ai_system::benchmark::add_benchmark_row(
        report,
        "reduction",
        "cpu",
        shape,
        cpu_result,
        compute_reduction_gbps(element_count, cpu_result.average_ms),
        "GB/s"
    );

    std::string error;
    if(ai_system::kernels::reduction_sum_cuda(values, cuda_sum, error)) {
        const bool matches = [&]() {
            const ai_system::profiling::ScopedNvtxRange phase_range("cuda_correctness");
            return std::fabs(cpu_sum - cuda_sum) < 1.0e-2f;
        }();
        ai_system::benchmark::add_validation_row(
            report,
            "reduction",
            "cuda",
            "correctness",
            matches ? "PASS" : "FAIL",
            "cpu=" + ai_system::benchmark::format_decimal(cpu_sum, 6) +
                ", gpu=" + ai_system::benchmark::format_decimal(cuda_sum, 6)
        );

        const auto cuda_result = [&]() {
            const ai_system::profiling::ScopedNvtxRange phase_range("cuda_benchmark");
            return ai_system::benchmark::run_benchmark("reduction/cuda", config, [&]() {
                std::string local_error;
                if(!ai_system::kernels::reduction_sum_cuda(values, cuda_sum, local_error)) {
                    throw std::runtime_error(local_error);
                }
            });
        }();
        ai_system::benchmark::add_benchmark_row(
            report,
            "reduction",
            "cuda",
            shape,
            cuda_result,
            compute_reduction_gbps(element_count, cuda_result.average_ms),
            "GB/s"
        );

        return matches ? 0 : 1;
    }

    ai_system::benchmark::add_validation_row(
        report,
        "reduction",
        "cuda",
        "runtime",
        AI_SYSTEM_HAS_CUDA ? "FAIL" : "SKIP",
        error
    );
    return AI_SYSTEM_HAS_CUDA ? 1 : 0;
}

int run_gemm_case(const LabOptions& options, ai_system::benchmark::BenchmarkReport& report) {
    const ai_system::profiling::ScopedNvtxRange case_range("gemm");

    const std::size_t m = options.gemm_m;
    const std::size_t n = options.gemm_n;
    const std::size_t k = options.gemm_k;
    const std::string shape = format_gemm_shape(m, n, k);

    std::vector<float> lhs(m * k);
    std::vector<float> rhs(k * n);
    std::vector<float> cpu_out;

    {
        const ai_system::profiling::ScopedNvtxRange phase_range("prepare_inputs");
        ai_system::kernels::fill_random(lhs, -1.0f, 1.0f, 41u);
        ai_system::kernels::fill_random(rhs, -1.0f, 1.0f, 53u);
        ai_system::kernels::naive_gemm_cpu(m, n, k, lhs, rhs, cpu_out);
    }

    const auto config = make_benchmark_config(options);
    const auto cpu_result = [&]() {
        const ai_system::profiling::ScopedNvtxRange phase_range("cpu_benchmark");
        return ai_system::benchmark::run_benchmark("gemm/cpu_naive", config, [&]() {
            ai_system::kernels::naive_gemm_cpu(m, n, k, lhs, rhs, cpu_out);
        });
    }();
    ai_system::benchmark::add_benchmark_row(
        report,
        "gemm",
        "cpu_naive",
        shape,
        cpu_result,
        compute_gemm_gflops(m, n, k, cpu_result.average_ms),
        "GFLOPS"
    );

    auto run_gpu_gemm_variant = [&](const std::string& impl_name,
                                    const std::string& benchmark_name,
                                    auto&& gemm_fn,
                                    float absolute_tolerance,
                                    float relative_tolerance,
                                    const std::string& tolerance_label) {
        std::vector<float> gpu_out;
        std::string error;
        if(gemm_fn(m, n, k, lhs, rhs, gpu_out, error)) {
            const bool matches = [&]() {
                const ai_system::profiling::ScopedNvtxRange phase_range("cuda_correctness");
                return ai_system::kernels::allclose(cpu_out, gpu_out, absolute_tolerance, relative_tolerance);
            }();

            ai_system::benchmark::add_validation_row(
                report,
                "gemm",
                impl_name,
                "correctness",
                matches ? "PASS" : "FAIL",
                matches ? tolerance_label : "CPU/GPU outputs diverged beyond tolerance."
            );

            const auto gpu_result = [&]() {
                const ai_system::profiling::ScopedNvtxRange phase_range("cuda_benchmark");
                return ai_system::benchmark::run_benchmark(benchmark_name, config, [&]() {
                    std::string local_error;
                    if(!gemm_fn(m, n, k, lhs, rhs, gpu_out, local_error)) {
                        throw std::runtime_error(local_error);
                    }
                });
            }();

            ai_system::benchmark::add_benchmark_row(
                report,
                "gemm",
                impl_name,
                shape,
                gpu_result,
                compute_gemm_gflops(m, n, k, gpu_result.average_ms),
                "GFLOPS"
            );

            return matches ? 0 : 1;
        }

        ai_system::benchmark::add_validation_row(
            report,
            "gemm",
            impl_name,
            "runtime",
            AI_SYSTEM_HAS_CUDA ? "FAIL" : "SKIP",
            error
        );
        return AI_SYSTEM_HAS_CUDA ? 1 : 0;
    };

    int failures = 0;
    failures += run_gpu_gemm_variant(
        "cuda_naive",
        "gemm/cuda_naive",
        ai_system::kernels::naive_gemm_cuda,
        1.0e-3f,
        1.0e-3f,
        "allclose(abs=1e-3, rel=1e-3)."
    );
    failures += run_gpu_gemm_variant(
        "cublas_sgemm",
        "gemm/cublas_sgemm",
        ai_system::kernels::cublas_sgemm_cuda,
        1.0e-4f,
        1.0e-4f,
        "allclose(abs=1e-4, rel=1e-4)."
    );
    failures += run_gpu_gemm_variant(
        "cublas_hgemm",
        "gemm/cublas_hgemm",
        ai_system::kernels::cublas_hgemm_cuda,
        5.0e-1f,
        5.0e-1f,
        "allclose(abs=5e-1, rel=5e-1)."
    );
    failures += run_gpu_gemm_variant(
        "cublas_tensor_core",
        "gemm/cublas_tensor_core",
        ai_system::kernels::cublas_tensor_core_gemm_cuda,
        5.0e-2f,
        5.0e-2f,
        "allclose(abs=5e-2, rel=5e-2)."
    );

    return failures;
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

    ai_system::benchmark::BenchmarkReport report;
    int failures = 0;
    {
        const ai_system::profiling::ScopedNvtxRange workload_range("profiled_workload");
        failures += run_vector_add_case(options, report);
        failures += run_reduction_case(options, report);
        failures += run_gemm_case(options, report);
    }

    ai_system::benchmark::print_benchmark_table(report);
    ai_system::benchmark::print_validation_table(report);

    return failures == 0 ? 0 : 1;
}
