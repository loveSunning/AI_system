#include "ai_system/benchmark/benchmark_report.hpp"
#include "ai_system/benchmark/benchmark_runner.hpp"
#include "ai_system/config.hpp"
#include "ai_system/kernels/basic_kernels.hpp"
#include "ai_system/profiling/nvtx.hpp"
#include "ai_system/runtime/gpu_info.hpp"
#include "gemm_lab.hpp"

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t kMaxCpuGemmDimension = 2048;

struct Options {
    std::size_t m {1024};
    std::size_t n {1024};
    std::size_t k {1024};
    ai_system::labs::gemm::GemmLabTileConfig gemm_tile;
    std::size_t warmup_iterations {2};
    std::size_t measured_iterations {6};
    bool include_e2e {false};
};

struct GemmTolerance {
    float absolute {0.0f};
    float relative {0.0f};
    std::string_view label;
};

constexpr GemmTolerance kFp32GemmTolerance {1.0e-3f, 1.0e-3f, "allclose(abs=1e-3, rel=1e-3)."};
constexpr GemmTolerance kCublasSgemmTolerance {1.0e-4f, 1.0e-4f, "allclose(abs=1e-4, rel=1e-4)."};

void print_usage() {
    std::cout << "AI_system SGEMM benchmark lab\n"
              << "  --gemm-m M       GEMM rows of lhs/out\n"
              << "  --gemm-n N       GEMM columns of rhs/out\n"
              << "  --gemm-k K       GEMM shared dimension\n"
              << "  --gemm-tile-m M  GEMM lab output tile rows; supported: 8, 16, 32, 64, 128\n"
              << "  --gemm-tile-n N  GEMM lab output tile columns; supported: 8, 16, 32, 64, 128\n"
              << "  --gemm-tile-k K  GEMM lab reduction tile; supported: 8, 16, 32\n"
              << "  --gemm-reg-m M  Register-tiled GEMM per-thread rows; supported: 1, 2, 4\n"
              << "  --gemm-reg-n N  Register-tiled GEMM per-thread columns; supported: 1, 2, 4\n"
              << "  --warmup I       Warmup iterations for each benchmark\n"
              << "  --iters I        Measured iterations for each benchmark\n"
              << "  --include-e2e    Also run end-to-end SGEMM benchmarks\n"
              << "  --help           Show this help message\n";
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

bool parse_int_argument(const char* raw_value, const char* option_name, int& output) {
    std::size_t parsed = 0;
    if(!parse_size_argument(raw_value, option_name, parsed)) {
        return false;
    }
    if(parsed > static_cast<std::size_t>((std::numeric_limits<int>::max)())) {
        std::cerr << "Value for " << option_name << " exceeds int range: " << raw_value << "\n";
        return false;
    }

    output = static_cast<int>(parsed);
    return true;
}

bool parse_options(int argc, char** argv, Options& options) {
    for(int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        auto require_size_value = [&](std::size_t& destination) -> bool {
            if(index + 1 >= argc) {
                std::cerr << "Missing value for " << argument << "\n";
                return false;
            }
            ++index;
            return parse_size_argument(argv[index], argv[index - 1], destination);
        };
        auto require_int_value = [&](int& destination) -> bool {
            if(index + 1 >= argc) {
                std::cerr << "Missing value for " << argument << "\n";
                return false;
            }
            ++index;
            return parse_int_argument(argv[index], argv[index - 1], destination);
        };

        if(argument == "--help") {
            print_usage();
            return false;
        }
        if(argument == "--include-e2e") {
            options.include_e2e = true;
            continue;
        }
        if(argument == "--gemm-m") {
            if(!require_size_value(options.m)) {
                return false;
            }
            continue;
        }
        if(argument == "--gemm-n") {
            if(!require_size_value(options.n)) {
                return false;
            }
            continue;
        }
        if(argument == "--gemm-k") {
            if(!require_size_value(options.k)) {
                return false;
            }
            continue;
        }
        if(argument == "--gemm-tile-m") {
            if(!require_int_value(options.gemm_tile.block_m)) {
                return false;
            }
            continue;
        }
        if(argument == "--gemm-tile-n") {
            if(!require_int_value(options.gemm_tile.block_n)) {
                return false;
            }
            continue;
        }
        if(argument == "--gemm-tile-k") {
            if(!require_int_value(options.gemm_tile.block_k)) {
                return false;
            }
            continue;
        }
        if(argument == "--gemm-reg-m") {
            if(!require_int_value(options.gemm_tile.register_m)) {
                return false;
            }
            continue;
        }
        if(argument == "--gemm-reg-n") {
            if(!require_int_value(options.gemm_tile.register_n)) {
                return false;
            }
            continue;
        }
        if(argument == "--warmup") {
            if(!require_size_value(options.warmup_iterations)) {
                return false;
            }
            continue;
        }
        if(argument == "--iters") {
            if(!require_size_value(options.measured_iterations)) {
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

ai_system::benchmark::BenchmarkConfig make_benchmark_config(const Options& options) {
    return ai_system::benchmark::BenchmarkConfig {options.warmup_iterations, options.measured_iterations};
}

std::string format_gemm_shape(std::size_t m, std::size_t n, std::size_t k) {
    return std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k);
}

std::string format_gemm_tile_shape(ai_system::labs::gemm::GemmLabTileConfig tile_config) {
    return std::to_string(tile_config.block_m) + "x" + std::to_string(tile_config.block_n) + "x" +
        std::to_string(tile_config.block_k);
}

std::string format_gemm_register_shape(ai_system::labs::gemm::GemmLabTileConfig tile_config) {
    return std::to_string(tile_config.register_m) + "x" + std::to_string(tile_config.register_n);
}

double compute_gemm_gflops(std::size_t m, std::size_t n, std::size_t k, double average_ms) {
    if(average_ms <= 0.0) {
        return 0.0;
    }

    const double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    return flops / 1.0e9 / (average_ms / 1000.0);
}

bool should_include_cpu_naive_gemm(std::size_t m, std::size_t n, std::size_t k) {
    return m <= kMaxCpuGemmDimension && n <= kMaxCpuGemmDimension && k <= kMaxCpuGemmDimension;
}

bool should_skip_gemm_lab_variant(bool allow_skip, const std::string& impl_name, const std::string& error) {
    if(!allow_skip) {
        return false;
    }
    if(error.find("not implemented") != std::string::npos) {
        return true;
    }
    return impl_name == "tiled_gemm_block" && error.find("requires block_m * block_n <= 1024") != std::string::npos;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if(!parse_options(argc, argv, options)) {
        return argc > 1 && std::string_view(argv[1]) == "--help" ? 0 : 1;
    }

    const ai_system::profiling::ScopedNvtxRange lab_range("sgemm_benchmark_lab");

    const std::size_t m = options.m;
    const std::size_t n = options.n;
    const std::size_t k = options.k;
    const std::string shape = format_gemm_shape(m, n, k);
    const std::string gemm_tile_shape = format_gemm_tile_shape(options.gemm_tile);
    const std::string gemm_register_shape = format_gemm_register_shape(options.gemm_tile);
    const auto config = make_benchmark_config(options);

    std::cout << "AI_system SGEMM benchmark lab\n";
    std::cout << "Configured architectures: " << AI_SYSTEM_CUDA_ARCHITECTURES << "\n";
    std::cout << "GEMM shape: " << shape << "\n";
    std::cout << "GEMM lab tile: " << gemm_tile_shape << "\n";
    std::cout << "GEMM register tile: " << gemm_register_shape << "\n";
    std::cout << "SGEMM e2e benchmarks: " << (options.include_e2e ? "enabled" : "disabled") << "\n";
    std::cout << "Warmup iterations: " << options.warmup_iterations << "\n";
    std::cout << "Measured iterations: " << options.measured_iterations << "\n";
    std::cout << ai_system::runtime::summarize_gpus(ai_system::runtime::query_gpus());

    std::vector<float> lhs(m * k);
    std::vector<float> rhs(k * n);
    std::vector<float> reference_out;
    std::string reference_impl_name;

    {
        const ai_system::profiling::ScopedNvtxRange phase_range("prepare_inputs");
        ai_system::kernels::fill_random(lhs, -1.0f, 1.0f, 41u);
        ai_system::kernels::fill_random(rhs, -1.0f, 1.0f, 53u);
    }

    ai_system::benchmark::BenchmarkReport report;
    bool has_reference = false;
    int failures = 0;

    if(should_include_cpu_naive_gemm(m, n, k)) {
        ai_system::kernels::naive_gemm_cpu(m, n, k, lhs, rhs, reference_out);
        reference_impl_name = "cpu_naive";
        has_reference = true;
    } else {
        const std::string detail =
            "Skipped because max(M,N,K) > " + std::to_string(kMaxCpuGemmDimension) + "; CPU baseline disabled.";
        if(options.include_e2e) {
            ai_system::benchmark::add_validation_row(report, "sgemm_e2e", "cpu_naive", "size_gate", "SKIP", detail);
        }
        ai_system::benchmark::add_validation_row(report, "sgemm_kernel_only", "cpu_naive", "size_gate", "SKIP", detail);

        std::string reference_error;
        if(ai_system::kernels::cublas_sgemm_cuda(m, n, k, lhs, rhs, reference_out, reference_error)) {
            reference_impl_name = "cublas_sgemm";
            has_reference = true;
        } else if(AI_SYSTEM_HAS_CUDA) {
            if(options.include_e2e) {
                ai_system::benchmark::add_validation_row(
                    report,
                    "sgemm_e2e",
                    "reference",
                    "runtime",
                    "FAIL",
                    "Failed to build cuBLAS SGEMM reference: " + reference_error
                );
            }
            ai_system::benchmark::add_validation_row(
                report,
                "sgemm_kernel_only",
                "reference",
                "runtime",
                "FAIL",
                "Failed to build cuBLAS SGEMM reference: " + reference_error
            );
            ++failures;
        }
    }

    auto format_reference_detail = [&](std::string_view tolerance_label) -> std::string {
        if(reference_impl_name.empty()) {
            return std::string(tolerance_label);
        }
        return std::string(tolerance_label) + " reference=" + reference_impl_name + ".";
    };

    auto compare_against_reference = [&](const std::vector<float>& gpu_out, const GemmTolerance& tolerance) {
        if(!has_reference) {
            return false;
        }
        return ai_system::kernels::allclose(reference_out, gpu_out, tolerance.absolute, tolerance.relative);
    };

    auto run_e2e_variant = [&](const std::string& impl_name,
                               const std::string& benchmark_name,
                               auto&& gemm_fn,
                               const GemmTolerance& tolerance,
                               std::string tile_shape = "none",
                               bool allow_unimplemented_skip = false,
                               std::string register_shape = "none") {
        std::vector<float> gpu_out;
        std::string error;
        if(gemm_fn(m, n, k, lhs, rhs, gpu_out, error)) {
            const bool matches = [&]() -> bool {
                if(!has_reference) {
                    return false;
                }
                const ai_system::profiling::ScopedNvtxRange phase_range("sgemm_e2e_correctness");
                return compare_against_reference(gpu_out, tolerance);
            }();

            ai_system::benchmark::add_validation_row(
                report,
                "sgemm_e2e",
                impl_name,
                "correctness",
                has_reference ? (matches ? "PASS" : "FAIL") : "SKIP",
                has_reference ? (matches ? format_reference_detail(tolerance.label)
                                         : "Reference/GPU outputs diverged beyond tolerance.")
                              : "No correctness reference is available."
            );

            const auto result = [&]() {
                const ai_system::profiling::ScopedNvtxRange phase_range("sgemm_e2e_benchmark");
                return ai_system::benchmark::run_benchmark(benchmark_name, config, [&]() {
                    std::string local_error;
                    if(!gemm_fn(m, n, k, lhs, rhs, gpu_out, local_error)) {
                        throw std::runtime_error(local_error);
                    }
                });
            }();

            ai_system::benchmark::add_benchmark_row(
                report,
                "sgemm_e2e",
                impl_name,
                shape,
                result,
                compute_gemm_gflops(m, n, k, result.average_ms),
                "GFLOPS",
                tile_shape,
                register_shape
            );

            return has_reference ? (matches ? 0 : 1) : 0;
        }

        const bool skip_unimplemented = should_skip_gemm_lab_variant(allow_unimplemented_skip, impl_name, error);
        ai_system::benchmark::add_validation_row(
            report,
            "sgemm_e2e",
            impl_name,
            "runtime",
            skip_unimplemented || !AI_SYSTEM_HAS_CUDA ? "SKIP" : "FAIL",
            error
        );
        return AI_SYSTEM_HAS_CUDA && !skip_unimplemented ? 1 : 0;
    };

    auto run_kernel_only_variant = [&](const std::string& impl_name,
                                       const std::string& benchmark_name,
                                       auto&& make_runner,
                                       auto&& prepare_runner,
                                       const GemmTolerance& tolerance,
                                       std::string tile_shape = "none",
                                       bool allow_unimplemented_skip = false,
                                       std::string register_shape = "none") {
        auto runner = make_runner();
        std::string error;
        if(!prepare_runner(runner, error)) {
            const bool skip_unimplemented = should_skip_gemm_lab_variant(allow_unimplemented_skip, impl_name, error);
            ai_system::benchmark::add_validation_row(
                report,
                "sgemm_kernel_only",
                impl_name,
                "prepare",
                skip_unimplemented || !AI_SYSTEM_HAS_CUDA ? "SKIP" : "FAIL",
                error
            );
            return AI_SYSTEM_HAS_CUDA && !skip_unimplemented ? 1 : 0;
        }

        std::vector<float> gpu_out;
        if(!runner.run(error) || !runner.copy_output(gpu_out, error)) {
            const bool skip_unimplemented = should_skip_gemm_lab_variant(allow_unimplemented_skip, impl_name, error);
            ai_system::benchmark::add_validation_row(
                report,
                "sgemm_kernel_only",
                impl_name,
                "runtime",
                skip_unimplemented || !AI_SYSTEM_HAS_CUDA ? "SKIP" : "FAIL",
                error
            );
            return AI_SYSTEM_HAS_CUDA && !skip_unimplemented ? 1 : 0;
        }

        const bool matches = [&]() -> bool {
            if(!has_reference) {
                return false;
            }
            const ai_system::profiling::ScopedNvtxRange phase_range("sgemm_kernel_only_correctness");
            return compare_against_reference(gpu_out, tolerance);
        }();
        ai_system::benchmark::add_validation_row(
            report,
            "sgemm_kernel_only",
            impl_name,
            "correctness",
            has_reference ? (matches ? "PASS" : "FAIL") : "SKIP",
            has_reference ? (matches ? format_reference_detail(tolerance.label)
                                     : "Reference/GPU outputs diverged beyond tolerance.")
                          : "No correctness reference is available."
        );

        const auto result = [&]() {
            const ai_system::profiling::ScopedNvtxRange phase_range("sgemm_kernel_only_benchmark");
            return ai_system::benchmark::run_timed_benchmark(
                benchmark_name,
                config,
                [&]() {
                    std::string local_error;
                    if(!runner.run(local_error)) {
                        throw std::runtime_error(local_error);
                    }
                },
                [&]() -> double {
                    double elapsed_ms = 0.0;
                    std::string local_error;
                    if(!runner.run_timed(elapsed_ms, local_error)) {
                        throw std::runtime_error(local_error);
                    }
                    return elapsed_ms;
                }
            );
        }();

        ai_system::benchmark::add_benchmark_row(
            report,
            "sgemm_kernel_only",
            impl_name,
            shape,
            result,
            compute_gemm_gflops(m, n, k, result.average_ms),
            "GFLOPS",
            tile_shape,
            register_shape
        );

        return has_reference ? (matches ? 0 : 1) : 0;
    };

    auto make_core_runner = []() {
        return ai_system::kernels::PreparedGemmKernelRunner {};
    };
    auto make_lab_runner = []() {
        return ai_system::labs::gemm::PreparedGemmLabRunner {};
    };
    auto prepare_core_backend = [&](ai_system::kernels::GemmBackend backend) {
        return [&, backend](auto& runner, std::string& error) {
            return runner.prepare(backend, m, n, k, lhs, rhs, error);
        };
    };
    auto prepare_tiled_gemm_block = [&](auto& runner, std::string& error) {
        return runner.prepare(
            ai_system::labs::gemm::GemmLabBackend::TiledGemmBlock,
            m,
            n,
            k,
            lhs,
            rhs,
            error,
            options.gemm_tile
        );
    };
    auto prepare_tiled_gemm_register = [&](auto& runner, std::string& error) {
        return runner.prepare(
            ai_system::labs::gemm::GemmLabBackend::TiledGemmRegister,
            m,
            n,
            k,
            lhs,
            rhs,
            error,
            options.gemm_tile
        );
    };

    if(options.include_e2e) {
        failures += run_e2e_variant(
            "cuda_naive",
            "sgemm/e2e/cuda_naive",
            ai_system::kernels::naive_gemm_cuda,
            kFp32GemmTolerance
        );
        failures += run_e2e_variant(
            "tiled_gemm_block",
            "sgemm/e2e/tiled_gemm_block",
            [&](std::size_t requested_m,
                std::size_t requested_n,
                std::size_t requested_k,
                const std::vector<float>& requested_lhs,
                const std::vector<float>& requested_rhs,
                std::vector<float>& requested_out,
                std::string& error) {
                return ai_system::labs::gemm::tiled_gemm_block_cuda(
                    requested_m,
                    requested_n,
                    requested_k,
                    requested_lhs,
                    requested_rhs,
                    requested_out,
                    error,
                    options.gemm_tile
                );
            },
            kFp32GemmTolerance,
            gemm_tile_shape,
            true
        );
        failures += run_e2e_variant(
            "tiled_gemm_register",
            "sgemm/e2e/tiled_gemm_register",
            [&](std::size_t requested_m,
                std::size_t requested_n,
                std::size_t requested_k,
                const std::vector<float>& requested_lhs,
                const std::vector<float>& requested_rhs,
                std::vector<float>& requested_out,
                std::string& error) {
                return ai_system::labs::gemm::tiled_gemm_register_cuda(
                    requested_m,
                    requested_n,
                    requested_k,
                    requested_lhs,
                    requested_rhs,
                    requested_out,
                    error,
                    options.gemm_tile
                );
            },
            kFp32GemmTolerance,
            gemm_tile_shape,
            true,
            gemm_register_shape
        );
        failures += run_e2e_variant(
            "cublas_sgemm",
            "sgemm/e2e/cublas_sgemm",
            ai_system::kernels::cublas_sgemm_cuda,
            kCublasSgemmTolerance
        );
    }

    failures += run_kernel_only_variant(
        "cuda_naive",
        "sgemm/kernel_only/cuda_naive",
        make_core_runner,
        prepare_core_backend(ai_system::kernels::GemmBackend::CudaNaive),
        kFp32GemmTolerance
    );
    failures += run_kernel_only_variant(
        "tiled_gemm_block",
        "sgemm/kernel_only/tiled_gemm_block",
        make_lab_runner,
        prepare_tiled_gemm_block,
        kFp32GemmTolerance,
        gemm_tile_shape,
        true
    );
    failures += run_kernel_only_variant(
        "tiled_gemm_register",
        "sgemm/kernel_only/tiled_gemm_register",
        make_lab_runner,
        prepare_tiled_gemm_register,
        kFp32GemmTolerance,
        gemm_tile_shape,
        true,
        gemm_register_shape
    );
    failures += run_kernel_only_variant(
        "cublas_sgemm",
        "sgemm/kernel_only/cublas_sgemm",
        make_core_runner,
        prepare_core_backend(ai_system::kernels::GemmBackend::CublasSgemm),
        kCublasSgemmTolerance
    );

    ai_system::benchmark::print_benchmark_table(report);
    ai_system::benchmark::print_validation_table(report);

    return failures == 0 ? 0 : 1;
}
