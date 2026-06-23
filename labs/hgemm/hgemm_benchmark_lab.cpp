#include "hgemm_lab.hpp"

#include "ai_system/benchmark/benchmark_report.hpp"
#include "ai_system/benchmark/benchmark_runner.hpp"
#include "ai_system/config.hpp"
#include "ai_system/kernels/basic_kernels.hpp"
#include "ai_system/profiling/nvtx.hpp"
#include "ai_system/runtime/gpu_info.hpp"

#include <cstddef>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::size_t m {4096};
    std::size_t n {4096};
    std::size_t k {4096};
    std::size_t warmup_iterations {2};
    std::size_t measured_iterations {5};
    std::string kernel_name {"all"};
    bool skip_correctness {false};
    bool list_kernels {false};
    bool help {false};
    ai_system::labs::hgemm::HgemmLaunchOptions launch_options;
};

void print_usage() {
    std::cout << "AI_system HGEMM benchmark lab\n"
              << "  --gemm-m M             GEMM rows of A/C; default: 4096\n"
              << "  --gemm-n N             GEMM columns of B/C; default: 4096\n"
              << "  --gemm-k K             GEMM shared dimension; default: 4096\n"
              << "  --kernel NAME          Kernel to run, or all; default: all\n"
              << "  --list-kernels         Print compiled HGEMM launcher names\n"
              << "  --stages I             Stage option for staged launchers; default: 2\n"
              << "  --swizzle              Enable staged swizzle option; default\n"
              << "  --no-swizzle           Disable staged swizzle option\n"
              << "  --swizzle-stride I     Staged swizzle stride; default: 2048\n"
              << "  --warmup I             Warmup iterations for each benchmark; default: 2\n"
              << "  --iters I              Measured iterations for each benchmark; default: 5\n"
              << "  --no-correctness       Skip reference output comparison\n"
              << "  --help                 Show this help message\n";
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
        auto require_string_value = [&](std::string& destination) -> bool {
            if(index + 1 >= argc) {
                std::cerr << "Missing value for " << argument << "\n";
                return false;
            }
            ++index;
            destination = argv[index];
            return true;
        };

        if(argument == "--help") {
            options.help = true;
            return true;
        }
        if(argument == "--list-kernels") {
            options.list_kernels = true;
            continue;
        }
        if(argument == "--no-correctness") {
            options.skip_correctness = true;
            continue;
        }
        if(argument == "--swizzle") {
            options.launch_options.swizzle = true;
            continue;
        }
        if(argument == "--no-swizzle") {
            options.launch_options.swizzle = false;
            continue;
        }
        if(argument == "--kernel") {
            if(!require_string_value(options.kernel_name)) {
                return false;
            }
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
        if(argument == "--stages") {
            if(!require_int_value(options.launch_options.stages)) {
                return false;
            }
            continue;
        }
        if(argument == "--swizzle-stride") {
            if(!require_int_value(options.launch_options.swizzle_stride)) {
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

std::string format_shape(std::size_t m, std::size_t n, std::size_t k) {
    return std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k);
}

double compute_gemm_gflops(std::size_t m, std::size_t n, std::size_t k, double average_ms) {
    if(average_ms <= 0.0) {
        return 0.0;
    }
    const double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    return flops / 1.0e9 / (average_ms / 1000.0);
}

std::string format_hgemm_register_shape(const ai_system::labs::hgemm::HgemmKernelInfo& info) {
    std::string label(info.register_shape.empty() ? std::string_view("none") : info.register_shape);
    label += "/f16acc";
    return label;
}

ai_system::benchmark::BenchmarkConfig make_benchmark_config(const Options& options) {
    return ai_system::benchmark::BenchmarkConfig {options.warmup_iterations, options.measured_iterations};
}

void print_kernel_list() {
    const auto& infos = ai_system::labs::hgemm::hgemm_kernel_infos();
    if(infos.empty()) {
        std::cout << "No HGEMM kernels are available in this build.\n";
        return;
    }

    for(const auto& info : infos) {
        std::cout << info.name << "  tile=" << info.tile_shape << "  register=" << format_hgemm_register_shape(info)
                  << "  ncu_regex=" << info.ncu_regex << "\n";
    }
}

std::vector<const ai_system::labs::hgemm::HgemmKernelInfo*> select_kernels(const Options& options) {
    std::vector<const ai_system::labs::hgemm::HgemmKernelInfo*> selected;
    const auto& infos = ai_system::labs::hgemm::hgemm_kernel_infos();
    if(options.kernel_name == "all") {
        selected.reserve(infos.size());
        for(const auto& info : infos) {
            selected.push_back(&info);
        }
        return selected;
    }

    if(const auto* info = ai_system::labs::hgemm::find_hgemm_kernel_info(options.kernel_name)) {
        selected.push_back(info);
    }
    return selected;
}

struct CompareStats {
    bool matches {false};
    std::size_t compared {0};
    std::size_t mismatches {0};
    std::size_t first_mismatch {0};
    std::size_t max_abs_index {0};
    std::size_t max_rel_index {0};
    float first_reference {0.0f};
    float first_output {0.0f};
    float max_abs_error {0.0f};
    float max_rel_error {0.0f};
};

CompareStats compare_outputs(
    const std::vector<float>& reference,
    const std::vector<float>& output,
    float absolute_tolerance,
    float relative_tolerance
) {
    CompareStats stats;
    if(reference.size() != output.size()) {
        stats.mismatches = (std::max)(reference.size(), output.size());
        return stats;
    }

    stats.compared = reference.size();
    stats.matches = true;
    for(std::size_t index = 0; index < reference.size(); ++index) {
        const float difference = std::fabs(reference[index] - output[index]);
        const float denominator = (std::max)(std::fabs(reference[index]), std::fabs(output[index]));
        const float relative_error = denominator > 0.0f ? difference / denominator : 0.0f;
        const float tolerance = absolute_tolerance + relative_tolerance * denominator;

        if(difference > stats.max_abs_error) {
            stats.max_abs_error = difference;
            stats.max_abs_index = index;
        }
        if(relative_error > stats.max_rel_error) {
            stats.max_rel_error = relative_error;
            stats.max_rel_index = index;
        }
        if(difference > tolerance) {
            if(stats.mismatches == 0) {
                stats.first_mismatch = index;
                stats.first_reference = reference[index];
                stats.first_output = output[index];
            }
            ++stats.mismatches;
            stats.matches = false;
        }
    }

    return stats;
}

std::string format_compare_detail(
    const CompareStats& stats,
    const char* reference_name,
    float absolute_tolerance,
    float relative_tolerance
) {
    std::ostringstream stream;
    stream << std::scientific << std::setprecision(3);
    if(stats.matches) {
        stream << "allclose(abs=" << absolute_tolerance << ", rel=" << relative_tolerance
               << "), reference=" << reference_name << ". max_abs=" << stats.max_abs_error
               << "@" << stats.max_abs_index << ", max_rel=" << stats.max_rel_error
               << "@" << stats.max_rel_index << ".";
    } else {
        stream << "mismatches=" << stats.mismatches << "/" << stats.compared
               << ", first=" << stats.first_mismatch << " ref=" << stats.first_reference
               << " out=" << stats.first_output << ", max_abs=" << stats.max_abs_error
               << "@" << stats.max_abs_index << ", max_rel=" << stats.max_rel_error
               << "@" << stats.max_rel_index << ", reference=" << reference_name << ".";
    }
    return stream.str();
}

bool is_ptx_mma_kernel(ai_system::labs::hgemm::HgemmKernel kernel) {
    using ai_system::labs::hgemm::HgemmKernel;
    switch(kernel) {
        case HgemmKernel::MmaM16n8k16Naive:
        case HgemmKernel::MmaM16n8k16Mma2x4Warp4x4:
        case HgemmKernel::MmaM16n8k16Mma2x4Warp4x4Stages:
        case HgemmKernel::MmaM16n8k16Mma2x4Warp4x4StagesDsmem:
        case HgemmKernel::MmaM16n8k16Mma2x4Warp4x4x2StagesDsmem:
        case HgemmKernel::MmaM16n8k16Mma2x4Warp4x4x2StagesDsmemX4:
        case HgemmKernel::MmaM16n8k16Mma2x4Warp4x4x2StagesDsmemRr:
        case HgemmKernel::MmaM16n8k16Mma2x4Warp4x4StagesDsmemTn:
        case HgemmKernel::MmaStagesBlockSwizzleTnCute:
        case HgemmKernel::MmaM16n8k16Mma2x4Warp4x4x2StagesDsmemSwizzle:
        case HgemmKernel::MmaM16n8k16Mma2x4Warp4x4StagesDsmemTnSwizzle:
        case HgemmKernel::MmaM16n8k16Mma2x4Warp4x4x2StagesDsmemTnSwizzleX2:
        case HgemmKernel::MmaM16n8k16Mma2x4Warp4x4x2StagesDsmemTnSwizzleX4:
            return true;
        default:
            return false;
    }
}

struct ComparisonTolerance {
    float absolute {2.5e-1f};
    float relative {5.0e-2f};
};

ComparisonTolerance comparison_tolerance(ai_system::labs::hgemm::HgemmKernel kernel) {
    if(is_ptx_mma_kernel(kernel)) {
        // PTX MMA uses tensor-core reduction order.  For large K, a few
        // near-zero outputs can differ from both scalar half-FMA and cuBLAS
        // enough that the generic 0.25 absolute tolerance is too tight.
        return ComparisonTolerance {5.0e-1f, 5.0e-2f};
    }
    return {};
}

bool uses_cublas_reference(ai_system::labs::hgemm::HgemmKernel kernel) {
    using ai_system::labs::hgemm::HgemmKernel;
    if(is_ptx_mma_kernel(kernel)) {
        return true;
    }
    switch(kernel) {
        case HgemmKernel::CublasTensorOpNn:
        case HgemmKernel::CublasTensorOpTn:
        case HgemmKernel::WmmaM16n16k16Naive:
        case HgemmKernel::WmmaM16n16k16Mma4x2:
        case HgemmKernel::WmmaM16n16k16Mma4x2Warp2x4:
        case HgemmKernel::WmmaM16n16k16Mma4x2Warp2x4DbufAsync:
        case HgemmKernel::WmmaM32n8k16Mma2x4Warp2x4DbufAsync:
        case HgemmKernel::WmmaM16n16k16Mma4x2Warp2x4Stages:
        case HgemmKernel::WmmaM16n16k16Mma4x2Warp2x4StagesDsmem:
        case HgemmKernel::WmmaM16n16k16Mma4x2Warp4x4StagesDsmem:
        case HgemmKernel::WmmaM16n16k16Mma4x4Warp4x4StagesDsmem:
            return true;
        default:
            return false;
    }
}

bool uses_cublas_tn_reference(ai_system::labs::hgemm::HgemmKernel kernel) {
    using ai_system::labs::hgemm::HgemmKernel;
    switch(kernel) {
        case HgemmKernel::CublasTensorOpTn:
        case HgemmKernel::MmaM16n8k16Mma2x4Warp4x4StagesDsmemTn:
        case HgemmKernel::MmaStagesBlockSwizzleTnCute:
        case HgemmKernel::MmaM16n8k16Mma2x4Warp4x4StagesDsmemTnSwizzle:
        case HgemmKernel::MmaM16n8k16Mma2x4Warp4x4x2StagesDsmemTnSwizzleX2:
        case HgemmKernel::MmaM16n8k16Mma2x4Warp4x4x2StagesDsmemTnSwizzleX4:
            return true;
        default:
            return false;
    }
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if(!parse_options(argc, argv, options)) {
        return 1;
    }
    if(options.help) {
        print_usage();
        return 0;
    }
    if(options.list_kernels) {
        print_kernel_list();
        return 0;
    }

    const auto selected = select_kernels(options);
    if(selected.empty()) {
        std::cerr << "No HGEMM kernel matched --kernel " << options.kernel_name << "\n";
        print_usage();
        return 1;
    }

    const ai_system::profiling::ScopedNvtxRange lab_range("hgemm_benchmark_lab");
    const auto config = make_benchmark_config(options);
    const std::string shape = format_shape(options.m, options.n, options.k);

    std::cout << "AI_system HGEMM benchmark lab\n";
    std::cout << "Configured architectures: " << AI_SYSTEM_CUDA_ARCHITECTURES << "\n";
    std::cout << "GEMM shape: " << shape << "\n";
    std::cout << "Kernel selection: " << options.kernel_name << "\n";
    std::cout << "Staged options: stages=" << options.launch_options.stages
              << ", swizzle=" << (options.launch_options.swizzle ? "true" : "false")
              << ", swizzle_stride=" << options.launch_options.swizzle_stride << "\n";
    std::cout << "Correctness: " << (options.skip_correctness ? "disabled" : "enabled") << "\n";
    std::cout << "Warmup iterations: " << options.warmup_iterations << "\n";
    std::cout << "Measured iterations: " << options.measured_iterations << "\n";
    std::cout << ai_system::runtime::summarize_gpus(ai_system::runtime::query_gpus());

    std::vector<float> lhs(options.m * options.k);
    std::vector<float> rhs(options.k * options.n);
    {
        const ai_system::profiling::ScopedNvtxRange phase_range("hgemm_prepare_inputs");
        ai_system::kernels::fill_random(lhs, -1.0f, 1.0f, 97u);
        ai_system::kernels::fill_random(rhs, -1.0f, 1.0f, 131u);
    }

    ai_system::benchmark::BenchmarkReport report;
    std::vector<float> half_reference_out;
    std::vector<float> cublas_nn_reference_out;
    std::vector<float> cublas_tn_reference_out;
    bool has_half_reference = false;
    bool has_cublas_nn_reference = false;
    bool has_cublas_tn_reference = false;
    int failures = 0;

    if(!options.skip_correctness) {
        bool needs_half_reference = false;
        bool needs_cublas_nn_reference = false;
        bool needs_cublas_tn_reference = false;
        for(const auto* info : selected) {
            if(uses_cublas_tn_reference(info->kernel)) {
                needs_cublas_tn_reference = true;
            } else if(uses_cublas_reference(info->kernel)) {
                needs_cublas_nn_reference = true;
            } else {
                needs_half_reference = true;
            }
        }

        auto prepare_reference = [&](ai_system::labs::hgemm::HgemmKernel kernel,
                                     std::string_view name,
                                     std::string_view success_detail,
                                     std::vector<float>& output,
                                     bool& has_reference) {
            ai_system::labs::hgemm::PreparedHgemmLabRunner reference_runner;
            std::string error;
            if(reference_runner.prepare(
                   kernel,
                   options.m,
                   options.n,
                   options.k,
                   lhs,
                   rhs,
                   error,
                   options.launch_options
               ) &&
               reference_runner.run(error) && reference_runner.copy_output(output, error)) {
                has_reference = true;
                ai_system::benchmark::add_validation_row(
                    report,
                    "hgemm_kernel_only",
                    std::string(name),
                    "reference",
                    "PASS",
                    std::string(success_detail)
                );
            } else {
                ai_system::benchmark::add_validation_row(
                    report,
                    "hgemm_kernel_only",
                    std::string(name),
                    "reference",
                    AI_SYSTEM_HAS_CUDA ? "FAIL" : "SKIP",
                    error
                );
                failures += AI_SYSTEM_HAS_CUDA ? 1 : 0;
            }
        };

        if(needs_half_reference) {
            prepare_reference(
                ai_system::labs::hgemm::HgemmKernel::NaiveF16,
                "hgemm_naive_f16",
                "half-accumulate HGEMM reference generated.",
                half_reference_out,
                has_half_reference
            );
        }
        if(needs_cublas_nn_reference) {
            prepare_reference(
                ai_system::labs::hgemm::HgemmKernel::CublasTensorOpNn,
                "hgemm_cublas_tensor_op_nn",
                "cuBLAS Tensor Core half-accumulate HGEMM reference generated.",
                cublas_nn_reference_out,
                has_cublas_nn_reference
            );
        }
        if(needs_cublas_tn_reference) {
            prepare_reference(
                ai_system::labs::hgemm::HgemmKernel::CublasTensorOpTn,
                "hgemm_cublas_tensor_op_tn",
                "cuBLAS Tensor Core half-accumulate HGEMM TN reference generated.",
                cublas_tn_reference_out,
                has_cublas_tn_reference
            );
        }
    }

    for(const auto* info : selected) {
        ai_system::labs::hgemm::PreparedHgemmLabRunner runner;
        std::string error;
        if(!runner.prepare(info->kernel, options.m, options.n, options.k, lhs, rhs, error, options.launch_options)) {
            ai_system::benchmark::add_validation_row(
                report,
                "hgemm_kernel_only",
                std::string(info->name),
                "prepare",
                AI_SYSTEM_HAS_CUDA ? "FAIL" : "SKIP",
                error
            );
            failures += AI_SYSTEM_HAS_CUDA ? 1 : 0;
            continue;
        }

        const bool use_cublas_tn_reference = uses_cublas_tn_reference(info->kernel);
        const bool use_cublas_reference = uses_cublas_reference(info->kernel);
        const bool has_reference = use_cublas_tn_reference ? has_cublas_tn_reference :
            (use_cublas_reference ? has_cublas_nn_reference : has_half_reference);
        const auto& reference_out = use_cublas_tn_reference ? cublas_tn_reference_out :
            (use_cublas_reference ? cublas_nn_reference_out : half_reference_out);
        const char* reference_name = use_cublas_tn_reference ? "hgemm_cublas_tensor_op_tn" :
            (use_cublas_reference ? "hgemm_cublas_tensor_op_nn" : "hgemm_naive_f16");

        if(has_reference) {
            std::vector<float> gpu_out;
            if(runner.run(error) && runner.copy_output(gpu_out, error)) {
                const ComparisonTolerance tolerance = comparison_tolerance(info->kernel);
                const CompareStats comparison =
                    compare_outputs(reference_out, gpu_out, tolerance.absolute, tolerance.relative);
                ai_system::benchmark::add_validation_row(
                    report,
                    "hgemm_kernel_only",
                    std::string(info->name),
                    "correctness",
                    comparison.matches ? "PASS" : "FAIL",
                    format_compare_detail(comparison, reference_name, tolerance.absolute, tolerance.relative)
                );
                failures += comparison.matches ? 0 : 1;
            } else {
                ai_system::benchmark::add_validation_row(
                    report,
                    "hgemm_kernel_only",
                    std::string(info->name),
                    "runtime",
                    AI_SYSTEM_HAS_CUDA ? "FAIL" : "SKIP",
                    error
                );
                failures += AI_SYSTEM_HAS_CUDA ? 1 : 0;
                continue;
            }
        } else {
            ai_system::benchmark::add_validation_row(
                report,
                "hgemm_kernel_only",
                std::string(info->name),
                "correctness",
                "SKIP",
                options.skip_correctness ? "Disabled by --no-correctness." : "No correctness reference is available."
            );
        }

        const auto result = [&]() {
            const ai_system::profiling::ScopedNvtxRange phase_range("hgemm_kernel_only_benchmark");
            return ai_system::benchmark::run_timed_benchmark(
                "hgemm/kernel_only/" + std::string(info->name),
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
            "hgemm_kernel_only",
            std::string(info->name),
            shape,
            result,
            compute_gemm_gflops(options.m, options.n, options.k, result.average_ms),
            "GFLOPS",
            std::string(info->tile_shape),
            format_hgemm_register_shape(*info)
        );
    }

    ai_system::benchmark::print_benchmark_table(report);
    ai_system::benchmark::print_validation_table(report);

    return failures == 0 ? 0 : 1;
}
