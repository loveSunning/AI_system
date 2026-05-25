#include "ai_system/config.hpp"
#include "ai_system/plan/learning_plan.hpp"
#include "ai_system/runtime/gpu_info.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <string_view>

namespace {

struct CliOptions {
    int gemm_register_tile_m {4};
    int gemm_register_tile_n {4};
};

void print_usage() {
    std::cout << "AI_system CLI\n"
              << "  --summary                 Print project and build summary (default)\n"
              << "  --list-plan               Print the month-by-month learning phases\n"
              << "  --print-gpus              Print detected local GPUs\n"
              << "  --gemm-reg-m M  Register-tiled GEMM per-thread rows; supported: 1, 2, 4\n"
              << "  --gemm-reg-n N  Register-tiled GEMM per-thread columns; supported: 1, 2, 4\n"
              << "  --help                    Show this help message\n";
}

bool is_supported_register_tile_dimension(int value) {
    return value == 1 || value == 2 || value == 4;
}

bool parse_int_argument(const char* raw_value, const char* option_name, int& output) {
    if(raw_value == nullptr || *raw_value == '\0') {
        std::cerr << "Invalid value for " << option_name << ": " << (raw_value == nullptr ? "<null>" : raw_value) << "\n";
        return false;
    }

    char* end = nullptr;
    const long parsed = std::strtol(raw_value, &end, 10);
    if(end == nullptr || *end != '\0') {
        std::cerr << "Invalid value for " << option_name << ": " << raw_value << "\n";
        return false;
    }
    if(parsed < 1L || parsed > static_cast<long>((std::numeric_limits<int>::max)())) {
        std::cerr << "Value for " << option_name << " must be a positive int: " << raw_value << "\n";
        return false;
    }

    const int parsed_value = static_cast<int>(parsed);
    if(!is_supported_register_tile_dimension(parsed_value)) {
        std::cerr << "Value for " << option_name << " must be one of 1, 2, or 4: " << raw_value << "\n";
        return false;
    }

    output = parsed_value;
    return true;
}

bool parse_options(int argc, char** argv, std::string_view& command, CliOptions& options) {
    command = "--summary";

    for(int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        auto require_int_value = [&](int& destination) -> bool {
            if(index + 1 >= argc) {
                std::cerr << "Missing value for " << argument << "\n";
                return false;
            }
            ++index;
            return parse_int_argument(argv[index], argv[index - 1], destination);
        };

        if(argument == "--summary" || argument == "--list-plan" || argument == "--print-gpus" || argument == "--help") {
            command = argument;
            continue;
        }
        if(argument == "--gemm-reg-m") {
            if(!require_int_value(options.gemm_register_tile_m)) {
                return false;
            }
            continue;
        }
        if(argument == "--gemm-reg-n") {
            if(!require_int_value(options.gemm_register_tile_n)) {
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

void print_summary(const CliOptions& options) {
    std::cout << "Project: " << AI_SYSTEM_PROJECT_NAME << " v" << AI_SYSTEM_PROJECT_VERSION << "\n"
              << "CUDA enabled: " << (AI_SYSTEM_HAS_CUDA ? "yes" : "no") << "\n"
              << "GPU profile: " << AI_SYSTEM_GPU_PROFILE << "\n"
              << "Configured architectures: " << AI_SYSTEM_CUDA_ARCHITECTURES << "\n"
              << "Configured labels: " << AI_SYSTEM_CONFIGURED_GPU_LABELS << "\n"
              << "GEMM register tile: " << options.gemm_register_tile_m << "x" << options.gemm_register_tile_n << "\n";
}

void print_plan() {
    const auto& phases = ai_system::plan::learning_plan();
    for(const auto& phase : phases) {
        std::cout << phase.month << " | " << phase.weeks << " | " << phase.topic << "\n"
                  << "  dir: " << phase.directory << "\n"
                  << "  deliverable: " << phase.deliverable << "\n";
    }
}

void print_gpus() {
    const auto gpus = ai_system::runtime::query_gpus();
    std::cout << ai_system::runtime::summarize_gpus(gpus);
}

}  // namespace

int main(int argc, char** argv) {
    CliOptions options;
    std::string_view command;
    if(!parse_options(argc, argv, command, options)) {
        return 1;
    }

    if(command == "--summary") {
        print_summary(options);
        return 0;
    }

    if(command == "--list-plan") {
        print_plan();
        return 0;
    }

    if(command == "--print-gpus") {
        print_gpus();
        return 0;
    }

    if(command == "--help") {
        print_usage();
        return 0;
    }

    std::cerr << "Unknown argument: " << command << "\n";
    print_usage();
    return 1;
}
