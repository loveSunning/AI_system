#include "ai_system/config.hpp"
#include "ai_system/plan/learning_plan.hpp"
#include "ai_system/runtime/gpu_info.hpp"

#include <iostream>
#include <string_view>
#include <vector>

namespace {

void print_usage() {
    std::cout << "AI_system CLI\n"
              << "  --summary     Print project and build summary (default)\n"
              << "  --list-plan   Print the month-by-month learning phases\n"
              << "  --print-gpus  Print detected local GPUs\n"
              << "  --help        Show this help message\n";
}

void print_summary() {
    std::cout << "Project: " << AI_SYSTEM_PROJECT_NAME << " v" << AI_SYSTEM_PROJECT_VERSION << "\n"
              << "CUDA enabled: " << (AI_SYSTEM_HAS_CUDA ? "yes" : "no") << "\n"
              << "GPU profile: " << AI_SYSTEM_GPU_PROFILE << "\n"
              << "Configured architectures: " << AI_SYSTEM_CUDA_ARCHITECTURES << "\n"
              << "Configured labels: " << AI_SYSTEM_CONFIGURED_GPU_LABELS << "\n";
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
    std::vector<std::string_view> args;
    for(int index = 1; index < argc; ++index) {
        args.emplace_back(argv[index]);
    }

    if(args.empty() || args.front() == "--summary") {
        print_summary();
        return 0;
    }

    if(args.front() == "--list-plan") {
        print_plan();
        return 0;
    }

    if(args.front() == "--print-gpus") {
        print_gpus();
        return 0;
    }

    if(args.front() == "--help") {
        print_usage();
        return 0;
    }

    std::cerr << "Unknown argument: " << args.front() << "\n";
    print_usage();
    return 1;
}
