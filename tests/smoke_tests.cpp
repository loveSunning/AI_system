#include "ai_system/kernels/basic_kernels.hpp"
#include "ai_system/plan/learning_plan.hpp"
#include "ai_system/runtime/gpu_info.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool expect(bool condition, const std::string& message) {
    if(!condition) {
        std::cerr << "Expectation failed: " << message << "\n";
        return false;
    }
    return true;
}

}  // namespace

int main() {
    bool success = true;

    {
        const std::vector<float> lhs {1.0f, 2.0f, 3.0f};
        const std::vector<float> rhs {4.0f, 5.0f, 6.0f};
        std::vector<float> out;
        ai_system::kernels::vector_add_cpu(lhs, rhs, out);
        success &= expect(out == std::vector<float>({5.0f, 7.0f, 9.0f}), "vector_add_cpu should add element-wise.");
    }

    {
        const std::vector<float> values {1.0f, -2.0f, 3.5f};
        const float sum = ai_system::kernels::reduction_sum_cpu(values);
        success &= expect(std::fabs(sum - 2.5f) < 1.0e-6f, "reduction_sum_cpu should accumulate values.");
    }

    {
        const std::vector<float> lhs {1.0f, 2.0f, 3.0f, 4.0f};
        const std::vector<float> rhs {5.0f, 6.0f, 7.0f, 8.0f};
        std::vector<float> out;
        ai_system::kernels::naive_gemm_cpu(2, 2, 2, lhs, rhs, out);
        success &= expect(
            out == std::vector<float>({19.0f, 22.0f, 43.0f, 50.0f}),
            "naive_gemm_cpu should produce a correct 2x2 matrix multiply."
        );
    }

    {
        const auto& phases = ai_system::plan::learning_plan();
        success &= expect(phases.size() == 12, "learning_plan should expose 12 monthly phases.");
    }

    {
        const auto gpus = ai_system::runtime::query_gpus();
        const auto summary = ai_system::runtime::summarize_gpus(gpus);
        success &= expect(!summary.empty(), "GPU summary should always return printable text.");
    }

    return success ? 0 : 1;
}
