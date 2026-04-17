#include "ai_system/config.hpp"
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

#if AI_SYSTEM_HAS_CUDA
    {
        const std::vector<float> lhs {1.0f, 2.0f, 3.0f, 4.0f};
        const std::vector<float> rhs {5.0f, 6.0f, 7.0f, 8.0f};
        const std::vector<float> reference {19.0f, 22.0f, 43.0f, 50.0f};
        std::vector<float> out;
        std::string error;
        ai_system::kernels::PreparedGemmKernelRunner runner;

        success &= expect(
            runner.prepare(ai_system::kernels::GemmBackend::CudaNaive, 2, 2, 2, lhs, rhs, error),
            "PreparedGemmKernelRunner should prepare the CUDA naive backend."
        );
        success &= expect(runner.run(error), "PreparedGemmKernelRunner should launch the CUDA naive backend.");
        success &= expect(runner.copy_output(out, error), "PreparedGemmKernelRunner should copy results back to the host.");
        success &= expect(
            ai_system::kernels::allclose(reference, out, 1.0e-4f, 1.0e-4f),
            "PreparedGemmKernelRunner should match the CPU reference."
        );
    }

    {
        const std::vector<float> lhs {1.0f, 2.0f, 3.0f, 4.0f};
        const std::vector<float> rhs {5.0f, 6.0f, 7.0f, 8.0f};
        const std::vector<float> reference {19.0f, 22.0f, 43.0f, 50.0f};
        std::vector<float> out;
        std::string error;

        success &= expect(
            ai_system::kernels::cublas_sgemm_cuda(2, 2, 2, lhs, rhs, out, error),
            "cublas_sgemm_cuda should run successfully."
        );
        success &= expect(
            ai_system::kernels::allclose(reference, out, 1.0e-4f, 1.0e-4f),
            "cublas_sgemm_cuda should match the CPU reference."
        );
    }

    {
        const std::vector<float> lhs {1.0f, 2.0f, 3.0f, 4.0f};
        const std::vector<float> rhs {5.0f, 6.0f, 7.0f, 8.0f};
        const std::vector<float> reference {19.0f, 22.0f, 43.0f, 50.0f};
        std::vector<float> out;
        std::string error;

        success &= expect(
            ai_system::kernels::cublas_hgemm_cuda(2, 2, 2, lhs, rhs, out, error),
            "cublas_hgemm_cuda should run successfully."
        );
        success &= expect(
            ai_system::kernels::allclose(reference, out, 1.0e-2f, 1.0e-2f),
            "cublas_hgemm_cuda should stay close to the CPU reference."
        );
    }

    {
        const std::vector<float> lhs {1.0f, 2.0f, 3.0f, 4.0f};
        const std::vector<float> rhs {5.0f, 6.0f, 7.0f, 8.0f};
        const std::vector<float> reference {19.0f, 22.0f, 43.0f, 50.0f};
        std::vector<float> out;
        std::string error;

        success &= expect(
            ai_system::kernels::cublas_tensor_core_gemm_cuda(2, 2, 2, lhs, rhs, out, error),
            "cublas_tensor_core_gemm_cuda should run successfully."
        );
        success &= expect(
            ai_system::kernels::allclose(reference, out, 1.0e-2f, 1.0e-2f),
            "cublas_tensor_core_gemm_cuda should stay close to the CPU reference."
        );
    }
#endif

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
