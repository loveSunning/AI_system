#include "ai_system/config.hpp"
#include "ai_system/kernels/basic_kernels.hpp"
#include "ai_system/profiling/nvtx.hpp"
#include "gemm_lab.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kDefaultM = 128;
constexpr std::size_t kDefaultN = 128;
constexpr std::size_t kDefaultK = 128;

}  // namespace

int main() {
    const ai_system::profiling::ScopedNvtxRange lab_range("gemm_tiled_v1_lab");

    std::vector<float> lhs(kDefaultM * kDefaultK);
    std::vector<float> rhs(kDefaultK * kDefaultN);
    std::vector<float> reference;
    std::vector<float> tiled_out;

    {
        const ai_system::profiling::ScopedNvtxRange phase_range("prepare_inputs");
        ai_system::kernels::fill_random(lhs, -1.0f, 1.0f, 101u);
        ai_system::kernels::fill_random(rhs, -1.0f, 1.0f, 103u);
        ai_system::kernels::naive_gemm_cpu(kDefaultM, kDefaultN, kDefaultK, lhs, rhs, reference);
    }

    std::string error;
    {
        const ai_system::profiling::ScopedNvtxRange phase_range("tiled_gemm_v1_cuda");
        if(!ai_system::labs::gemm::tiled_gemm_v1_cuda(kDefaultM, kDefaultN, kDefaultK, lhs, rhs, tiled_out, error)) {
            const bool unimplemented = error.find("not implemented") != std::string::npos;
            std::cout << "tiled_gemm_v1: " << (!AI_SYSTEM_HAS_CUDA || unimplemented ? "SKIP" : "FAIL") << " - " << error
                      << "\n";
            return !AI_SYSTEM_HAS_CUDA || unimplemented ? 0 : 1;
        }
    }

    const bool matches = [&]() {
        const ai_system::profiling::ScopedNvtxRange phase_range("result_check");
        return ai_system::kernels::allclose(reference, tiled_out, 1.0e-3f, 1.0e-3f);
    }();

    std::cout << "tiled_gemm_v1: " << (matches ? "PASS" : "FAIL") << "\n";
    return matches ? 0 : 1;
}
