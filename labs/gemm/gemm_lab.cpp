#include "gemm_lab.hpp"

#include "ai_system/config.hpp"

#if !AI_SYSTEM_HAS_CUDA

namespace ai_system::labs::gemm {

struct PreparedGemmLabRunner::Impl {};

PreparedGemmLabRunner::PreparedGemmLabRunner() = default;
PreparedGemmLabRunner::~PreparedGemmLabRunner() = default;
PreparedGemmLabRunner::PreparedGemmLabRunner(PreparedGemmLabRunner&& other) noexcept = default;
PreparedGemmLabRunner& PreparedGemmLabRunner::operator=(PreparedGemmLabRunner&& other) noexcept = default;

bool PreparedGemmLabRunner::prepare(
    GemmLabBackend,
    std::size_t,
    std::size_t,
    std::size_t,
    const std::vector<float>&,
    const std::vector<float>&,
    std::string& error
) {
    error = "CUDA support is disabled in this build.";
    return false;
}

bool PreparedGemmLabRunner::run(std::string& error) {
    error = "CUDA support is disabled in this build.";
    return false;
}

bool PreparedGemmLabRunner::run_timed(double& elapsed_ms, std::string& error) {
    elapsed_ms = 0.0;
    error = "CUDA support is disabled in this build.";
    return false;
}

bool PreparedGemmLabRunner::copy_output(std::vector<float>&, std::string& error) const {
    error = "CUDA support is disabled in this build.";
    return false;
}

bool gemm_lab_backend_available(GemmLabBackend) {
    return false;
}

bool tiled_gemm_v1_cuda(
    std::size_t,
    std::size_t,
    std::size_t,
    const std::vector<float>&,
    const std::vector<float>&,
    std::vector<float>&,
    std::string& error
) {
    error = "CUDA support is disabled in this build.";
    return false;
}

bool tiled_gemm_v1_kernel_available() {
    return gemm_lab_backend_available(GemmLabBackend::TiledGemmV1);
}

}  // namespace ai_system::labs::gemm

#endif
