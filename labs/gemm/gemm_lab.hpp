#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace ai_system::labs::gemm {

enum class GemmLabBackend {
    TiledGemmV1
};

class PreparedGemmLabRunner {
public:
    PreparedGemmLabRunner();
    ~PreparedGemmLabRunner();

    PreparedGemmLabRunner(const PreparedGemmLabRunner&) = delete;
    PreparedGemmLabRunner& operator=(const PreparedGemmLabRunner&) = delete;
    PreparedGemmLabRunner(PreparedGemmLabRunner&& other) noexcept;
    PreparedGemmLabRunner& operator=(PreparedGemmLabRunner&& other) noexcept;

    bool prepare(
        GemmLabBackend backend,
        std::size_t m,
        std::size_t n,
        std::size_t k,
        const std::vector<float>& lhs,
        const std::vector<float>& rhs,
        std::string& error
    );

    bool run(std::string& error);
    bool run_timed(double& elapsed_ms, std::string& error);
    bool copy_output(std::vector<float>& out, std::string& error) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

bool gemm_lab_backend_available(GemmLabBackend backend);

bool tiled_gemm_v1_cuda(
    std::size_t m,
    std::size_t n,
    std::size_t k,
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    std::vector<float>& out,
    std::string& error
);

bool tiled_gemm_v1_kernel_available();

}  // namespace ai_system::labs::gemm
