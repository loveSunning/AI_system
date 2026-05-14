#pragma once

#include <cstddef>
#include <string>

namespace ai_system::labs::gemm::detail {

bool launch_tiled_gemm_v1(
    const float* lhs,
    const float* rhs,
    float* out,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    std::string& error
);

bool tiled_gemm_v1_kernel_available();

}  // namespace ai_system::labs::gemm::detail
