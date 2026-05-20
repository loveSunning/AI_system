#pragma once

#include "gemm_lab.hpp"

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
    GemmLabTileConfig tile_config,
    std::string& error
);

bool is_tiled_gemm_v1_kernel_implemented();

}  // namespace ai_system::labs::gemm::detail
