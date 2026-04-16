#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ai_system::kernels {

void fill_random(
    std::vector<float>& values,
    float min_value = -1.0f,
    float max_value = 1.0f,
    std::uint32_t seed = 7u
);

void vector_add_cpu(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    std::vector<float>& out
);

bool vector_add_cuda(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    std::vector<float>& out,
    std::string& error
);

float reduction_sum_cpu(const std::vector<float>& values);

bool reduction_sum_cuda(
    const std::vector<float>& values,
    float& result,
    std::string& error
);

void naive_gemm_cpu(
    std::size_t m,
    std::size_t n,
    std::size_t k,
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    std::vector<float>& out
);

bool naive_gemm_cuda(
    std::size_t m,
    std::size_t n,
    std::size_t k,
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    std::vector<float>& out,
    std::string& error
);

bool cublas_sgemm_cuda(
    std::size_t m,
    std::size_t n,
    std::size_t k,
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    std::vector<float>& out,
    std::string& error
);

bool cublas_hgemm_cuda(
    std::size_t m,
    std::size_t n,
    std::size_t k,
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    std::vector<float>& out,
    std::string& error
);

bool cublas_tensor_core_gemm_cuda(
    std::size_t m,
    std::size_t n,
    std::size_t k,
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    std::vector<float>& out,
    std::string& error
);

bool allclose(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    float absolute_tolerance = 1.0e-4f,
    float relative_tolerance = 1.0e-4f
);

}  // namespace ai_system::kernels
