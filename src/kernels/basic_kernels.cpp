#include "ai_system/config.hpp"
#include "ai_system/kernels/basic_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace ai_system::kernels {

void fill_random(
    std::vector<float>& values,
    float min_value,
    float max_value,
    std::uint32_t seed
) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(min_value, max_value);

    for(float& value : values) {
        value = distribution(generator);
    }
}

void vector_add_cpu(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    std::vector<float>& out
) {
    if(lhs.size() != rhs.size()) {
        throw std::invalid_argument("vector_add_cpu requires lhs.size() == rhs.size().");
    }

    out.resize(lhs.size());
    for(std::size_t index = 0; index < lhs.size(); ++index) {
        out[index] = lhs[index] + rhs[index];
    }
}

float reduction_sum_cpu(const std::vector<float>& values) {
    float sum = 0.0f;
    for(const float value : values) {
        sum += value;
    }
    return sum;
}

void naive_gemm_cpu(
    std::size_t m,
    std::size_t n,
    std::size_t k,
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    std::vector<float>& out
) {
    if(lhs.size() != m * k) {
        throw std::invalid_argument("naive_gemm_cpu requires lhs.size() == m * k.");
    }
    if(rhs.size() != k * n) {
        throw std::invalid_argument("naive_gemm_cpu requires rhs.size() == k * n.");
    }

    out.assign(m * n, 0.0f);

    for(std::size_t row = 0; row < m; ++row) {
        for(std::size_t column = 0; column < n; ++column) {
            float accumulator = 0.0f;
            for(std::size_t inner = 0; inner < k; ++inner) {
                accumulator += lhs[row * k + inner] * rhs[inner * n + column];
            }
            out[row * n + column] = accumulator;
        }
    }
}

bool allclose(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    float absolute_tolerance,
    float relative_tolerance
) {
    if(lhs.size() != rhs.size()) {
        return false;
    }

    for(std::size_t index = 0; index < lhs.size(); ++index) {
        const float difference = std::fabs(lhs[index] - rhs[index]);
        const float tolerance =
            absolute_tolerance + relative_tolerance * std::max(std::fabs(lhs[index]), std::fabs(rhs[index]));
        if(difference > tolerance) {
            return false;
        }
    }

    return true;
}

#if !AI_SYSTEM_HAS_CUDA

bool vector_add_cuda(
    const std::vector<float>&,
    const std::vector<float>&,
    std::vector<float>&,
    std::string& error
) {
    error = "CUDA support is disabled in this build.";
    return false;
}

bool reduction_sum_cuda(
    const std::vector<float>&,
    float&,
    std::string& error
) {
    error = "CUDA support is disabled in this build.";
    return false;
}

bool naive_gemm_cuda(
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

#endif

}  // namespace ai_system::kernels
