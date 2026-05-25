#include "gemm_lab.hpp"

#include "gemm_lab_kernels.hpp"

#include "ai_system/cuda/runtime.hpp"
#include "ai_system/profiling/nvtx.hpp"

#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace ai_system::labs::gemm {

namespace {

bool validate_gemm_inputs(
    std::size_t m,
    std::size_t n,
    std::size_t k,
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    std::string& error
) {
    if(m == 0 || n == 0 || k == 0) {
        error = "GEMM lab requires non-zero M, N, and K.";
        return false;
    }
    if(lhs.size() != m * k) {
        error = "GEMM lab requires lhs.size() == m * k.";
        return false;
    }
    if(rhs.size() != k * n) {
        error = "GEMM lab requires rhs.size() == k * n.";
        return false;
    }
    if(m > static_cast<std::size_t>((std::numeric_limits<unsigned int>::max)()) ||
       n > static_cast<std::size_t>((std::numeric_limits<unsigned int>::max)())) {
        error = "GEMM lab grid dimensions exceed unsigned int limits.";
        return false;
    }
    return true;
}

bool is_supported_gemm_output_tile_dimension(int value) {
    return value == 8 || value == 16 || value == 32 || value == 64 || value == 128;
}

bool is_supported_gemm_reduction_tile_dimension(int value) {
    return value == 8 || value == 16 || value == 32;
}

bool is_supported_gemm_register_tile_shape(int register_m, int register_n) {
    return (register_m == 2 && register_n == 2) || (register_m == 4 && register_n == 4) ||
        (register_m == 4 && register_n == 8) || (register_m == 8 && register_n == 4) ||
        (register_m == 8 && register_n == 8);
}

bool validate_gemm_output_tile_dimensions(GemmLabTileConfig tile_config, const char* backend_label, std::string& error) {
    if(!is_supported_gemm_output_tile_dimension(tile_config.block_m) ||
       !is_supported_gemm_output_tile_dimension(tile_config.block_n)) {
        error = std::string(backend_label) + " output tile dimensions must each be one of 8, 16, 32, 64, or 128.";
        return false;
    }

    if(!is_supported_gemm_reduction_tile_dimension(tile_config.block_k)) {
        error = std::string(backend_label) + " reduction tile dimension must be one of 8, 16, or 32.";
        return false;
    }

    return true;
}

bool validate_tiled_gemm_block_tile_config(GemmLabTileConfig tile_config, std::string& error) {
    if(!validate_gemm_output_tile_dimensions(tile_config, "tiled_gemm_block", error)) {
        return false;
    }

    if(tile_config.block_m * tile_config.block_n > 1024) {
        error = "tiled_gemm_block requires block_m * block_n <= 1024.";
        return false;
    }

    return true;
}

bool validate_tiled_gemm_register_tile_config(GemmLabTileConfig tile_config, std::string& error) {
    if(!validate_gemm_output_tile_dimensions(tile_config, "tiled_gemm_register", error)) {
        return false;
    }

    if(!is_supported_gemm_register_tile_shape(tile_config.register_m, tile_config.register_n)) {
        error = "tiled_gemm_register register tile must be one of 2x2, 4x4, 4x8, 8x4, or 8x8.";
        return false;
    }

    if(tile_config.block_m % tile_config.register_m != 0 || tile_config.block_n % tile_config.register_n != 0) {
        error = "tiled_gemm_register requires block_m/block_n to be divisible by register_m/register_n.";
        return false;
    }

    const int threads_per_block = (tile_config.block_m / tile_config.register_m) *
        (tile_config.block_n / tile_config.register_n);
    if(threads_per_block <= 0 || threads_per_block > 1024) {
        error = "tiled_gemm_register derived thread block size must be between 1 and 1024.";
        return false;
    }

    return true;
}

bool validate_backend_config(GemmLabBackend backend, GemmLabTileConfig tile_config, std::string& error) {
    switch(backend) {
        case GemmLabBackend::TiledGemmBlock:
            return validate_tiled_gemm_block_tile_config(tile_config, error);
        case GemmLabBackend::TiledGemmRegister:
            return validate_tiled_gemm_register_tile_config(tile_config, error);
        case GemmLabBackend::TiledGemmV2:
        case GemmLabBackend::ManualTensorCoreV1:
            return true;
    }

    error = "Unsupported GEMM lab backend.";
    return false;
}

const char* backend_name(GemmLabBackend backend) {
    switch(backend) {
        case GemmLabBackend::TiledGemmBlock:
            return "tiled_gemm_block";
        case GemmLabBackend::TiledGemmRegister:
            return "tiled_gemm_register";
        case GemmLabBackend::TiledGemmV2:
            return "tiled_gemm_v2";
        case GemmLabBackend::ManualTensorCoreV1:
            return "manual_tensor_core_v1";
    }

    return "unknown_gemm_lab_backend";
}

}  // namespace

struct PreparedGemmLabRunner::Impl {
    GemmLabBackend backend {GemmLabBackend::TiledGemmBlock};
    std::size_t m {0};
    std::size_t n {0};
    std::size_t k {0};
    GemmLabTileConfig tile_config;
    bool prepared {false};
    ai_system::cuda_utils::DeviceBuffer<float> lhs_device;
    ai_system::cuda_utils::DeviceBuffer<float> rhs_device;
    ai_system::cuda_utils::DeviceBuffer<float> out_device;
    ai_system::cuda_utils::EventPair events;

    bool prepare(
        GemmLabBackend requested_backend,
        std::size_t requested_m,
        std::size_t requested_n,
        std::size_t requested_k,
        const std::vector<float>& lhs,
        const std::vector<float>& rhs,
        std::string& error,
        GemmLabTileConfig requested_tile_config
    ) {
        const ai_system::profiling::ScopedNvtxRange phase_range("gemm_lab_prepare_h2d");

        if(!validate_gemm_inputs(requested_m, requested_n, requested_k, lhs, rhs, error)) {
            return false;
        }
        if(!validate_backend_config(requested_backend, requested_tile_config, error)) {
            return false;
        }
        if(!events.ensure(error)) {
            return false;
        }

        backend = requested_backend;
        m = requested_m;
        n = requested_n;
        k = requested_k;
        tile_config = requested_tile_config;
        prepared = false;

        lhs_device.reset();
        rhs_device.reset();
        out_device.reset();

        if(!lhs_device.allocate(lhs.size(), error) ||
           !rhs_device.allocate(rhs.size(), error) ||
           !out_device.allocate(m * n, error)) {
            return false;
        }

        if(!ai_system::cuda_utils::copy_to_device(lhs_device.get(), lhs, error) ||
           !ai_system::cuda_utils::copy_to_device(rhs_device.get(), rhs, error)) {
            return false;
        }

        prepared = true;
        return true;
    }

    bool run(std::string& error) {
        if(!launch(error)) {
            return false;
        }
        return ai_system::cuda_utils::synchronize(error);
    }

    bool run_timed(double& elapsed_ms, std::string& error) {
        elapsed_ms = 0.0;
        if(!prepared) {
            error = "PreparedGemmLabRunner::prepare must succeed before run_timed.";
            return false;
        }

        if(!events.record_start(error)) {
            return false;
        }
        if(!launch(error)) {
            return false;
        }
        if(!events.record_stop(error)) {
            return false;
        }
        if(!events.synchronize_stop(error)) {
            return false;
        }

        float event_ms = 0.0f;
        if(!events.elapsed_ms(event_ms, error)) {
            return false;
        }

        elapsed_ms = static_cast<double>(event_ms);
        return true;
    }

    bool copy_output(std::vector<float>& out, std::string& error) const {
        if(!prepared) {
            error = "PreparedGemmLabRunner::prepare must succeed before copy_output.";
            return false;
        }

        out.assign(m * n, 0.0f);
        return ai_system::cuda_utils::copy_to_host(out, out_device.get(), error);
    }

private:
    bool launch(std::string& error) {
        if(!prepared) {
            error = "PreparedGemmLabRunner::prepare must succeed before run.";
            return false;
        }

        switch(backend) {
            case GemmLabBackend::TiledGemmBlock:
                return detail::launch_tiled_gemm_block(
                    lhs_device.get(),
                    rhs_device.get(),
                    out_device.get(),
                    m,
                    n,
                    k,
                    tile_config,
                    error
                );
            case GemmLabBackend::TiledGemmRegister:
                return detail::launch_tiled_gemm_register(
                    lhs_device.get(),
                    rhs_device.get(),
                    out_device.get(),
                    m,
                    n,
                    k,
                    tile_config,
                    error
                );
            case GemmLabBackend::TiledGemmV2:
            case GemmLabBackend::ManualTensorCoreV1:
                error = std::string(backend_name(backend)) + " launcher is not wired yet.";
                return false;
        }

        error = "Unsupported GEMM lab backend.";
        return false;
    }
};

PreparedGemmLabRunner::PreparedGemmLabRunner() : impl_(std::make_unique<Impl>()) {}
PreparedGemmLabRunner::~PreparedGemmLabRunner() = default;
PreparedGemmLabRunner::PreparedGemmLabRunner(PreparedGemmLabRunner&& other) noexcept = default;
PreparedGemmLabRunner& PreparedGemmLabRunner::operator=(PreparedGemmLabRunner&& other) noexcept = default;

bool PreparedGemmLabRunner::prepare(
    GemmLabBackend backend,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    std::string& error,
    GemmLabTileConfig tile_config
) {
    return impl_->prepare(backend, m, n, k, lhs, rhs, error, tile_config);
}

bool PreparedGemmLabRunner::run(std::string& error) {
    return impl_->run(error);
}

bool PreparedGemmLabRunner::run_timed(double& elapsed_ms, std::string& error) {
    return impl_->run_timed(elapsed_ms, error);
}

bool PreparedGemmLabRunner::copy_output(std::vector<float>& out, std::string& error) const {
    return impl_->copy_output(out, error);
}

bool gemm_lab_backend_available(GemmLabBackend backend) {
    switch(backend) {
        case GemmLabBackend::TiledGemmBlock:
            return detail::is_tiled_gemm_block_kernel_implemented();
        case GemmLabBackend::TiledGemmRegister:
            return detail::is_tiled_gemm_register_kernel_implemented();
        case GemmLabBackend::TiledGemmV2:
        case GemmLabBackend::ManualTensorCoreV1:
            return false;
    }

    return false;
}

bool tiled_gemm_block_cuda(
    std::size_t m,
    std::size_t n,
    std::size_t k,
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    std::vector<float>& out,
    std::string& error,
    GemmLabTileConfig tile_config
) {
    const ai_system::profiling::ScopedNvtxRange e2e_range("tiled_gemm_block_e2e");

    PreparedGemmLabRunner runner;
    {
        const ai_system::profiling::ScopedNvtxRange phase_range("tiled_gemm_block_prepare");
        if(!runner.prepare(GemmLabBackend::TiledGemmBlock, m, n, k, lhs, rhs, error, tile_config)) {
            return false;
        }
    }
    {
        const ai_system::profiling::ScopedNvtxRange phase_range("tiled_gemm_block_run");
        if(!runner.run(error)) {
            return false;
        }
    }
    {
        const ai_system::profiling::ScopedNvtxRange phase_range("tiled_gemm_block_d2h");
        return runner.copy_output(out, error);
    }
}

bool tiled_gemm_register_cuda(
    std::size_t m,
    std::size_t n,
    std::size_t k,
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    std::vector<float>& out,
    std::string& error,
    GemmLabTileConfig tile_config
) {
    const ai_system::profiling::ScopedNvtxRange e2e_range("tiled_gemm_register_e2e");

    PreparedGemmLabRunner runner;
    {
        const ai_system::profiling::ScopedNvtxRange phase_range("tiled_gemm_register_prepare");
        if(!runner.prepare(GemmLabBackend::TiledGemmRegister, m, n, k, lhs, rhs, error, tile_config)) {
            return false;
        }
    }
    {
        const ai_system::profiling::ScopedNvtxRange phase_range("tiled_gemm_register_run");
        if(!runner.run(error)) {
            return false;
        }
    }
    {
        const ai_system::profiling::ScopedNvtxRange phase_range("tiled_gemm_register_d2h");
        return runner.copy_output(out, error);
    }
}

}  // namespace ai_system::labs::gemm
