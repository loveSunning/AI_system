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

}  // namespace

struct PreparedGemmLabRunner::Impl {
    GemmLabBackend backend {GemmLabBackend::TiledGemmV1};
    std::size_t m {0};
    std::size_t n {0};
    std::size_t k {0};
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
        std::string& error
    ) {
        const ai_system::profiling::ScopedNvtxRange phase_range("gemm_lab_prepare_h2d");

        if(!validate_gemm_inputs(requested_m, requested_n, requested_k, lhs, rhs, error)) {
            return false;
        }
        if(!events.ensure(error)) {
            return false;
        }

        backend = requested_backend;
        m = requested_m;
        n = requested_n;
        k = requested_k;
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
            case GemmLabBackend::TiledGemmV1:
                return detail::launch_tiled_gemm_v1(lhs_device.get(), rhs_device.get(), out_device.get(), m, n, k, error);
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
    std::string& error
) {
    return impl_->prepare(backend, m, n, k, lhs, rhs, error);
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
        case GemmLabBackend::TiledGemmV1:
            return detail::tiled_gemm_v1_kernel_available();
    }

    return false;
}

bool tiled_gemm_v1_cuda(
    std::size_t m,
    std::size_t n,
    std::size_t k,
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    std::vector<float>& out,
    std::string& error
) {
    const ai_system::profiling::ScopedNvtxRange e2e_range("tiled_gemm_v1_e2e");

    PreparedGemmLabRunner runner;
    {
        const ai_system::profiling::ScopedNvtxRange phase_range("tiled_gemm_v1_prepare");
        if(!runner.prepare(GemmLabBackend::TiledGemmV1, m, n, k, lhs, rhs, error)) {
            return false;
        }
    }
    {
        const ai_system::profiling::ScopedNvtxRange phase_range("tiled_gemm_v1_run");
        if(!runner.run(error)) {
            return false;
        }
    }
    {
        const ai_system::profiling::ScopedNvtxRange phase_range("tiled_gemm_v1_d2h");
        return runner.copy_output(out, error);
    }
}

bool tiled_gemm_v1_kernel_available() {
    return gemm_lab_backend_available(GemmLabBackend::TiledGemmV1);
}

}  // namespace ai_system::labs::gemm
