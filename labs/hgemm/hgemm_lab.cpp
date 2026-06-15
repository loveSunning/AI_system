#include "hgemm_lab.hpp"

#include "ai_system/config.hpp"

#include <algorithm>
#include <utility>

#if !AI_SYSTEM_HAS_CUDA

namespace ai_system::labs::hgemm {

namespace {

bool cuda_disabled(std::string& error) {
    error = "CUDA support is disabled in this build.";
    return false;
}

}  // namespace

const std::vector<HgemmKernelInfo>& hgemm_kernel_infos() {
    static const std::vector<HgemmKernelInfo> infos {};
    return infos;
}

const HgemmKernelInfo* find_hgemm_kernel_info(HgemmKernel) {
    return nullptr;
}

const HgemmKernelInfo* find_hgemm_kernel_info(std::string_view) {
    return nullptr;
}

struct PreparedHgemmLabRunner::Impl {};

PreparedHgemmLabRunner::PreparedHgemmLabRunner() = default;
PreparedHgemmLabRunner::~PreparedHgemmLabRunner() = default;
PreparedHgemmLabRunner::PreparedHgemmLabRunner(PreparedHgemmLabRunner&& other) noexcept = default;
PreparedHgemmLabRunner& PreparedHgemmLabRunner::operator=(PreparedHgemmLabRunner&& other) noexcept = default;

bool PreparedHgemmLabRunner::prepare(
    HgemmKernel,
    std::size_t,
    std::size_t,
    std::size_t,
    const std::vector<float>&,
    const std::vector<float>&,
    std::string& error,
    HgemmLaunchOptions
) {
    return cuda_disabled(error);
}

bool PreparedHgemmLabRunner::run(std::string& error) {
    return cuda_disabled(error);
}

bool PreparedHgemmLabRunner::run_timed(double& elapsed_ms, std::string& error) {
    elapsed_ms = 0.0;
    return cuda_disabled(error);
}

bool PreparedHgemmLabRunner::copy_output(std::vector<float>&, std::string& error) const {
    return cuda_disabled(error);
}

bool launch_hgemm_kernel(
    HgemmKernel,
    const half*,
    const half*,
    half*,
    int,
    int,
    int,
    std::string& error,
    HgemmLaunchOptions
) {
    return cuda_disabled(error);
}

bool hgemm_naive_f16(const half*, const half*, half*, int, int, int, std::string& error) {
    return cuda_disabled(error);
}

bool hgemm_sliced_k_f16(const half*, const half*, half*, int, int, int, std::string& error) {
    return cuda_disabled(error);
}

bool hgemm_t_8x8_sliced_k_f16x4(const half*, const half*, half*, int, int, int, std::string& error) {
    return cuda_disabled(error);
}

bool hgemm_t_8x8_sliced_k_f16x4_pack(const half*, const half*, half*, int, int, int, std::string& error) {
    return cuda_disabled(error);
}

bool hgemm_t_8x8_sliced_k_f16x4_bcf(const half*, const half*, half*, int, int, int, std::string& error) {
    return cuda_disabled(error);
}

bool hgemm_t_8x8_sliced_k_f16x4_pack_bcf(const half*, const half*, half*, int, int, int, std::string& error) {
    return cuda_disabled(error);
}

bool hgemm_t_8x8_sliced_k_f16x8_pack_bcf(const half*, const half*, half*, int, int, int, std::string& error) {
    return cuda_disabled(error);
}

bool hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf(const half*, const half*, half*, int, int, int, std::string& error) {
    return cuda_disabled(error);
}

bool hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf(const half*, const half*, half*, int, int, int, std::string& error) {
    return cuda_disabled(error);
}

bool hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async(const half*, const half*, half*, int, int, int, std::string& error) {
    return cuda_disabled(error);
}

bool hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf(const half*, const half*, half*, int, int, int, std::string& error) {
    return cuda_disabled(error);
}

bool hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_async(const half*, const half*, half*, int, int, int, std::string& error) {
    return cuda_disabled(error);
}

bool hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf(const half*, const half*, half*, int, int, int, std::string& error) {
    return cuda_disabled(error);
}

bool hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_async(const half*, const half*, half*, int, int, int, std::string& error) {
    return cuda_disabled(error);
}

bool hgemm_cublas_tensor_op_nn(const half*, const half*, half*, int, int, int, std::string& error) {
    return cuda_disabled(error);
}

bool hgemm_cublas_tensor_op_tn(const half*, const half*, half*, int, int, int, std::string& error) {
    return cuda_disabled(error);
}

bool hgemm_wmma_m16n16k16_naive(const half*, const half*, half*, int, int, int, std::string& error) {
    return cuda_disabled(error);
}

bool hgemm_wmma_m16n16k16_mma4x2(const half*, const half*, half*, int, int, int, std::string& error) {
    return cuda_disabled(error);
}

bool hgemm_wmma_m16n16k16_mma4x2_warp2x4(const half*, const half*, half*, int, int, int, std::string& error) {
    return cuda_disabled(error);
}

bool hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async(const half*, const half*, half*, int, int, int, std::string& error) {
    return cuda_disabled(error);
}

bool hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async(const half*, const half*, half*, int, int, int, std::string& error) {
    return cuda_disabled(error);
}

bool hgemm_mma_m16n8k16_naive(const half*, const half*, half*, int, int, int, std::string& error) {
    return cuda_disabled(error);
}

bool hgemm_mma_m16n8k16_mma2x4_warp4x4(const half*, const half*, half*, int, int, int, std::string& error) {
    return cuda_disabled(error);
}

bool hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages(
    const half*,
    const half*,
    half*,
    int,
    int,
    int,
    int,
    bool,
    int,
    std::string& error
) {
    return cuda_disabled(error);
}

bool hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem(
    const half*,
    const half*,
    half*,
    int,
    int,
    int,
    int,
    bool,
    int,
    std::string& error
) {
    return cuda_disabled(error);
}

bool hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem(
    const half*,
    const half*,
    half*,
    int,
    int,
    int,
    int,
    bool,
    int,
    std::string& error
) {
    return cuda_disabled(error);
}

bool hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem(
    const half*,
    const half*,
    half*,
    int,
    int,
    int,
    int,
    bool,
    int,
    std::string& error
) {
    return cuda_disabled(error);
}

bool hgemm_mma_m16n8k16_mma2x4_warp4x4_stages(
    const half*,
    const half*,
    half*,
    int,
    int,
    int,
    int,
    bool,
    int,
    std::string& error
) {
    return cuda_disabled(error);
}

bool hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem(
    const half*,
    const half*,
    half*,
    int,
    int,
    int,
    int,
    bool,
    int,
    std::string& error
) {
    return cuda_disabled(error);
}

bool hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem(
    const half*,
    const half*,
    half*,
    int,
    int,
    int,
    int,
    bool,
    int,
    std::string& error
) {
    return cuda_disabled(error);
}

bool hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4(
    const half*,
    const half*,
    half*,
    int,
    int,
    int,
    int,
    bool,
    int,
    std::string& error
) {
    return cuda_disabled(error);
}

bool hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr(
    const half*,
    const half*,
    half*,
    int,
    int,
    int,
    int,
    bool,
    int,
    std::string& error
) {
    return cuda_disabled(error);
}

bool hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn(
    const half*,
    const half*,
    half*,
    int,
    int,
    int,
    int,
    bool,
    int,
    std::string& error
) {
    return cuda_disabled(error);
}

bool hgemm_mma_stages_block_swizzle_tn_cute(
    const half*,
    const half*,
    half*,
    int,
    int,
    int,
    int,
    bool,
    int,
    std::string& error
) {
    return cuda_disabled(error);
}

bool hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_swizzle(
    const half*,
    const half*,
    half*,
    int,
    int,
    int,
    int,
    bool,
    int,
    std::string& error
) {
    return cuda_disabled(error);
}

bool hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x4(
    const half*,
    const half*,
    half*,
    int,
    int,
    int,
    int,
    bool,
    int,
    std::string& error
) {
    return cuda_disabled(error);
}

}  // namespace ai_system::labs::hgemm

#endif
