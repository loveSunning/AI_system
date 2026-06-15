#pragma once

#include "ai_system/config.hpp"

#if AI_SYSTEM_HAS_CUDA
#include <cuda_fp16.h>
#else
#include <cstdint>
using half = std::uint16_t;
#endif

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ai_system::labs::hgemm {

enum class HgemmKernel {
    NaiveF16,
    SlicedKF16,
    T8x8SlicedKF16x4,
    T8x8SlicedKF16x4Pack,
    T8x8SlicedKF16x4Bcf,
    T8x8SlicedKF16x4PackBcf,
    T8x8SlicedKF16x8PackBcf,
    T8x8SlicedKF16x8PackBcfDbuf,
    T8x8SlicedK16F16x8PackDbuf,
    T8x8SlicedK16F16x8PackDbufAsync,
    T8x8SlicedK32F16x8PackDbuf,
    T8x8SlicedK32F16x8PackDbufAsync,
    T16x8SlicedK32F16x8PackDbuf,
    T16x8SlicedK32F16x8PackDbufAsync,
    CublasTensorOpNn,
    CublasTensorOpTn,
    WmmaM16n16k16Naive,
    WmmaM16n16k16Mma4x2,
    WmmaM16n16k16Mma4x2Warp2x4,
    WmmaM16n16k16Mma4x2Warp2x4DbufAsync,
    WmmaM32n8k16Mma2x4Warp2x4DbufAsync,
    WmmaM16n16k16Mma4x2Warp2x4Stages,
    WmmaM16n16k16Mma4x2Warp2x4StagesDsmem,
    WmmaM16n16k16Mma4x2Warp4x4StagesDsmem,
    WmmaM16n16k16Mma4x4Warp4x4StagesDsmem,
    MmaM16n8k16Naive,
    MmaM16n8k16Mma2x4Warp4x4,
    MmaM16n8k16Mma2x4Warp4x4Stages,
    MmaM16n8k16Mma2x4Warp4x4StagesDsmem,
    MmaM16n8k16Mma2x4Warp4x4x2StagesDsmem,
    MmaM16n8k16Mma2x4Warp4x4x2StagesDsmemX4,
    MmaM16n8k16Mma2x4Warp4x4x2StagesDsmemRr,
    MmaM16n8k16Mma2x4Warp4x4StagesDsmemTn,
    MmaStagesBlockSwizzleTnCute,
    MmaM16n8k16Mma2x4Warp4x4x2StagesDsmemSwizzle,
    MmaM16n8k16Mma2x4Warp4x4x2StagesDsmemTnSwizzleX4
};

struct HgemmKernelInfo {
    HgemmKernel kernel;
    std::string_view name;
    std::string_view ncu_regex;
    std::string_view tile_shape;
    std::string_view register_shape;
    bool accepts_stage_options;
};

struct HgemmLaunchOptions {
    int stages {2};
    bool swizzle {true};
    int swizzle_stride {2048};
};

class PreparedHgemmLabRunner {
public:
    PreparedHgemmLabRunner();
    ~PreparedHgemmLabRunner();

    PreparedHgemmLabRunner(const PreparedHgemmLabRunner&) = delete;
    PreparedHgemmLabRunner& operator=(const PreparedHgemmLabRunner&) = delete;
    PreparedHgemmLabRunner(PreparedHgemmLabRunner&& other) noexcept;
    PreparedHgemmLabRunner& operator=(PreparedHgemmLabRunner&& other) noexcept;

    bool prepare(
        HgemmKernel kernel,
        std::size_t m,
        std::size_t n,
        std::size_t k,
        const std::vector<float>& lhs,
        const std::vector<float>& rhs,
        std::string& error,
        HgemmLaunchOptions launch_options = {}
    );

    bool run(std::string& error);
    bool run_timed(double& elapsed_ms, std::string& error);
    bool copy_output(std::vector<float>& out, std::string& error) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

const std::vector<HgemmKernelInfo>& hgemm_kernel_infos();
const HgemmKernelInfo* find_hgemm_kernel_info(HgemmKernel kernel);
const HgemmKernelInfo* find_hgemm_kernel_info(std::string_view name);

bool launch_hgemm_kernel(
    HgemmKernel kernel,
    const half* a,
    const half* b,
    half* c,
    int m,
    int n,
    int k,
    std::string& error,
    HgemmLaunchOptions launch_options = {}
);

bool hgemm_naive_f16(const half* a, const half* b, half* c, int m, int n, int k, std::string& error);
bool hgemm_sliced_k_f16(const half* a, const half* b, half* c, int m, int n, int k, std::string& error);
bool hgemm_t_8x8_sliced_k_f16x4(const half* a, const half* b, half* c, int m, int n, int k, std::string& error);
bool hgemm_t_8x8_sliced_k_f16x4_pack(const half* a, const half* b, half* c, int m, int n, int k, std::string& error);
bool hgemm_t_8x8_sliced_k_f16x4_bcf(const half* a, const half* b, half* c, int m, int n, int k, std::string& error);
bool hgemm_t_8x8_sliced_k_f16x4_pack_bcf(const half* a, const half* b, half* c, int m, int n, int k, std::string& error);
bool hgemm_t_8x8_sliced_k_f16x8_pack_bcf(const half* a, const half* b, half* c, int m, int n, int k, std::string& error);
bool hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf(const half* a, const half* b, half* c, int m, int n, int k, std::string& error);
bool hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf(const half* a, const half* b, half* c, int m, int n, int k, std::string& error);
bool hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async(const half* a, const half* b, half* c, int m, int n, int k, std::string& error);
bool hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf(const half* a, const half* b, half* c, int m, int n, int k, std::string& error);
bool hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_async(const half* a, const half* b, half* c, int m, int n, int k, std::string& error);
bool hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf(const half* a, const half* b, half* c, int m, int n, int k, std::string& error);
bool hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_async(const half* a, const half* b, half* c, int m, int n, int k, std::string& error);
bool hgemm_cublas_tensor_op_nn(const half* a, const half* b, half* c, int m, int n, int k, std::string& error);
bool hgemm_cublas_tensor_op_tn(const half* a, const half* b, half* c, int m, int n, int k, std::string& error);
bool hgemm_wmma_m16n16k16_naive(const half* a, const half* b, half* c, int m, int n, int k, std::string& error);
bool hgemm_wmma_m16n16k16_mma4x2(const half* a, const half* b, half* c, int m, int n, int k, std::string& error);
bool hgemm_wmma_m16n16k16_mma4x2_warp2x4(const half* a, const half* b, half* c, int m, int n, int k, std::string& error);
bool hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async(const half* a, const half* b, half* c, int m, int n, int k, std::string& error);
bool hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async(const half* a, const half* b, half* c, int m, int n, int k, std::string& error);
bool hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages(const half* a, const half* b, half* c, int m, int n, int k, int stages, bool swizzle, int swizzle_stride, std::string& error);
bool hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem(const half* a, const half* b, half* c, int m, int n, int k, int stages, bool swizzle, int swizzle_stride, std::string& error);
bool hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem(const half* a, const half* b, half* c, int m, int n, int k, int stages, bool swizzle, int swizzle_stride, std::string& error);
bool hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem(const half* a, const half* b, half* c, int m, int n, int k, int stages, bool swizzle, int swizzle_stride, std::string& error);
bool hgemm_mma_m16n8k16_naive(const half* a, const half* b, half* c, int m, int n, int k, std::string& error);
bool hgemm_mma_m16n8k16_mma2x4_warp4x4(const half* a, const half* b, half* c, int m, int n, int k, std::string& error);
bool hgemm_mma_m16n8k16_mma2x4_warp4x4_stages(const half* a, const half* b, half* c, int m, int n, int k, int stages, bool swizzle, int swizzle_stride, std::string& error);
bool hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem(const half* a, const half* b, half* c, int m, int n, int k, int stages, bool swizzle, int swizzle_stride, std::string& error);
bool hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem(const half* a, const half* b, half* c, int m, int n, int k, int stages, bool swizzle, int swizzle_stride, std::string& error);
bool hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4(const half* a, const half* b, half* c, int m, int n, int k, int stages, bool swizzle, int swizzle_stride, std::string& error);
bool hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr(const half* a, const half* b, half* c, int m, int n, int k, int stages, bool swizzle, int swizzle_stride, std::string& error);
bool hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn(const half* a, const half* b, half* c, int m, int n, int k, int stages, bool swizzle, int swizzle_stride, std::string& error);
bool hgemm_mma_stages_block_swizzle_tn_cute(const half* a, const half* b, half* c, int m, int n, int k, int stages, bool swizzle, int swizzle_stride, std::string& error);
bool hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_swizzle(const half* a, const half* b, half* c, int m, int n, int k, int stages, bool swizzle, int swizzle_stride, std::string& error);
bool hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x4(const half* a, const half* b, half* c, int m, int n, int k, int stages, bool swizzle, int swizzle_stride, std::string& error);

}  // namespace ai_system::labs::hgemm
