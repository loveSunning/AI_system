#include "ai_system/plan/learning_plan.hpp"

namespace ai_system::plan {

const std::vector<LearningPhase>& learning_plan() {
    static const std::vector<LearningPhase> phases {
        {"第1个月", "W01-W04", "性能工程基础（benchmark / profiling / NVTX）", "labs/perf_engineering", "cuda-kernel-lab v0.1"},
        {"第2个月", "W05-W08", "GEMM 深入（层级分块 / Tensor Core / Autotune）", "labs/gemm", "GEMM 专项报告 v1"},
        {"第3个月", "W09-W12", "Triton 入门（vector add / softmax / matmul / fused op）", "labs/triton", "triton-playground v0.1"},
        {"第4个月", "W13-W16", "Triton 进阶与 Attention primitives", "labs/triton", "attention-primitives v0.1"},
        {"第5个月", "W17-W20", "CUTLASS GEMM 与 epilogue 参数扫描", "labs/cutlass", "cutlass-gemm-study v0.1"},
        {"第6个月", "W21-W24", "CuTe layout / tensor / pipeline 抽象", "labs/cute", "cute-notes v0.1"},
        {"第7个月", "W25-W28", "FlashAttention IO-aware forward/backward", "labs/flash_attention", "flash-attn-mini v0.1"},
        {"第8个月", "W29-W32", "Qwen3-8B 基线、W4A16 量化、Profiling 与 vLLM 上线", "capstone/quantization_profiling + capstone/runtime_training", "P1 量化诊断报告 + P3 API/压测 demo"},
        {"第9个月", "W33-W36", "FlashInfer prefill/decode、SGLang 对照与推理作品集", "capstone/kernel_training + capstone/runtime_training", "P2 FlashInfer 实验 + P3 双框架报告"},
        {"第10个月", "W37-W40", "PyTorch custom op、Liger、Qwen3 LoRA/QLoRA 到服务", "integrations/pytorch + capstone/kernel_training + capstone/runtime_training", "P2 算子报告 + P3 SFT/服务闭环"},
        {"第11个月", "W41-W44", "DDP/NCCL、Transformer Engine、Megatron-Core 训练架构", "capstone/kernel_training + capstone/runtime_training", "TE 数值对照 + MCore 训练/恢复与多卡实验"},
        {"第12个月", "W45-W52", "W45-W48 国产平台迁移；W49-W52 可靠性、四案例复现与求职", "capstone/domestic_migration + capstone", "P4 真实设备迁移报告 + 四案例最终版"}
    };

    return phases;
}

}  // namespace ai_system::plan
