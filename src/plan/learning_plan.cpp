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
        {"第8个月", "W29-W32", "PyTorch custom op + torch.compile 接入", "integrations/pytorch", "torch-custom-op-lab v0.1"},
        {"第9个月", "W33-W36", "TVM Quick Start / TensorIR / MetaSchedule", "compilers/tvm", "tvm-lab v0.1"},
        {"第10个月", "W37-W40", "TVM 进阶 + RK3588 工程闭环", "compilers/tvm + edge/rk3588", "rk3588-edge-lab v0.1"},
        {"第11个月", "W41-W44", "MLIR Toy Tutorial + pass / lowering", "compilers/mlir", "mlir-toy-notes v0.1"},
        {"第12个月", "W45-W52", "TPU-MLIR + 年度 Capstone", "accelerators/tpu_mlir + capstone", "年度最终报告 + demo"}
    };

    return phases;
}

}  // namespace ai_system::plan
