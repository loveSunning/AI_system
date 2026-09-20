# AI_system 学习路线图

W01–W28 保留项目原有计划。W29 起依据 `00.AI-system-plan-optimized-v3.xlsx` 调整为推理与训练架构主线，详见 [每周计划与验收](learning-plan-w29-w52.md)。月份沿用累计学习阶段编号；新的 24 周日期为 2026-09-21 至 2027-03-07。

| 月份 | 周次 | 主题 | 目录 | 预期产物 |
| --- | --- | --- | --- | --- |
| 第1个月 | W01-W04 | 性能工程基础：benchmark、profiling、NVTX、基础 kernel | `labs/perf_engineering` | `cuda-kernel-lab v0.1` |
| 第2个月 | W05-W08 | GEMM 深入：层级分块、Tensor Core、autotune | `labs/gemm` | `GEMM 专项报告 v1` |
| 第3个月 | W09-W12 | Triton 入门：vector add、softmax、matmul、fused kernel | `labs/triton` | `triton-playground v0.1` |
| 第4个月 | W13-W16 | Triton 进阶：norm、online softmax、attention primitives | `labs/triton` | `attention-primitives v0.1` |
| 第5个月 | W17-W20 | CUTLASS：device::Gemm、参数扫描、epilogue | `labs/cutlass` | `cutlass-gemm-study v0.1` |
| 第6个月 | W21-W24 | CuTe：layout、tensor、pipeline、copy/partition | `labs/cute` | `cute-notes v0.1` |
| 第7个月 | W25-W28 | FlashAttention：IO-aware attention、causal mask、online softmax | `labs/flash_attention` | `flash-attn-mini v0.1` |
| 第8个月 | W29-W32 | Qwen3-8B 基线、W4A16 量化、Profiling 与 vLLM 上线 | `capstone/quantization_profiling` + `capstone/runtime_training` | P1 量化诊断报告 + P3 API/压测 demo |
| 第9个月 | W33-W36 | FlashInfer prefill/decode、SGLang 对照与推理作品集 | `capstone/kernel_training` + `capstone/runtime_training` | P2 FlashInfer 实验 + P3 双框架报告 |
| 第10个月 | W37-W40 | PyTorch custom op、Liger、Qwen3 LoRA/QLoRA 到服务 | `integrations/pytorch` + `capstone/kernel_training` + `capstone/runtime_training` | P2 算子报告 + P3 SFT/服务闭环 |
| 第11个月 | W41-W44 | DDP/NCCL、Transformer Engine、Megatron-Core 训练架构 | `capstone/kernel_training` + `capstone/runtime_training` | TE 数值对照 + MCore 训练/恢复与多卡实验 |
| 第12个月 | W45-W52 | W45-W48 国产平台迁移；W49-W52 可靠性、四案例复现与求职 | `capstone/domestic_migration` + `capstone` | P4 真实设备迁移报告 + 四案例最终版 |

## 执行与产物

- W29–W40 为 P0 就业主线；W41–W48 为 P1 训练与异构平台进阶；W49–W52 为 P0 复现与求职收尾。
- W32 达到案例验收后开始试投递，W36 起持续投递；不等待全部框架学完。
- 四案例目录与详细任务见 [Capstone](../capstone/README.md)，新增任务状态沿用表格的“未开始”。
- TVM、MLIR、TPU-MLIR、RK3588 延后至 W52 后按岗位需要学习，原目录继续保留。
- 无同型号多卡或国产设备时，相关实验标为待硬件验证，不影响核心案例与求职推进。
