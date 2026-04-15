# AI_system 学习路线图

这份路线图来自 `AI-system-plan.xlsx`，我把它收敛成了适合仓库落地的模块视图，方便你一边学一边持续往项目里加代码。

| 月份 | 周次 | 主题 | 目录 | 预期产物 |
| --- | --- | --- | --- | --- |
| 第1个月 | W01-W04 | 性能工程基础：benchmark、profiling、NVTX、基础 kernel | `labs/perf_engineering` | `cuda-kernel-lab v0.1` |
| 第2个月 | W05-W08 | GEMM 深入：层级分块、Tensor Core、autotune | `labs/gemm` | `GEMM 专项报告 v1` |
| 第3个月 | W09-W12 | Triton 入门：vector add、softmax、matmul、fused kernel | `labs/triton` | `triton-playground v0.1` |
| 第4个月 | W13-W16 | Triton 进阶：norm、online softmax、attention primitives | `labs/triton` | `attention-primitives v0.1` |
| 第5个月 | W17-W20 | CUTLASS：device::Gemm、参数扫描、epilogue | `labs/cutlass` | `cutlass-gemm-study v0.1` |
| 第6个月 | W21-W24 | CuTe：layout、tensor、pipeline、copy/partition | `labs/cute` | `cute-notes v0.1` |
| 第7个月 | W25-W28 | FlashAttention：IO-aware attention、causal mask、online softmax | `labs/flash_attention` | `flash-attn-mini v0.1` |
| 第8个月 | W29-W32 | PyTorch 接入：custom op、torch.compile、dispatcher/meta/autograd | `integrations/pytorch` | `torch-custom-op-lab v0.1` |
| 第9个月 | W33-W36 | TVM 入门：Quick Start、TensorIR、MetaSchedule | `compilers/tvm` | `tvm-lab v0.1` |
| 第10个月 | W37-W40 | TVM 进阶 + RK3588：schedule、端到端、部署闭环 | `compilers/tvm` + `edge/rk3588` | `rk3588-edge-lab v0.1` |
| 第11个月 | W41-W44 | MLIR：Toy Tutorial、pattern/pass、lowering | `compilers/mlir` | `mlir-toy-notes v0.1` |
| 第12个月 | W45-W52 | TPU-MLIR + Capstone：量化、custom op、跨平台对比、最终作品集 | `accelerators/tpu_mlir` + `capstone` | 年度最终报告 + demo |

## 当前实现状态

- 已完成：第 1 个月的跨平台 CMake 基础设施和首批 CUDA 样例。
- 已预留：后续所有月份的目录和 README 占位。
- 下一步最推荐：从 `labs/gemm` 开始，把 `naive GEMM` 迭代成 `tiled GEMM v1/v2`。
