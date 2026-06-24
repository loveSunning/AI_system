# Triton Plan

本计划来自 `00.AI-system-plan-optimized-v2.xlsx` 的 Triton 相关阶段：`W09-W12` 做 Triton 基础，`W13-W14` 做 Attention 零件。旧路线中较长的 `W09-W16` Triton 阶段已在优化版中压缩，`W15` 起进入 CuTe。

## 阶段目标

| 阶段 | 周次 | 日期范围 | 核心目标 | 主要产出 | 验收标准 |
| --- | --- | --- | --- | --- | --- |
| Triton 基础 | W09-W12 | 2026-06-15~2026-07-12 | 掌握 Triton DSL、matmul、persistent、fusion | `triton-playground v0.1` | 正确性 + benchmark + `program_id` 映射解释 |
| Attention 零件 | W13-W14 | 2026-07-13~2026-07-26 | online softmax 与 attention forward 分步正确性 | `attention-primitives v0.1` | 对齐 PyTorch SDPA/softmax，完成稳定性推导 |

## W09: Vector Add + Fused Softmax

- 学习重点：官方教程、`program_id`、`tl.arange`、mask、`tl.load/store`。
- 动手任务：实现 Triton `vector add` 和 `fused softmax`。
- 验收：正确性通过，并与 PyTorch reference 做 benchmark 对比。
- 记录：不要追极限性能，优先写清楚 block size、num warps、mask 的语义。

## W10: Matmul Baseline + Autotune

- 学习重点：`pid_m/pid_n`、grouped ordering、L2 locality、`tl.dot`。
- 动手任务：实现 Triton matmul baseline，加入 autotune 配置。
- 验收：与 CUDA v2 GEMM、cuBLAS、`torch.matmul` 做至少一组可复现对比曲线。
- 记录：shape、`BLOCK_M/N/K`、`num_warps`、`num_stages`、autotune winner。

## W11: Persistent Matmul

- 学习重点：persistent matmul、program id 映射、tile scheduler、常驻线程块思路。
- 动手任务：实现 persistent matmul 最小版本，并画出 mapping。
- 验收：输出 mapping 图和性能表。
- 记录：说明哪些 shape 受益，哪些 shape 不适合 persistent。

## W12: Fused Kernels

- 学习重点：常用融合算子的读写模式、dtype、误差容忍。
- 动手任务：实现 `RMSNorm` 和 `MatMul+bias+SiLU`。
- 验收：形成 `triton-playground v0.1`，包含 correctness harness 和 benchmark harness。
- 记录：完整记录 `atol/rtol`、dtype、shape、reference 逻辑。

## W13: Online Softmax

- 学习重点：streaming max、streaming sum-exp、数值稳定性。
- 动手任务：实现 `online_softmax`，NumPy reference + Triton 或 CUDA 实现均可。
- 验收：文档推导稳定性，并与 `torch.softmax` 对齐。
- 记录：保存 shape 表、误差范围、mask 处理方式。

## W14: Attention Forward 分步版

- 学习重点：`QK^T -> mask -> softmax -> PV` 的 shape、scale、mask 和访存路径。
- 动手任务：实现 attention forward 分步版本，先正确，暂不追求融合。
- 验收：形成 `attention-primitives v0.1`，多 shape 正确性通过。
- 记录：画出 Q/K/V/O shape 与访存路径，为 FlashAttention 阶段复用。

## 最小验收清单

| 阶段 | 必须产出 | 必须会解释 | 必须保存的证据 | 通过标准 |
| --- | --- | --- | --- | --- |
| Triton 基础 | `vector add`、`softmax`、`matmul`、`persistent`、`RMSNorm`、`MatMul+bias+SiLU` | `program_id`、`tl.arange`、mask、`tl.dot`、autotune、persistent tile mapping | `benchmark.csv`、correctness log、mapping 图、README | 和 PyTorch、cuBLAS 或手写 CUDA 有至少一组可复现对比 |
| Attention 零件 | `online_softmax`、attention forward 分步版 | streaming max/sum-exp 公式、mask、scale、Q/K/V/O shape | 稳定性推导、PyTorch 对齐测试、shape 表 | 多 shape 正确性通过，误差范围记录清楚 |

## 建议提交节奏

1. 每完成一个 kernel，先落 correctness，再落 benchmark。
2. 每次 benchmark 固定 warmup、iters、shape 和 dtype，不把不可复现的单次结果写入结论。
3. 每周结束更新对应 notes，把“为什么这样映射”和“为什么这个 shape 快/慢”写清楚。
4. 阶段结束时把 `out/triton/benchmarks/` 的 CSV、`reports/` 的复盘和 `notes/` 的图表串起来，作为后续 CuTe/CUTLASS/FlashAttention 对照基线。
