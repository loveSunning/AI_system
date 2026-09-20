# Capstone

W29 起逐步积累四个案例，W49–W52 完成可靠性验证、复现、作品集和求职收尾。排期来源为 `00.AI-system-plan-optimized-v3.xlsx`；详见 [每周计划与验收](../docs/learning-plan-w29-w52.md)。

| 案例 | 任务入口 |
| --- | --- |
| P1 Qwen3-8B 量化与性能诊断 | [quantization_profiling](quantization_profiling/README.md) |
| P2 Kernel、FlashInfer 与训练算子优化 | [kernel_training](kernel_training/README.md) |
| P3 vLLM、SGLang 服务与训练架构闭环 | [runtime_training](runtime_training/README.md) |
| P4 Qwen3-8B 国产平台迁移 | [domestic_migration](domestic_migration/README.md) |

每个案例保存代码、依赖与模型 revision、配置、原始 CSV、trace/日志、质量结果、报告与限制。W50 在另一环境复跑代表 case，每条性能结论能定位到实验 ID。

W49 对 P3 做 30 分钟混合请求压测及取消、超时、OOM/重启恢复验证；W51 根据岗位反馈补短板；W52 补失败实验并交付最终演示。国产平台以至少一个真实可用平台（优先 Ascend）为目标，无硬件则明确延期，不以准备工作代替部署验收。TVM/MLIR/TPU-MLIR/RK3588 已移至 W52 后选修。
