# PyTorch 接入

对应 v3 计划 `W37`，在 W38–W40 结合训练与服务验证端到端收益。

复用已有 RMSNorm/RoPE（或 SwiGLU）kernel，完成 dispatcher、fake/meta、autograd、opcheck 与 torch.compile 接入。至少测评两个算子、将其中一个接入模型，覆盖正确性、异常 shape、eager/compile 和回退路径，并交付可安装包。

建议目录：

- `cpp_extension/`
- `cuda_ops/`
- `python_tests/`
- `compile_bench/`

任务与验收见 [P2 Kernel 与训练优化](../../capstone/kernel_training/README.md) 和 [每周计划](../../docs/learning-plan-w29-w52.md)。
