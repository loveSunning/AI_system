# Notes

这里放解释性材料，不放大段原始日志。优先沉淀三类内容：

- 映射：`program_id` 如何映射到 tile、row、block、attention head。
- 数值：softmax、online softmax、RMSNorm 的稳定性与误差范围。
- 性能：shape、block size、warp、stage、L2 locality 对结果的影响。

当前占位文档：

- [program-id-mapping.md](./program-id-mapping.md)
- [online-softmax.md](./online-softmax.md)
- [fused-attention.md](./fused-attention.md)
- [flash-attention-v2.md](./flash-attention-v2.md)
