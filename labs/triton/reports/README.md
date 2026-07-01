# Reports

这里保存阶段复盘和人工整理后的证据，不建议直接提交超大的原始 profiler 文件。

推荐文件：

- `w09-summary.md`
- `w10-matmul-autotune.md`
- `w11-persistent-mapping.md`
- `w12-fused-ops.md`
- `w12-triton-playground-v0.1.md`
- `w13-online-softmax.md`
- `w14-attention-primitives-v0.1.md`

每份报告至少回答：

1. 做了哪些 shape 和 dtype。
2. correctness 是否通过，误差范围是多少。
3. benchmark 如何跑，reference 是谁。
4. 当前实现快或慢的主要原因。
5. 下一阶段要复用或修正什么。
