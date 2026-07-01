# Online Softmax

本笔记用于沉淀 W13 的 online softmax 推导和实验结果。

## 普通 Softmax

对一行输入 `x`：

```text
m = max(x)
y_i = exp(x_i - m) / sum_j exp(x_j - m)
```

减去行最大值可以避免指数溢出。

## Streaming 形式

分块扫描时维护两个状态：

```text
m_old: 已处理元素的最大值
l_old: sum(exp(x - m_old))
```

读入新 block 后：

```text
m_new = max(m_old, max(block))
l_new = l_old * exp(m_old - m_new) + sum(exp(block - m_new))
```

最终输出仍然除以全局归一化项。这个形式是后续 FlashAttention online softmax 的基础。

## 当前实现

已落地两个版本：

- Torch 教学版：`triton_playground.ops.torch_online_softmax`，用 Python loop 按列块执行 online recurrence。
- Triton 版：`triton_playground.ops.online_softmax`，一行一个 program，第一遍按 `BLOCK_SIZE` 分块得到最终 `m/l`，第二遍写回输出。

当前 benchmark 会和两条 baseline 对比：

- `triton_fused`：W09 fused softmax，整行放进一个 block。
- `torch_softmax`：`torch.softmax(x, dim=-1)`。

运行：

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python pytest tests/test_online_softmax.py
PYTHONPATH=python python3 scripts/bench_online_softmax.py --rows 4096 --cols 1024 --dtype float32 --block-size 1024
PYTHONPATH=python python3 scripts/bench_online_softmax.py --sweep --plot --rows 4096 --min-cols-power 7 --max-cols-power 13 --dtype float32 --block-size 1024
```

默认输出：

```text
out/triton/benchmarks/w13_online_softmax.csv
out/triton/benchmarks/plots/
```

性能预期：

- `triton_online` 是两遍扫描，同一行会被读两次，通常不应该比 W09 `triton_fused` 更快。
- `torch_online` 会触发多次 PyTorch op 和 Python loop，主要用于验证算法语义。
- online softmax 的价值不是当前单个 softmax kernel 的速度，而是后续 attention 中可以流式处理 scores，避免物化完整 `QK^T`。

## 实验记录模板

| shape | dtype | mask | reference | atol | rtol | max error | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `(1, 1)` / `(17, 129)` / `(32, 1000)` / `(8, 2049)` | `float32`, `float16` | none | `torch.softmax` | `1e-6` / `1e-2` | `1e-5` / `1e-2` | see pytest | implemented |
