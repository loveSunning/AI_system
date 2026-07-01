# W13 Online Softmax

本阶段目标是把 online softmax 的状态更新从公式落到可测试、可 benchmark 的代码里，为后续 attention forward 做准备。

## 实现入口

| op | kernel | API | test | benchmark |
| --- | --- | --- | --- | --- |
| Online Softmax | `python/triton_playground/kernels/online_softmax.py` | `python/triton_playground/ops/online_softmax.py` | `tests/test_online_softmax.py` | `scripts/bench_online_softmax.py` |

## 算法

普通 softmax：

```text
m = max(x)
y_i = exp(x_i - m) / sum_j exp(x_j - m)
```

online recurrence：

```text
m_new = max(m_old, max(block))
l_new = l_old * exp(m_old - m_new) + sum(exp(block - m_new))
```

当前 Triton 实现用两遍扫描：

- 第一遍按 `BLOCK_SIZE` 扫描每一行，得到最终 `running_max` 和 `running_sum`。
- 第二遍重新读取输入，计算并写回 `exp(x - running_max) / running_sum`。

## Correctness

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python pytest tests/test_online_softmax.py
```

覆盖点：

- dtype: `float32`, `float16`
- shape: 小列数、非 2 的幂列数、跨 block 的长行
- reference: `torch.softmax(x, dim=-1)`
- cross-check: `online_softmax` 和 W09 `fused_softmax` 输出一致

## Benchmark

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python python3 scripts/bench_online_softmax.py --rows 4096 --cols 1024 --dtype float32 --block-size 1024
PYTHONPATH=python python3 scripts/bench_online_softmax.py --sweep --plot --rows 4096 --min-cols-power 7 --max-cols-power 13 --dtype float32 --block-size 1024
```

默认输出：

```text
out/triton/benchmarks/w13_online_softmax.csv
out/triton/benchmarks/plots/
```

Providers:

- `triton_online`: Triton two-pass online softmax.
- `torch_online`: PyTorch loop implementation of the same recurrence.
- `triton_fused`: W09 Triton fused softmax baseline.
- `torch_softmax`: PyTorch softmax baseline.

## 预期结论

`triton_online` 为了演示 streaming recurrence，会读取每行两遍，通常不会比 `triton_fused` 更快。它的价值在于把 `m/l` 状态更新训练清楚，后续 attention 可以把这个状态带进 `QK^T -> softmax -> PV` 的分块流水里，避免物化完整 scores。
