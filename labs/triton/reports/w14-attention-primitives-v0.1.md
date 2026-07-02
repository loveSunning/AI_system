# W14 Attention Primitives v0.1

本阶段实现的是 stepwise/materialized attention forward，用来把 `QK^T -> mask -> softmax -> PV` 拆成可测试、可 benchmark 的 Triton primitives。它不是 FlashAttention，当前会显式物化 `scores` 和 `probs`。

## 实现入口

| op | kernel | API | test | benchmark |
| --- | --- | --- | --- | --- |
| Attention Forward | `python/triton_playground/kernels/attention_forward.py` | `python/triton_playground/ops/attention_forward.py` | `tests/test_attention_forward.py` | `scripts/bench_attention_forward.py` |

## 算法

```text
scores = Q @ K^T * scale
scores = causal_mask(scores)  # optional
probs = softmax(scores)
out = probs @ V
```

Triton 拆成三个 kernel：

- `_qk_scores_kernel`: tile 化计算 `scores[B, H, S, S]`，scores 使用 fp32。
- `_attention_softmax_kernel`: 一行 scores 一个 program，输出 `probs`。
- `_pv_out_kernel`: tile 化计算 `out = probs @ V`。

## Correctness

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python pytest tests/test_attention_forward.py
```

覆盖点：

- shape: `(B,H,S,D)` 包含小 S、多 head、非单 batch、较长序列。
- dtype: `float16`。
- mask: non-causal 和 causal。
- reference: `torch_attention` 物化 baseline。
- extra checks: `probs.sum(dim=-1) ~= 1`，causal 上三角概率接近 0。

## Benchmark

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python python3 scripts/bench_attention_forward.py --batch 1 --heads 8 --seq 256 --dim 64 --dtype float16
PYTHONPATH=python python3 scripts/bench_attention_forward.py --batch 1 --heads 8 --seq 256 --dim 64 --dtype float16 --causal
PYTHONPATH=python python3 scripts/bench_attention_forward.py --sweep --plot --batch 1 --heads 8 --dim 64 --dtype float16
```

默认输出：

```text
out/triton/benchmarks/w14_attention_forward.csv
out/triton/benchmarks/plots/
```

Providers:

- `triton_stepwise`: Triton materialized attention primitives。
- `torch_attention`: PyTorch materialized `matmul -> softmax -> matmul` baseline。

CSV 中的 `notes` 会记录估算的物化显存：

```text
scores_fp32 = B * H * S * S * 4 bytes
probs = B * H * S * S * sizeof(dtype) bytes
```

## 当前限制

- 只支持 contiguous CUDA `float16` / `float32` tensor。
- `q/k/v` shape 必须相同，固定为 `[B, H, S, D]`。
- softmax 当前要求 `softmax_block >= S`，适合教学版 materialized attention。
- 没有做 online softmax + PV 融合，因此显存和 kernel launch 都不是 FlashAttention 路线。

## 下一步

下一步把 W13 online softmax 的 `m/l` 状态带进 attention tile 循环中，避免物化完整 `scores/probs`，再逐步走向 fused attention forward。
