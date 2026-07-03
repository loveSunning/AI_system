# W14 FlashAttention v2

本报告记录 `flash_attention_v2` 的实现和后续 benchmark 结果。当前代码主要参照 Triton 官方教程 06 的 FlashAttention v2 路线，和项目已有的 FlashAttention-1 风格学习版并存。

## Scope

已落地：

- `python/triton_playground/kernels/flash_attention_v2.py`
- `python/triton_playground/ops/flash_attention_v2.py`
- `tests/test_flash_attention_v2.py`
- `scripts/bench_flash_attention_v2.py`
- `notes/flash-attention-v2.md`

核心特性：

- Forward 使用 online softmax，不物化 `scores/probs`。
- 使用 `tl.math.exp2` 和 base-2 `lse`。
- 使用 `STAGE` 拆分 causal attention。
- Forward 使用 `@triton.autotune`。
- 支持 TensorDescriptor API 和 host descriptor pre-hook。
- Backward 使用 Triton kernel 计算 `dQ/dK/dV`。

当前约束：

- `float16` only。
- `S` 必须是 128 的倍数。
- 暂不支持 dropout、attention bias、FP8、varlen、GQA/MQA。

## Correctness

运行：

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python pytest tests/test_flash_attention_v2.py
```

当前测试覆盖：

- forward non-causal / causal 对比 PyTorch materialized reference。
- forward 对比项目内 FlashAttention-1 风格实现。
- backward non-causal / causal 对比 PyTorch autograd reference。
- 非 128 倍数 sequence length 的输入校验。

## Benchmark

单点：

```bash
PYTHONPATH=python python3 scripts/bench_flash_attention_v2.py --batch 1 --heads 8 --seq 1024 --dim 64 --dtype float16
PYTHONPATH=python python3 scripts/bench_flash_attention_v2.py --batch 1 --heads 8 --seq 1024 --dim 64 --dtype float16 --causal
PYTHONPATH=python python3 scripts/bench_flash_attention_v2.py --mode backward --batch 1 --heads 8 --seq 1024 --dim 64 --dtype float16
```

曲线：

```bash
PYTHONPATH=python python3 scripts/bench_flash_attention_v2.py --sweep --plot --batch 1 --heads 8 --dim 64 --dtype float16
PYTHONPATH=python python3 scripts/bench_flash_attention_v2.py --sweep --plot --mode backward --batch 1 --heads 8 --dim 64 --dtype float16
```

默认输出：

```text
out/triton/benchmarks/w14_flash_attention_v2.csv
out/triton/benchmarks/plots/
```

默认 providers：

- `flash_attention_v2`
- `flash_attention_v1`
- `torch_attention`

## 待记录结果

跑完 benchmark 后建议补充：

- GPU 型号、CUDA、Triton、PyTorch 版本。
- non-causal forward 曲线。
- causal forward 曲线。
- backward 曲线。
- v2 是否快过 v1，在哪些 `S/D/H` 上明显。
- v2 和 PyTorch materialized baseline 的差距。
- autotune winner，可从 Triton 编译日志或手工打印配置继续补。

