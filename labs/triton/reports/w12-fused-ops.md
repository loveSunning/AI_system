# W12 Fused Ops

W12 的目标不是追一个单点性能数字，而是把常见 row-wise / element-wise fused op 做成可测试、可 benchmark、可复盘的 Triton 小体系。当前阶段包含四类算子：

- Dropout: 显式 mask 版本和 low-memory seeded 版本。
- LayerNorm: affine LayerNorm forward/backward。
- RMSNorm: production-style Triton 版本和 naive Triton 版本。
- MatMul + Bias + SiLU: matmul 主循环加 fused epilogue。

这些算子共同服务下一阶段 Online Softmax / Attention：它们反复训练 `program_id` 到 row 的映射、`tl.arange` 向量化、mask load/store、row-wise reduction、forward 中间量保存、backward 中跨 row 的参数梯度 reduce。

## 实现入口

| op | kernel | API | test | benchmark |
| --- | --- | --- | --- | --- |
| Dropout | `python/triton_playground/kernels/dropout.py` | `python/triton_playground/ops/dropout.py` | `tests/test_dropout.py` | `scripts/bench_dropout.py` |
| LayerNorm | `python/triton_playground/kernels/layer_norm.py` | `python/triton_playground/ops/layer_norm.py` | `tests/test_layer_norm.py` | `scripts/bench_layer_norm.py` |
| RMSNorm | `python/triton_playground/kernels/rms_norm.py` | `python/triton_playground/ops/rms_norm.py` | `tests/test_rms_norm.py` | `scripts/bench_rms_norm.py` |
| MatMul + Bias + SiLU | `python/triton_playground/kernels/matmul_bias_silu.py` | `python/triton_playground/ops/matmul_bias_silu.py` | `tests/test_matmul_bias_silu.py` | `scripts/bench_matmul_bias_silu.py` |

聚合脚本：

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python python3 scripts/bench_w12_fused_ops.py --rows 4096 --cols 8192 --dtype float16 --mode backward
PYTHONPATH=python python3 scripts/bench_w12_fused_ops.py --sweep --plot --rows 4096 --cols 8192 --dtype float16 --mode backward
```

默认输出：

```text
out/triton/benchmarks/w12_dropout.csv
out/triton/benchmarks/w12_layer_norm.csv
out/triton/benchmarks/w12_rms_norm.csv
out/triton/benchmarks/w12_matmul_bias_silu.csv
out/triton/benchmarks/plots/
```

## Correctness 验收

推荐固定运行：

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python pytest tests/test_dropout.py
PYTHONPATH=python pytest tests/test_layer_norm.py
PYTHONPATH=python pytest tests/test_rms_norm.py
PYTHONPATH=python pytest tests/test_matmul_bias_silu.py
```

覆盖点：

- dtype: `float16`, `float32`。
- shape: 2D、多维 last-dim norm、非 2 的幂列数。
- reference: Dropout 使用显式 mask / deterministic seed 语义检查；LayerNorm 对齐 `torch.nn.functional.layer_norm`；RMSNorm 对齐 PyTorch expression `x * rsqrt(mean(x^2) + eps) * weight`；MatMul + Bias + SiLU 对齐 `torch.nn.functional.silu(a @ b + bias)`。
- 输入限制: CUDA tensor、contiguous、最后一维非空。
- norm 限制: 单行 feature 当前遵循 64KB fused-row limit。

## 算法模式

### Dropout

显式 mask 版本：

```text
out = where(keep_mask, x / (1 - p), 0)
```

Low-memory seeded 版本：

```text
random = tl.rand(seed, offsets)
keep = random > p
out = where(keep, x / (1 - p), 0)
```

要点：

- 显式 mask 版本适合教学和确定性 correctness。
- seeded 版本不物化 mask tensor，减少显存占用。
- PyTorch dropout 的随机序列不和 Triton seed bitwise 对齐，因此 benchmark 主要比较性能和统计语义。

### LayerNorm

Forward:

```text
mean = mean(x)
rstd = rsqrt(mean((x - mean)^2) + eps)
y = (x - mean) * rstd * weight + bias
```

Backward:

```text
xhat = (x - mean) * rstd
wdy = weight * dy
dx = rstd * (wdy - mean(wdy) - xhat * mean(wdy * xhat))
dweight = sum(dy * xhat, over rows)
dbias = sum(dy, over rows)
```

实现拆分：

- forward 一行一个 program，保存 `mean/rstd`。
- backward 第一阶段 fused 计算 `dx + partial dweight/dbias`。
- backward 第二阶段 reduce partial buffer 得到最终 `dweight/dbias`。

### RMSNorm

Forward:

```text
rstd = rsqrt(mean(x^2) + eps)
y = x * rstd * weight
```

Backward:

```text
xhat = x * rstd
wdy = weight * dy
dx = rstd * (wdy - xhat * mean(wdy * xhat))
dweight = sum(dy * xhat, over rows)
```

实现拆分：

- `triton_prod`: backward 第一阶段 fused 计算 `dx + partial dweight`，第二阶段 reduce `dweight`。
- `triton_naive`: backward 拆为独立 `dx` kernel 和按列块扫描所有 row 的 `dweight` reduce kernel。
- `torch`: PyTorch expression baseline，非 fused，通常会物化多个中间张量。

### MatMul + Bias + SiLU

```text
z = A @ B + bias
y = z * sigmoid(z)
```

实现拆分：

- matmul mainloop 沿用 W10 的 `pid_m/pid_n`、grouped ordering 和 `tl.dot`。
- epilogue 在 accumulator 上直接加 `bias` 并计算 SiLU，避免额外写出再读回 bias-add 中间张量。
- benchmark 对比 `triton_fused`、`torch_expression` 和 `torch_compile`；`TFLOP/s` 只按 matmul FLOPs 估算。

## Benchmark 命令

单点：

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python python3 scripts/bench_dropout.py --n-elements 16777216 --dtype float32 --p 0.5
PYTHONPATH=python python3 scripts/bench_layer_norm.py --rows 4096 --cols 8192 --dtype float16 --mode backward
PYTHONPATH=python python3 scripts/bench_rms_norm.py --rows 4096 --cols 8192 --dtype float16 --mode backward
PYTHONPATH=python python3 scripts/bench_matmul_bias_silu.py --m 1024 --n 1024 --k 1024 --dtype float16
```

Sweep + plot:

```bash
PYTHONPATH=python python3 scripts/bench_dropout.py --sweep --plot --min-power 12 --max-power 28 --dtype float32 --p 0.5
PYTHONPATH=python python3 scripts/bench_layer_norm.py --sweep --plot --rows 4096 --min-cols 1024 --max-cols 16384 --cols-step 512 --dtype float16 --mode backward
PYTHONPATH=python python3 scripts/bench_rms_norm.py --sweep --plot --rows 4096 --min-cols 1024 --max-cols 16384 --cols-step 512 --dtype float16 --mode backward
PYTHONPATH=python python3 scripts/bench_matmul_bias_silu.py --sweep --plot --min-power 10 --max-power 13 --k 1024 --dtype float16
```

聚合脚本：

```bash
PYTHONPATH=python python3 scripts/bench_w12_fused_ops.py --dry-run
PYTHONPATH=python python3 scripts/bench_w12_fused_ops.py --rows 4096 --cols 8192 --dtype float16 --mode backward
PYTHONPATH=python python3 scripts/bench_w12_fused_ops.py --sweep --plot --rows 4096 --cols 8192 --dtype float16 --mode backward
```

## 性能解释

- Norm forward/backward 多数是 memory-bound，少物化中间张量通常比多 kernel PyTorch expression 更重要。
- Triton fused RMSNorm 比 PyTorch expression 快很多是预期现象，因为 PyTorch expression 会产生 `float`, `pow`, `mean`, `rsqrt`, multiply, cast 等多个算子和中间张量。
- `GB/s_est` 是统一估算值，不是真实硬件带宽。对 non-fused baseline，真实内存流量会比估算值更大，因此图上的 GB/s_est 会显得低。
- RMSNorm 的 `triton_naive` 有时接近或超过 `triton_prod`，说明 partial buffer + lock accumulation 的同步成本在某些 shape 下不一定划算；production-style 的价值在于更接近可扩展 fused backward 路线。
- LayerNorm backward 比 RMSNorm backward 多了 mean 相关项和 `dbias`，通常更复杂，也更容易受 reduce 策略影响。
- MatMul + Bias + SiLU 的 fused epilogue 可以少一次中间张量写回/读回；但整体仍主要受 matmul mainloop、tile 配置和 cuBLAS/PyTorch 底层库优化程度影响。

## 当前限制

- 只支持 contiguous CUDA tensor。
- Norm 的 normalized dimension 固定为最后一维。
- Norm 的 fused row block 遵循 64KB 限制；超大 hidden size 后续需要多 CTA row split 或分块 reduce。
- MatMul + Bias + SiLU 当前是 fixed-config 学习版，没有接入 autotune；默认配置不保证覆盖所有 shape 的最佳性能。
- 当前 benchmark 主要是 kernel-level microbenchmark，没有覆盖端到端模型图优化。
- PyTorch baseline 里 RMSNorm 使用 expression，不等价于 Apex/FlashAttention/xFormers 的 fused RMSNorm。

## W13 衔接

W12 的核心训练是“一次性 row-wise reduction”。Online Softmax 会把这个模式推进一步：

```text
row-wise sum/max -> block-wise streaming max/sum-exp
```

进入 W13 时重点关注：

- streaming max 如何合并历史 block 和当前 block。
- streaming sum-exp 如何在 max 改变后重缩放。
- 为什么 softmax 可以分块计算而不物化完整 scores。
- 这些状态变量如何映射到 FlashAttention 的 `m_i`, `l_i`, `acc`。

完成 W12 后，下一步建议直接进入 `notes/online-softmax.md` 和 `scripts/bench_online_softmax.py` 这条线。
