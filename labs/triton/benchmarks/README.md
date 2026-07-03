# Benchmarks

这里保留 benchmark 可复现实验说明。脚本生成的 CSV 和图片不要放在本目录，默认输出到仓库级：

```text
out/triton/benchmarks/
```

建议 CSV 字段固定为：

```text
stage,op,impl,shape,dtype,config,warmup,iters,avg_ms,min_ms,max_ms,throughput,unit,reference,device,notes
```

推荐文件：

- `w09_vector_add_softmax.csv`
- `w10_matmul.csv`
- `w11_grouped_gemm.csv`
- `w11_persistent_matmul.csv`
- `w12_fused_ops.csv`
- `w12_dropout.csv`
- `w12_layer_norm.csv`
- `w12_rms_norm.csv`
- `w12_matmul_bias_silu.csv`
- `w13_online_softmax.csv`
- `w14_attention_forward.csv`
- `w14_fused_attention.csv`
- `w14_flash_attention_v2.csv`

同一个 CSV 中保留 PyTorch、cuBLAS 或手写 CUDA reference 的结果，避免只看 Triton 单点数字。

W09 benchmark 脚本使用 Triton 官方教程中的 `triton.testing.do_bench` 和 `triton.testing.perf_report` 风格：

- CSV 中 `avg_ms/min_ms/max_ms` 对应 `do_bench` 的 median/p20/p80 计时结果。
- `--sweep` 会把多组 shape 写入 CSV。
- `--plot` 会生成 Triton `perf_report` 对比图，默认保存到 `out/triton/benchmarks/plots/`。
- `scripts/plot_w09_benchmarks.py` 可以直接从 CSV 生成 PNG 图，适合保存到报告中。

Softmax 对比包含三种实现：

- `triton`：一个 Triton program 处理一行的 fused softmax。
- `torch`：`torch.softmax(x, dim=-1)`。
- `naive`：PyTorch 分步 `max -> exp -> sum -> div`，会物化中间张量。

推荐流程：

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python python3 scripts/bench_vector_add.py --sweep --plot --min-power 12 --max-power 28 --dtype float32
PYTHONPATH=python python3 scripts/bench_fused_softmax.py --sweep --plot --rows 4096 --min-cols-power 7 --max-cols-power 12 --dtype float32
python3 scripts/plot_w09_benchmarks.py --dtype float32
```

W13 online softmax benchmark：

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python python3 scripts/bench_online_softmax.py --rows 4096 --cols 1024 --dtype float32 --block-size 1024
PYTHONPATH=python python3 scripts/bench_online_softmax.py --sweep --plot --rows 4096 --min-cols-power 7 --max-cols-power 13 --dtype float32 --block-size 1024
```

对比项：

- `triton_online`：两遍扫描的 Triton online softmax，一行一个 program，按 `BLOCK_SIZE` 分块。
- `torch_online`：PyTorch loop 版 online recurrence，用来对照算法语义，不是性能目标。
- `triton_fused`：W09 fused softmax baseline，整行放进一个 block。
- `torch_softmax`：`torch.softmax(x, dim=-1)` baseline。

W10 matmul benchmark：

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python python3 scripts/bench_matmul.py --m 1024 --n 1024 --k 1024 --dtype float16
PYTHONPATH=python python3 scripts/bench_matmul.py --sweep --plot --min-power 8 --max-power 12 --dtype float16
```

默认输出：

```text
out/triton/benchmarks/w10_matmul.csv
out/triton/benchmarks/plots/
```

W11 grouped GEMM benchmark：

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python python3 scripts/bench_grouped_gemm.py --group-size 4 --m 1024 --n 1024 --k 1024 --pattern vary_n
PYTHONPATH=python python3 scripts/bench_grouped_gemm.py --sweep --plot --min-group-size 1 --max-group-size 8 --m 1024 --n 1024 --k 1024 --pattern vary_n
```

对比项：

- `triton`：一次 grouped GEMM launch，包含 metadata tensor 构造开销。
- `torch_loop`：Python loop 中逐个调用 `torch.matmul(a, b)`。

W11 persistent matmul benchmark：

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python python3 scripts/bench_persistent_matmul.py --m 8192 --n 8192 --k 512 --dtype float16
PYTHONPATH=python python3 scripts/bench_persistent_matmul.py --sweep --plot --min-power 10 --max-power 13 --k 512 --dtype float16
```

对比项：

- `persistent_fixed`：非 TMA persistent scheduling，默认使用 4090D-friendly fixed tile。
- `persistent_autotune`：非 TMA persistent scheduling，使用受 shared-memory 约束的 autotune 配置。
- `triton_fixed`：普通 one-program-per-tile Triton matmul baseline。
- `torch`：`torch.matmul(a, b)`。

Dropout benchmark：

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python python3 scripts/bench_dropout.py --n-elements 16777216 --dtype float32 --p 0.5
PYTHONPATH=python python3 scripts/bench_dropout.py --sweep --plot --min-power 12 --max-power 28 --dtype float32 --p 0.5
```

对比项：

- `triton_mask`：显式 keep-mask baseline，需要读 `x` 和 `keep_mask`，写 `out`。
- `triton_seeded_low_memory`：只保存 seed，kernel 内用 `tl.rand(seed, offsets)` 生成 mask。
- `torch`：`torch.nn.functional.dropout(x, p=p, training=True)`。

LayerNorm benchmark：
```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python python3 scripts/bench_layer_norm.py --rows 4096 --cols 8192 --dtype float16 --mode backward
PYTHONPATH=python python3 scripts/bench_layer_norm.py --sweep --plot --rows 4096 --min-cols 1024 --max-cols 16384 --cols-step 512 --dtype float16 --mode backward
```

对比项：

- `triton`：Triton affine LayerNorm，前向保存 `mean/rstd`，后向分两阶段 reduce `dw/db`。
- `torch`：`torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)`。
- `--mode forward` 只测前向，`--mode backward` 复用一次前向图后只测 backward。
RMSNorm benchmark:

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python python3 scripts/bench_rms_norm.py --rows 4096 --cols 8192 --dtype float16 --mode backward
PYTHONPATH=python python3 scripts/bench_rms_norm.py --sweep --plot --rows 4096 --min-cols 1024 --max-cols 16384 --cols-step 512 --dtype float16 --mode backward
```

Providers:

- `triton_prod`: production-style Triton RMSNorm, backward fuses `dx` and partial `dweight`.
- `triton_naive`: simple Triton RMSNorm, backward uses separate `dx` and column-wise `dweight` reduction kernels.
- `torch`: PyTorch expression baseline, `x * rsqrt(mean(x^2) + eps) * weight`.
- `--mode forward` benchmarks forward only; `--mode backward` reuses one forward graph and benchmarks backward.

MatMul + Bias + SiLU benchmark:

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python python3 scripts/bench_matmul_bias_silu.py --m 1024 --n 1024 --k 1024 --dtype float16
PYTHONPATH=python python3 scripts/bench_matmul_bias_silu.py --sweep --plot --min-power 10 --max-power 13 --k 1024 --dtype float16
```

Providers:

- `triton_fused`: Triton matmul mainloop with fused bias + SiLU epilogue.
- `torch_expression`: PyTorch `torch.nn.functional.silu(a @ b + bias)` expression baseline.
- `torch_compile`: `torch.compile` version of the PyTorch expression when available.
- `TFLOP/s` counts matmul FLOPs only, so it is best used for same-shape relative comparison.

W14 attention forward benchmark:

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python python3 scripts/bench_attention_forward.py --batch 1 --heads 8 --seq 256 --dim 64 --dtype float16
PYTHONPATH=python python3 scripts/bench_attention_forward.py --batch 1 --heads 8 --seq 256 --dim 64 --dtype float16 --causal
PYTHONPATH=python python3 scripts/bench_attention_forward.py --sweep --plot --batch 1 --heads 8 --dim 64 --dtype float16
```

Providers:

- `triton_stepwise`: materialized Triton attention, split into QK scores, softmax, and PV output kernels.
- `torch_attention`: PyTorch materialized baseline, `matmul -> softmax -> matmul`.
- CSV `notes` records the estimated materialized `scores` fp32 memory and `probs` memory.

W14 fused attention benchmark:

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python python3 scripts/bench_fused_attention.py --batch 1 --heads 8 --seq 256 --dim 64 --dtype float16
PYTHONPATH=python python3 scripts/bench_fused_attention.py --batch 1 --heads 8 --seq 256 --dim 64 --dtype float16 --causal
PYTHONPATH=python python3 scripts/bench_fused_attention.py --mode backward --batch 1 --heads 8 --seq 256 --dim 64 --dtype float16
PYTHONPATH=python python3 scripts/bench_fused_attention.py --batch 1 --heads 8 --seq 256 --dim 64 --dtype float16 --dropout-p 0.1 --dropout-seed 123
PYTHONPATH=python python3 scripts/bench_fused_attention.py --sweep --plot --batch 1 --heads 8 --dim 64 --dtype float16
```

Providers:

- `flash_attention`: Triton FlashAttention-1 style autograd API with forward and backward kernels.
- `triton_fused`: one fused Triton forward-only kernel with online softmax state `m/l/acc`.
- `triton_stepwise`: materialized Triton baseline from `scripts/bench_attention_forward.py`.
- `torch_attention`: PyTorch materialized baseline.
- Fused attention does not materialize `scores/probs/dropout_mask`; materialized baselines report estimated `scores/probs` memory in CSV `notes`.

W14 FlashAttention v2 benchmark:

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python python3 scripts/bench_flash_attention_v2.py --batch 1 --heads 8 --seq 1024 --dim 64 --dtype float16
PYTHONPATH=python python3 scripts/bench_flash_attention_v2.py --batch 1 --heads 8 --seq 1024 --dim 64 --dtype float16 --causal
PYTHONPATH=python python3 scripts/bench_flash_attention_v2.py --mode backward --batch 1 --heads 8 --seq 1024 --dim 64 --dtype float16
PYTHONPATH=python python3 scripts/bench_flash_attention_v2.py --sweep --plot --batch 1 --heads 8 --dim 64 --dtype float16
```

Providers:

- `flash_attention_v2`: official tutorial 06 style path with `exp2/log2`, `STAGE` causal split, autotune, and TensorDescriptor support.
- `flash_attention_v1`: the project learning implementation from `ops/fused_attention.py`.
- `torch_attention`: PyTorch materialized baseline.
- This path currently supports `float16` and sequence lengths that are multiples of 128. It does not include dropout; use `bench_fused_attention.py` for the v1 dropout path.
