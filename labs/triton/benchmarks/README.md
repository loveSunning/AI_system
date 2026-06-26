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
- `w11_persistent_matmul.csv`
- `w12_fused_ops.csv`
- `w12_dropout.csv`
- `w12_layer_norm.csv`
- `w13_online_softmax.csv`
- `w14_attention_forward.csv`

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
