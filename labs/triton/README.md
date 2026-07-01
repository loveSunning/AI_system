# Triton Lab

本目录对应优化版学习计划中的 `W09-W14`：

- `W09-W12`：Triton 基础，产出 `triton-playground v0.1`
- `W13-W14`：Attention 零件，产出 `attention-primitives v0.1`

目标不是马上追极限性能，而是把 Triton DSL 的执行模型、tile 映射、正确性校验和 benchmark 证据沉淀下来，为后续 CuTe、CUTLASS、FlashAttention 做对照基线。

## 目录结构

```text
labs/triton/
|-- README.md
|-- PLAN.md
|-- requirements.txt
|-- python/
|   |-- README.md
|   `-- triton_playground/
|       |-- __init__.py
|       |-- benchmarking/
|       |   `-- __init__.py
|       |-- kernels/
|       |   `-- __init__.py
|       |-- ops/
|       |   `-- __init__.py
|       `-- testing/
|           `-- __init__.py
|-- tests/
|   `-- README.md
|-- benchmarks/
|   `-- README.md
|-- notes/
|   |-- README.md
|   |-- online-softmax.md
|   `-- program-id-mapping.md
|-- reports/
|   `-- README.md
`-- scripts/
    `-- README.md
```

## 环境建议

计划主线是 NVIDIA + Linux，推荐在 Ubuntu 22.04、WSL2 或 Linux 服务器中运行 Triton 实验。Windows 目录可以继续作为代码工作区，真正运行 kernel 时优先放到 Linux CUDA 环境中。

```bash
cd /workspace/AI_system/labs/triton
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch 请按本机 CUDA 版本单独选择安装命令；`requirements.txt` 只放 Triton lab 需要的通用 Python 包，不强行固定 CUDA wheel。

## 学习主线

1. `vector add`：理解 `program_id`、`tl.arange`、mask、`tl.load/store`。
2. `fused softmax`：理解 row-wise 并行、数值稳定性、带 mask 的 load/store。
3. `matmul baseline + autotune`：理解 `pid_m/pid_n`、grouped ordering、`tl.dot`、`num_warps`、`num_stages`。
4. `persistent matmul`：理解 persistent tile scheduler 和不同 shape 的适配边界。
5. `RMSNorm` 与 `MatMul+bias+SiLU`：把常见融合算子做成可测试、可 benchmark 的实验。
6. `online softmax` 与分步 `attention forward`：先正确拆解 `QK^T -> mask -> softmax -> PV`，再为 FlashAttention 做准备。

详细周计划和验收标准见 [PLAN.md](./PLAN.md)。

## W09 Vector Add / Fused Softmax 入口

当前 W09 已按三层结构落地：

- Kernel: [python/triton_playground/kernels/vector_add.py](./python/triton_playground/kernels/vector_add.py)
- Kernel: [python/triton_playground/kernels/fused_softmax.py](./python/triton_playground/kernels/fused_softmax.py)
- API: [python/triton_playground/ops/vector_add.py](./python/triton_playground/ops/vector_add.py)
- API: [python/triton_playground/ops/fused_softmax.py](./python/triton_playground/ops/fused_softmax.py)
- Test: [tests/test_vector_add.py](./tests/test_vector_add.py)
- Test: [tests/test_fused_softmax.py](./tests/test_fused_softmax.py)
- Benchmark: [scripts/bench_vector_add.py](./scripts/bench_vector_add.py)
- Benchmark: [scripts/bench_fused_softmax.py](./scripts/bench_fused_softmax.py)

运行测试：

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python pytest tests/test_vector_add.py
PYTHONPATH=python pytest tests/test_fused_softmax.py
```

运行 benchmark：

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python python3 scripts/bench_vector_add.py --n-elements 16777216 --dtype float32 --block-size 1024
PYTHONPATH=python python3 scripts/bench_fused_softmax.py --rows 4096 --cols 1024 --dtype float32
```

运行 Triton 官方 `perf_report` 风格 sweep，并把结果保存到仓库级 `out/triton/benchmarks/`：

```bash
PYTHONPATH=python python3 scripts/bench_vector_add.py --sweep --plot --min-power 12 --max-power 28 --dtype float32
PYTHONPATH=python python3 scripts/bench_fused_softmax.py --sweep --plot --rows 4096 --min-cols-power 7 --max-cols-power 12 --dtype float32
python3 scripts/plot_w09_benchmarks.py --dtype float32
```

`fused_softmax` benchmark 会同时比较 `triton`、`torch.softmax` 和 `naive` 三种实现。

默认输出位置：

```text
/workspace/AI_system/out/triton/benchmarks/w09_vector_add_softmax.csv
/workspace/AI_system/out/triton/benchmarks/plots/
```

## W10 Matmul 入口

当前 matmul 已按 baseline + autotune 落地：

- Kernel: [python/triton_playground/kernels/matmul.py](./python/triton_playground/kernels/matmul.py)
- API: [python/triton_playground/ops/matmul.py](./python/triton_playground/ops/matmul.py)
- Test: [tests/test_matmul.py](./tests/test_matmul.py)
- Benchmark: [scripts/bench_matmul.py](./scripts/bench_matmul.py)

运行测试：

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python pytest tests/test_matmul.py
```

运行单点 benchmark：

```bash
PYTHONPATH=python python3 scripts/bench_matmul.py --m 1024 --n 1024 --k 1024 --dtype float16
```

运行 sweep 和曲线图：

```bash
PYTHONPATH=python python3 scripts/bench_matmul.py --sweep --plot --min-power 8 --max-power 12 --dtype float16
```

## W11 Grouped GEMM 入口

参照官方 Group GEMM 教程，当前已落地非 TMA 版本的 grouped GEMM：

- Kernel: [python/triton_playground/kernels/grouped_gemm.py](./python/triton_playground/kernels/grouped_gemm.py)
- API: [python/triton_playground/ops/grouped_gemm.py](./python/triton_playground/ops/grouped_gemm.py)
- Test: [tests/test_grouped_gemm.py](./tests/test_grouped_gemm.py)
- Benchmark: [scripts/bench_grouped_gemm.py](./scripts/bench_grouped_gemm.py)

实现要点：

- Python 侧把每个 GEMM 的 A/B/C 指针、`M/N/K`、leading dimension 打包成 device metadata tensor。
- Kernel 侧启动固定 `NUM_SM` 个 CTA，用全局 tile 序号遍历整组 GEMM；每个 CTA 完成一个 tile 后按 `NUM_SM` 跨步继续。
- 与官方示例不同，这里对 `M/N/K` 边界做了 mask，支持非 block size 整除的 shape。
- 当前先支持 contiguous CUDA `float16` 输入，输出为每个 GEMM 对应的 `float16` C tensor。

运行测试：

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python pytest tests/test_grouped_gemm.py
```

运行单点 benchmark：

```bash
PYTHONPATH=python python3 scripts/bench_grouped_gemm.py --group-size 4 --m 1024 --n 1024 --k 1024 --pattern vary_n
```

运行 sweep 和曲线图：

```bash
PYTHONPATH=python python3 scripts/bench_grouped_gemm.py --sweep --plot --min-group-size 1 --max-group-size 8 --m 1024 --n 1024 --k 1024 --pattern vary_n
```

## W11 Persistent Matmul 入口

参照官方 Persistent Matmul 教程，当前已落地面向 RTX 4090D 的非 TMA persistent matmul：

- Kernel: [python/triton_playground/kernels/persistent_matmul.py](./python/triton_playground/kernels/persistent_matmul.py)
- API: [python/triton_playground/ops/persistent_matmul.py](./python/triton_playground/ops/persistent_matmul.py)
- Test: [tests/test_persistent_matmul.py](./tests/test_persistent_matmul.py)
- Benchmark: [scripts/bench_persistent_matmul.py](./scripts/bench_persistent_matmul.py)

实现要点：

- 4090D 是 Ada / SM89，不使用 Hopper/Blackwell 侧重的 TMA 和 warp-specialize 路线。
- Kernel 只启动 `min(NUM_SMS, num_tiles)` 个 program，每个 program 以 `NUM_SMS` 为步长循环处理多个 output tile。
- 默认 fixed 配置为 `BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_warps=4, num_stages=4`，控制 shared memory 占用，适合 4090/4090D 这类卡。
- autotune 配置限定在 4090D 友好的 shared-memory 范围内，避免教程中部分大 tile/TMA 配置在 4090 系列上失败。

运行测试：

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python pytest tests/test_persistent_matmul.py
```

运行单点 benchmark：

```bash
PYTHONPATH=python python3 scripts/bench_persistent_matmul.py --m 8192 --n 8192 --k 512 --dtype float16
```

运行 sweep 和曲线图：

```bash
PYTHONPATH=python python3 scripts/bench_persistent_matmul.py --sweep --plot --min-power 10 --max-power 13 --k 512 --dtype float16
```

## Dropout 入口

参照官方 Low-Memory Dropout 教程，当前已落地两种 Triton dropout：

- 普通 dropout: [python/triton_playground/kernels/dropout.py](./python/triton_playground/kernels/dropout.py) 中的显式 `keep_mask` 版本。
- Low-memory dropout: 同一文件中的 `seeded_dropout` 版本，只保存 seed，不物化 mask tensor。
- API: [python/triton_playground/ops/dropout.py](./python/triton_playground/ops/dropout.py)
- Test: [tests/test_dropout.py](./tests/test_dropout.py)
- Benchmark: [scripts/bench_dropout.py](./scripts/bench_dropout.py)

运行测试：

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python pytest tests/test_dropout.py
```

运行单点 benchmark：

```bash
PYTHONPATH=python python3 scripts/bench_dropout.py --n-elements 16777216 --dtype float32 --p 0.5
```

运行 sweep 和曲线图：

```bash
PYTHONPATH=python python3 scripts/bench_dropout.py --sweep --plot --min-power 12 --max-power 28 --dtype float32 --p 0.5
```

## MatMul + Bias + SiLU 入口

当前已落地 fused matmul epilogue 版本：
- Kernel: [python/triton_playground/kernels/matmul_bias_silu.py](./python/triton_playground/kernels/matmul_bias_silu.py)
- API: [python/triton_playground/ops/matmul_bias_silu.py](./python/triton_playground/ops/matmul_bias_silu.py)
- Test: [tests/test_matmul_bias_silu.py](./tests/test_matmul_bias_silu.py)
- Benchmark: [scripts/bench_matmul_bias_silu.py](./scripts/bench_matmul_bias_silu.py)
- W12 report: [reports/w12-fused-ops.md](./reports/w12-fused-ops.md)

实现要点：
- 主循环沿用 W10 matmul 的 `pid_m/pid_n` grouped ordering 和 `tl.dot`。
- epilogue 直接计算 `z = A @ B + bias` 和 `SiLU(z) = z * sigmoid(z)`，避免单独物化 bias-add 中间结果。
- 当前 benchmark 的 `TFLOP/s` 只按 matmul FLOPs 估算，bias 和 SiLU 的 element-wise FLOPs 不计入。

运行测试：
```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python pytest tests/test_matmul_bias_silu.py
```

运行单点 benchmark：
```bash
PYTHONPATH=python python3 scripts/bench_matmul_bias_silu.py --m 1024 --n 1024 --k 1024 --dtype float16
```

运行 sweep 和曲线图：
```bash
PYTHONPATH=python python3 scripts/bench_matmul_bias_silu.py --sweep --plot --min-power 10 --max-power 13 --k 1024 --dtype float16
```

## LayerNorm 入口

参照官方 Layer Normalization 教程，当前已落地 affine LayerNorm 的 Triton 前向和后向：

- Kernel: [python/triton_playground/kernels/layer_norm.py](./python/triton_playground/kernels/layer_norm.py)
- API: [python/triton_playground/ops/layer_norm.py](./python/triton_playground/ops/layer_norm.py)
- Test: [tests/test_layer_norm.py](./tests/test_layer_norm.py)
- Benchmark: [scripts/bench_layer_norm.py](./scripts/bench_layer_norm.py)

实现边界：

- 归一化维度固定为最后一维，`weight` 和 `bias` 的形状必须为 `(x.shape[-1],)`。
- 当前支持 contiguous CUDA tensor，dtype 为 `float16` 或 `float32`。
- 与官方教程一致，单行 feature 使用小于等于 64KB 的 fused block；超过时需要拆分或换多 CTA 实现。
- 后向分两阶段：第一阶段每行计算 `dx` 并用 lock 累积 partial `dw/db`，第二阶段 reduce 得到最终 `dw/db`。

运行测试：

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python pytest tests/test_layer_norm.py
```

运行单点 benchmark：

```bash
PYTHONPATH=python python3 scripts/bench_layer_norm.py --rows 4096 --cols 8192 --dtype float16 --mode backward
PYTHONPATH=python python3 scripts/bench_layer_norm.py --rows 4096 --cols 8192 --dtype float16 --mode forward
```

运行 sweep 和曲线图：

```bash
PYTHONPATH=python python3 scripts/bench_layer_norm.py --sweep --plot --rows 4096 --min-cols 1024 --max-cols 16384 --cols-step 512 --dtype float16 --mode backward
```

## RMSNorm 入口

当前已落地 RMSNorm 的 production-style Triton 版本和 naive Triton 版本：

- Kernel: [python/triton_playground/kernels/rms_norm.py](./python/triton_playground/kernels/rms_norm.py)
- API: [python/triton_playground/ops/rms_norm.py](./python/triton_playground/ops/rms_norm.py)
- Test: [tests/test_rms_norm.py](./tests/test_rms_norm.py)
- Benchmark: [scripts/bench_rms_norm.py](./scripts/bench_rms_norm.py)
- W12 report: [reports/w12-fused-ops.md](./reports/w12-fused-ops.md)

实现边界：

- 归一化维度固定为最后一维，`weight` 的形状必须为 `(x.shape[-1],)`。
- 当前支持 contiguous CUDA tensor，dtype 为 `float16` 或 `float32`。
- 与 LayerNorm lab 保持一致，单行 feature 使用小于等于 64KB 的 fused block。
- production-style backward 使用 `dx + partial dweight` fused kernel 和最终 `dweight` reduce kernel。
- naive backward 使用独立 `dx` kernel 和按列块扫描所有 row 的 `dweight` reduce kernel，主要用于教学和 benchmark 对比。

运行测试：

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python pytest tests/test_rms_norm.py
```

运行单点 benchmark：

```bash
PYTHONPATH=python python3 scripts/bench_rms_norm.py --rows 4096 --cols 8192 --dtype float16 --mode backward
PYTHONPATH=python python3 scripts/bench_rms_norm.py --rows 4096 --cols 8192 --dtype float16 --mode forward
```

运行 sweep 和曲线图：

```bash
PYTHONPATH=python python3 scripts/bench_rms_norm.py --sweep --plot --rows 4096 --min-cols 1024 --max-cols 16384 --cols-step 512 --dtype float16 --mode backward
```

## 证据要求

每个 landed kernel 至少保存四类证据：

- 正确性：shape、dtype、`atol/rtol`、reference、PASS/FAIL。
- 性能：`benchmark.csv`，包含 shape、配置、耗时、吞吐或带宽。
- 映射：`program_id` 到 tile 的解释，必要时画图。
- 结论：哪些 shape 适合当前实现，哪些不适合，下一步应该改哪里。

benchmark 原始结果和图片默认放到仓库级 `out/triton/benchmarks/`；阶段复盘和人工整理后的结论放到 `reports/`，解释性材料放到 `notes/`。
