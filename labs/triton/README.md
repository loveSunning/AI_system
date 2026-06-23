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

## W09 Vector Add 入口

当前 `vector add` 已按三层结构落地：

- Kernel: [python/triton_playground/kernels/vector_add.py](./python/triton_playground/kernels/vector_add.py)
- API: [python/triton_playground/ops/vector_add.py](./python/triton_playground/ops/vector_add.py)
- Test: [tests/test_vector_add.py](./tests/test_vector_add.py)
- Benchmark: [scripts/bench_vector_add.py](./scripts/bench_vector_add.py)

运行测试：

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python pytest tests/test_vector_add.py
```

运行 benchmark：

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python python scripts/bench_vector_add.py --n-elements 16777216 --dtype float32 --block-size 1024
```

## 证据要求

每个 landed kernel 至少保存四类证据：

- 正确性：shape、dtype、`atol/rtol`、reference、PASS/FAIL。
- 性能：`benchmark.csv`，包含 shape、配置、耗时、吞吐或带宽。
- 映射：`program_id` 到 tile 的解释，必要时画图。
- 结论：哪些 shape 适合当前实现，哪些不适合，下一步应该改哪里。

这些证据分别放到 `benchmarks/`、`reports/` 和 `notes/` 中。
