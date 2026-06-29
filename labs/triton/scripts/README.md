# Scripts

这里放可复现命令脚本，例如安装检查、批量测试、benchmark sweep 和结果整理。

建议后续补齐：

- `check_env.py`：打印 Python、Torch、Triton、CUDA、GPU 信息。
- `run_tests.ps1` / `run_tests.sh`：统一运行 pytest。
- `run_benchmarks.ps1` / `run_benchmarks.sh`：统一导出 `out/triton/benchmarks/*.csv`。
- `summarize_benchmarks.py`：把 CSV 汇总成阶段报告表格。

脚本要能从 `labs/triton` 目录直接运行，也要在 README 中写清楚输出文件位置。

当前已落地：

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python python3 scripts/bench_vector_add.py --n-elements 16777216 --dtype float32 --block-size 1024
PYTHONPATH=python python3 scripts/bench_fused_softmax.py --rows 4096 --cols 1024 --dtype float32
PYTHONPATH=python python3 scripts/bench_matmul.py --m 4096 --n 4096 --k 4096 --dtype float16
PYTHONPATH=python python3 scripts/bench_grouped_gemm.py --group-size 4 --m 1024 --n 1024 --k 1024 --pattern vary_n
PYTHONPATH=python python3 scripts/bench_dropout.py --n-elements 16777216 --dtype float32 --p 0.5
PYTHONPATH=python python3 scripts/bench_layer_norm.py --rows 4096 --cols 8192 --dtype float16 --mode backward
```

生成 sweep 数据和 Triton `perf_report` 图：

```bash
PYTHONPATH=python python3 scripts/bench_vector_add.py --sweep --plot --min-power 12 --max-power 28 --dtype float32
PYTHONPATH=python python3 scripts/bench_fused_softmax.py --sweep --plot --rows 4096 --min-cols-power 7 --max-cols-power 12 --dtype float32
PYTHONPATH=python python3 scripts/bench_matmul.py --sweep --plot --min-power 8 --max-power 12 --dtype float16
PYTHONPATH=python python3 scripts/bench_grouped_gemm.py --sweep --plot --min-group-size 1 --max-group-size 8 --m 1024 --n 1024 --k 1024 --pattern vary_n
PYTHONPATH=python python3 scripts/bench_dropout.py --sweep --plot --min-power 12 --max-power 28 --dtype float32 --p 0.5
PYTHONPATH=python python3 scripts/bench_layer_norm.py --sweep --plot --rows 4096 --min-cols 1024 --max-cols 16384 --cols-step 512 --dtype float16 --mode backward
```

生成 Triton `perf_report` 图：

```bash
PYTHONPATH=python python3 scripts/bench_vector_add.py --plot --min-power 12 --max-power 28 --dtype float32
PYTHONPATH=python python3 scripts/bench_fused_softmax.py --plot --rows 4096 --min-cols-power 7 --max-cols-power 12 --dtype float32
PYTHONPATH=python python3 scripts/bench_matmul.py --plot --min-power 8 --max-power 12 --dtype float16
PYTHONPATH=python python3 scripts/bench_grouped_gemm.py --plot --min-group-size 1 --max-group-size 8 --m 1024 --n 1024 --k 1024 --pattern vary_n
PYTHONPATH=python python3 scripts/bench_dropout.py --plot --min-power 12 --max-power 28 --dtype float32 --p 0.5
PYTHONPATH=python python3 scripts/bench_layer_norm.py --plot --rows 4096 --min-cols 1024 --max-cols 16384 --cols-step 512 --dtype float16 --mode backward
```

生成 sweep 数据：

```bash
PYTHONPATH=python python3 scripts/bench_vector_add.py --sweep --min-power 12 --max-power 28 --dtype float32
PYTHONPATH=python python3 scripts/bench_fused_softmax.py --sweep --rows 4096 --min-cols-power 7 --max-cols-power 12 --dtype float32
PYTHONPATH=python python3 scripts/bench_matmul.py --sweep --min-power 8 --max-power 12 --dtype float16
PYTHONPATH=python python3 scripts/bench_grouped_gemm.py --sweep --min-group-size 1 --max-group-size 8 --m 1024 --n 1024 --k 1024 --pattern vary_n
PYTHONPATH=python python3 scripts/bench_dropout.py --sweep --min-power 12 --max-power 28 --dtype float32 --p 0.5
PYTHONPATH=python python3 scripts/bench_layer_norm.py --sweep --rows 4096 --min-cols 1024 --max-cols 16384 --cols-step 512 --dtype float16 --mode backward
```

默认输出位置：

```text
/workspace/AI_system/out/triton/benchmarks/w09_vector_add_softmax.csv
/workspace/AI_system/out/triton/benchmarks/w10_matmul.csv
/workspace/AI_system/out/triton/benchmarks/w11_grouped_gemm.csv
/workspace/AI_system/out/triton/benchmarks/w12_dropout.csv
/workspace/AI_system/out/triton/benchmarks/w12_layer_norm.csv
/workspace/AI_system/out/triton/benchmarks/plots/
```

从 CSV 稳定生成 PNG 图：

```bash
python3 scripts/plot_w09_benchmarks.py --dtype float32
python3 scripts/plot_w09_benchmarks.py --op fused_softmax --metric throughput --dtype float32
```
