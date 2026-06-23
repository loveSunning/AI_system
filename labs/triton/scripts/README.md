# Scripts

这里放可复现命令脚本，例如安装检查、批量测试、benchmark sweep 和结果整理。

建议后续补齐：

- `check_env.py`：打印 Python、Torch、Triton、CUDA、GPU 信息。
- `run_tests.ps1` / `run_tests.sh`：统一运行 pytest。
- `run_benchmarks.ps1` / `run_benchmarks.sh`：统一导出 `benchmarks/*.csv`。
- `summarize_benchmarks.py`：把 CSV 汇总成阶段报告表格。

脚本要能从 `labs/triton` 目录直接运行，也要在 README 中写清楚输出文件位置。

当前已落地：

```bash
cd /workspace/AI_system/labs/triton
PYTHONPATH=python python scripts/bench_vector_add.py --n-elements 16777216 --dtype float32 --block-size 1024
```
