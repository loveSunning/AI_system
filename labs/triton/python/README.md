# Python Workspace

这里放 Triton lab 的可执行 Python 代码。建议保持三层边界：

- `kernels/`：只放 Triton JIT kernel 和低层 launcher。
- `ops/`：放面向实验的 Python API，例如 `rmsnorm(x, weight)`、`matmul_bias_silu(a, b, bias)`。
- `benchmarking/`：放计时、shape sweep、CSV 输出等通用工具。
- `testing/`：放 reference、误差检查、随机输入生成等测试辅助。

后续脚本可以从仓库根目录或本目录运行，但 README 和 benchmark 里要固定命令，保证结果能复现。
