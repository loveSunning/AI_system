# Python Workspace

这里放 Triton lab 的可执行 Python 代码。建议保持三层边界：

- `kernels/`：只放 Triton JIT kernel 和低层 launcher。
- `ops/`：放面向实验的 Python API，例如 `rmsnorm(x, weight)`、`matmul_bias_silu(a, b, bias)`。
- `benchmarking/`：放计时、shape sweep、CSV 输出等通用工具。
- `testing/`：放 reference、误差检查、随机输入生成等测试辅助。

后续脚本可以从仓库根目录或本目录运行，但 README 和 benchmark 里要固定命令，保证结果能复现。

## 已落地入口

- `triton_playground.kernels.vector_add`：W09 vector add 的 Triton JIT kernel 和 launcher。
- `triton_playground.ops.vector_add`：面向测试、benchmark 和后续实验调用的 API。
