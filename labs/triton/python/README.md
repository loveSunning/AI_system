# Python Workspace

这里放 Triton lab 的可执行 Python 代码。建议保持三层边界：

- `kernels/`：只放 Triton JIT kernel 和低层 launcher。
- `ops/`：放面向实验的 Python API，例如 `rmsnorm(x, weight)`、`matmul_bias_silu(a, b, bias)`。
- `benchmarking/`：放计时、shape sweep、CSV 输出等通用工具。
- `testing/`：放 reference、误差检查、随机输入生成等测试辅助。

后续脚本可以从仓库根目录或本目录运行，但 README 和 benchmark 里要固定命令，保证结果能复现。

## 已落地入口

- `triton_playground.kernels.vector_add`：W09 vector add 的 Triton JIT kernel 和 launcher。
- `triton_playground.kernels.fused_softmax`：W09 fused softmax 的 row-wise Triton JIT kernel 和 launcher。
- `triton_playground.kernels.matmul`：W10 matmul 的 fixed/autotuned Triton JIT kernel 和 CUDA autotune config。
- `triton_playground.kernels.grouped_gemm`：W11 grouped GEMM 的 fixed-CTA device-side scheduler kernel。
- `triton_playground.kernels.persistent_matmul`：W11 persistent matmul 的非 TMA fixed/autotuned kernel。
- `triton_playground.kernels.dropout`：显式 mask dropout 和 low-memory seeded dropout。
- `triton_playground.kernels.layer_norm`：affine LayerNorm 前向、`dx` 后向和 `dw/db` reduce kernel。
- `triton_playground.kernels.rms_norm`：production-style 和 naive RMSNorm 前向、`dx` 后向和 `dweight` reduce kernel。
- `triton_playground.kernels.matmul_bias_silu`：W12 matmul mainloop + fused bias/SiLU epilogue kernel。
- `triton_playground.kernels.online_softmax`：W13 两遍扫描 online softmax kernel。
- `triton_playground.kernels.attention_forward`：W14 stepwise attention forward 的 QK、softmax、PV kernels。
- `triton_playground.kernels.fused_attention`：W14 online softmax fused attention forward kernel。
- `triton_playground.ops.vector_add`：面向测试、benchmark 和后续实验调用的 API。
- `triton_playground.ops.fused_softmax`：面向测试、benchmark 和后续实验调用的 API。
- `triton_playground.ops.matmul`：面向测试、benchmark 和后续实验调用的 API。
- `triton_playground.ops.grouped_gemm`：面向多 GEMM 同次 launch 实验的 API。
- `triton_playground.ops.persistent_matmul`：面向 4090D 友好 persistent matmul 实验的 API。
- `triton_playground.ops.dropout`：面向测试、benchmark 和后续实验调用的 API。
- `triton_playground.ops.layer_norm`：对齐 `torch.nn.functional.layer_norm` 的最后一维 affine LayerNorm API。
- `triton_playground.ops.rms_norm`：面向 W12 fused ops 的 RMSNorm production/naive API。
- `triton_playground.ops.matmul_bias_silu`：面向 W12 fused epilogue 实验的 `SiLU(A @ B + bias)` API。
- `triton_playground.ops.online_softmax`：面向 W13 online softmax 实验的 Triton API 和 Torch 教学版。
- `triton_playground.ops.attention_forward`：面向 W14 attention forward 实验的 Torch reference 和 Triton stepwise API。
- `triton_playground.ops.fused_attention`：面向 W14 fused attention forward 实验的 Triton API 和 Torch reference 包装。
