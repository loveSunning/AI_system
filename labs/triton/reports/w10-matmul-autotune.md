# W10 Matmul Autotune

实现位置：

- Kernel: `python/triton_playground/kernels/matmul.py`
- API: `python/triton_playground/ops/matmul.py`
- Test: `tests/test_matmul.py`
- Benchmark: `scripts/bench_matmul.py`

## 当前目标

- `triton_fixed`：固定 `BLOCK_M/N/K` 的 grouped-ordering matmul baseline。
- `triton_autotune`：参照官方 `get_cuda_autotune_config()` 风格，自动搜索多组 tile / warp / stage 配置。
- `torch`：`torch.matmul` baseline，CUDA 环境下通常走 PyTorch/cuBLAS 路径。

## 结果模板

| impl | shape | dtype | config | median_ms | p20_ms | p80_ms | TFLOP/s | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| triton_autotune | TODO | TODO | `get_cuda_autotune_config` | TODO | TODO | TODO | TODO | TODO |
| triton_fixed | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| torch | TODO | TODO | `torch_matmul` | TODO | TODO | TODO | TODO | TODO |

## 需要记录

- `pid_m/pid_n` 如何从一维 `program_id` 映射到 C 矩阵 tile。
- grouped ordering 如何改善相邻 tile 的 A/B 数据复用。
- `BLOCK_M/N/K`、`GROUP_M`、`num_warps`、`num_stages` 对 TFLOP/s 的影响。
- autotune winner：如果当前 Triton 版本能打印 winner，记录具体配置；否则记录 sweep 曲线中最快的一行。
