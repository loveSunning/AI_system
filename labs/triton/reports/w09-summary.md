# W09 Summary

## Vector Add

实现位置：

- Kernel: `python/triton_playground/kernels/vector_add.py`
- API: `python/triton_playground/ops/vector_add.py`
- Test: `tests/test_vector_add.py`
- Benchmark: `scripts/bench_vector_add.py`

## 当前结论

TODO: 跑完 benchmark 后填写。

| op | impl | shape | dtype | config | avg_ms | throughput | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vector_add | triton | TODO | TODO | `BLOCK_SIZE=1024` | TODO | TODO | TODO |
| vector_add | torch | TODO | TODO | `torch_add` | TODO | TODO | TODO |

## Fused Softmax

实现位置：

- Kernel: `python/triton_playground/kernels/fused_softmax.py`
- API: `python/triton_playground/ops/fused_softmax.py`
- Test: `tests/test_fused_softmax.py`
- Benchmark: `scripts/bench_fused_softmax.py`

当前 fused softmax 已在 RTX 4090 D 上通过 correctness，并完成 `4096x1024 fp32` benchmark。Triton 入门版达到 PyTorch baseline 约 `46.7%` 的估算吞吐，说明 row-wise program、mask 和 block 内 reduction 路径已经跑通；性能差距主要来自 PyTorch/CUDA 内置 softmax 更成熟的实现与调度优化。

| op | impl | shape | dtype | config | avg_ms | throughput | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fused_softmax | triton | `4096x1024` | `float32` | `BLOCK_SIZE=1024;num_warps=4` | `0.063035` | `532.315 GB/s_est` | PASS |
| fused_softmax | torch | `4096x1024` | `float32` | `torch_softmax` | `0.029420` | `1140.550 GB/s_est` | PASS |
| fused_softmax | naive | TODO | TODO | `torch_max_exp_sum_div` | TODO | TODO | TODO |

## Program ID 解释

`vector_add` 使用一维 grid：

```text
pid = tl.program_id(0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
```

每个 Triton program 处理连续的 `BLOCK_SIZE` 个元素。最后一个 program 可能超过真实长度，所以所有 `tl.load` 和 `tl.store` 都带同一个 `mask`。

`fused_softmax` 也是一维 grid，但 `program_id` 映射到 row：

```text
row_id = tl.program_id(0)
offsets = tl.arange(0, BLOCK_SIZE)
mask = offsets < n_cols
```

每个 Triton program 处理一整行。padding 列用 `other=-inf` 参与 load，这样不会影响行最大值和归一化分母。
