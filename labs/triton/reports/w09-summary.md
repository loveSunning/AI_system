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

TODO: 跑完 benchmark 后填写。

| op | impl | shape | dtype | config | avg_ms | throughput | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fused_softmax | triton | TODO | TODO | `BLOCK_SIZE=next_power_of_2(n_cols)` | TODO | TODO | TODO |
| fused_softmax | torch | TODO | TODO | `torch_softmax` | TODO | TODO | TODO |

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
