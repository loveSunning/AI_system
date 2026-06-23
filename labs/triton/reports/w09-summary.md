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

## Program ID 解释

`vector_add` 使用一维 grid：

```text
pid = tl.program_id(0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
```

每个 Triton program 处理连续的 `BLOCK_SIZE` 个元素。最后一个 program 可能超过真实长度，所以所有 `tl.load` 和 `tl.store` 都带同一个 `mask`。
