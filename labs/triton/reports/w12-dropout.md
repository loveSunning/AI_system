# W12 Dropout

实现位置：

- Kernel: `python/triton_playground/kernels/dropout.py`
- API: `python/triton_playground/ops/dropout.py`
- Test: `tests/test_dropout.py`
- Benchmark: `scripts/bench_dropout.py`

## 实现口径

- `triton_mask`：普通 dropout，输入显式 `keep_mask`，输出 `where(keep, x / (1 - p), 0)`。
- `triton_seeded_low_memory`：low-memory dropout，输入只保存 `seed`，kernel 内用 `tl.rand(seed, offsets)` 生成 mask。
- `torch`：`torch.nn.functional.dropout(x, p=p, training=True)`。

## 结果模板

| impl | shape | dtype | p | config | median_ms | p20_ms | p80_ms | throughput | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| triton_mask | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| triton_seeded_low_memory | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| torch | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

## 需要记录

- 显式 mask 版本需要额外 mask tensor，内存状态更大。
- low-memory seeded 版本只保存 seed，同 seed 多次调用产生同一 mask。
- PyTorch dropout 作为成熟 baseline，但随机 mask 不和 Triton seed bitwise 对齐，只比较性能和统计性质。
