# Program ID Mapping

本笔记用于记录 Triton kernel 中 `tl.program_id` 到数据 tile 的映射关系。

## Vector Add

- `pid = tl.program_id(0)`
- `offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`
- `mask = offsets < n_elements`

记录重点：最后一个 block 的 mask 是否正确，BLOCK_SIZE 如何影响 occupancy 和内存吞吐。

## Matmul

- `pid_m` 映射 C 矩阵的 M 维 tile。
- `pid_n` 映射 C 矩阵的 N 维 tile。
- grouped ordering 用于改善 L2 locality。

需要补图：

```text
program_id -> group_id -> pid_m/pid_n -> C tile
```

## Persistent Matmul

记录重点：

- tile scheduler 如何把多个 output tile 分配给有限数量的 persistent programs。
- 当前 shape 下 program 数量、SM 数量、tile 数量之间的关系。
- persistent 版本相对 baseline 的收益或退化原因。
