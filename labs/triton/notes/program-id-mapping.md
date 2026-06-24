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

当前 W10 matmul 使用一维 `program_id` 加 grouped ordering：

```text
num_pid_m = ceil(M / BLOCK_M)
num_pid_n = ceil(N / BLOCK_N)
num_pid_in_group = GROUP_M * num_pid_n

group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_M)

pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

直觉：

- C 的每个 output tile 是 `[BLOCK_M, BLOCK_N]`。
- 同一个 `group_id` 内优先扫一组相邻的 M tile，让多个 C tile 更容易复用 A/B 的 L2 cache 数据。
- `GROUP_M=1` 接近普通 row-major tile ordering；更大的 `GROUP_M` 可能改善 L2 locality，但也要结合 shape 和 tile 配置 benchmark。

## Fused Softmax

- `row_id = tl.program_id(0)` 映射输入矩阵的一行。
- `offsets = tl.arange(0, BLOCK_SIZE)` 映射这一行内的列。
- `mask = offsets < n_cols` 处理列数不是 2 的幂的情况。
- `tl.load(..., other=-inf)` 让 padding 列在 `tl.max` 和 `tl.sum` 中不影响真实 softmax。

记录重点：`BLOCK_SIZE` 是 `n_cols` 的 next power of 2。每个 program 在一行内部完成 `max -> exp -> sum -> div -> store`，减少中间张量读写。

## Persistent Matmul

记录重点：

- tile scheduler 如何把多个 output tile 分配给有限数量的 persistent programs。
- 当前 shape 下 program 数量、SM 数量、tile 数量之间的关系。
- persistent 版本相对 baseline 的收益或退化原因。
