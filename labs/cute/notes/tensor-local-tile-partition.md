# Tensor / local_tile / partition

目标：能解释 global memory tensor、shared memory tensor、register fragment 的 layout 映射，并能把它们和 GEMM 数据流对应起来。

对应官方文档：

- https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/00_quickstart.html
- https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/03_tensor.html
- https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html

## 一句话模型

CuTe `Tensor = Engine + Layout`。

- `Engine` 管“数据在哪里”：global memory pointer、shared memory pointer、register owning array，或者普通 iterator。
- `Layout` 管“逻辑坐标怎么变成线性 offset”：例如 `(m,k) -> m * stride_m + k * stride_k`。
- `Tensor(coord)` 的本质是：先用 `Layout(coord)` 算 offset，再从 `Engine` 持有的 iterator 上做偏移和解引用。

所以学习 Tensor 时不要先想“矩阵长什么样”，先问三个问题：

- 逻辑 shape 是什么？
- stride 把坐标映射到了哪个 offset？
- engine 指向哪一种 memory space？

## 三种 Tensor

### Global Memory Tensor

GEMM kernel 入口拿到的是普通指针，例如 `A`、`B`、`C`。CuTe 会用 `make_gmem_ptr` 给指针加上 global memory tag，再用 `make_tensor` 配上 shape/stride：

```cpp
auto gmem_layout = make_layout(make_shape(Int<8>{}, Int<12>{}),
                               make_stride(Int<12>{}, Int<1>{}));
auto gA = make_tensor(make_gmem_ptr(global_storage.data()), gmem_layout);
```

此时：

```text
gA: gmem_ptr[...] o (_8,_12):(_12,_1)
```

解释：

- `(_8,_12)` 是逻辑 shape，也就是 8 行、12 列。
- `(_12,_1)` 是 row-major stride。
- `gA(5,7)` 的 offset 是 `5 * 12 + 7 * 1 = 67`。
- 如果测试数据写成 `value = m * 100 + k`，那么 `gA(5,7) == 507`。

GEMM 中常见的 full tensor 语义是：

```cpp
Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA);  // (M,K)
Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB);  // (N,K)
Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC);  // (M,N)
```

注意 CuTe GEMM 注释里常把 `B` 写成 `(N,K)`，不是传统 BLAS 叙述里的 `(K,N)`。这是为了让 A/B/C 的模式统一写成 A:`(M,K)`、B:`(N,K)`、C:`(M,N)`。

## local_tile：从 full tensor 切出 CTA tile

`local_tile(tensor, tiler, coord)` 的语义是：先把 full tensor 按 `tiler` 分块，再用 `coord` 选择其中一个块，最后保留 tile 内部的 shape。

示例：

```cpp
auto cta_tiler = Shape<_4, _3>{};
auto cta_coord = make_coord(1, 2);
auto tAgA = local_tile(gA, cta_tiler, cta_coord);
```

对 `gA: (_8,_12):(_12,_1)` 来说：

- `Shape<_4,_3>` 表示每个 tile 是 4 行、3 列。
- `cta_coord = (1,2)` 表示选择第 1 个 M tile、第 2 个 K tile。
- tile 起点是 `(m,k) = (1 * 4, 2 * 3) = (4,6)`。
- `tAgA(1,1)` 对应 full tensor 的 `gA(5,7)`。

输出类似：

```text
tAgA: gmem_ptr[...] o (_4,_3):(_12,_1)
```

关键点是：`tAgA` 的 data pointer 已经偏移到 tile 起点，layout 仍描述 tile 内坐标如何移动。也就是说：

```text
tAgA(1,1) == gA(4 + 1, 6 + 1) == gA(5,7)
```

在实现上，`local_tile` 是 `inner_partition` 的别名：它保留 tiler 内部的 tile mode，切掉外层 remainder mode。官方文档常把它用于 CTA 级别：

```cpp
auto ctaA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
auto ctaB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
auto ctaC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)
```

这里的 `Step` 是 projection：同一个 `(M,N,K)` CTA tiler 被投影到 A/B/C 各自需要的模式上。

## Shared Memory Tensor

shared memory tensor 通常来自静态 shared buffer：

```cpp
auto smem_layout = make_layout(make_shape(Int<4>{}, Int<3>{}),
                               make_stride(Int<1>{}, Int<4>{}));
std::array<int, decltype(cosize(smem_layout))::value> shared_storage{};
auto sA = make_tensor(make_smem_ptr(shared_storage.data()), smem_layout);
```

输出类似：

```text
sA: smem_ptr[...] o (_4,_3):(_1,_4)
```

解释：

- `sA` 的 shape 仍是 tile shape `(_4,_3)`。
- stride 变成了 `(_1,_4)`，也就是 column-major 风格。
- `sA(1,1)` 的 offset 是 `1 * 1 + 1 * 4 = 5`。

这正是 shared memory layout 要解决的问题：global tile 的逻辑元素可以被复制到 shared tile 的同一个逻辑坐标，但 shared memory 的物理布局可以不同，用来优化 vectorized copy、shared bank conflict、后续 MMA 读布局等。

```cpp
for (int i = 0; i < size(sA); ++i) {
  sA(i) = tAgA(i);
}
```

这段按线性逻辑顺序复制，用来教学足够清楚。后续进入 `TiledCopy` 时，会把这个循环替换成“每个 thread 拿自己的 partition，再用 copy atom 搬运”。

## Register Fragment

register fragment 是线程私有的小 tensor，通常是 owning tensor：

```cpp
auto rA = make_tensor<int>(Shape<_4, _3>{}, LayoutRight{});
```

输出类似：

```text
rA: ptr[...] o (_4,_3):(_3,_1)
```

解释：

- `ptr[...]` 说明它不是 gmem/smem tag，而是普通 owning storage。
- shape 是 fragment 逻辑 shape。
- stride 是 register fragment 内部的线性排列方式。

真实 GEMM 里，register fragment 往往不是整个 CTA tile，而是某个 thread、warp 或 MMA atom 分到的一小片数据。你现在先掌握“register tensor 也有 layout”这一点：即使数据在寄存器里，CuTe 仍然用同一套坐标系统解释它。

## local_partition：从 tile 分给 thread

`local_partition(tensor, thread_layout, thr_idx)` 的语义是：用一个 thread layout 描述 thread id 如何落在 tile 坐标上，然后取出某个 thread 负责的 subtensor。

示例：

```cpp
auto thr_layout = make_layout(make_shape(Int<2>{}, Int<3>{}),
                              make_stride(Int<3>{}, Int<1>{}));
int thr_idx = 4;
auto tAsA = local_partition(sA, thr_layout, thr_idx);
```

输出类似：

```text
thr_layout: (_2,_3):(_3,_1)
tAsA: smem_ptr[...] o (_2,_1):(_2,_0)
```

解释：

- `thr_layout` 把 6 个 thread 排成 `(2,3)` 的逻辑网格。
- `thr_idx = 4` 在 row-major thread layout 里对应 thread 坐标 `(1,1)`。
- `local_partition` 保留的是 tile 外剩下的 repetition mode，所以这个 thread 得到的 subtensor shape 是 `(_2,_1)`。
- 在 demo 中，`tAsA(0,0)` 对应 `sA(1,1)`，值仍然是 `507`。

和 `local_tile` 对比：

| 操作 | 常用层级 | 保留什么 | 可以怎么记 |
| --- | --- | --- | --- |
| `local_tile` / `inner_partition` | CTA/block | tile 内部 | 给这个 CTA 一整块 tile |
| `local_partition` / `outer_partition` | thread/warp | 某个 agent 的重复访问片段 | 给这个 thread 它负责的元素 |

## 一条完整映射链

本仓库 demo：

```text
labs/cute/examples/cute_tensor_tile_demo.cu
```

构造了下面这条链：

```text
gA full tensor
  shape  = (_8,_12)
  stride = (_12,_1)
  value  = gA(5,7) = 507

local_tile(gA, Shape<_4,_3>, make_coord(1,2))
  tAgA shape  = (_4,_3)
  tAgA origin = gA(4,6)
  tAgA(1,1)  = gA(5,7) = 507

sA shared tensor
  shape  = (_4,_3)
  stride = (_1,_4)
  sA(1,1) = 507

rA register tensor
  shape  = (_4,_3)
  stride = (_3,_1)
  rA(1,1) = 507

local_partition(sA, thr_layout, 4)
  thread 4 gets a subtensor
  tAsA(0,0) = sA(1,1) = 507
```

运行：

```powershell
cmake --build --preset windows-vs2022-cuda-release --config Release --target cute_tensor_tile_demo
.\out\build\windows-vs2022-cuda-release\labs\cute\Release\cute_tensor_tile_demo.exe
```

```bash
cmake --build --preset linux-make-cuda-release --target cute_tensor_tile_demo -j"$(nproc)"
./out/build/linux-make-cuda-release/labs/cute/cute_tensor_tile_demo
```

预期最后看到：

```text
tensor/local_tile/partition check passed
```

## 学习检查

你能独立回答这些问题，就说明这一节过关：

- `Tensor` 的 `Engine` 和 `Layout` 分别负责什么？
- 为什么 `make_gmem_ptr` 和 `make_smem_ptr` 只是给 pointer 加 memory-space 语义，而不是改变 shape/stride？
- `local_tile(gA, Shape<_4,_3>{}, make_coord(1,2))` 的 tile 起点为什么是 `(4,6)`？
- 为什么 `tAgA(1,1)`、`sA(1,1)`、`rA(1,1)` 可以读到同一个逻辑值，但它们的 stride 不一样？
- `local_tile` 和 `local_partition` 一个保留 tile，一个保留 per-thread subtensor，这句话能不能用 `zipped_divide + slice` 解释出来？

下一步进入 `TiledCopy` 时，把这份笔记里的教学循环：

```cpp
for (int i = 0; i < size(sA); ++i) {
  sA(i) = tAgA(i);
}
```

替换成：

```text
thread layout -> per-thread partition -> Copy_Atom / TiledCopy
```

这就是从“能解释 layout 映射”走向“能写高效 copy pipeline”的过渡。
