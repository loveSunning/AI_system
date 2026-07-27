# CuTe TiledCopy: Global -> Shared -> Register

这份笔记对应学习计划 W16。它把 W15 已经掌握的 `Layout`、`Tensor`、`local_tile`
和 `local_partition` 继续推进到真实 CUDA kernel 中的数据搬运：

```text
global tensor
    -> CTA tile
    -> per-thread source partition
    -> per-thread destination partition
    -> shared tensor
    -> register fragment
```

配套程序：

```text
examples/cute_copy_g2s_naive.cu
examples/cute_copy_g2s_cpasync.cu
examples/cute_copy_s2r.cu
examples/cute_smem_swizzle_demo.cu
```

## 1. 固定的 GEMM 参数

W16 延续前一阶段的 GEMM 参数：

```text
M = N = K = 2048
CTA tile = 128 x 128 x 32
A tile = 128 x 32
B tile = 128 x 32
threads = 128
Element = half
```

一个 A tile 包含：

```text
128 * 32 = 4096 half
4096 * 2 = 8192 bytes
```

分给 128 个线程后：

```text
4096 / 128 = 32 half/thread
```

如果使用 128-bit copy：

```text
128 bit = 16 bytes = 8 half
32 / 8 = 4 copy instructions/thread
```

## 2. 从 local_partition 到 TiledCopy

手写线程划分：

```cpp
Tensor tAgA = local_partition(gA, ThreadLayout{}, threadIdx.x);
Tensor tAsA = local_partition(sA, ThreadLayout{}, threadIdx.x);
copy(tAgA, tAsA);
```

这里的 `ThreadLayout` 只回答：

```text
thread_id -> tile coordinate
```

`TiledCopy` 增加了 value mode：

```text
(thread_id, value_id) -> tile coordinate
```

对应代码：

```cpp
auto tiled_copy = make_tiled_copy(
    Copy_Atom<UniversalCopy<uint128_t>, half_t>{},
    Layout<Shape<_32, _4>, Stride<_4, _1>>{},
    Layout<Shape<_1, _8>>{});
```

三个参数分别表示：

1. `Copy_Atom`：一次底层 copy instruction 搬什么类型、多少位。
2. `ThrLayout`：128 个线程如何排列到 tile 上。
3. `ValLayout`：每个线程一次 instruction 内的 value 如何排列。

这个配置产生：

```text
Tiler_MN       = (32,32)
TiledLayout_TV = ((4,32),8):((256,1),32)
```

一个 `TiledCopy` tile 覆盖 `32x32=1024` 个 half。完整 `128x32` A tile
沿 M 方向重复 4 次，所以每个线程最终得到 4 组 8 个连续 half。

例如：

```text
thread 0:
(0,0..7), (32,0..7), (64,0..7), (96,0..7)

thread 1:
(0,8..15), (32,8..15), (64,8..15), (96,8..15)

thread 127:
(31,24..31), (63,24..31), (95,24..31), (127,24..31)
```

`cute_copy_g2s_naive` 会枚举全部 128 个线程的坐标并验证：

```text
coverage=4096/4096
duplicates=0
missing=0
```

## 3. Copy_Atom、TiledCopy、ThrCopy

三者处于不同层级：

```text
Copy_Atom
  一次 copy instruction 的线程/value 映射

TiledCopy
  把 Copy_Atom 平铺成一个协作线程组

ThrCopy
  从 TiledCopy 中选定一个具体 thread_id
```

标准调用链：

```cpp
ThrCopy thr_copy = tiled_copy.get_slice(threadIdx.x);

Tensor tAgA = thr_copy.partition_S(gA);
Tensor tAsA = thr_copy.partition_D(sA);

copy(tiled_copy, tAgA, tAsA);
```

`get_slice` 不读取数据。它只是把当前 `threadIdx.x` 绑定到 `TiledCopy`。

`partition_S` 和 `partition_D` 也不执行 copy。它们按照 Copy Atom 的 source/destination
映射分别生成 per-thread tensor view。真正发出 load/store instruction 的是 `copy`。

## 4. partition_S 与 partition_D

可以把它们理解成：

```text
partition_S(source tensor)
    -> 当前线程应该从源 tensor 的哪些逻辑坐标读取

partition_D(destination tensor)
    -> 当前线程应该向目标 tensor 的哪些逻辑坐标写入
```

虽然 global 和 shared tensor 的物理 layout 可以不同，但是 source partition 和
destination partition 的逻辑 shape 必须兼容。这样同一个 `(copy_value, repeat...)`
在 source 和 destination 中表示同一个逻辑元素。

典型结果：

```text
tAgA: (CPY, CPY_M, CPY_K)
tAsA: (CPY, CPY_M, CPY_K)
```

其中：

- `CPY` 是 Copy Atom 一次 instruction 内的 value mode。
- `CPY_M/CPY_K` 是 Copy Atom 在完整 tile 上重复后的 rest modes。

## 5. scalar 与 128-bit vector copy

`cute_copy_g2s_naive` 包含三条路径：

```text
scalar local_partition
scalar TiledCopy
128-bit TiledCopy
```

标量 `TiledCopy`：

```cpp
Copy_Atom<UniversalCopy<half_t>, half_t>
```

一次只复制一个 half，因而每线程需要约 32 条 copy。

向量化 `TiledCopy`：

```cpp
Copy_Atom<UniversalCopy<uint128_t>, half_t>
```

一次复制 8 个 half，每线程只需要 4 条 copy。

向量化成立需要同时满足：

- global 地址按 16 bytes 对齐。
- shared 地址按 16 bytes 对齐。
- value mode 对应连续物理地址。
- leading dimension 保持每一行的起始地址对齐。
- 边界处理不能让 16-byte instruction 访问未分配内存。

`ValLayout` 描述逻辑 value 顺序，但最终能否生成向量指令还要看 source/destination
layout 的物理连续性和 alignment。

## 6. global memory coalescing

本例的 global A layout 是：

```text
(M,K):(lda,1)
```

因此 K 是连续维度。128-bit value vector 被放在 K 维：

```cpp
Layout<Shape<_1, _8>>
```

一个 warp 中每 4 个线程共同覆盖一行的 32 个 half：

```text
thread group 0: K=0..7
thread group 1: K=8..15
thread group 2: K=16..23
thread group 3: K=24..31
```

如果把 8-value vector 放到 M 维，那么逻辑上仍可能覆盖完整 tile，但 global 地址会按
`lda` 跳跃，不再是一次连续 16-byte load。

## 7. cp.async

同步向量 copy：

```cpp
Copy_Atom<UniversalCopy<uint128_t>, half_t>
```

异步 global-to-shared copy：

```cpp
Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>
Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, half_t>
```

调用顺序：

```cpp
copy(tiled_copy, tAgA, tAsA);
cp_async_fence();
cp_async_wait<0>();
__syncthreads();
```

含义：

```text
copy
  当前线程发出 cp.async instruction

cp_async_fence
  当前线程关闭并提交这一组 cp.async

cp_async_wait<0>
  当前线程等待自己提交的所有 group 完成

__syncthreads
  等待 CTA 中所有线程到达屏障，使 shared tile 可被整个 CTA 使用
```

`cp_async_wait<0>()` 只约束当前线程发出的异步操作，不能替代 CTA barrier。
`__syncthreads()` 也不能代替 `cp_async_wait` 来完成异步 copy group。

缓存策略：

- `CACHEALWAYS`：允许数据缓存在可用 cache 层级。
- `CACHEGLOBAL`：对 16-byte copy 使用 global/L2 缓存倾向。

单次 copy demo 很难体现 `cp.async` 的真正优势。它的主要价值是在多 stage GEMM 中让：

```text
global -> shared
shared -> register
MMA compute
```

发生重叠。多 stage pipeline 属于 W18。

## 8. Shared -> Register

`cute_copy_s2r` 使用相同的 TiledCopy 映射：

```cpp
ThrCopy thr_copy = tiled_copy.get_slice(threadIdx.x);
Tensor tXsA = thr_copy.partition_S(sA);
Tensor tXrA = make_fragment_like(tXsA);
copy(tiled_copy, tXsA, tXrA);
```

`tXsA` 是 shared-memory view，仍然引用 shared storage。

`tXrA` 是 owning tensor。它具有静态 shape，并在 kernel 的线程作用域中创建，因此通常
由编译器放入寄存器。寄存器是否发生 spill 最终仍需查看编译器资源报告或 profiler。

本例把所有 fragment 写回 global，并同时写回每个 value 对应的 `(m,k)`，验证：

```text
4096/4096 coordinates covered
duplicates=0
fragment value == source value
```

## 9. Shared-memory bank conflict

`cute_smem_swizzle_demo` 使用 `32x32 float`，让一个 warp 执行：

```text
lane t reads logical (t,column)
```

对于 4-byte float：

```text
bank = physical_offset % 32
```

普通 row-major：

```text
layout = (32,32):(32,1)
offset(t,0) = t * 32
bank(t,0) = 0
```

32 个 lane 全部命中 bank 0，形成 32-way conflict。

Padding：

```text
layout = (32,32):(33,1)
offset(t,0) = t * 33
bank(t,0) = t
```

Swizzle：

```cpp
composition(Swizzle<5,0,5>{}, PlainLayout{})
```

它把 row offset 的高 5 bit XOR 到低 5 bit：

```text
plain offset  = t << 5
swizzle offset = (t << 5) XOR t
bank = t
```

因此逻辑坐标和值不变，但物理 offset 和 bank 分布发生变化。

三种 layout 的 storage size：

```text
plain    cosize = 1024
padded   cosize = 1055
swizzle  cosize = 1024
```

Padding 消除了冲突，但增加了 shared storage；Swizzle 在这个例子中不增加 `cosize`。

## 10. Ragged Tile 与 Predication

非整除测试使用：

```text
logical M = 2053
logical K = 2051
CTA coord = (16,64)
valid part of 128x32 tile = 5x3 = 15 elements
```

先构造 coordinate tensor：

```cpp
Tensor cA = make_identity_tensor(problem_shape);
Tensor pA = cute::lazy::transform(cA, [&](auto coord) {
  return elem_less(coord, problem_shape);
});
```

然后对 data tensor 和 predicate tensor 应用相同的 `local_tile` 与 `partition_S`：

```cpp
Tensor gP = local_tile(pA, BlockShape{}, cta_coord);
Tensor tAgP = thr_copy.partition_S(gP);
copy_if(tiled_copy, tAgP, tAgA, tAsA);
```

128-bit Copy Atom 的 predicate 粒度是整条 16-byte instruction。测试使用 padded allocation：

```text
logical K = 2051
lda = 2056
```

这样末尾 vector 中逻辑无效的 lane 仍处于已分配、已清零的 padding 内存中。shared tile
在 copy 前也被清零，因此最终验证得到：

```text
valid=15
zero-filled=4081
```

## 11. 直接构建和运行

Windows：

```powershell
cmake -S . --preset windows-vs2022-cuda-release -DAI_SYSTEM_CUTLASS_ROOT="D:\workspace\learing\AI_system\3rdparty\cutlass"

cmake --build --preset windows-vs2022-cuda-release --config Release --target cute_copy_g2s_naive cute_copy_g2s_cpasync cute_copy_s2r cute_smem_swizzle_demo

.\out\build\windows-vs2022-cuda-release\labs\cute\Release\cute_copy_g2s_naive.exe
.\out\build\windows-vs2022-cuda-release\labs\cute\Release\cute_copy_g2s_cpasync.exe
.\out\build\windows-vs2022-cuda-release\labs\cute\Release\cute_copy_s2r.exe
.\out\build\windows-vs2022-cuda-release\labs\cute\Release\cute_smem_swizzle_demo.exe
```

Linux / WSL：

```bash
cmake -S . --preset linux-make-cuda-release -DAI_SYSTEM_CUTLASS_ROOT="${PWD}/3rdparty/cutlass"

cmake --build --preset linux-make-cuda-release --target cute_copy_g2s_naive cute_copy_g2s_cpasync cute_copy_s2r cute_smem_swizzle_demo -j"$(nproc)"

./out/build/linux-make-cuda-release/labs/cute/cute_copy_g2s_naive
./out/build/linux-make-cuda-release/labs/cute/cute_copy_g2s_cpasync
./out/build/linux-make-cuda-release/labs/cute/cute_copy_s2r
./out/build/linux-make-cuda-release/labs/cute/cute_smem_swizzle_demo
```

## 12. W16 验收问题

完成 W16 后，应该能独立回答：

1. `Copy_Atom`、`TiledCopy`、`ThrCopy` 分别描述哪个层级？
2. `ThrLayout` 和 `ValLayout` 如何形成 `(thread,value)->coordinate` 映射？
3. `partition_S` 和 `partition_D` 为什么返回兼容 shape？
4. 一次 128-bit copy 对 FP16 包含几个元素？
5. value vector 为什么必须放在物理连续的 K 维？
6. `cp_async_fence`、`cp_async_wait`、`__syncthreads` 分别同步什么？
7. shared padding 和 swizzle 如何改变 bank，但不改变逻辑值？
8. register fragment 为什么是每线程私有 owning tensor？
9. vectorized predication 为什么需要考虑 instruction 粒度和 padded allocation？

回答完这些问题后，再进入 W17 的 `MMA_Atom`、`TiledMMA`、`ThrMMA` 和
MMA-compatible fragment。
