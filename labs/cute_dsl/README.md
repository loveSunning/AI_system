# CuTe DSL 实验

本目录放置使用 Python CuTe DSL 编写的 GPU kernel。它和 `labs/cute/` 中的 CuTe C++ 示例是两套前端，但共享 Layout、Tensor、Copy、MMA 等核心概念。

## 01：向量加法

文件：`examples/01_vector_add.py`

计算：

```text
out[i] = x[i] + y[i]
```

这个最小示例展示：

1. `@cute.kernel` 定义 GPU device kernel。
2. `@cute.jit` 定义 host 侧网格计算和 kernel launch。
3. `from_dlpack` 将 PyTorch CUDA Tensor 零拷贝转换成 `cute.Tensor`。
4. `cute.compile` 将 DSL 编译为 GPU kernel。
5. 使用 PyTorch 参考结果验证正确性。

### 在 cutedsl_dev 容器中运行

进入容器：

```bash
cd /path/to/AI_system/docker/container
docker compose -f docker-compose-cutedsl-dev.yaml exec cutedsl_dev bash
```

容器内运行：

```bash
cd /workspace
python labs/cute_dsl/examples/01_vector_add.py
```

指定向量长度：

```bash
python labs/cute_dsl/examples/01_vector_add.py --num-elements 1048576
```

指定预热和计时次数：

```bash
python labs/cute_dsl/examples/01_vector_add.py \
    --num-elements 1048576 \
    --warmup-iterations 20 \
    --iterations 200
```

预热不会计入执行时间。正式计时使用与 CuTe DSL kernel 相同 CUDA stream 上的 CUDA Event，统计多次 kernel launch 的 GPU 总时间和平均时间，不包含 JIT 编译时间。

预期关键输出：

```text
GPU: NVIDIA GeForce RTX 4090 D
Compute capability: (8, 9)
Compiling CuTe DSL kernel (the first run can take a few seconds)...
Compilation time: 2.345 s
Launching kernel for correctness check...
Max absolute error: 0.000e+00
Warming up: 10 iterations...
Benchmark iterations: 100
Total GPU time: 1.234 ms
Average kernel time: 12.340 us (0.012340 ms)
Effective memory throughput: 972.00 GB/s
PASS
```

第一次运行包含 JIT 编译时间，之后相同参数和类型通常会命中 CuTe DSL 编译缓存。

## 02：二维分块矩阵加法

文件：`examples/02_tiled_matrix_add.py`

计算：

```text
C[i, j] = A[i, j] + B[i, j]
```

与 01 的「一个线程算一个元素」不同，02 展示 CuTe 的核心抽象：Layout、Tile、Copy。

1. `make_layout_tv(thr_layout, val_layout)` 由线程布局 `(4, 32)`（128 线程）和值布局 `(4, 4)`（每线程 4×4 个元素）推导出 CTA tile 形状 `(16, 128)`：每个 CTA 处理一个 16×128 的二维块，按 128 位向量化读写。
2. `zipped_divide` 把 `(M, N)` 问题切成 `((TileM, TileN), (RestM, RestN))`，kernel 里用 `((None, None), bidx)` 取出当前 CTA 对应的 tile。
3. `make_tiled_copy_tv` + `partition_S` 把 tile 分给块内 128 个线程。
4. `make_rmem_tensor_like` 分配寄存器片段（fragment），`cute.copy` 完成 gmem→rmem 的加载与 rmem→gmem 的存储。
5. `make_identity_tensor` 生成坐标张量，`cute.elem_less(coord, shape)` 逐元素生成边界谓词，因此 M、N 不是 tile 整数倍时（如 1000×1200）也不会越界。

### 在 cutedsl_dev 容器中运行

```bash
cd /workspace
python labs/cute_dsl/examples/02_tiled_matrix_add.py
```

指定形状和计时参数：

```bash
python labs/cute_dsl/examples/02_tiled_matrix_add.py \
    --M 1000 --N 1200 \
    --warmup-iterations 20 --iterations 200
```

`1000×1200` 不是 tile `(16, 128)` 的整数倍，用来验证谓词边界处理。

预期关键输出：

```text
GPU: NVIDIA GeForce RTX 4090 D
Compute capability: (8, 9)
CTA tile: 16 x 128, threads per CTA: 128
Compiling CuTe DSL kernel (the first run can take a few seconds)...
Compilation time: 2.800 s
Launching kernel for correctness check...
Max absolute error: 0.000e+00
Warming up: 10 iterations...
Benchmark iterations: 100
Total GPU time: 0.900 ms
Average kernel time: 9.000 us (0.009000 ms)
Effective memory throughput: 1092.27 GB/s
PASS
```

预热和计时方式与 01 相同：预热不计入时间，正式计时使用同一 CUDA stream 上的 CUDA Event。

## 01 代码执行关系

```text
PyTorch CUDA Tensor
        │ from_dlpack（零拷贝）
        ▼
    cute.Tensor
        │
        ▼
@cute.jit launch_vector_add
        │ grid / block
        ▼
@cute.kernel vector_add_kernel
        │ JIT compile
        ▼
CUDA kernel on RTX 4090 D (SM89)
```

默认长度 `1,000,003` 不是线程块大小 256 的整数倍，用来验证：

```python
if idx < num_elements:
```

确实阻止最后一个线程块越界访问。

## 02 代码执行关系

```text
PyTorch CUDA Tensor A / B / C
        │ from_dlpack（零拷贝）
        ▼
    cute.Tensor (M, N)
        │ make_layout_tv → tiler (16, 128)，zipped_divide 分块
        ▼
@cute.jit launch_tiled_add
        │ grid = RestM * RestN 个 CTA，block = 128 线程
        ▼
@cute.kernel tiled_add_kernel
        │ partition_S 分给线程 → cute.copy 到寄存器 → C = A + B
        ▼
CUDA kernel on RTX 4090 D (SM89)
```

每个 CTA 处理一个 `(16, 128)` 的二维块，块内 128 个线程各自负责 `(4, 4)` 的子块；边界处由坐标张量生成的谓词屏蔽越界元素。
