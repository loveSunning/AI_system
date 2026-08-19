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

预期关键输出：

```text
GPU: NVIDIA GeForce RTX 4090 D
Compute capability: (8, 9)
Compiling CuTe DSL kernel (the first run can take a few seconds)...
Launching kernel...
Max absolute error: 0.000e+00
PASS
```

第一次运行包含 JIT 编译时间，之后相同参数和类型通常会命中 CuTe DSL 编译缓存。

## 代码执行关系

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
