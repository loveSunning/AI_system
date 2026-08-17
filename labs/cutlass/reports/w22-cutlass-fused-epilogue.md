# W22 CUTLASS Fused Epilogue：Bias + ReLU

## 1. 实验目标

学习周：2026-09-14 至 2026-09-20。实现与首次验证日期：2026-08-14。

本实验只实现一个可解释的 epilogue 变体：

```text
D[i,j] = ReLU(alpha * sum_k(A[i,k] * B[k,j]) + bias[j])
```

目标不是继续扫描模板参数，而是在完全相同的 GEMM mainloop 下比较：

```text
unfused: CUTLASS GEMM -> temporary -> 独立 bias+ReLU kernel -> D
fused:   CUTLASS GEMM -> bias+ReLU epilogue -> D
```

两个实现分别使用 CUTLASS 2.x Device API 和 CUTLASS 3.x
Kernel/Collective API，保持相同数据类型、layout、tile 和 MMA 指令。

## 2. 平台与版本

| 项目 | Windows 实测 | Linux 目标 |
| --- | --- | --- |
| GPU | NVIDIA GeForce RTX 5060 | NVIDIA GeForce RTX 4090D |
| Compute Capability | SM120 | SM89 |
| OS | Windows 10/11 | Linux |
| CUDA | 13.0 | 建议 12.8+ |
| CUTLASS | 4.5.2，commit `db1c288993354c88e551c40c19a8fb93a774a241` | 同一 checkout |
| 算法最低架构 | `cutlass::arch::Sm80` | `cutlass::arch::Sm80` |
| 机器码目标 | `sm_120` | `sm_89` |

这里的 `Sm80` 是所选 CUTLASS 算法的最低硬件能力。实际构建仍分别生成
SM120 或 SM89 机器码。公共路径使用 FP16 `mma.sync m16n8k16` 和
`cp.async`，没有使用 SM90 TMA/WGMMA 或 SM120 原生窄精度 MMA。

## 3. 固定 GEMM 配置

| 配置 | 值 |
| --- | --- |
| A | FP16 row-major |
| B | FP16 column-major |
| D | FP32 row-major |
| Accumulator | FP32 |
| CTA tile | `128 x 128 x 32` |
| Warp tile | `64 x 64 x 32` |
| MMA | `16 x 8 x 16` |
| Pipeline | 3 stages |
| Bias | FP32 `bias[N]`，沿 M 广播 |
| Activation | ReLU，threshold 0 |

### 3.1 CUTLASS 2.x

2.x 使用：

```text
cutlass::epilogue::thread::LinearCombinationRelu<
    float, 4, float, float, ScaleType::NoBetaScaling>
```

输出是 row-major，因此 `TensorRef(bias, 0)` 的地址计算为：

```text
offset(i, j) = i * 0 + j = j
```

同一个 `bias[j]` 被所有 M 行复用。

### 3.2 CUTLASS 3.x

3.x 保持现有 `CollectiveMainloop`，把 thread epilogue 放入：

```text
collective::DefaultEpilogue<LinearCombinationRelu<...>>
```

source tensor 使用：

```text
StrideC{0, 1, 0}
```

即 M stride 和 batch stride 为零，N stride 为一。CUTLASS 4.5.2 的旧式
`LinearCombinationRelu` 仍使用 fragment call，而当前 `DefaultEpilogue` 使用
scalar call；源码中的薄适配层只完成 scalar 与单元素 fragment 的转换，数学操作
和参数仍由 CUTLASS functor 完成。

## 4. 为什么 fusion 减少 global memory traffic

只计算 GEMM 输出侧的 FP32 矩阵流量，A/B mainloop 读取在两条路径中相同：

| 路径 | 大矩阵操作 | 近似字节数 |
| --- | --- | ---: |
| Unfused GEMM | 写 temporary | `4MN` |
| Unfused elementwise | 读 temporary、写 D | `8MN` |
| Fused epilogue | 直接写 D | `4MN` |

bias 向量只有 `4N` bytes，而且可能被 cache 重用。忽略 cache line 和 transaction
粒度后：

```text
unfused ~= 12MN + 4N bytes
fused   ~=  4MN + 4N bytes
saved   ~=  8MN bytes
```

当 `M=N=4096` 时，理论上每次调用少传输：

```text
8 * 4096 * 4096 = 134217728 bytes = 128 MiB
```

这是理论 traffic 模型，不等价于所有 shape 都获得三倍性能。K 越大，GEMM
mainloop 的 Tensor Core 计算占比越高，固定的 128 MiB 输出侧节省越容易被掩盖。

## 5. 正确性设计

为了避免 `A=B=1` 导致 ReLU 永远处于正区间，bias 按列构造为：

```text
accumulator_term = alpha * K
bias[j] = -accumulator_term + (j 为奇数 ? 1 : -1)
```

所以逻辑输出必定交替为：

```text
j 为偶数: ReLU(-1) = 0
j 为奇数: ReLU(+1) = 1
```

校验 kernel 同时检查：

- unfused 输出是否等于 0/1；
- fused 输出是否等于 0/1；
- fused 与 unfused 是否一致；
- 是否存在 NaN 或 Inf。

`130 x 130 x 130` 会补齐为 `256 x 256 x 160`，两套 API 在 RTX 5060
SM120 上均以 `mismatches=0` 通过。

## 6. Benchmark 方法

- Release 构建；
- 每条路径独立 warmup 5 次；
- CUDA Event 计时 20 次；
- unfused 时间覆盖 GEMM 和独立 bias+ReLU kernel；
- fused 时间覆盖一个包含 fused epilogue 的 GEMM kernel；
- TFLOP/s 只计算 `2MNK` GEMM FLOPs，不把 ReLU 当成额外 FLOPs；
- 每组运行后执行完整逻辑输出校验。

Windows 命令示例：

```powershell
$bin = ".\out\build\windows-vs2022-cuda-release\labs\cutlass\Release"

& "$bin\cutlass_2x_gemm_bias_relu.exe" `
  --m=4096 --n=4096 --k=256 --warmup=5 --iterations=20
& "$bin\cutlass_3x_gemm_bias_relu.exe" `
  --m=4096 --n=4096 --k=256 --warmup=5 --iterations=20
```

## 7. RTX 5060 / Windows 结果

| API | M | N | K | Unfused ms | Fused ms | Speedup | 校验 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2.x | 4096 | 4096 | 256 | 0.595 | 0.238 | 2.495x | PASS |
| 2.x | 4096 | 4096 | 1024 | 1.188 | 0.844 | 1.406x | PASS |
| 2.x | 4096 | 4096 | 4096 | 3.653 | 3.501 | 1.044x | PASS |
| 3.x | 4096 | 4096 | 256 | 0.632 | 0.251 | 2.520x | PASS |
| 3.x | 4096 | 4096 | 1024 | 1.246 | 0.846 | 1.473x | PASS |
| 3.x | 4096 | 4096 | 4096 | 3.748 | 3.383 | 1.108x | PASS |

这些是一次本机观察值，不是 GPU 峰值承诺。温度、时钟、后台负载和运行顺序
都会影响绝对时间。

结果符合 traffic 模型：

- K=256 时 mainloop 较短，避免 temporary 写回与重读带来约 2.5x 加速；
- K=1024 时计算占比上升，收益降到约 1.4x；
- K=4096 时 GEMM 计算占主导，fusion 仍减少 traffic 和一次 kernel launch，
  但端到端收益只剩约 1.04x 至 1.11x。

## 8. RTX 4090D / Linux 复现

Linux preset 固定使用 `AI_SYSTEM_GPU_PROFILE=4090d`，生成 `sm_89`：

```bash
cd /path/to/AI_system

bash ./labs/cutlass/scripts/configure.sh --profile 4090d
bash ./labs/cutlass/scripts/build.sh --profile 4090d \
  --target cutlass_gemm_examples

bin=./out/build/linux-make-cuda-release/labs/cutlass

"${bin}/cutlass_2x_gemm_bias_relu" \
  --m=130 --n=130 --k=130 --warmup=1 --iterations=1
"${bin}/cutlass_3x_gemm_bias_relu" \
  --m=130 --n=130 --k=130 --warmup=1 --iterations=1

for k in 256 1024 4096; do
  "${bin}/cutlass_2x_gemm_bias_relu" \
    --m=4096 --n=4096 --k="${k}" --warmup=5 --iterations=20
  "${bin}/cutlass_3x_gemm_bias_relu" \
    --m=4096 --n=4096 --k="${k}" --warmup=5 --iterations=20
done
```

Windows SM120 的运行结果不能替代 Linux SM89 实测。Linux 执行后应把同样六行
数据追加到本报告，重点检查随 K 增大而收益下降的趋势是否一致，不要求绝对时间
与 RTX 5060 相同。

两个新增目标已在 Windows CUDA 13.0 工具链下额外使用
`--generate-code=arch=compute_89,code=[compute_89,sm_89]` 完成交叉编译，说明
SM89 device code 可以生成。这仍不能替代 Linux host 工具链和 RTX 4090D 实机运行。

## 9. 输出物

- `examples/cutlass_2x_gemm_bias_relu.cu`
- `examples/cutlass_3x_gemm_bias_relu.cu`
- `examples/bias_relu_lab_common.hpp`
- `reports/w22-cutlass-fused-epilogue.md`
- `CMakeLists.txt` 中的 executable 和 CTest 注册
- `README.md` 中的 Windows RTX 5060 与 Linux RTX 4090D 命令

## 10. 结论与边界

本实验说明了 epilogue fusion 的直接价值：accumulator 在 CUTLASS epilogue 中
完成 bias 和 ReLU 后只写最终 D，避免 temporary 的一次写回和一次重读。

范围刻意停在一个变体：

- 不增加 SiLU/GELU；
- 不实现 EVT；
- 不引入 aux tensor 或 residual；
- 不重新扫描 tile、stage 或 instruction shape；
- 不把 SM120 特化路径与 SM89 公共路径混在同一次比较中。

因此代码可以逐行解释，性能差异也能够直接归因于 kernel fusion 和输出侧
global memory traffic 的变化。

## 11. 参考资料

- [CUTLASS Quickstart](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/quickstart.html)
- [Efficient GEMM in CUDA：Epilogue](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html#epilogue)
- [CUTLASS example 12：GEMM bias + ReLU](https://github.com/NVIDIA/cutlass/tree/main/examples/12_gemm_bias_relu)
