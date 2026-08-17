# CUTLASS 2.x / 3.x Tensor Core GEMM 学习实验

本实验用两套 CUTLASS C++ API 实现同一个 GEMM：

```text
D = alpha * A * B + beta * C
A、B: FP16
Accumulator、C、D: FP32
默认逻辑 shape: M=N=K=4096
```

目标不是追求当前 GPU 的最高性能，而是让 CUTLASS 2.x 与 3.x 的层级、
mainloop、epilogue 和 pipeline 能够逐行对照。

## 1. 目标平台

| 平台 | GPU | Compute Capability | CMake 架构值 | 推荐 CUDA |
| --- | --- | ---: | ---: | --- |
| Windows 10/11 | RTX 5060 Blackwell | 12.0 | `120` | CUDA 12.8+ |
| Linux | RTX 4090D Ada | 8.9 | `89` | CUDA 11.8+；本仓库建议 12.8+ |

这里必须注意：RTX 5060 是 `sm_120`，RTX 4090D 是 `sm_89`，不能反过来。
Windows/Linux 只是本实验的部署约定，SM 编号由 GPU 决定，与操作系统无关。

本地默认依赖 `3rdparty/cutlass`。当前验证使用 CUTLASS 4.5.2；其中仍保留
CUTLASS 2.x 兼容 API。因此“2.x 实现”是指使用 2.x API 和算法组织方式，
并不要求另外下载一个旧版 CUTLASS。这样两套实现可以在同一套头文件和工具链中公平对照。

## 2. 文件结构

```text
labs/cutlass/
├── CMakeLists.txt
├── examples/
│   ├── cutlass_2x_gemm.cu       # 2.x Device API
│   ├── cutlass_3x_gemm.cu       # 3.x Kernel/Collective/CuTe API
│   ├── cutlass_2x_gemm_bias_relu.cu  # 2.x fused bias+ReLU 与 unfused 对照
│   ├── cutlass_3x_gemm_bias_relu.cu  # 3.x fused bias+ReLU 与 unfused 对照
│   ├── bias_relu_lab_common.hpp      # bias+ReLU 初始化、独立 kernel、计时与校验
│   ├── gemm_lab_common.hpp      # 公共 CLI、显存、初始化、计时与校验
│   └── cutlass_header_probe.cu
├── notes/
└── scripts/
```

## 3. 两个实现对照

| 概念 | CUTLASS 2.x 示例 | CUTLASS 3.x 示例 |
| --- | --- | --- |
| 用户入口 | `cutlass::gemm::device::Gemm` | `GemmUniversalAdapter` |
| Kernel 层 | 由 Device API 内部生成 | `kernel::GemmUniversal` |
| CTA mainloop | `MmaMultistage` | `collective::CollectiveMma` |
| CTA tile | `ThreadblockShape<128,128,32>` | `TileShape = Shape<_128,_128,_32>` |
| Warp tile | `WarpShape<64,64,32>` | `TiledMMA` 的 Atom/Value layout |
| MMA 指令 | `InstructionShape<16,8,16>` | `MMA_Atom<SM80_16x8x16_...>` |
| GMEM→SMEM | 专用 iterator，内部使用 `cp.async` | 显式 `GmemTiledCopy` + `Copy_Atom` |
| SMEM→寄存器 | 专用 warp iterator | `SmemCopyAtom`，使用 `ldmatrix` |
| Pipeline | `Stages=3` | `MainloopSm80CpAsync<3>` |
| 普通 Epilogue | `thread::LinearCombination` | `collective::DefaultEpilogue` 包装同一操作 |
| Fused Epilogue | `thread::LinearCombinationRelu` | `DefaultEpilogue<LinearCombinationRelu>` |
| Bias 广播 | row-major `TensorRef(bias, 0)` | `StrideC{0, 1, 0}` |

两个 kernel 都使用 `OpClassTensorOp` 和 `m16n8k16` FP16 Tensor Core MMA，
都是 FP32 累加，不是 SIMT CUDA Core GEMM。

### 2.x 层级

```text
Device GEMM
└── CTA tile: 128 x 128 x 32
    ├── 4 个 warp tile: 每个 64 x 64 x 32
    │   └── mma.sync: 16 x 8 x 16
    └── Epilogue: alpha * accumulator + beta * C
```

一个 CTA 的 M/N 平面由 `2 x 2 = 4` 个 warp 覆盖。一个 warp 对一个
`64x64x32` tile 的概念性 MMA 数量为：

```text
(64/16) * (64/8) * (32/16) = 64 条 mma.sync
```

### 3.x 层级

```text
GemmUniversalAdapter
└── kernel::GemmUniversal
    ├── collective::CollectiveMma
    │   ├── TileShape 128 x 128 x 32
    │   ├── TiledCopy / CopyAtom
    │   └── TiledMMA / MMAAtom m16n8k16
    └── collective::DefaultEpilogue
```

CUTLASS 3.x 没有取消 CTA、warp 或 Tensor Core；它只是不再把公共 API
强绑定到这些硬件层级。这样同一套 Kernel/Collective/Atom 词汇才能继续描述
Hopper warpgroup 和 Blackwell 的新指令组织。

## 4. 默认 4096³ 与自定义 shape

默认 `4096×4096×4096` 可以被 FP16 128-bit 向量宽度、K tile 32 和 CTA tile
128 整除，因此逻辑问题和实际执行问题完全一致，不需要额外 padding：

```text
逻辑问题:  4096 x 4096 x 4096
执行问题:  4096 x 4096 x 4096
```

程序仍支持通过 CLI 输入任意正整数 shape。对于 `130³` 之类的非整 tile
问题，公共代码会自动补齐执行 shape、将 K padding 置零，并且只校验逻辑输出
区域。这样既能保持向量化 Tensor Core kernel 的对齐要求，也便于学习边界处理。

输出的 TFLOP/s 始终按用户输入的逻辑 shape 计算；它是学习用的对比指标，
不应当作该 GPU 的峰值 benchmark。

## 5. Mainloop、Epilogue 与三阶段 Pipeline

对于每个 CTA 输出 tile，M/N 坐标固定，mainloop 沿 K 维推进：

```text
for each K tile:
    A/B: global memory -> shared memory
    A/B: shared memory -> warp registers
    accumulator += MMA(A fragment, B fragment)
```

本例的执行 K 为 4096，CTA K tile 为 32，因此 mainloop 有 `4096/32=128`
次 K-tile 迭代。`Stages=3` 表示 shared memory 中有三个循环复用的 stage：

```text
stage 0: 当前参与 MMA
stage 1: 已加载，等待消费
stage 2: cp.async 正在预取更后的 K tile
```

这是一种一般化的多缓冲。双缓冲只是 `Stages=2` 的特殊情况。

Mainloop 完成后 accumulator 仍按 Tensor Core 最方便的寄存器 fragment 布局
分散在线程中。Epilogue 单独负责：

```text
寄存器 accumulator
-> 线程间/共享内存重排
-> alpha * accumulator + beta * C
-> 合并写回 D
```

因此 mainloop 和 epilogue 的分离不是 2.x 专属；3.x 仍然把它们显式组织为
`CollectiveMainloop + CollectiveEpilogue`。

## 6. 架构适配策略

本实验选择了一条两张卡都能执行的公共路径：

```text
算法最低架构: Sm80
底层 MMA:      mma.sync m16n8k16
编译目标:      sm_89 或 sm_120
```

`ArchTag=Sm80` 表示该算法要求的最低硬件能力；`CMAKE_CUDA_ARCHITECTURES`
表示实际生成哪种 GPU 机器码。将 SM80 算法编译成 `sm_120` 并不等于生成
SM80 二进制再由 Blackwell 模拟，而是为 SM120 编译一条兼容的 `mma.sync` 路径。

| 能力 | RTX 4090D / SM89 | RTX 5060 / SM120 | 本实验 |
| --- | --- | --- | --- |
| FP16 `mma.sync` | 支持 | 支持 | 使用 |
| `cp.async` | 支持 | 支持 | 使用 |
| SM90 TMA/WGMMA | 不支持 | 不是本实验的公共路径 | 不使用 |
| SM120 原生窄精度 MMA | 不支持 | 支持 | 不使用 |
| CTA cluster multicast | 不支持 | GeForce SM120 有限制 | 不使用 |

为什么 3.x 示例没有直接写 `CollectiveBuilder<Sm120,...half_t>`？当前 CUTLASS
SM120 原生 builder 主要面向 F8/F6/F4 和 block-scaled 路径，不能用同一个
FP16 配置在 SM89 上做等价对比。若改用 NVFP4/FP8，数据类型、scale factor、
误差模型和 epilogue 都发生变化，就无法只比较 2.x/3.x API 差异。

后续做性能专项时，应另外增加两条架构特化路线：SM89 FP8 Tensor Core，及
SM120 原生 F8/F6/F4、TMA、warp-specialized ping-pong/cooperative kernel。

## 7. 环境检查

CUTLASS 不是 CUDA Toolkit 的一部分。本仓库默认从以下位置读取头文件：

```text
3rdparty/cutlass/include/cutlass
3rdparty/cutlass/include/cute
```

Windows：

```powershell
cd D:\workspace\learing\AI_system
.\labs\cutlass\scripts\check_env.ps1
```

Linux：

```bash
cd /path/to/AI_system
bash ./labs/cutlass/scripts/check_env.sh
```

重点确认 `nvcc --list-gpu-code` 中存在目标 `sm_120` 或 `sm_89`。

## 8. Windows：RTX 5060 / SM120

### 配置与编译

```powershell
cd D:\workspace\learing\AI_system

cmake -S . --preset windows-vs2022-cuda-release `
  -DAI_SYSTEM_CUTLASS_ROOT="D:\workspace\learing\AI_system\3rdparty\cutlass"

cmake --build --preset windows-vs2022-cuda-release `
  --config Release --target cutlass_gemm_examples
```

也可以使用封装脚本：

```powershell
.\labs\cutlass\scripts\configure.ps1
.\labs\cutlass\scripts\build.ps1
```

### 运行

```powershell
$bin = ".\out\build\windows-vs2022-cuda-release\labs\cutlass\Release"

& "$bin\cutlass_2x_gemm.exe"
& "$bin\cutlass_3x_gemm.exe"
& "$bin\cutlass_2x_gemm_bias_relu.exe"
& "$bin\cutlass_3x_gemm_bias_relu.exe"
```

快速 smoke test：

```powershell
& "$bin\cutlass_2x_gemm.exe" --m=130 --n=130 --k=130 --warmup=1 --iterations=1
& "$bin\cutlass_3x_gemm.exe" --m=130 --n=130 --k=130 --warmup=1 --iterations=1
& "$bin\cutlass_2x_gemm_bias_relu.exe" --m=130 --n=130 --k=130 --warmup=1 --iterations=1
& "$bin\cutlass_3x_gemm_bias_relu.exe" --m=130 --n=130 --k=130 --warmup=1 --iterations=1
```

W22 fused epilogue 性能对照：

```powershell
foreach ($k in 256, 1024, 4096) {
  & "$bin\cutlass_2x_gemm_bias_relu.exe" `
    --m=4096 --n=4096 --k=$k --warmup=5 --iterations=20
  & "$bin\cutlass_3x_gemm_bias_relu.exe" `
    --m=4096 --n=4096 --k=$k --warmup=5 --iterations=20
}
```

## 9. Linux：RTX 4090D / SM89

### 配置与编译

```bash
cd /path/to/AI_system

cmake -S . --preset linux-make-cuda-release \
  -DAI_SYSTEM_CUTLASS_ROOT="${PWD}/3rdparty/cutlass"

cmake --build --preset linux-make-cuda-release \
  --target cutlass_gemm_examples -j
```

也可以使用封装脚本：

```bash
bash ./labs/cutlass/scripts/configure.sh
bash ./labs/cutlass/scripts/build.sh
```

### 运行

```bash
bin=./out/build/linux-make-cuda-release/labs/cutlass

"${bin}/cutlass_2x_gemm"
"${bin}/cutlass_3x_gemm"
"${bin}/cutlass_2x_gemm_bias_relu"
"${bin}/cutlass_3x_gemm_bias_relu"
```

W22 fused epilogue 性能对照：

```bash
for k in 256 1024 4096; do
  "${bin}/cutlass_2x_gemm_bias_relu" \
    --m=4096 --n=4096 --k="${k}" --warmup=5 --iterations=20
  "${bin}/cutlass_3x_gemm_bias_relu" \
    --m=4096 --n=4096 --k="${k}" --warmup=5 --iterations=20
done
```

## 10. CLI 参数

两套程序参数完全一致：

```text
--m=<int>             逻辑 M，默认 4096
--n=<int>             逻辑 N，默认 4096
--k=<int>             逻辑 K，默认 4096
--warmup=<int>        预热次数，默认 2
--iterations=<int>    计时次数，默认 10
--alpha=<float>       默认 1
--beta=<float>        默认 0
--no-verify           跳过完整逻辑输出校验
--help                显示帮助
```

两个 bias+ReLU 程序也使用相同 CLI，但固定计算：

```text
unfused: T = alpha * A * B; D = ReLU(T + bias)
fused:   D = ReLU(alpha * A * B + bias)
```

其中 `--beta` 必须保持为 `0`；bias 是不经过 beta 缩放的 epilogue source。
每个程序一次运行会依次测量 unfused 和 fused 两条路径，因此不需要额外的
`--mode` 参数。

示例：

```powershell
& "$bin\cutlass_3x_gemm.exe" --m=4096 --n=4096 --k=4096 --warmup=5 --iterations=20
```

输入初始化为 `A=B=C=1`，因此每个逻辑输出的期望值为：

```text
expected = alpha * K + beta
```

校验在 GPU 上遍历整个逻辑输出区域并统计 mismatch，不需要复制约 67 MB 的
输出矩阵回主机。

## 11. CTest

启用 `AI_SYSTEM_ENABLE_TESTS=ON` 时会注册普通 GEMM 和 fused bias+ReLU 的
`130³` smoke test：

```powershell
ctest --test-dir .\out\build\windows-vs2022-cuda-release `
  -C Release -L cutlass --output-on-failure
```

```bash
ctest --test-dir ./out/build/linux-make-cuda-release \
  -L cutlass --output-on-failure
```

## 12. 本机验证记录

在 Windows、RTX 5060 `sm_120`、CUDA 13.0、CUTLASS 4.5.2 上，两个目标均已
完成 Release 编译，并通过：

- 非整 tile 的 `130³` smoke test；
- 默认 `4096³` 全输出校验；
- FP16 Tensor Core、FP32 accumulate、FP32 output 路径。

W22 新增的 2.x/3.x bias+ReLU 目标也已在相同环境通过：

- `130³` 非整 tile 的 fused/unfused 全逻辑输出校验；
- bias 使逻辑列交替得到 ReLU 输出 `0/1`，正负分支均被覆盖；
- `4096×4096×{256,1024,4096}` 的端到端性能对照；
- fused 路径单 kernel，unfused 路径为 GEMM 加独立 bias+ReLU kernel。

完整数据与 traffic 推导见
[`reports/w22-cutlass-fused-epilogue.md`](reports/w22-cutlass-fused-epilogue.md)。

一次 5 次迭代的观察值约为 40–41 TFLOP/s。该数字会随温度、功耗、后台负载
和时钟变化，只应当作构建与执行成功的证据。

Linux/RTX 4090D 需要在对应机器上执行第 9 节命令完成实机验证；当前 Windows
机器只能验证 SM120，不能替代 SM89 的运行测试。两套源码已额外通过
`compute_89,sm_89` 交叉编译，包括新增的 2.x/3.x bias+ReLU 目标，确认 Ada
目标代码能够生成；Linux host 编译与运行仍需在 4090D 机器上完成。

## 13. 常见问题

### `nvcc fatal: Unsupported gpu architecture 'compute_120'`

CUDA Toolkit 太旧。RTX 5060/SM120 使用 CUDA 12.8 或更新版本。

### `include/cutlass/cutlass.h not found`

确认子目录存在，或显式传入：

```text
-DAI_SYSTEM_CUTLASS_ROOT=<CUTLASS checkout>
```

### `misaligned address` 或 `kErrorMisalignedOperand`

默认 `4096³` 不需要 padding。若输入自定义非整 tile shape，不要删除公共代码
中的自动补齐；它用于满足 FP16 128-bit 向量访问、K tile 和 CTA tile 的对齐要求。

### Windows 编译出现大量 C4819 警告

这是 CUDA/CUTLASS 头文件在中文系统代码页下的字符集警告，不影响本例生成。
CMake 已为完整 CuTe target 加入 MSVC conformance 和 `/bigobj` 参数。

### 为什么两个版本性能很接近？

它们最终描述的是同一种 SM80-compatible `mma.sync` 算法。3.x 的主要变化是
组合方式和扩展点，不保证仅因 API 更新就自动变快。

## 14. 参考资料

- [Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)
- [CUTLASS 3.x Design](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cutlass_3x_design.html)
- [CUTLASS 3.x GEMM API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html)
- [CuTe GEMM tutorial](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html)
