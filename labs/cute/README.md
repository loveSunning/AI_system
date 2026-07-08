# CuTe Lab

CuTe 是深入阅读 CUTLASS 3.x/4.x 之前的第一站。本实验目录先从 `Layout -> offset` 的代数关系开始，再逐步扩展到 tensor tiling、copy partition、`TiledMMA` 和完整 HGEMM pipeline。

当前可直接编译运行的 demo 是：

```text
cute_layout_mapping
cute_layout_algebra_demo
cute_tensor_tile_demo
```

- `cute_layout_mapping` 是 host-side `Layout` smoke test，用 CuTe `Layout` 验证 `(M,K)`、`(N,K)` 和 shared-memory stage layout 的 offset 计算。
- `cute_layout_algebra_demo` 是 host-side `Layout Algebra` smoke test，对应 NVIDIA `02_layout_algebra`，验证 `coalesce`、`composition`、`complement`、divide 和 product。
- `cute_tensor_tile_demo` 是 host-side `Tensor/local_tile/partition` smoke test，用同一个逻辑值串起 global tensor、shared tensor、register fragment 和 per-thread partition。

## 目录结构

```text
labs/cute/
|-- CMakeLists.txt
|-- README.md
|-- examples/
|   |-- README.md
|   |-- cute_layout_algebra_demo.cu
|   |-- cute_layout_mapping.cu
|   `-- cute_tensor_tile_demo.cu
|-- notes/
|   |-- README.md
|   |-- learning-plan.md
|   `-- windows-linux-build.md
|-- scripts/
|   |-- README.md
|   |-- build.ps1
|   |-- build.sh
|   |-- check_env.ps1
|   |-- check_env.sh
|   |-- configure.ps1
|   `-- configure.sh
`-- reports/
    `-- README.md
```

## 依赖

CuTe 是 header-only，随 CUTLASS 一起分发。本仓库默认从下面的本地路径读取 CuTe 头文件：

```text
3rdparty/cutlass/include/cute
```

CMake cache 变量是：

```text
AI_SYSTEM_CUTLASS_ROOT=D:\workspace\learing\AI_system\3rdparty\cutlass
```

如果你想临时使用另一份 CUTLASS checkout，可以用以下任意一种方式覆盖：

```powershell
$env:CUTLASS_ROOT="D:\path\to\cutlass"
cmake -S . --preset windows-vs2022-cuda-release -DAI_SYSTEM_CUTLASS_ROOT="$env:CUTLASS_ROOT"
```

```bash
export CUTLASS_ROOT=/path/to/cutlass
cmake -S . --preset linux-make-cuda-release -DAI_SYSTEM_CUTLASS_ROOT="${CUTLASS_ROOT}"
```

最关键的检查项是这个文件必须存在：

```text
<CUTLASS_ROOT>/include/cute/tensor.hpp
```

官方参考：

- https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/00_quickstart.html
- https://docs.nvidia.com/cutlass/latest/media/docs/cpp/quickstart.html

## GPU 和 CMake preset

本仓库刻意为两个常用学习环境各保留一个 CUDA preset：

| 环境 | CMake preset | GPU profile | CUDA arch |
| --- | --- | --- | --- |
| Windows 10/11 + Visual Studio 2022 | `windows-vs2022-cuda-release` | RTX 5060 | `sm_120` |
| Linux / WSL + Make | `linux-make-cuda-release` | RTX 4090D | `sm_89` |

如果 `nvcc` 不认识对应的架构，例如 `sm_120` 或 `sm_89`，优先升级 CUDA Toolkit。`check_env` 脚本会显示 `nvcc --list-gpu-code` 是否列出目标架构。

## 直接使用 CMake

下面是最直接、最透明的路径：从仓库根目录配置、编译、运行 demo。

### Windows

适用于 Windows 10/11、Visual Studio 2022、CUDA Toolkit 已安装且 `nvcc` 可用。

```powershell
cd D:\workspace\learing\AI_system

# 可选：确认工具链、GPU、CUTLASS/CuTe 头文件和 sm_120 支持。
.\labs\cute\scripts\check_env.ps1

# 配置：生成 VS2022 x64 Release build tree。
cmake -S . --preset windows-vs2022-cuda-release -DAI_SYSTEM_CUTLASS_ROOT="D:\workspace\learing\AI_system\3rdparty\cutlass"

# 编译：只编译当前 CuTe demo，避免顺手构建整个仓库。
cmake --build --preset windows-vs2022-cuda-release --config Release --target cute_layout_mapping
cmake --build --preset windows-vs2022-cuda-release --config Release --target cute_layout_algebra_demo
cmake --build --preset windows-vs2022-cuda-release --config Release --target cute_tensor_tile_demo

# 运行。
.\out\build\windows-vs2022-cuda-release\labs\cute\Release\cute_layout_mapping.exe
.\out\build\windows-vs2022-cuda-release\labs\cute\Release\cute_layout_algebra_demo.exe
.\out\build\windows-vs2022-cuda-release\labs\cute\Release\cute_tensor_tile_demo.exe
```

期望最后看到：

```text
layout mapping check passed
```

如果你已经设置了 `CUTLASS_ROOT`，配置命令也可以写成：

```powershell
cmake -S . --preset windows-vs2022-cuda-release -DAI_SYSTEM_CUTLASS_ROOT="$env:CUTLASS_ROOT"
```

### Linux / WSL

适用于 Linux 或 WSL，CUDA Toolkit、`nvcc`、GCC 和 Make 已可用。

```bash
cd /workspace/AI_system

# 可选：确认工具链、GPU、CUTLASS/CuTe 头文件和 sm_89 支持。
labs/cute/scripts/check_env.sh

# 配置：生成 Unix Makefiles Release build tree。
cmake -S . --preset linux-make-cuda-release -DAI_SYSTEM_CUTLASS_ROOT="${PWD}/3rdparty/cutlass"

# 编译：只编译当前 CuTe demo。
cmake --build --preset linux-make-cuda-release --target cute_layout_mapping -j"$(nproc)"
cmake --build --preset linux-make-cuda-release --target cute_layout_algebra_demo -j"$(nproc)"
cmake --build --preset linux-make-cuda-release --target cute_tensor_tile_demo -j"$(nproc)"

# 运行。
./out/build/linux-make-cuda-release/labs/cute/cute_layout_mapping
./out/build/linux-make-cuda-release/labs/cute/cute_layout_algebra_demo
./out/build/linux-make-cuda-release/labs/cute/cute_tensor_tile_demo
```

期望最后看到：

```text
layout mapping check passed
```

如果你已经设置了 `CUTLASS_ROOT`，配置命令也可以写成：

```bash
cmake -S . --preset linux-make-cuda-release -DAI_SYSTEM_CUTLASS_ROOT="${CUTLASS_ROOT}"
```

## 使用封装脚本

脚本只是对上面 CMake 命令的薄封装，适合日常快速使用。

Windows：

```powershell
cd D:\workspace\learing\AI_system
.\labs\cute\scripts\check_env.ps1
.\labs\cute\scripts\configure.ps1
.\labs\cute\scripts\build.ps1
.\out\build\windows-vs2022-cuda-release\labs\cute\Release\cute_layout_mapping.exe
```

Linux / WSL：

```bash
cd /workspace/AI_system
labs/cute/scripts/check_env.sh
labs/cute/scripts/configure.sh
labs/cute/scripts/build.sh
./out/build/linux-make-cuda-release/labs/cute/cute_layout_mapping
```

覆盖 CUTLASS 路径：

```powershell
.\labs\cute\scripts\configure.ps1 -CutlassRoot "D:\path\to\cutlass"
```

```bash
labs/cute/scripts/configure.sh --cutlass-root /path/to/cutlass
```

指定 build target：

```powershell
.\labs\cute\scripts\build.ps1 -Target cute_layout_mapping -Configuration Release
.\labs\cute\scripts\build.ps1 -Target cute_layout_algebra_demo -Configuration Release
.\labs\cute\scripts\build.ps1 -Target cute_tensor_tile_demo -Configuration Release
```

```bash
labs/cute/scripts/build.sh --target cute_layout_mapping
labs/cute/scripts/build.sh --target cute_layout_algebra_demo
labs/cute/scripts/build.sh --target cute_tensor_tile_demo
```

## Demo 说明

`examples/cute_layout_mapping.cu` 会构造三个 CuTe layout：

| Layout | Shape | Stride | 验证内容 |
| --- | --- | --- | --- |
| `mk_row_major` | `(4,8)` | `(8,1)` | `mk_row_major(2,3) == 19` |
| `nk_col_major` | `(8,4)` | `(1,8)` | `nk_col_major(2,3) == 26` |
| `smem_bk_stage` | `(16,32,2)` | `(64,1,32)` | `smem_bk_stage(3,5,1) == 229` |

这个 demo 不追求 GPU 性能；它的目标是把 CuTe 的 `make_shape`、`make_stride`、`make_layout` 和手写 offset 公式对齐。后续写 `TiledCopy` 和 `TiledMMA` 时，所有 tile/thread/value 映射都会回到这个基本问题：一个逻辑坐标最后对应哪一个线性地址。

`examples/cute_layout_algebra_demo.cu` 对应 NVIDIA `02_layout_algebra` 这一章，覆盖：

```text
coalesce
composition
by-mode composition
complement
logical_divide / zipped_divide
logical_product
blocked_product / raked_product
```

它验证的核心性质包括：

- `coalesce(layout)(i) == layout(i)`。
- `composition(A,B)(i) == A(B(i))`。
- `layout<0>(zipped_divide(A,B)) == composition(A,B)`。
- `logical_product(A,B)` 的 mode-0 兼容 `A`，mode-1 兼容 `B`。

`examples/cute_tensor_tile_demo.cu` 会构造一条更接近 GEMM mainloop 的教学链：

```text
global tensor -> local_tile -> shared tensor -> register fragment -> local_partition
```

它使用 `M=N=K=2048`、CTA tile `128x128x32`、CTA coord `(3,5,7)`，验证同一个逻辑元素在不同 memory space 和不同 layout 中的映射：

| Tensor | Memory tag | Layout | 验证内容 |
| --- | --- | --- | --- |
| `gA` | `gmem_ptr` | `(_2048,_2048):(_2048,_1)` | `gA(389,231) == 38900231` |
| `tAgA` | `gmem_ptr` | `(_128,_32):(_2048,_1)` | `local_tile` 后 `tAgA(5,7) == 38900231` |
| `sA` | `smem_ptr` | `(_128,_32):(_1,_128)` | shared memory 改物理布局但保留逻辑坐标 |
| `rA` | `ptr` | `(_16,_16):(_16,_1)` | MMA A register fragment |
| `rB` | `ptr` | `(_8,_16):(_16,_1)` | MMA B register fragment |
| `rC` | `ptr` | `(_16,_8):(_8,_1)` | MMA C accumulator fragment |
| `tAsA` | `smem_ptr` | per-thread subtensor | `local_partition` 后 `tAsA(0,0) == 38900229` |

配套笔记见 `notes/tensor-local-tile-partition.md`。

## 常见问题

`labs/cute examples` 被跳过：

检查配置输出里是否出现 `include/cute/tensor.hpp not found`。如果出现，确认 `3rdparty/cutlass` 已存在，或者显式传入 `-DAI_SYSTEM_CUTLASS_ROOT=<CUTLASS_ROOT>`。

`nvcc fatal : Unsupported gpu architecture`：

当前 preset 绑定了固定 GPU profile。Windows preset 使用 `sm_120`，Linux preset 使用 `sm_89`。如果当前 CUDA Toolkit 不支持该架构，需要升级 CUDA Toolkit，或临时用根工程支持的 `AI_SYSTEM_GPU_PROFILE` 覆盖到你的机器支持的架构。

Visual Studio generator 找不到：

确认安装了 Visual Studio 2022 和 C++/CUDA 相关组件。也可以先运行：

```powershell
cmake --list-presets
cmake --version
nvcc --version
```

Linux 找不到 `nvcc`：

确认 CUDA Toolkit 已安装，并且 `nvcc` 所在目录在 `PATH` 中：

```bash
which nvcc
nvcc --version
```

修改了源码但运行结果没变：

只重新编译当前目标即可：

```powershell
cmake --build --preset windows-vs2022-cuda-release --config Release --target cute_layout_mapping
cmake --build --preset windows-vs2022-cuda-release --config Release --target cute_layout_algebra_demo
cmake --build --preset windows-vs2022-cuda-release --config Release --target cute_tensor_tile_demo
```

```bash
cmake --build --preset linux-make-cuda-release --target cute_layout_mapping -j"$(nproc)"
cmake --build --preset linux-make-cuda-release --target cute_layout_algebra_demo -j"$(nproc)"
cmake --build --preset linux-make-cuda-release --target cute_tensor_tile_demo -j"$(nproc)"
```

## 里程碑

第一个里程碑是能独立解释并运行 `cute_layout_mapping`：

- 知道 CuTe `Layout` 是 shape + stride 的组合。
- 能手算 `(i,j)` 或 `(i,j,k)` 到线性 offset 的映射。
- 能用 CMake preset 单独构建一个 CuTe demo。
- 能判断构建失败是 CUDA Toolkit、GPU arch、CMake preset，还是 CUTLASS/CuTe 头文件路径问题。

第二个里程碑是能独立解释并运行 `cute_tensor_tile_demo`：

- 知道 `Tensor = Engine + Layout`。
- 能解释 `make_gmem_ptr`、`make_smem_ptr` 和 register owning tensor 的差异。
- 能解释 `local_tile` 如何从 full tensor 得到 CTA tile。
- 能解释 `local_partition` 如何从 tile 得到 per-thread subtensor。

中间里程碑是能独立解释并运行 `cute_layout_algebra_demo`：

- 知道 `coalesce` 是不改变 1D 映射的 layout 简化。
- 知道 `composition(A,B)` 是 `A(B(i))`。
- 知道 `complement` 描述 tiler 没有覆盖的 rest layout。
- 知道 `logical_divide`、`zipped_divide` 如何把 tile mode 和 rest mode 分开。
- 知道 `logical_product`、`blocked_product`、`raked_product` 如何表达 tile 重复和排列。

完成后再进入 `notes/learning-plan.md` 中的 `TiledCopy`、`TiledMMA` 和 HGEMM pipeline 阶段。
