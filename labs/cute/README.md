# CuTe Lab

CuTe 是深入阅读 CUTLASS 3.x/4.x 之前的第一站。本实验目录先从 `Layout -> offset` 的代数关系开始，再逐步扩展到 tensor tiling、copy partition、`TiledMMA` 和完整 HGEMM pipeline。

当前可直接编译运行的 demo 是：

```text
cute_layout_mapping
cute_layout_algebra_demo
cute_tensor_tile_demo
cute_copy_g2s_naive
cute_copy_g2s_cpasync
cute_copy_s2r
cute_smem_swizzle_demo
cute_gemm_sm80_demo
cute_gemm_sm70_demo
cute_gemm_v2_fma_demo
```

- `cute_layout_mapping` 是 host-side `Layout` smoke test，用 CuTe `Layout` 验证 `(M,K)`、`(N,K)` 和 shared-memory stage layout 的 offset 计算。
- `cute_layout_algebra_demo` 是 host-side `Layout Algebra` smoke test，对应 NVIDIA `02_layout_algebra`，验证 `coalesce`、`composition`、`complement`、divide 和 product。
- `cute_tensor_tile_demo` 是 host-side `Tensor/local_tile/partition` smoke test，用同一个逻辑值串起 global tensor、shared tensor、register fragment 和 per-thread partition。
- `cute_copy_g2s_naive` 比较 scalar `local_partition`、scalar `TiledCopy` 和 128-bit `TiledCopy`，并验证 TV mapping 与 ragged tile。
- `cute_copy_g2s_cpasync` 演示 `cp.async` 的 cache policy、async group 和同步顺序。
- `cute_copy_s2r` 验证 shared-memory partition 到 per-thread register fragment 的坐标和值。
- `cute_smem_swizzle_demo` 对比普通、padding、swizzle shared layout 的 offset、bank 和实际访问耗时。
- `cute_gemm_sm80_demo` 使用 Ampere MMA、128-bit `cp.async`、`ldmatrix`、三阶段 shared-memory buffer 和 register fragment pipeline 实现 NT/TN HGEMM。
- `cute_gemm_sm70_demo` 使用 Volta `m8n8k4` MMA、向量化 G2R→S copy、双阶段 shared-memory buffer 和 register pipeline 实现 NT/TN HGEMM。
- `cute_gemm_v2_fma_demo` 不使用 MMA、`cp.async` 或 `ldmatrix`，使用向量化 copy、双阶段 shared-memory buffer 和 FP32 scalar FMA 实现 NT/TN GEMM。

## 目录结构

```text
labs/cute/
|-- CMakeLists.txt
|-- README.md
|-- examples/
|   |-- README.md
|   |-- cute_layout_algebra_demo.cu
|   |-- cute_layout_mapping.cu
|   |-- cute_tensor_tile_demo.cu
|   |-- cute_copy_g2s_naive.cu
|   |-- cute_copy_g2s_cpasync.cu
|   |-- cute_copy_s2r.cu
|   |-- cute_smem_swizzle_demo.cu
|   |-- cute_gemm_demo_common.hpp
|   |-- cute_gemm_sm80_demo.cu
|   |-- cute_gemm_sm70_demo.cu
|   `-- cute_gemm_v2_fma_demo.cu
|-- notes/
|   |-- README.md
|   |-- learning-plan.md
|   |-- tensor-local-tile-partition.md
|   |-- tiled-copy.md
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
    |-- README.md
    `-- w16-copy-report.md
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
- https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html
- https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html

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
cmake --build --preset windows-vs2022-cuda-release --config Release --target cute_copy_g2s_naive cute_copy_g2s_cpasync cute_copy_s2r cute_smem_swizzle_demo
cmake --build --preset windows-vs2022-cuda-release --config Release --target cute_gemm_sm80_demo cute_gemm_sm70_demo cute_gemm_v2_fma_demo

# 运行。
.\out\build\windows-vs2022-cuda-release\labs\cute\Release\cute_layout_mapping.exe
.\out\build\windows-vs2022-cuda-release\labs\cute\Release\cute_layout_algebra_demo.exe
.\out\build\windows-vs2022-cuda-release\labs\cute\Release\cute_tensor_tile_demo.exe
.\out\build\windows-vs2022-cuda-release\labs\cute\Release\cute_copy_g2s_naive.exe
.\out\build\windows-vs2022-cuda-release\labs\cute\Release\cute_copy_g2s_cpasync.exe
.\out\build\windows-vs2022-cuda-release\labs\cute\Release\cute_copy_s2r.exe
.\out\build\windows-vs2022-cuda-release\labs\cute\Release\cute_smem_swizzle_demo.exe
.\out\build\windows-vs2022-cuda-release\labs\cute\Release\cute_gemm_sm80_demo.exe
.\out\build\windows-vs2022-cuda-release\labs\cute\Release\cute_gemm_sm70_demo.exe
.\out\build\windows-vs2022-cuda-release\labs\cute\Release\cute_gemm_v2_fma_demo.exe
```

期望最后看到：

```text
layout mapping check passed
layout algebra check passed
tensor/local_tile/partition check passed
W16 global-to-shared copy checks passed
W16 cp.async checks passed
W16 shared-to-register checks passed
W16 shared-memory swizzle checks passed
cute_gemm_sm80_mma layout=NT ... mismatches=0
cute_gemm_sm80_mma layout=TN ... mismatches=0
cute_gemm_v2_fma   layout=NT ... mismatches=0
cute_gemm_v2_fma   layout=TN ... mismatches=0
```

Windows 默认 GPU 是 `sm_120`，所以 SM70 demo 会明确打印 `SKIPPED`，不会在不兼容设备上启动 Volta kernel。

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
cmake --build --preset linux-make-cuda-release --target cute_copy_g2s_naive cute_copy_g2s_cpasync cute_copy_s2r cute_smem_swizzle_demo -j"$(nproc)"
cmake --build --preset linux-make-cuda-release --target cute_gemm_sm80_demo cute_gemm_sm70_demo cute_gemm_v2_fma_demo -j"$(nproc)"

# 运行。
./out/build/linux-make-cuda-release/labs/cute/cute_layout_mapping
./out/build/linux-make-cuda-release/labs/cute/cute_layout_algebra_demo
./out/build/linux-make-cuda-release/labs/cute/cute_tensor_tile_demo
./out/build/linux-make-cuda-release/labs/cute/cute_copy_g2s_naive
./out/build/linux-make-cuda-release/labs/cute/cute_copy_g2s_cpasync
./out/build/linux-make-cuda-release/labs/cute/cute_copy_s2r
./out/build/linux-make-cuda-release/labs/cute/cute_smem_swizzle_demo
./out/build/linux-make-cuda-release/labs/cute/cute_gemm_sm80_demo
./out/build/linux-make-cuda-release/labs/cute/cute_gemm_sm70_demo
./out/build/linux-make-cuda-release/labs/cute/cute_gemm_v2_fma_demo
```

期望最后看到：

```text
layout mapping check passed
layout algebra check passed
tensor/local_tile/partition check passed
W16 global-to-shared copy checks passed
W16 cp.async checks passed
W16 shared-to-register checks passed
W16 shared-memory swizzle checks passed
cute_gemm_sm80_mma layout=NT ... mismatches=0
cute_gemm_sm80_mma layout=TN ... mismatches=0
cute_gemm_v2_fma   layout=NT ... mismatches=0
cute_gemm_v2_fma   layout=TN ... mismatches=0
```

Linux 默认 GPU 是 `sm_89`，所以 SM70 demo 同样会打印 `SKIPPED`。

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
.\labs\cute\scripts\build.ps1 -Target cute_copy_g2s_naive -Configuration Release
.\labs\cute\scripts\build.ps1 -Target cute_copy_g2s_cpasync -Configuration Release
.\labs\cute\scripts\build.ps1 -Target cute_copy_s2r -Configuration Release
.\labs\cute\scripts\build.ps1 -Target cute_smem_swizzle_demo -Configuration Release
.\labs\cute\scripts\build.ps1 -Target cute_gemm_sm80_demo -Configuration Release
.\labs\cute\scripts\build.ps1 -Target cute_gemm_sm70_demo -Configuration Release
.\labs\cute\scripts\build.ps1 -Target cute_gemm_v2_fma_demo -Configuration Release
```

```bash
labs/cute/scripts/build.sh --target cute_layout_mapping
labs/cute/scripts/build.sh --target cute_layout_algebra_demo
labs/cute/scripts/build.sh --target cute_tensor_tile_demo
labs/cute/scripts/build.sh --target cute_copy_g2s_naive
labs/cute/scripts/build.sh --target cute_copy_g2s_cpasync
labs/cute/scripts/build.sh --target cute_copy_s2r
labs/cute/scripts/build.sh --target cute_smem_swizzle_demo
labs/cute/scripts/build.sh --target cute_gemm_sm80_demo
labs/cute/scripts/build.sh --target cute_gemm_sm70_demo
labs/cute/scripts/build.sh --target cute_gemm_v2_fma_demo
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

四个 W16 demo 共同实现：

```text
global tensor
    -> local_tile
    -> Copy_Atom / TiledCopy / ThrCopy
    -> partition_S / partition_D
    -> shared tensor
    -> register fragment
```

| Target | 主要输出 |
| --- | --- |
| `cute_copy_g2s_naive` | 128-bit TV layout、thread 0/1/31/32/127 坐标、4096 元素覆盖、三种同步 copy、ragged `copy_if` |
| `cute_copy_g2s_cpasync` | `CACHEALWAYS/CACHEGLOBAL`、`fence/wait/syncthreads`、aligned/ragged correctness |
| `cute_copy_s2r` | 所有 register fragment 的坐标覆盖，以及 thread 45 的 shared offset/value |
| `cute_smem_swizzle_demo` | plain/padded/swizzled offset 与 bank 表、bank conflict 和微基准 |

W16 使用 `half` A tile `128x32`、128 个线程和 128-bit copy。每线程负责 32 个
half，也就是 4 条 16-byte copy instruction。完整概念、代码映射和边界处理见
`notes/tiled-copy.md`，本机验证记录见 `reports/w16-copy-report.md`。

## W17 GEMM 编译、运行与调试

三个 W17 demo 使用相同的逻辑问题、输入类型、正确性检查和性能统计：

```text
A: (M,K)
B: (N,K)
C(m,n) = sum_k A(m,k) * B(n,k)
input/output: FP16
accumulator: FP32
baseline: cublasGemmEx
```

NT 使用 A/B stride `(1,M)`、`(1,N)`，TN 使用 A/B stride `(K,1)`。两种布局只改变物理存储方式，不改变 A、B 的逻辑值和 GEMM 结果。

| Target | CTA tile | Threads | G2S | S2R | Compute | Stage buffer |
| --- | --- | ---: | --- | --- | --- | --- |
| `cute_gemm_sm80_demo` | `128x128x64` | 128 | 128-bit `cp.async` | TN `LDSM_N`，NT `LDSM_T` | `m16n8k16` MMA | 3-stage SMEM + register fragment |
| `cute_gemm_sm70_demo` | `128x128x32` | 128 | 128-bit G2R→S | 普通 register copy | `m8n8k4` MMA | 2-stage SMEM + G2R/S2R register |
| `cute_gemm_v2_fma_demo` | `128x128x8` | 256 | 64-bit G2R→S | half→float register copy | FP32 scalar FMA | 2-stage SMEM + G2R/S2R register |

三个程序的命令行一致：

```text
<executable> [M] [N] [K] [nt|tn|both] [iterations] [warmups]
```

默认值为：

```text
M = 4096
N = 4096
K = 4096
layout = both
iterations = 20
warmups = 5
```

教学 kernel 只处理完整 CTA tile，暂不包含边界 predicate：

| Target | Shape 约束 |
| --- | --- |
| `cute_gemm_sm80_demo` | `M%128==0`、`N%128==0`、`K%64==0` |
| `cute_gemm_sm70_demo` | `M%128==0`、`N%128==0`、`K%32==0` |
| `cute_gemm_v2_fma_demo` | `M%128==0`、`N%128==0`、`K%8==0` |

为了快速验证，可以先运行较小问题：

```powershell
.\out\build\windows-vs2022-cuda-release\labs\cute\Release\cute_gemm_sm80_demo.exe 256 256 128 both 3 1
.\out\build\windows-vs2022-cuda-release\labs\cute\Release\cute_gemm_v2_fma_demo.exe 256 256 64 both 3 1
```

```bash
./out/build/linux-make-cuda-release/labs/cute/cute_gemm_sm80_demo 256 256 128 both 3 1
./out/build/linux-make-cuda-release/labs/cute/cute_gemm_v2_fma_demo 256 256 64 both 3 1
```

每一行输出包含：

```text
kernel time
kernel TFLOP/s
cuBLAS time
cuBLAS TFLOP/s
relative cuBLAS percentage
max_abs / max_rel / mismatches
```

正确性通过的关键标志是：

```text
mismatches=0
```

### Lineinfo 构建

分析 SASS、Nsight Compute source correlation 或 register/shared-memory 指标时，使用优化构建加 `-lineinfo`，不要使用会显著改变性能特征的 device debug `-G`。

Windows：

```powershell
cmake -S . --preset windows-vs2022-cuda-release-lineinfo
cmake --build --preset windows-vs2022-cuda-release-lineinfo --config Release --target cute_gemm_sm80_demo cute_gemm_v2_fma_demo
```

Linux / WSL：

```bash
cmake -S . --preset linux-make-cuda-release-lineinfo
cmake --build --preset linux-make-cuda-release-lineinfo --target cute_gemm_sm80_demo cute_gemm_v2_fma_demo -j"$(nproc)"
```

使用 Nsight Compute 时把迭代次数降到 1，避免重复采集同一个 kernel：

```powershell
ncu --set basic .\out\build\windows-vs2022-cuda-release-lineinfo\labs\cute\Release\cute_gemm_sm80_demo.exe 1024 1024 1024 tn 1 0
```

```bash
ncu --set basic ./out/build/linux-make-cuda-release-lineinfo/labs/cute/cute_gemm_sm80_demo 1024 1024 1024 tn 1 0
```

重点检查：

- launch 的 threads/block、register/thread 和 dynamic shared memory。
- G2S 是否出现连续的 16-byte global transaction。
- shared-memory bank conflict。
- Tensor Core pipe、FMA pipe 和 memory pipe 的利用率。
- `cp.async`、S2R load 和 MMA 是否按预期交错。

### SASS 检查

Windows SM80：

```powershell
$exe = ".\out\build\windows-vs2022-cuda-release\labs\cute\Release\cute_gemm_sm80_demo.exe"
cuobjdump --dump-sass $exe | Select-String "HMMA|LDSM|LDGSTS"
```

在当前工具链和 GPU 目标下，预期看到：

```text
HMMA.16816.F32
LDSM.16.MT88.4
LDGSTS
```

其中 `LDGSTS` 是 `cp.async` 落到 SASS 后的表现。NT/TN 的 `LDSM` 修饰形式可能不同，不应只按完整字符串判断。

V2 FMA：

```powershell
$exe = ".\out\build\windows-vs2022-cuda-release\labs\cute\Release\cute_gemm_v2_fma_demo.exe"
$sass = cuobjdump --dump-sass $exe
"FFMA=" + (($sass | Select-String "FFMA").Count)
"HMMA=" + (($sass | Select-String "HMMA").Count)
"LDSM=" + (($sass | Select-String "LDSM").Count)
"LDGSTS=" + (($sass | Select-String "LDGSTS").Count)
```

V2 应存在 `FFMA`，并且 `HMMA`、`LDSM`、`LDGSTS` 都为 0。Linux 可以使用：

```bash
cuobjdump --dump-sass \
  ./out/build/linux-make-cuda-release/labs/cute/cute_gemm_sm80_demo \
  | grep -E 'HMMA|LDSM|LDGSTS'
```

### SM70 编译和运行限制

Windows RTX 5060 是 `sm_120`，Linux RTX 4090D 是 `sm_89`。SM70 demo 在这两个环境都会打印 `SKIPPED`，避免在不兼容设备上启动 Volta kernel。

CUDA 13 已不提供 `sm_70` code generation，但仍支持用 `sm_75` 编译检查真实的 `m8n8k4` kernel body：

```powershell
cmake -S . -B out/build/windows-sm75-check -G "Visual Studio 17 2022" -A x64 `
  -DAI_SYSTEM_ENABLE_CUDA=ON `
  -DAI_SYSTEM_BUILD_LABS=ON `
  -DAI_SYSTEM_ENABLE_TESTS=OFF `
  -DAI_SYSTEM_GPU_PROFILE=75

cmake --build out/build/windows-sm75-check --config Release --target cute_gemm_sm70_demo

$exe = ".\out\build\windows-sm75-check\labs\cute\Release\cute_gemm_sm70_demo.exe"
cuobjdump --dump-sass $exe | Select-String "HMMA.884"
```

该检查预期出现 `HMMA.884.F32.F32.STEP0..3`，但不能替代 Volta 实机性能测试。真正运行 SM70 需要 compute capability 7.x GPU，并且 `nvcc --list-gpu-code` 必须包含对应目标架构。

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

GEMM 输出 `misaligned address`：

首先检查 M/N/K 是否满足 CTA tile 约束，以及 A/B device pointer 是否至少满足 copy atom 的 16-byte 或 8-byte alignment。对于向量化 G2S，source 和 destination 的同一组 value 都必须物理连续并满足对齐；给 shared layout 增加 padding 后，也要重新检查每个 K slice 和 stage 起点的对齐。

SM80 launch 报 dynamic shared-memory 错误：

SM80 demo 的三阶段 A/B tile 需要较大的 dynamic shared memory。检查 GPU 的 per-block shared-memory 上限，以及 `cudaFuncAttributeMaxDynamicSharedMemorySize`、shared-memory carveout 和实际 `SharedStorage` 大小。

SM70 demo 只打印 `SKIPPED`：

这是非 7.x GPU 上的预期行为。默认 `sm_120`/`sm_89` preset 用于编译当前机器可运行的代码，不代表能够执行 Volta `m8n8k4`。使用前面的 `sm_75` 检查构建验证 kernel 和 SASS，使用 7.x GPU 获取实际正确性和性能数据。

修改了源码但运行结果没变：

只重新编译当前目标即可：

```powershell
cmake --build --preset windows-vs2022-cuda-release --config Release --target cute_layout_mapping
cmake --build --preset windows-vs2022-cuda-release --config Release --target cute_layout_algebra_demo
cmake --build --preset windows-vs2022-cuda-release --config Release --target cute_tensor_tile_demo
cmake --build --preset windows-vs2022-cuda-release --config Release --target cute_copy_g2s_naive cute_copy_g2s_cpasync cute_copy_s2r cute_smem_swizzle_demo
cmake --build --preset windows-vs2022-cuda-release --config Release --target cute_gemm_sm80_demo cute_gemm_sm70_demo cute_gemm_v2_fma_demo
```

```bash
cmake --build --preset linux-make-cuda-release --target cute_layout_mapping -j"$(nproc)"
cmake --build --preset linux-make-cuda-release --target cute_layout_algebra_demo -j"$(nproc)"
cmake --build --preset linux-make-cuda-release --target cute_tensor_tile_demo -j"$(nproc)"
cmake --build --preset linux-make-cuda-release --target cute_copy_g2s_naive cute_copy_g2s_cpasync cute_copy_s2r cute_smem_swizzle_demo -j"$(nproc)"
cmake --build --preset linux-make-cuda-release --target cute_gemm_sm80_demo cute_gemm_sm70_demo cute_gemm_v2_fma_demo -j"$(nproc)"
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

W16 里程碑是能独立解释并运行四个 copy demo：

- 能从 `ThrLayout + ValLayout` 手算 `(thread_id,value_id) -> tile coordinate`。
- 能区分 `Copy_Atom`、`TiledCopy`、`ThrCopy`、`partition_S` 和 `partition_D`。
- 能解释 128-bit vector copy 的连续性和 alignment 要求。
- 能解释 `cp_async_fence`、`cp_async_wait` 和 `__syncthreads` 的不同作用域。
- 能计算 shared-memory bank，并解释 padding 和 swizzle 的取舍。
- 能使用 identity coordinate tensor 和 `copy_if` 处理 ragged tile。

W17 里程碑是能独立解释、运行和反汇编三个 GEMM demo：

- 能区分 `MMA_Operation`、`MMA_Atom`、`TiledMMA` 和 `ThrMMA`。
- 能解释 `partition_A/B/C`、`make_fragment_A/B/C` 和每线程 fragment 的 shape。
- 能解释 CTA tile、MMA atom tile、warp tile 和 K-block 之间的关系。
- 能说明 NT/TN stride 如何影响 G2S CopyTile、shared layout 和 `LDSM_N/LDSM_T`。
- 能画出 Gmem→Smem→Register→MMA/FMA→C 的完整数据流。
- 能解释 shared-memory stage buffer 与 G2R/S2R register buffer 分别隐藏哪一段延迟。
- 能用 cuBLAS baseline 验证正确性并计算相对性能。
- 能在 SASS 中确认 SM80 的 `LDGSTS/LDSM/HMMA`、SM70 的 `HMMA.884` 和 V2 的 `FFMA`。

完成 W17 后，可以继续加入边界 predicate、不同 stage 数、FP16/FP32 accumulator 对比和更完整的 epilogue。
