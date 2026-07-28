# CuTe Scripts

本目录提供 `labs/cute` 的环境检查、CMake 配置和单目标构建脚本。脚本是根工程
CMake preset 的轻量封装，不会下载 CUDA、CUTLASS 或其他依赖。

## 脚本一览

| 平台 | 环境检查 | 配置 | 构建 |
|---|---|---|---|
| Windows | `check_env.ps1` | `configure.ps1` | `build.ps1` |
| Linux / WSL | `check_env.sh` | `configure.sh` | `build.sh` |

平台默认配置：

| 平台 | CMake preset | GPU profile | CUDA architecture |
|---|---|---|---|
| Windows | `windows-vs2022-cuda-release` | RTX 5060 | `sm_120` |
| Linux / WSL | `linux-make-cuda-release` | RTX 4090D | `sm_89` |

CUTLASS 默认使用 `<repo>/3rdparty/cutlass`。如果设置了 `CUTLASS_ROOT`，脚本优先
使用该环境变量指定的 checkout。

## 快速开始

### Windows

在仓库根目录运行：

```powershell
.\labs\cute\scripts\check_env.ps1
.\labs\cute\scripts\configure.ps1
.\labs\cute\scripts\build.ps1 -Target cute_gemm_sm80_demo
.\out\build\windows-vs2022-cuda-release\labs\cute\Release\cute_gemm_sm80_demo.exe
```

默认构建配置为 `Release`。覆盖 CUTLASS 路径：

```powershell
.\labs\cute\scripts\check_env.ps1 -CutlassRoot "D:\path\to\cutlass"
.\labs\cute\scripts\configure.ps1 -CutlassRoot "D:\path\to\cutlass"
```

### Linux / WSL

在仓库根目录运行：

```bash
labs/cute/scripts/check_env.sh
labs/cute/scripts/configure.sh
labs/cute/scripts/build.sh --target cute_gemm_sm80_demo
./out/build/linux-make-cuda-release/labs/cute/cute_gemm_sm80_demo
```

覆盖 CUTLASS 路径：

```bash
labs/cute/scripts/check_env.sh /path/to/cutlass
labs/cute/scripts/configure.sh --cutlass-root /path/to/cutlass
```

## Demo Targets

基础布局与 Tensor：

| Target | 内容 |
|---|---|
| `cute_layout_mapping` | Layout 坐标到线性 offset 的映射 |
| `cute_layout_algebra_demo` | Layout composition、divide、product 等代数操作 |
| `cute_tensor_tile_demo` | Tensor、`local_tile`、partition 与 fragment |

W16 Copy：

| Target | 内容 |
|---|---|
| `cute_copy_g2s_naive` | scalar、TiledCopy 和 128-bit G2S copy |
| `cute_copy_g2s_cpasync` | SM80 `cp.async` 与边界 predicate |
| `cute_copy_s2r` | shared-memory 到 register fragment |
| `cute_smem_swizzle_demo` | shared layout、swizzle 和 bank conflict |

W17 GEMM：

| Target | CTA tile | 数据路径 | 计算路径 |
|---|---:|---|---|
| `cute_gemm_sm80_demo` | `128x128x64` | 128-bit `cp.async` + `ldmatrix` | Ampere `m16n8k16` MMA |
| `cute_gemm_sm70_demo` | `128x128x32` | 128-bit G2R→S，普通 S2R | Volta `m8n8k4` MMA |
| `cute_gemm_v2_fma_demo` | `128x128x8` | 64-bit G2R→S，half→float S2R | scalar FP32 FMA |

构建其他目标时，只需要替换 `-Target` 或 `--target`：

```powershell
.\labs\cute\scripts\build.ps1 -Target cute_gemm_v2_fma_demo
```

```bash
labs/cute/scripts/build.sh --target cute_gemm_v2_fma_demo
```

## GEMM 参数

三个 GEMM demo 使用相同接口：

```text
<executable> [M] [N] [K] [nt|tn|both] [iterations] [warmups]
```

默认值：

```text
M = 4096
N = 4096
K = 4096
layout = both
iterations = 20
warmups = 5
```

例如，只运行 `1024x1024x1024` 的 TN 路径：

```powershell
.\out\build\windows-vs2022-cuda-release\labs\cute\Release\cute_gemm_sm80_demo.exe 1024 1024 1024 tn 20 5
```

```bash
./out/build/linux-make-cuda-release/labs/cute/cute_gemm_sm80_demo 1024 1024 1024 tn 20 5
```

每个程序都会：

1. 为 NT/TN 生成相同的逻辑 A、B 矩阵。
2. 使用 FP16 输入、FP32 累加和 FP16 输出。
3. 使用 `cublasGemmEx` 生成 reference 并测量 cuBLAS baseline。
4. 输出 kernel 时间、TFLOP/s、相对 cuBLAS 百分比和数值误差。

教学 kernel 只处理完整 CTA tile，不包含边界 predicate：

| Demo | Shape 约束 |
|---|---|
| SM80 | `M % 128 == 0`、`N % 128 == 0`、`K % 64 == 0` |
| SM70 | `M % 128 == 0`、`N % 128 == 0`、`K % 32 == 0` |
| V2 FMA | `M % 128 == 0`、`N % 128 == 0`、`K % 8 == 0` |

## SM70 限制

Windows 和 Linux 默认机器分别是 `sm_120` 和 `sm_89`，因此
`cute_gemm_sm70_demo` 会打印 `SKIPPED`，不会在不兼容设备上启动 Volta kernel。

运行 SM70 性能测试需要：

- compute capability 7.x GPU；
- `nvcc --list-gpu-code` 中包含目标架构；
- 配置 CMake 使用对应的 `AI_SYSTEM_GPU_PROFILE`。

CUDA 13 已不提供 `sm_70` code generation，但仍可使用 `sm_75` 对真实
`m8n8k4` kernel body 做编译和 SASS 检查。这个检查不能替代 Volta 实机性能测试。

## SASS 检查

Windows SM80：

```powershell
$exe = ".\out\build\windows-vs2022-cuda-release\labs\cute\Release\cute_gemm_sm80_demo.exe"
cuobjdump --dump-sass $exe | Select-String "HMMA|LDSM|LDGSTS"
```

预期看到：

```text
HMMA.16816.F32
LDSM.16.MT88.4
LDGSTS
```

V2 FMA：

```powershell
$exe = ".\out\build\windows-vs2022-cuda-release\labs\cute\Release\cute_gemm_v2_fma_demo.exe"
cuobjdump --dump-sass $exe | Select-String "FFMA|HMMA|LDSM|LDGSTS"
```

预期存在 `FFMA`，不存在 `HMMA`、`LDSM` 和 `LDGSTS`。

Linux：

```bash
cuobjdump --dump-sass \
  ./out/build/linux-make-cuda-release/labs/cute/cute_gemm_sm80_demo \
  | grep -E 'HMMA|LDSM|LDGSTS'
```

## 等价 CMake 命令

Windows：

```powershell
cmake -S . --preset windows-vs2022-cuda-release `
  -DAI_SYSTEM_CUTLASS_ROOT="$PWD\3rdparty\cutlass"

cmake --build --preset windows-vs2022-cuda-release --config Release `
  --target cute_gemm_sm80_demo cute_gemm_sm70_demo cute_gemm_v2_fma_demo
```

Linux / WSL：

```bash
cmake -S . --preset linux-make-cuda-release \
  -DAI_SYSTEM_CUTLASS_ROOT="${PWD}/3rdparty/cutlass"

cmake --build --preset linux-make-cuda-release \
  --target cute_gemm_sm80_demo cute_gemm_sm70_demo cute_gemm_v2_fma_demo \
  -j"$(nproc)"
```

更完整的构建说明见：

- `labs/cute/README.md`
- `labs/cute/notes/windows-linux-build.md`
