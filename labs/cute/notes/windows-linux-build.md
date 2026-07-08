# CuTe Windows / Linux Build Notes

这份笔记记录 CuTe lab 的构建前置条件、直接 CMake 命令、脚本命令和常见问题。主入口文档见 `labs/cute/README.md`。

## 构建输入

CuTe 是 header-only，随 CUTLASS 分发。本 lab 只需要 CUTLASS checkout 中的头文件：

```text
3rdparty/cutlass/include/cute/tensor.hpp
```

CMake 变量：

```text
AI_SYSTEM_CUTLASS_ROOT=<repo>/3rdparty/cutlass
```

如果配置阶段找不到这个路径，`labs/cute` 会被跳过，不会生成 `cute_layout_mapping` 目标。

## 基础要求

通用要求：

- CUDA Toolkit with `nvcc`.
- CMake 3.22+，与根工程 `CMakePresets.json` 保持一致。
- C++17 host compiler，CuTe demo 目标要求 `cxx_std_17`。
- 本地 CUTLASS checkout，默认在 `3rdparty/cutlass`。

Windows baseline：

- Windows 10 或 Windows 11。
- Visual Studio 2022，包含 C++ 工具链。
- CUDA Toolkit 版本需要支持目标架构 `sm_120`。
- 构建大 CUTLASS tree 前建议启用 Windows long path。本 CuTe demo 只用头文件，路径压力较小。

Linux / WSL baseline：

- Linux 或 WSL。
- 与当前 CUDA Toolkit 兼容的 GCC。
- Make，因为当前 Linux preset 使用 `Unix Makefiles`。
- CUDA Toolkit 版本需要支持目标架构 `sm_89`。

## GPU Profiles

根工程通过 `AI_SYSTEM_GPU_PROFILE` 映射 CUDA architecture：

```text
Windows preset -> RTX 5060  -> sm_120 -> CMAKE_CUDA_ARCHITECTURES=120
Linux preset   -> RTX 4090D -> sm_89  -> CMAKE_CUDA_ARCHITECTURES=89
```

对应 preset：

| OS | Configure preset | Build preset | 生成目录 |
| --- | --- | --- | --- |
| Windows | `windows-vs2022-cuda-release` | `windows-vs2022-cuda-release` | `out/build/windows-vs2022-cuda-release` |
| Linux / WSL | `linux-make-cuda-release` | `linux-make-cuda-release` | `out/build/linux-make-cuda-release` |

如果 `nvcc` 报 `Unsupported gpu architecture`，先运行环境检查脚本确认目标 SM 是否被当前 CUDA Toolkit 支持：

```powershell
.\labs\cute\scripts\check_env.ps1
```

```bash
labs/cute/scripts/check_env.sh
```

## Windows: direct CMake

从仓库根目录执行：

```powershell
cd D:\workspace\learing\AI_system

.\labs\cute\scripts\check_env.ps1

cmake -S . --preset windows-vs2022-cuda-release -DAI_SYSTEM_CUTLASS_ROOT="D:\workspace\learing\AI_system\3rdparty\cutlass"

cmake --build --preset windows-vs2022-cuda-release --config Release --target cute_layout_mapping
cmake --build --preset windows-vs2022-cuda-release --config Release --target cute_layout_algebra_demo
cmake --build --preset windows-vs2022-cuda-release --config Release --target cute_tensor_tile_demo

.\out\build\windows-vs2022-cuda-release\labs\cute\Release\cute_layout_mapping.exe
.\out\build\windows-vs2022-cuda-release\labs\cute\Release\cute_layout_algebra_demo.exe
.\out\build\windows-vs2022-cuda-release\labs\cute\Release\cute_tensor_tile_demo.exe
```

预期运行结果包含：

```text
CuTe layout mapping smoke test
layout mapping check passed
```

如果用环境变量指定 CUTLASS：

```powershell
$env:CUTLASS_ROOT="D:\path\to\cutlass"
cmake -S . --preset windows-vs2022-cuda-release -DAI_SYSTEM_CUTLASS_ROOT="$env:CUTLASS_ROOT"
```

## Linux / WSL: direct CMake

从仓库根目录执行：

```bash
cd /workspace/AI_system

labs/cute/scripts/check_env.sh

cmake -S . --preset linux-make-cuda-release -DAI_SYSTEM_CUTLASS_ROOT="${PWD}/3rdparty/cutlass"

cmake --build --preset linux-make-cuda-release --target cute_layout_mapping -j"$(nproc)"
cmake --build --preset linux-make-cuda-release --target cute_layout_algebra_demo -j"$(nproc)"
cmake --build --preset linux-make-cuda-release --target cute_tensor_tile_demo -j"$(nproc)"

./out/build/linux-make-cuda-release/labs/cute/cute_layout_mapping
./out/build/linux-make-cuda-release/labs/cute/cute_layout_algebra_demo
./out/build/linux-make-cuda-release/labs/cute/cute_tensor_tile_demo
```

预期运行结果包含：

```text
CuTe layout mapping smoke test
layout mapping check passed
```

如果用环境变量指定 CUTLASS：

```bash
export CUTLASS_ROOT=/path/to/cutlass
cmake -S . --preset linux-make-cuda-release -DAI_SYSTEM_CUTLASS_ROOT="${CUTLASS_ROOT}"
```

## 脚本等价命令

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

脚本和 direct CMake 的关系：

- `configure.ps1` 等价于 `cmake -S <repo> --preset windows-vs2022-cuda-release -DAI_SYSTEM_CUTLASS_ROOT=<path>`。
- `build.ps1` 等价于 `cmake --build --preset windows-vs2022-cuda-release --config Release --target cute_layout_mapping`。
- `build.ps1 -Target cute_layout_algebra_demo` 会编译 Layout Algebra demo。
- `build.ps1 -Target cute_tensor_tile_demo` 会编译 Tensor/local_tile/partition demo。
- `configure.sh` 等价于 `cmake -S <repo> --preset linux-make-cuda-release -DAI_SYSTEM_CUTLASS_ROOT=<path>`。
- `build.sh` 等价于 `cmake --build --preset linux-make-cuda-release --target cute_layout_mapping`。
- `build.sh --target cute_layout_algebra_demo` 会编译 Layout Algebra demo。
- `build.sh --target cute_tensor_tile_demo` 会编译 Tensor/local_tile/partition demo。

## 清理和重新配置

当你切换 CUDA Toolkit、CUTLASS 路径或 GPU profile 后，建议删除对应 build tree 后重新配置。

Windows：

```powershell
Remove-Item -Recurse -Force .\out\build\windows-vs2022-cuda-release
cmake -S . --preset windows-vs2022-cuda-release -DAI_SYSTEM_CUTLASS_ROOT="D:\workspace\learing\AI_system\3rdparty\cutlass"
```

Linux / WSL：

```bash
rm -rf ./out/build/linux-make-cuda-release
cmake -S . --preset linux-make-cuda-release -DAI_SYSTEM_CUTLASS_ROOT="${PWD}/3rdparty/cutlass"
```

## Troubleshooting Checklist

`cute_layout_mapping` target 不存在：

- 看 CMake configure 输出是否出现 `Skipping labs/cute examples`。
- 确认 `AI_SYSTEM_ENABLE_CUDA=ON`。
- 确认 `AI_SYSTEM_BUILD_LABS=ON`。
- 确认 `<CUTLASS_ROOT>/include/cute/tensor.hpp` 存在。

CMake 找不到 CUDA：

- 确认 `nvcc --version` 可运行。
- Windows 确认 CUDA Toolkit 和 Visual Studio 2022 集成正常。
- Linux/WSL 确认 `PATH` 包含 CUDA `bin` 目录。

编译时报 GPU arch 不支持：

- Windows preset 目标是 `sm_120`。
- Linux preset 目标是 `sm_89`。
- 运行 `nvcc --list-gpu-code` 查看当前 Toolkit 支持列表。

运行 exe 失败或路径不对：

- Windows Visual Studio 多配置生成器会把 exe 放在 target build 子目录下的 `Release` 目录：`out/build/windows-vs2022-cuda-release/labs/cute/Release/`。
- Linux Make 单配置生成器会把 exe 放在 target 所在 build 子目录：`out/build/linux-make-cuda-release/labs/cute/`。

修改代码后是否需要重新 configure：

- 只改 `.cu` 或 `.hpp`：通常只需要重新 build。
- 改 CMake preset、CUTLASS 路径、CUDA Toolkit、GPU profile：建议重新 configure，必要时清理 build tree。
