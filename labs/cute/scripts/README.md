# CuTe Scripts

这些脚本只是根工程 CMake preset 的薄封装。完整 direct CMake 命令见：

```text
labs/cute/README.md
labs/cute/notes/windows-linux-build.md
```

## Windows

默认目标：

- Preset: `windows-vs2022-cuda-release`
- GPU profile: RTX 5060 / `sm_120`
- Default target: `cute_layout_mapping`
- Algebra target: `cute_layout_algebra_demo`
- Tensor target: `cute_tensor_tile_demo`
- Configuration: `Release`
- CUTLASS root: `$env:CUTLASS_ROOT`，否则 `<repo>\3rdparty\cutlass`

```powershell
cd D:\workspace\learing\AI_system
.\labs\cute\scripts\check_env.ps1
.\labs\cute\scripts\configure.ps1
.\labs\cute\scripts\build.ps1
.\out\build\windows-vs2022-cuda-release\labs\cute\Release\cute_layout_mapping.exe
```

覆盖 CUTLASS root：

```powershell
.\labs\cute\scripts\check_env.ps1 -CutlassRoot "D:\path\to\cutlass"
.\labs\cute\scripts\configure.ps1 -CutlassRoot "D:\path\to\cutlass"
```

指定 build target 或 configuration：

```powershell
.\labs\cute\scripts\build.ps1 -Target cute_layout_mapping -Configuration Release
.\labs\cute\scripts\build.ps1 -Target cute_layout_algebra_demo -Configuration Release
.\labs\cute\scripts\build.ps1 -Target cute_tensor_tile_demo -Configuration Release
```

## Linux / WSL

默认目标：

- Preset: `linux-make-cuda-release`
- GPU profile: RTX 4090D / `sm_89`
- Default target: `cute_layout_mapping`
- Algebra target: `cute_layout_algebra_demo`
- Tensor target: `cute_tensor_tile_demo`
- CUTLASS root: `$CUTLASS_ROOT`，否则 `<repo>/3rdparty/cutlass`

```bash
cd /workspace/AI_system
labs/cute/scripts/check_env.sh
labs/cute/scripts/configure.sh
labs/cute/scripts/build.sh
./out/build/linux-make-cuda-release/labs/cute/cute_layout_mapping
```

覆盖 CUTLASS root：

```bash
labs/cute/scripts/check_env.sh /path/to/cutlass
labs/cute/scripts/configure.sh --cutlass-root /path/to/cutlass
```

指定 build target：

```bash
labs/cute/scripts/build.sh --target cute_layout_mapping
labs/cute/scripts/build.sh --target cute_layout_algebra_demo
labs/cute/scripts/build.sh --target cute_tensor_tile_demo
```

## 等价 CMake 命令

Windows:

```powershell
cmake -S . --preset windows-vs2022-cuda-release -DAI_SYSTEM_CUTLASS_ROOT="D:\workspace\learing\AI_system\3rdparty\cutlass"
cmake --build --preset windows-vs2022-cuda-release --config Release --target cute_layout_mapping
cmake --build --preset windows-vs2022-cuda-release --config Release --target cute_layout_algebra_demo
cmake --build --preset windows-vs2022-cuda-release --config Release --target cute_tensor_tile_demo
```

Linux / WSL:

```bash
cmake -S . --preset linux-make-cuda-release -DAI_SYSTEM_CUTLASS_ROOT="${PWD}/3rdparty/cutlass"
cmake --build --preset linux-make-cuda-release --target cute_layout_mapping -j"$(nproc)"
cmake --build --preset linux-make-cuda-release --target cute_layout_algebra_demo -j"$(nproc)"
cmake --build --preset linux-make-cuda-release --target cute_tensor_tile_demo -j"$(nproc)"
```

## 预期输出

运行 demo 后应看到：

```text
CuTe layout mapping smoke test
layout mapping check passed
CuTe layout algebra demo
layout algebra check passed
CuTe tensor/local_tile/partition demo
tensor/local_tile/partition check passed
```
