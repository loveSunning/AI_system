# CuTe Windows / Linux Build Notes

CuTe is header-only and is distributed inside CUTLASS. The build input for this lab is therefore a local CUTLASS checkout.

## Requirements

Shared requirements:

- CUDA Toolkit with `nvcc`.
- CMake 3.18+.
- C++17 host compiler.
- Local CUTLASS checkout under `3rdparty/cutlass`.

Windows baseline:

- Windows 10 or 11.
- Visual Studio 2022.
- Long path support enabled before building large CUTLASS trees.

Linux / WSL baseline:

- Recent GCC compatible with the installed CUDA Toolkit.
- Make or Ninja.

## GPU Profiles

The root project maps:

```text
Windows preset -> RTX 5060  -> sm_120
Linux preset   -> RTX 4090D -> sm_89
```

For RTX 5060, NVIDIA's CUDA GPU table lists compute capability `12.0`, so the Windows preset uses `CMAKE_CUDA_ARCHITECTURES=120`.

For RTX 4090D, the Linux preset uses Ada Lovelace `sm_89`. If `nvcc` rejects the expected target SM, upgrade the CUDA Toolkit.

## Windows Commands

```powershell
cd D:\workspace\learing\AI_system
.\labs\cute\scripts\check_env.ps1
.\labs\cute\scripts\configure.ps1
.\labs\cute\scripts\build.ps1
.\out\build\windows-vs2022-cuda-release\Release\cute_layout_mapping.exe
```

## Linux / WSL Commands

```bash
cd /workspace/AI_system
labs/cute/scripts/check_env.sh
labs/cute/scripts/configure.sh
labs/cute/scripts/build.sh
./out/build/linux-make-cuda-release/labs/cute/cute_layout_mapping
```
