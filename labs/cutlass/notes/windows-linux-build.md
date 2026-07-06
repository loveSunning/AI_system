# CUTLASS Windows / Linux Build Notes

## Official Support Boundary

NVIDIA CUTLASS documentation includes a Windows + Visual Studio build path. The practical Windows baseline for this lab is:

- Windows 10 or 11.
- Visual Studio 2019 16.11.27 or Visual Studio 2022.
- CUDA Toolkit at least 12.2 for CUTLASS Windows builds; use a toolkit that recognizes `sm_120` for RTX 5060.
- CMake 3.18+.
- Python 3.6+.
- Long path support enabled on Windows before cloning/building large CUTLASS trees.

Linux / WSL remains the easier path for heavy profiler sweeps. This repository intentionally maps Windows to RTX 5060 / `sm_120` and Linux to RTX 4090D / `sm_89`.

References:

- https://docs.nvidia.com/cutlass/latest/media/docs/cpp/build/building_in_windows_with_visual_studio.html
- https://docs.nvidia.com/cutlass/latest/media/docs/cpp/quickstart.html
- https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html

## Local Project Build

The local AI_system examples only need CUTLASS headers:

```text
${CUTLASS_ROOT}/include
${CUTLASS_ROOT}/tools/util/include
```

Windows:

```powershell
cd D:\workspace\learing\AI_system
$env:CUTLASS_ROOT = "D:\deps\cutlass"
.\labs\cutlass\scripts\check_env.ps1
.\labs\cutlass\scripts\configure.ps1 -CutlassRoot $env:CUTLASS_ROOT
.\labs\cutlass\scripts\build.ps1
.\out\build\windows-vs2022-cuda-release\Release\cutlass_header_probe.exe
```

Linux / WSL:

```bash
cd /workspace/AI_system
export CUTLASS_ROOT=/opt/cutlass
labs/cutlass/scripts/check_env.sh
labs/cutlass/scripts/configure.sh --cutlass-root "$CUTLASS_ROOT"
labs/cutlass/scripts/build.sh
./out/build/linux-make-cuda-release/labs/cutlass/cutlass_header_probe
```

## Official CUTLASS Profiler

Build profiler inside the CUTLASS checkout when you need full kernel generation:

```powershell
cd D:\deps\cutlass
cmake -S . -B build\windows-vs2022-5060 -G "Visual Studio 17 2022" -A x64 -DCUTLASS_NVCC_ARCHS=120 -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_UNITY_BUILD_ENABLED=ON
cmake --build build\windows-vs2022-5060 --config Release --target cutlass_profiler -j 4
.\build\windows-vs2022-5060\tools\profiler\Release\cutlass_profiler.exe --operation=Gemm --m=4096 --n=4096 --k=4096
```

```bash
cd /opt/cutlass
cmake -S . -B build/linux-4090d -DCUTLASS_NVCC_ARCHS=89 -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_UNITY_BUILD_ENABLED=ON
cmake --build build/linux-4090d --target cutlass_profiler -j
./build/linux-4090d/tools/profiler/cutlass_profiler --operation=Gemm --m=4096 --n=4096 --k=4096
```
