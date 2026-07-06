# CUTLASS Lab

This lab starts after the CuTe basics. It focuses on CUTLASS Quickstart, CUTLASS Profiler, GEMM API 3.x, Efficient GEMM, parameter sweeps, and fused epilogues.

## Structure

```text
labs/cutlass/
|-- CMakeLists.txt
|-- README.md
|-- examples/
|   |-- README.md
|   `-- cutlass_header_probe.cu
|-- notes/
|   |-- README.md
|   |-- learning-plan.md
|   |-- gpu-targets.md
|   `-- windows-linux-build.md
|-- scripts/
|   |-- README.md
|   |-- build.ps1
|   |-- build.sh
|   |-- check_env.ps1
|   |-- check_env.sh
|   |-- configure.ps1
|   |-- configure.sh
|   |-- run_profiler.ps1
|   `-- run_profiler.sh
|-- benchmarks/
|   `-- README.md
`-- reports/
    `-- README.md
```

## External Dependency

This repository uses a local CUTLASS 4.5 checkout:

```text
3rdparty/cutlass
```

You can override it with `CUTLASS_ROOT`, `-CutlassRoot`, `--cutlass-root`, or `-DAI_SYSTEM_CUTLASS_ROOT=...` when testing another checkout.

The local CMake targets include:

- `3rdparty/cutlass/include`
- `3rdparty/cutlass/tools/util/include`

## Windows Support

NVIDIA CUTLASS documentation states that Windows builds are supported with Visual Studio. The practical baseline for this lab is:

- Windows 10 or 11
- Visual Studio 2022
- CUDA Toolkit new enough for the target architecture
- CMake 3.18+
- Long path support enabled on Windows

For Windows, this lab targets RTX 5060. NVIDIA lists RTX 5060 under compute capability `12.0`, so the Windows preset uses `sm_120`.

For Linux / WSL, this lab targets RTX 4090D. NVIDIA lists RTX 4090-class Ada GPUs under compute capability `8.9`, so the Linux preset uses `sm_89`.

Official references:

- https://docs.nvidia.com/cutlass/latest/media/docs/cpp/build/building_in_windows_with_visual_studio.html
- https://docs.nvidia.com/cutlass/latest/media/docs/cpp/quickstart.html
- https://developer.nvidia.com/cuda/gpus

Before the first build, run `check_env.ps1` or `check_env.sh`. It checks the CUDA Toolkit, visible GPUs, `CUTLASS_ROOT`, whether CUTLASS/CuTe headers are present in the CUDA Toolkit include directory, and whether the installed `nvcc` lists the expected target SM.

## Build

Windows 10/11 + Visual Studio 2022 + RTX 5060:

```powershell
cd D:\workspace\learing\AI_system
.\labs\cutlass\scripts\configure.ps1
.\labs\cutlass\scripts\build.ps1
.\out\build\windows-vs2022-cuda-release\Release\cutlass_header_probe.exe
```

Linux / WSL + RTX 4090D:

```bash
cd /workspace/AI_system
labs/cutlass/scripts/configure.sh
labs/cutlass/scripts/build.sh
./out/build/linux-make-cuda-release/labs/cutlass/cutlass_header_probe
```

The root presets intentionally avoid dual-architecture builds here: Windows is RTX 5060 / `sm_120`, Linux is RTX 4090D / `sm_89`.

## First Milestone

Build `cutlass_header_probe` first. It only verifies that headers, CUDA runtime, target architecture selection, and device detection are wired correctly. After that, move to official examples and `cutlass_profiler`.
