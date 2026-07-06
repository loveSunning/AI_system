# CuTe Lab

CuTe is the first step before reading CUTLASS 3.x deeply. This lab focuses on layout algebra, tensor tiling, copy partitioning, and MMA abstraction.

## Structure

```text
labs/cute/
|-- CMakeLists.txt
|-- README.md
|-- examples/
|   |-- README.md
|   `-- cute_layout_mapping.cu
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

## External Dependency

CuTe is header-only and lives inside the local CUTLASS 4.5 checkout:

```text
3rdparty/cutlass/include/cute
```

The default CMake value for `AI_SYSTEM_CUTLASS_ROOT` is:

```text
D:\workspace\learing\AI_system\3rdparty\cutlass
```

You can override it with `CUTLASS_ROOT`, `-CutlassRoot`, `--cutlass-root`, or `-DAI_SYSTEM_CUTLASS_ROOT=...` when testing another checkout.

Official references:

- https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/00_quickstart.html
- https://docs.nvidia.com/cutlass/latest/media/docs/cpp/quickstart.html

## Build

Windows 10/11 + Visual Studio 2022 + RTX 5060 / `sm_120`:

```powershell
cd D:\workspace\learing\AI_system
.\labs\cute\scripts\configure.ps1
.\labs\cute\scripts\build.ps1
.\out\build\windows-vs2022-cuda-release\Release\cute_layout_mapping.exe
```

Linux / WSL + RTX 4090D / `sm_89`:

```bash
cd /workspace/AI_system
labs/cute/scripts/configure.sh
labs/cute/scripts/build.sh
./out/build/linux-make-cuda-release/labs/cute/cute_layout_mapping
```

This lab intentionally keeps one GPU target per OS: Windows uses RTX 5060 / `sm_120`, Linux uses RTX 4090D / `sm_89`.

Before the first build, run `check_env.ps1` or `check_env.sh`. It checks the CUDA Toolkit, visible GPUs, `CUTLASS_ROOT`, whether CUTLASS/CuTe headers are present in the CUDA Toolkit include directory, and whether the installed `nvcc` lists the expected target SM.

## First Milestone

Build and run `cute_layout_mapping`. It validates:

- `(M,K)` row-major offset mapping.
- `(N,K)` column-like offset mapping.
- `(BM,BK,stage)` shared-memory stage offset mapping.

The goal is to explain `Layout -> offset` by hand before moving to `TiledCopy` and `TiledMMA`.
