# CUTLASS Lab

This lab uses the local CUTLASS checkout at `3rdparty/cutlass`.

For normal study work, build only this repo's small smoke target. Do not build
the official CUTLASS profiler unless you explicitly need profiler sweeps.

## Direct CMake Build

Windows 10/11 + RTX 5060:

```powershell
cd D:\workspace\learing\AI_system
cmake --preset windows-vs2022-cuda-release
cmake --build --preset windows-vs2022-cuda-release --config Release --target cutlass_header_probe
.\out\build\windows-vs2022-cuda-release\labs\cutlass\Release\cutlass_header_probe.exe
```

Linux / WSL + RTX 4090D:

```bash
cd /workspace/AI_system
cmake --preset linux-make-cuda-release
cmake --build --preset linux-make-cuda-release --target cutlass_header_probe
./out/build/linux-make-cuda-release/labs/cutlass/cutlass_header_probe
```

Expected Linux CMake line:

```text
AI_system CUDA architectures: 89 (RTX 4090D)
```

Expected Windows CMake line:

```text
AI_system CUDA architectures: 120 (RTX 5060)
```

## Optional Wrappers

The wrapper scripts run the same commands with the repo defaults:

```powershell
.\labs\cutlass\scripts\configure.ps1
.\labs\cutlass\scripts\build.ps1
```

```bash
bash ./labs/cutlass/scripts/configure.sh
bash ./labs/cutlass/scripts/build.sh
```

Use `bash script.sh` on Linux if the executable bit is not set.

## Optional Profiler

The official `cutlass_profiler` is separate from this repo's smoke target. It
configures and builds a large generated CUTLASS kernel library, so it is much
slower and much noisier.

Linux / WSL + RTX 4090D:

```bash
bash ./labs/cutlass/scripts/configure_official_cutlass.sh
bash ./labs/cutlass/scripts/build_official_cutlass.sh
bash ./labs/cutlass/scripts/run_profiler.sh
```

Windows + RTX 5060:

```powershell
.\labs\cutlass\scripts\configure_official_cutlass.ps1
.\labs\cutlass\scripts\build_official_cutlass.ps1
.\labs\cutlass\scripts\run_profiler.ps1
```

## Dependency

CUTLASS is not part of the CUDA Toolkit. CUDA provides `nvcc`, runtime headers,
cuBLAS, etc. CUTLASS/CuTe headers come from this checkout:

```text
3rdparty/cutlass/include/cutlass
3rdparty/cutlass/include/cute
```

Check the dependency:

```bash
test -f 3rdparty/cutlass/include/cutlass/cutlass.h
test -f 3rdparty/cutlass/include/cute/tensor.hpp
```

```powershell
Test-Path .\3rdparty\cutlass\include\cutlass\cutlass.h
Test-Path .\3rdparty\cutlass\include\cute\tensor.hpp
```
