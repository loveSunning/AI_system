# CUTLASS Build Notes

There are two different builds. Keep them separate.

## 1. Local Smoke Target

This is the normal path for learning CuTe/CUTLASS integration in this repo.

Windows + RTX 5060:

```powershell
cmake --preset windows-vs2022-cuda-release
cmake --build --preset windows-vs2022-cuda-release --config Release --target cutlass_header_probe
.\out\build\windows-vs2022-cuda-release\labs\cutlass\Release\cutlass_header_probe.exe
```

Linux / WSL + RTX 4090D:

```bash
cmake --preset linux-make-cuda-release
cmake --build --preset linux-make-cuda-release --target cutlass_header_probe
./out/build/linux-make-cuda-release/labs/cutlass/cutlass_header_probe
```

This only compiles `labs/cutlass/examples/cutlass_header_probe.cu`, and only
checks that local CUTLASS headers, CUDA, and the target architecture are wired.

## 2. Official CUTLASS Profiler

This is optional. It configures the official CUTLASS build tree and generates a
large operation library for `cutlass_profiler`.

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

The configure step uses:

```text
Windows: CUTLASS_NVCC_ARCHS=120
Linux:   CUTLASS_NVCC_ARCHS=89
```

## Common Errors

`Permission denied` when running `./labs/cutlass/scripts/run_profiler.sh` means
the script executable bit is not set. Run it through bash:

```bash
bash ./labs/cutlass/scripts/run_profiler.sh
```

`cutlass_profiler was not found` means the official profiler has been configured
but not built yet. Run:

```bash
bash ./labs/cutlass/scripts/build_official_cutlass.sh
```

Seeing many `Generating ... cutlass_library_*.cu` lines is normal for the
official profiler. It is not needed for the local smoke target.
