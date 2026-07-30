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

### CUDA 12.8 `__nv_atomic_load_n` error

CUTLASS 4.5.2 contains this CUDA 12.8+ branch in
`include/cutlass/subbyte_reference.h`:

```cpp
Storage original = __nv_atomic_load_n(ptr_, __NV_ATOMIC_RELAXED);
```

An unrestricted `cutlass_profiler` build compiles
`tools/library/src/reference/gemm_int4.cu`. Its device reference templates
instantiate the line above with `Storage=uint16_t`; CUDA 12.8 can then report:

```text
error: too few arguments in function call
```

`00_basic_gemm` is not the source of the error. It appears because the same
build command also requests `cutlass_profiler`.

#### Recommended FP16 profiler build

For this lab, the INT4 reference providers and unrelated kernel families are
not required. Reconfigure the existing build directory; it does not need to be
deleted:

```bash
cd /workspace/AI_system/3rdparty/cutlass

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUTLASS_NVCC_ARCHS=89 \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_LIBRARY_OPERATIONS=gemm \
  '-DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_s16816gemm_f16_*' \
  '-DCUTLASS_LIBRARY_IGNORE_KERNELS=cutlass_tensorop_s16816gemm_f16_s8_*,cutlass_tensorop_s16816gemm_f16_u8_*' \
  -DCUTLASS_PROFILER_DISABLE_REFERENCE=ON \
  -DCUTLASS_UNITY_BUILD_ENABLED=ON

cmake --build build \
  --target 00_basic_gemm cutlass_profiler \
  -j12
```

Disabling CUTLASS reference providers does not disable the CUTLASS kernels.
Use cuBLAS to verify FP16 profiler results:

```bash
./build/tools/profiler/cutlass_profiler \
  --operation=Gemm \
  --m=4096 --n=4096 --k=4096 \
  --A=f16 --B=f16 --C=f16 \
  --accumulator=f32 \
  --verification-providers=cublas
```

The repository wrappers apply the same configuration:

```bash
cd /workspace/AI_system
bash ./labs/cutlass/scripts/configure_official_cutlass.sh
bash ./labs/cutlass/scripts/build_official_cutlass.sh
bash ./labs/cutlass/scripts/run_profiler.sh
```

#### Full reference profiler build

If INT4 host/device reference providers are required, keep
`CUTLASS_PROFILER_DISABLE_REFERENCE=OFF` and make the CUDA thread scope
explicit at `include/cutlass/subbyte_reference.h`:

```cpp
Storage original = __nv_atomic_load_n(
    ptr_, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
```

The device scope matches the following `atomicCAS` loop. After changing that
line, rebuild the existing tree:

```bash
cmake --build build \
  --target 00_basic_gemm cutlass_profiler \
  -j12
```
