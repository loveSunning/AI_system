# HGEMM Lab

This lab benchmarks half-precision GEMM launchers with raw CUDA `half*` device buffers.

Build:

```powershell
cmake --build --preset windows-vs2022-cuda-release --config Release
```

List kernels:

```powershell
out\build\windows-vs2022-cuda-release\labs\hgemm\Release\hgemm_benchmark_lab.exe --list-kernels
```

Run the default `4096x4096x4096` comparison:

```powershell
out\build\windows-vs2022-cuda-release\labs\hgemm\Release\hgemm_benchmark_lab.exe --warmup 2 --iters 5
```

Run one launcher:

```powershell
out\build\windows-vs2022-cuda-release\labs\hgemm\Release\hgemm_benchmark_lab.exe --kernel hgemm_wmma_m16n16k16_naive --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 --warmup 2 --iters 5
```

Profiling commands live in `docs/profiling/hgemm/README.md`.
