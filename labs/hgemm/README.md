# HGEMM Lab

This lab benchmarks half-precision GEMM launchers with raw CUDA `half*` device buffers.
All commands below are meant to be run from the repository root.

```powershell
cd D:\workspace\learing\AI_system
```

## Build

Normal CUDA Release build:

```powershell
cmake --preset windows-vs2022-cuda-release
cmake --build --preset windows-vs2022-cuda-release --config Release --target hgemm_benchmark_lab
```

Build a separate line-info configuration for Nsight Compute source/PTX/SASS correlation:

```powershell
cmake --preset windows-vs2022-cuda-release-lineinfo
cmake --build --preset windows-vs2022-cuda-release-lineinfo --config Release --target hgemm_benchmark_lab
```

Use this executable for normal Release:

```powershell
$Exe = "D:\workspace\learing\AI_system\out\build\windows-vs2022-cuda-release\labs\hgemm\Release\hgemm_benchmark_lab.exe"
```

Use this executable for `-lineinfo` profiling:

```powershell
$ExeLineInfo = "D:\workspace\learing\AI_system\out\build\windows-vs2022-cuda-release-lineinfo\labs\hgemm\Release\hgemm_benchmark_lab.exe"
```

## List Kernels

```powershell
& $Exe --list-kernels
```

The output includes launcher names, tile shapes, register shapes, and the Nsight Compute kernel regex.

## Correctness Tests

Run the default `4096x4096x4096` comparison:

```powershell
& $Exe --warmup 2 --iters 5
```

Run all kernels on an uneven shape to test boundary handling:

```powershell
& $Exe --gemm-m 257 --gemm-n 263 --gemm-k 65 --warmup 1 --iters 1
```

Run one launcher:

```powershell
& $Exe `
  --kernel hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_async `
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 `
  --warmup 2 --iters 5
```

Correctness uses one tolerance for every comparison: `allclose(abs=2.5e-1, rel=5e-2)`.
SIMT and inline-MMA half-accumulate kernels are compared against `hgemm_naive_f16`, which also accumulates in `half`.
cuBLAS and WMMA float-accumulate kernels are compared against `hgemm_cublas_tensor_op_nn`.

## Performance Comparison

Use `--kernel all` to compare every landed HGEMM launcher:

```powershell
& $Exe `
  --kernel all `
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 `
  --warmup 2 --iters 5
```

For a tighter comparison of the current SIMT double-buffer and `cp.async` kernels:

```powershell
$Kernels = @(
  "hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf",
  "hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async",
  "hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf",
  "hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_async",
  "hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf",
  "hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_async",
  "hgemm_cublas_tensor_op_nn"
)

foreach($Kernel in $Kernels) {
  & $Exe `
    --kernel $Kernel `
    --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 `
    --warmup 2 --iters 5
}
```

Once correctness is already established, add `--no-correctness` for cleaner kernel-only timing:

```powershell
& $Exe `
  --kernel hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async `
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 `
  --no-correctness `
  --warmup 5 --iters 20
```

## Profiling And SASS

Detailed Nsight Compute, `cuobjdump`, and `nvdisasm` commands live in:

```text
docs\profiling\hgemm\README.md
```

Build with `-lineinfo` before collecting NCU reports if you want the source, PTX, and SASS views to line up with `labs\hgemm\hgemm_lab.cu`.
