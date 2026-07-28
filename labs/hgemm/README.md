# HGEMM Lab

This lab benchmarks half-precision GEMM launchers with raw CUDA `half*` device buffers.
All commands below are meant to be run from the repository root.

```powershell
cd D:\workspace\learing\AI_system
```

On Ubuntu 22.04 + RTX 4090 D, use:

```bash
cd /workspace/AI_system
```

## Build

### Windows / RTX 5060

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

### Ubuntu 22.04 / RTX 4090 D

Normal CUDA Release build fixed to `sm_89`:

```bash
cmake --preset linux-make-cuda-release-4090d
cmake --build --preset linux-make-cuda-release-4090d --target hgemm_benchmark_lab -j"$(nproc)"
```

Build a separate line-info configuration for Nsight Compute source/PTX/SASS correlation:

```bash
cmake --preset linux-make-cuda-release-4090d-lineinfo
cmake --build --preset linux-make-cuda-release-4090d-lineinfo --target hgemm_benchmark_lab -j"$(nproc)"
```

Use these executables on Ubuntu:

```bash
export Exe=/workspace/AI_system/out/build/linux-make-cuda-release-4090d/labs/hgemm/hgemm_benchmark_lab
export ExeLineInfo=/workspace/AI_system/out/build/linux-make-cuda-release-4090d-lineinfo/labs/hgemm/hgemm_benchmark_lab
```

## List Kernels

```powershell
& $Exe --list-kernels
```

```bash
"$Exe" --list-kernels
```

The output includes launcher names, tile shapes, register shapes, and the Nsight Compute kernel regex.

## CuTe HGEMM TN v0.1

The real CuTe implementation is:

```text
labs/hgemm/cute_hgemm_tn_v01.cu
```

Its benchmark launcher is `hgemm_cute_tn_v01`. The aligned fast path uses CuTe
`Tensor`, `TiledCopy`, `cp.async`, a swizzled shared layout, `TiledMMA`,
`ldmatrix`, and `cute::gemm`. It supports `--stages 2|3|4` and block swizzle.
Non-divisible shapes use the documented correctness-first boundary path.

Run one aligned correctness test:

```powershell
& $Exe `
  --kernel hgemm_cute_tn_v01 `
  --gemm-m 256 --gemm-n 256 --gemm-k 256 `
  --stages 3 --swizzle --swizzle-stride 2048 `
  --warmup 1 --iters 1
```

Run the boundary test:

```powershell
& $Exe `
  --kernel hgemm_cute_tn_v01 `
  --gemm-m 257 --gemm-n 263 --gemm-k 65 `
  --stages 2 --swizzle `
  --warmup 1 --iters 1
```

Compare stage counts and block swizzle after correctness passes:

```powershell
foreach($Stages in 2,3,4) {
  foreach($UseSwizzle in $false,$true) {
    $SwizzleArg = if($UseSwizzle) { "--swizzle" } else { "--no-swizzle" }
    & $Exe `
      --kernel hgemm_cute_tn_v01 `
      --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 `
      --stages $Stages $SwizzleArg `
      --no-correctness --warmup 5 --iters 20
  }
}
```

The complete tensor mapping and pipeline walkthrough is in
[`cute_hgemm_tn_v01.md`](./cute_hgemm_tn_v01.md).

## Correctness Tests

Run the default `4096x4096x4096` comparison:

```powershell
& $Exe --warmup 2 --iters 5
```

```bash
"$Exe" --warmup 2 --iters 5
```

Run all kernels on an uneven shape to test boundary handling:

```powershell
& $Exe --gemm-m 257 --gemm-n 263 --gemm-k 65 --warmup 1 --iters 1
```

```bash
"$Exe" --gemm-m 257 --gemm-n 263 --gemm-k 65 --warmup 1 --iters 1
```

Run one launcher:

```powershell
& $Exe `
  --kernel hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_async `
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 `
  --warmup 2 --iters 5
```

Correctness uses `allclose(abs=2.5e-1, rel=5e-2)` by default. The lab's CUDA-core,
inline-MMA, WMMA, and cuBLAS HGEMM variants all use FP16 accumulation and write FP16 C.
Tensor Core paths compare against the cuBLAS Tensor Core half-accumulate reference; PTX
MMA kernels use the same reference with `allclose(abs=5e-1, rel=5e-2)` because their
reduction order can differ from the scalar half-FMA reference at large K.

## Performance Comparison

Use `--kernel all` to compare every landed HGEMM launcher:

```powershell
& $Exe `
  --kernel all `
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 `
  --warmup 2 --iters 5
```

```bash
"$Exe" \
  --kernel all \
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 \
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

## Debug With Compute Sanitizer

Resolve the tool and run a small aligned case:

```powershell
$ComputeSanitizer = (Get-Command compute-sanitizer -ErrorAction Stop).Source

& $ComputeSanitizer `
  --tool memcheck `
  --error-exitcode 1 `
  $Exe `
  --kernel hgemm_cute_tn_v01 `
  --gemm-m 128 --gemm-n 128 --gemm-k 128 `
  --stages 2 --no-swizzle `
  --no-correctness --warmup 1 --iters 1
```

Check barriers and asynchronous-copy synchronization separately:

```powershell
& $ComputeSanitizer --tool synccheck --error-exitcode 1 `
  $Exe --kernel hgemm_cute_tn_v01 `
  --gemm-m 128 --gemm-n 128 --gemm-k 256 `
  --stages 4 --no-swizzle `
  --no-correctness --warmup 1 --iters 1
```

Use an aligned shape when debugging the CuTe fast kernel. A ragged K shape
selects the boundary kernel and will not exercise `cp.async` or `mma.sync`.

## Nsight Compute: CUDA, PTX, And SASS

Build and use `$ExeLineInfo`, then choose output paths:

```powershell
$Repo = "D:\workspace\learing\AI_system"
$Ncu = (Get-Command ncu -ErrorAction Stop).Source
$Out = "$Repo\out\reports\hgemm"
New-Item -ItemType Directory -Force $Out

$KernelName = "hgemm_cute_tn_v01"
$KernelRegex = "cute_hgemm_tn_v01_kernel"
$Report = "$Out\ncu_${KernelName}_s3_4096"
```

Collect one measured fast-path launch. `-s 1` skips the benchmark warmup:

```powershell
& $Ncu `
  --set detailed `
  --import-source yes `
  --source-folders "$Repo\labs\hgemm,$Repo\include,$Repo\3rdparty\cutlass\include" `
  --target-processes all `
  --kernel-name-base demangled `
  -k "regex:$KernelRegex" `
  -s 1 -c 1 `
  -f -o $Report `
  $ExeLineInfo `
  --kernel $KernelName `
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 `
  --stages 3 --swizzle --swizzle-stride 2048 `
  --no-correctness --warmup 1 --iters 1
```

Open the report interactively:

```powershell
& $Ncu --open-in-ui -i "$Report.ncu-rep"
```

Print the correlated CUDA/SASS, PTX, and final SASS views in the terminal:

```powershell
& $Ncu -i "$Report.ncu-rep" --page source --print-source cuda,sass
& $Ncu -i "$Report.ncu-rep" --page source --print-source ptx
& $Ncu -i "$Report.ncu-rep" --page source --print-source sass
```

If collection reports `ERR_NVGPUCTRPERM`, either run the profiling PowerShell
as Administrator or enable unrestricted counters in NVIDIA Control Panel:
`Desktop > Enable Developer Settings`, then
`Developer > Manage GPU Performance Counters > Allow access ... to all users`.
See NVIDIA's
[performance-counter permission guide](https://developer.nvidia.com/ERR_NVGPUCTRPERM).

What to search for:

```text
CuTe operation              PTX                         SASS
cute::copy G2S              cp.async                    LDGSTS
cute::copy S2R              ldmatrix.sync               LDSM
cute::gemm                  mma.sync.aligned            MMA / HMMA
wait/fence/barrier          cp.async.wait_group/barrier DEPBAR / BAR
```

The PTX view is compiler input for the GPU backend. SASS is the actual
architecture-specific machine code. Always use SASS to confirm that an intended
instruction survived lowering.

## Nsight Systems Timeline

Nsight Systems is for launch order, NVTX ranges, CPU/CUDA API overhead, and the
GPU timeline. It does not replace NCU's per-kernel instruction analysis.

```powershell
$Nsys = (Get-Command nsys -ErrorAction Stop).Source
$NsysReport = "$Out\nsys_hgemm_cute_tn_v01_s3_4096"

& $Nsys profile `
  --trace=cuda,nvtx `
  --sample=none `
  --cpuctxsw=none `
  --cuda-memory-usage=true `
  --force-overwrite=true `
  -o $NsysReport `
  $Exe `
  --kernel hgemm_cute_tn_v01 `
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 `
  --stages 3 --swizzle `
  --no-correctness --warmup 2 --iters 10
```

Print useful summaries without opening the GUI:

```powershell
& $Nsys stats `
  --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum `
  "$NsysReport.nsys-rep"
```

In the timeline, locate `cute_hgemm_tn_v01_launch` and
`hgemm/kernel_only/hgemm_cute_tn_v01`. Check whether allocation/reference work
was excluded, whether launches are back-to-back, and whether CPU gaps distort
the benchmark.

## Static PTX And SASS

Detailed Nsight Compute, `cuobjdump`, and `nvdisasm` commands live in:

```text
docs\profiling\hgemm\README.md
```

On Ubuntu 22.04 + RTX 4090 D, use:

```text
docs/profiling/hgemm/ubuntu-22.04-rtx4090d.md
```

Build with `-lineinfo` before collecting NCU reports if you want the source, PTX, and SASS views to line up with the CUDA files under `labs/hgemm`.

For the new translation unit, a direct SM120 cubin build is:

```powershell
$Nvcc = "$env:CUDA_PATH\bin\nvcc.exe"
$Nvdisasm = "$env:CUDA_PATH\bin\nvdisasm.exe"
$Cubin = "$Repo\out\sass\hgemm\cute_hgemm_tn_v01_sm120.cubin"
$VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$VsRoot = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$VcToolsVersion = (Get-Content "$VsRoot\VC\Auxiliary\Build\Microsoft.VCToolsVersion.default.txt").Trim()
$Ccbin = "$VsRoot\VC\Tools\MSVC\$VcToolsVersion\bin\Hostx64\x64"
New-Item -ItemType Directory -Force (Split-Path $Cubin)

& $Nvcc `
  -ccbin $Ccbin `
  -std=c++20 --expt-relaxed-constexpr -O3 -lineinfo -arch=sm_120 `
  -I"$Repo\3rdparty\cutlass\include" `
  -I"$Repo\labs\hgemm" `
  -I"$Repo\include" `
  -I"$Repo\out\build\windows-vs2022-cuda-release-lineinfo\generated\include" `
  -cubin "$Repo\labs\hgemm\cute_hgemm_tn_v01.cu" `
  -o $Cubin

& $Nvdisasm `
  --print-code --separate-functions `
  --print-line-info-ptx --print-instruction-encoding `
  $Cubin
```
