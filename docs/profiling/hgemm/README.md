# HGEMM Profiling Commands

This note stores runnable benchmark and Nsight Compute commands for `labs/hgemm`.
The lab follows the launcher list from [xlite-dev/HGEMM](https://github.com/xlite-dev/HGEMM), but uses raw `half*` device inputs plus explicit `M/N/K` sizes instead of `torch::Tensor`.

Run commands from the repository root:

```powershell
cd D:\workspace\learing\AI_system
```

Build the CUDA Release preset first:

```powershell
cmake --build --preset windows-vs2022-cuda-release --config Release
```

For source/PTX/SASS correlation in Nsight Compute, use a separate `-lineinfo` build directory:

```powershell
cmake --preset windows-vs2022-cuda-release-lineinfo
cmake --build --preset windows-vs2022-cuda-release-lineinfo --config Release --target hgemm_benchmark_lab
```

## Common Paths

```powershell
$Repo = "D:\workspace\learing\AI_system"
$BuildLineInfo = "$Repo\out\build\windows-vs2022-cuda-release-lineinfo"
$Exe = "$Repo\out\build\windows-vs2022-cuda-release\labs\hgemm\Release\hgemm_benchmark_lab.exe"
$ExeLineInfo = "$Repo\out\build\windows-vs2022-cuda-release-lineinfo\labs\hgemm\Release\hgemm_benchmark_lab.exe"
$HgemmLibLineInfo = "$Repo\out\build\windows-vs2022-cuda-release-lineinfo\labs\hgemm\Release\ai_system_hgemm_lab.lib"
$Out = "$Repo\out\reports\hgemm"
$SassOut = "$Repo\out\sass\hgemm"
$Nvcc = "$env:CUDA_PATH\bin\nvcc.exe"
$Ncu = "C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.1\target\windows-desktop-win7-x64\ncu.exe"
$Cuobjdump = "$env:CUDA_PATH\bin\cuobjdump.exe"
$Nvdisasm = "$env:CUDA_PATH\bin\nvdisasm.exe"

New-Item -ItemType Directory -Force $Out
New-Item -ItemType Directory -Force $SassOut
```

## Baseline Benchmark

The command-line defaults are `M=N=K=4096`, so this is enough for the default full comparison:

```powershell
& $Exe --warmup 2 --iters 5
```

Run one kernel by name:

```powershell
& $Exe `
  --kernel hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf `
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 `
  --warmup 2 --iters 5

# 最快的一个cuda core kernel
& $Exe `
  --kernel hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async `
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 `
  --warmup 2 --iters 5
```

List all compiled launcher names and their Nsight Compute regex:

```powershell
& $Exe --list-kernels
```

Correctness uses the same tolerance for every row: `allclose(abs=2.5e-1, rel=5e-2)`.
The reference source depends on the accumulation mode:

- SIMT and inline-MMA half-accumulate kernels compare against `hgemm_naive_f16`, which also accumulates in `half`.
- cuBLAS and WMMA float-accumulate kernels compare against `hgemm_cublas_tensor_op_nn`.

For profiling, `--no-correctness --warmup 1 --iters 1` keeps the launch stream simple: one warmup launch and one measured launch. The `-s 1 -c 1` Nsight Compute options skip the warmup and collect the measured launch.

## Nsight Compute Sections

The section set below is a stable first pass for HGEMM:

- `SpeedOfLight`
- `LaunchStats`
- `Occupancy`
- `WarpStateStats`
- `SchedulerStats`
- `MemoryWorkloadAnalysis`
- `ComputeWorkloadAnalysis`
- `InstructionStats`

## Generate One NCU Report Per Kernel

This PowerShell loop emits and runs one Nsight Compute command per HGEMM launcher.

```powershell
$Kernels = @(
  @{ Name = "hgemm_naive_f16"; Regex = "hgemm_naive_f16_kernel" },
  @{ Name = "hgemm_sliced_k_f16"; Regex = "hgemm_sliced_k_f16_kernel" },
  @{ Name = "hgemm_t_8x8_sliced_k_f16x4"; Regex = "hgemm_t_8x8_sliced_k_f16x4_kernel" },
  @{ Name = "hgemm_t_8x8_sliced_k_f16x4_pack"; Regex = "hgemm_t_8x8_sliced_k_f16x4_pack_kernel" },
  @{ Name = "hgemm_t_8x8_sliced_k_f16x4_bcf"; Regex = "hgemm_t_8x8_sliced_k_f16x4_bcf_kernel" },
  @{ Name = "hgemm_t_8x8_sliced_k_f16x4_pack_bcf"; Regex = "hgemm_t_8x8_sliced_k_f16x4_pack_bcf_kernel" },
  @{ Name = "hgemm_t_8x8_sliced_k_f16x8_pack_bcf"; Regex = "hgemm_t_8x8_sliced_k_f16x8_pack_bcf_kernel" },
  @{ Name = "hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf"; Regex = "hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf_kernel" },
  @{ Name = "hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf"; Regex = "hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_kernel" },
  @{ Name = "hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async"; Regex = "hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async_kernel" },
  @{ Name = "hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf"; Regex = "hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_kernel" },
  @{ Name = "hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_async"; Regex = "hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_async_kernel" },
  @{ Name = "hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf"; Regex = "hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_kernel" },
  @{ Name = "hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_async"; Regex = "hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_async_kernel" },
  @{ Name = "hgemm_cublas_tensor_op_nn"; Regex = "gemm|hgemm|tensor" },
  @{ Name = "hgemm_cublas_tensor_op_tn"; Regex = "gemm|hgemm|tensor" },
  @{ Name = "hgemm_wmma_m16n16k16_naive"; Regex = "hgemm_wmma_m16n16k16_naive_kernel" },
  @{ Name = "hgemm_wmma_m16n16k16_mma4x2"; Regex = "hgemm_wmma_m16n16k16_mma4x2_kernel" },
  @{ Name = "hgemm_wmma_m16n16k16_mma4x2_warp2x4"; Regex = "hgemm_wmma_m16n16k16_mma4x2_warp2x4_kernel" },
  @{ Name = "hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async"; Regex = "hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_kernel" },
  @{ Name = "hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async"; Regex = "hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async_kernel" },
  @{ Name = "hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages"; Regex = "hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_kernel" },
  @{ Name = "hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem"; Regex = "hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem_kernel" },
  @{ Name = "hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem"; Regex = "hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem_kernel" },
  @{ Name = "hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem"; Regex = "hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem_kernel" },
  @{ Name = "hgemm_mma_m16n8k16_naive"; Regex = "hgemm_mma_m16n8k16_naive_kernel" },
  @{ Name = "hgemm_mma_m16n8k16_mma2x4_warp4x4"; Regex = "hgemm_mma_m16n8k16_mma2x4_warp4x4_kernel" },
  @{ Name = "hgemm_mma_m16n8k16_mma2x4_warp4x4_stages"; Regex = "hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_kernel" },
  @{ Name = "hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem"; Regex = "hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_kernel" },
  @{ Name = "hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem"; Regex = "hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_kernel" },
  @{ Name = "hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4"; Regex = "hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4_kernel" },
  @{ Name = "hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr"; Regex = "hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr_kernel" },
  @{ Name = "hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn"; Regex = "hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_kernel" },
  @{ Name = "hgemm_mma_stages_block_swizzle_tn_cute"; Regex = "hgemm_mma_stages_block_swizzle_tn_cute_kernel" },
  @{ Name = "hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_swizzle"; Regex = "hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_swizzle_kernel" },
  @{ Name = "hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x4"; Regex = "hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x4_kernel" }
)

foreach($Kernel in $Kernels) {
  & $Ncu `
    --section SpeedOfLight `
    --section LaunchStats `
    --section Occupancy `
    --section WarpStateStats `
    --section SchedulerStats `
    --section MemoryWorkloadAnalysis `
    --section ComputeWorkloadAnalysis `
    --section InstructionStats `
    --target-processes all `
    --kernel-name-base demangled `
    -k "regex:$($Kernel.Regex)" `
    -s 1 `
    -c 1 `
    -f `
    -o "$Out\ncu_$($Kernel.Name)_4096_sections" `
    $Exe `
    --kernel $Kernel.Name `
    --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 `
    --no-correctness `
    --warmup 1 --iters 1
}
```

## Single-Kernel NCU Template

Use this when iterating on one kernel:

```powershell
$KernelName = "hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async"
$KernelRegex = "hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async_kernel"

& $Ncu `
  --set detailed `
  --import-source yes `
  --source-folders "$Repo\labs\hgemm,$Repo\include" `
  --target-processes all `
  --kernel-name-base demangled `
  -k "regex:$KernelRegex" `
  -s 1 `
  -c 1 `
  -f `
  -o "$Out\ncu_${KernelName}_4096_set_detail" `
  $ExeLineInfo `
  --kernel $KernelName `
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 `
  --no-correctness `
  --warmup 1 --iters 1
```

Open the report in the Nsight Compute UI:

```powershell
& $Ncu --open-in-ui -i "$Out\ncu_${KernelName}_4096_set_detail.ncu-rep"
```

Print the source-correlated views in the terminal:

```powershell
& $Ncu -i "$Out\ncu_${KernelName}_4096_set_detail.ncu-rep" --page source --print-source cuda,sass
& $Ncu -i "$Out\ncu_${KernelName}_4096_set_detail.ncu-rep" --page source --print-source ptx
& $Ncu -i "$Out\ncu_${KernelName}_4096_set_detail.ncu-rep" --page source --print-source sass
```

Use `cuda,sass` when mapping instructions back to `hgemm_lab.cu`. Use `ptx` when checking the PTX that was embedded in the cubin. The `sass` view is the final machine code.

## Compare Landed HGEMM Kernels

Use this loop for a focused comparison of the SIMT double-buffer and `cp.async` kernels that are currently implemented:

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
    --no-correctness `
    --warmup 5 --iters 20
}
```

Run the same group with correctness enabled before trusting a new implementation:

```powershell
foreach($Kernel in $Kernels) {
  & $Exe `
    --kernel $Kernel `
    --gemm-m 257 --gemm-n 263 --gemm-k 65 `
    --warmup 1 --iters 1
}
```

## Static Disassembly With cuobjdump

Dump PTX, SASS, resource usage, and ELF metadata from the line-info HGEMM library:

```powershell
& $Cuobjdump --list-elf $HgemmLibLineInfo |
  Out-File -FilePath "$SassOut\cuobjdump_hgemm_elf_list.txt" -Encoding ascii

& $Cuobjdump --dump-elf-symbols $HgemmLibLineInfo |
  Out-File -FilePath "$SassOut\cuobjdump_hgemm_elf_symbols.txt" -Encoding ascii

& $Cuobjdump --dump-resource-usage $HgemmLibLineInfo |
  Out-File -FilePath "$SassOut\cuobjdump_hgemm_resource_usage.txt" -Encoding ascii

& $Cuobjdump --dump-ptx $HgemmLibLineInfo |
  Out-File -FilePath "$SassOut\cuobjdump_hgemm.ptx" -Encoding ascii

& $Cuobjdump --dump-sass --sort-functions --gpu-architecture sm_120 $HgemmLibLineInfo |
  Out-File -FilePath "$SassOut\cuobjdump_hgemm_sm120.sass" -Encoding ascii
```

Check for Tensor Core, async-copy, and memory instructions:

```powershell
Select-String -Path "$SassOut\cuobjdump_hgemm_sm120.sass" -Pattern "LDGSTS|MMA|HMMA|LDSM|LDG|STS|BAR"
```

`cp.async` usually appears as `LDGSTS` in SASS. For example, the `hgemm_t_*_dbuf_async` kernels should contain `LDGSTS...LTC128B` instructions when the compiler keeps the async path.

To dump one function only, first find the mangled symbol:

```powershell
$Mangled = (
  Select-String `
    -Path "$SassOut\cuobjdump_hgemm_elf_symbols.txt" `
    -Pattern "hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_async_kernel" |
    Select-Object -First 1
).Line -replace '^.*STO_ENTRY\s+', ''

& $Cuobjdump --dump-sass --gpu-architecture sm_120 --function $Mangled $HgemmLibLineInfo |
  Out-File -FilePath "$SassOut\cuobjdump_hgemm_k32_dbuf_async_sm120.sass" -Encoding ascii
```

## Static Disassembly With nvdisasm

`nvdisasm` works on cubin/ELF images. Extract cubins from the HGEMM library first:

```powershell
Push-Location $SassOut
& $Cuobjdump --extract-elf all $HgemmLibLineInfo
Pop-Location
```

Then disassemble every extracted cubin with line information:

```powershell
Get-ChildItem $SassOut -Filter "*.cubin" | ForEach-Object {
  $Output = Join-Path $SassOut ("nvdisasm_" + $_.BaseName + ".sass")
  & $Nvdisasm `
    --print-code `
    --separate-functions `
    --print-line-info-ptx `
    --print-instruction-encoding `
    $_.FullName |
    Out-File -FilePath $Output -Encoding ascii
}
```

If the extracted cubin contains line information, `--print-line-info-ptx` annotates SASS with PTX/source line mappings. If a cubin has no line annotations, rebuild with `-DCMAKE_CUDA_FLAGS="-lineinfo"` and disassemble the line-info library.

## Direct cubin Build With nvcc

When you want a standalone cubin for `hgemm_lab.cu`, compile the CUDA translation unit directly. This is useful for fast SASS inspection without walking the Visual Studio `.lib`.

```powershell
$Cubin = "$SassOut\nvcc_hgemm_lab_sm120.cubin"

& $Nvcc `
  -std=c++20 `
  --expt-relaxed-constexpr `
  -O3 `
  -lineinfo `
  -arch=sm_120 `
  -I"$Repo\labs\hgemm" `
  -I"$Repo\include" `
  -I"$BuildLineInfo\generated\include" `
  -cubin "$Repo\labs\hgemm\hgemm_lab.cu" `
  -o $Cubin
```

If `nvcc` cannot find `cl.exe`, run the command in a Visual Studio Developer PowerShell or initialize MSVC first:

```powershell
$Vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cmd /c "`"$Vcvars`" && `"$Nvcc`" -std=c++20 --expt-relaxed-constexpr -O3 -lineinfo -arch=sm_120 -I`"$Repo\labs\hgemm`" -I`"$Repo\include`" -I`"$BuildLineInfo\generated\include`" -cubin `"$Repo\labs\hgemm\hgemm_lab.cu`" -o `"$Cubin`""
```

Disassemble that cubin:

```powershell
& $Nvdisasm `
  --print-code `
  --separate-functions `
  --print-line-info-ptx `
  --print-instruction-encoding `
  $Cubin |
  Out-File -FilePath "$SassOut\nvdisasm_hgemm_lab_sm120.sass" -Encoding ascii
```

Search the standalone SASS:

```powershell
Select-String -Path "$SassOut\nvdisasm_hgemm_lab_sm120.sass" -Pattern "LDGSTS|MMA|HMMA|LDSM|LDG|STS|BAR"
```

## Reading Results

Start with:

1. `LaunchStats`: verify block size, register count, and shared memory.
2. `SpeedOfLight`: check whether the kernel is compute or memory limited.
3. `WarpStateStats` and `SchedulerStats`: look for stalls that explain low issue utilization.
4. `MemoryWorkloadAnalysis`: compare global load efficiency and shared-memory behavior.

The SIMT thread-tile kernels are intended as readable fixed-tile baselines. The WMMA launchers use the CUDA WMMA API, while the MMA launchers use an inline PTX `ldmatrix.sync` + `mma.sync.aligned.m16n8k16` path so the two Tensor Core styles stay separate and profileable.
