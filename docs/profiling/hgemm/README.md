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

## Common Paths

```powershell
$Repo = "D:\workspace\learing\AI_system"
$Exe = "$Repo\out\build\windows-vs2022-cuda-release\labs\hgemm\Release\hgemm_benchmark_lab.exe"
$Out = "$Repo\out\reports\hgemm"
$Ncu = "C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.1\target\windows-desktop-win7-x64\ncu.exe"

New-Item -ItemType Directory -Force $Out
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
```

List all compiled launcher names and their Nsight Compute regex:

```powershell
& $Exe --list-kernels
```

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
  @{ Name = "hgemm_sliced_k_f16"; Regex = "hgemm_sliced_k_f16" },
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
$KernelName = "hgemm_wmma_m16n16k16_naive"
$KernelRegex = "hgemm_wmma_m16n16k16_naive_kernel"

& $Ncu `
  --set detailed `
  --target-processes all `
  --kernel-name-base demangled `
  -k "regex:$KernelRegex" `
  -s 1 `
  -c 1 `
  -f `
  -o "$Out\ncu_${KernelName}_4096_set_detail" `
  $Exe `
  --kernel $KernelName `
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 `
  --no-correctness `
  --warmup 1 --iters 1
```

## Reading Results

Start with:

1. `LaunchStats`: verify block size, register count, and shared memory.
2. `SpeedOfLight`: check whether the kernel is compute or memory limited.
3. `WarpStateStats` and `SchedulerStats`: look for stalls that explain low issue utilization.
4. `MemoryWorkloadAnalysis`: compare global load efficiency and shared-memory behavior.

The SIMT thread-tile kernels are intended as readable fixed-tile baselines. The WMMA launchers use the CUDA WMMA API, while the MMA launchers use an inline PTX `ldmatrix.sync` + `mma.sync.aligned.m16n8k16` path so the two Tensor Core styles stay separate and profileable.
