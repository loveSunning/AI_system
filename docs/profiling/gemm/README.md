# GEMM Profiling Commands

This note stores runnable Nsight Systems and Nsight Compute commands for the GEMM lab.
Run the commands from the repository root:

```powershell
cd D:\workspace\learing\AI_system
```

The commands assume the CUDA Release build has already been generated:

```powershell
cmake --build --preset windows-vs2022-cuda-release --config Release
```

## Common Paths

```powershell
$Repo = "D:\workspace\learing\AI_system"
$Exe = "$Repo\out\build\windows-vs2022-cuda-release\labs\gemm\Release\sgemm_benchmark_lab.exe"
$Out = "$Repo\out\reports\gemm"
$Nsys = "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.3.2\target-windows-x64\nsys.exe"
$Ncu = "C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.1\target\windows-desktop-win7-x64\ncu.exe"

New-Item -ItemType Directory -Force $Out
```

## Baseline Benchmark

Use this before profiling to make sure the workload is correct and stable.
The `tiled_gemm_register` commands use the default 4x4 register tile. Supported register-tile pairs are 2x2, 4x4, 4x8, 8x4, and 8x8.
The `gemm_dbuffer_vload` kernel is included in the same benchmark binary when the tile is one of 32/64/128 for M/N, 8/16/32 for K, and the register tile is 4x4 or 8x8.

```powershell
& $Exe `
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 `
  --gemm-tile-m 32 --gemm-tile-n 32 --gemm-tile-k 32 `
  --gemm-reg-m 4 --gemm-reg-n 4 `
  --warmup 2 --iters 5
```

Recommended third-version run with `16x16` threads/block and `float4` vector loads:

```powershell
& $Exe `
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 `
  --gemm-tile-m 64 --gemm-tile-n 64 --gemm-tile-k 32 `
  --gemm-reg-m 4 --gemm-reg-n 4 `
  --warmup 2 --iters 5
```

Larger output tile with 8x8 register tiles:

```powershell
& $Exe `
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 `
  --gemm-tile-m 128 --gemm-tile-n 128 --gemm-tile-k 32 `
  --gemm-reg-m 8 --gemm-reg-n 8 `
  --warmup 2 --iters 5
```

Run end-to-end only when you want to include allocation, H2D, D2H, and host-side overhead:

```powershell
& $Exe `
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 `
  --gemm-tile-m 16 --gemm-tile-n 16 --gemm-tile-k 16 `
  --gemm-reg-m 4 --gemm-reg-n 4 `
  --include-e2e `
  --warmup 2 --iters 5
```

## Nsight Systems

Use `nsys` first to answer: where does time go on the timeline?

This is useful for checking:

- whether the benchmark is really measuring kernel-only work
- CUDA API synchronization cost
- memory copy and allocation cost when `--include-e2e` is enabled
- NVTX ranges such as `sgemm_kernel_only_benchmark`, `sgemm_e2e_benchmark`, and individual benchmark names

### Kernel-Only Timeline

```powershell
& $Nsys profile `
  --sample=none `
  --cpuctxsw=none `
  --trace=cuda,nvtx `
  --force-overwrite=true `
  -o "$Out\nsys_sgemm_kernel_only_4096_t16x16x16" `
  $Exe `
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 `
  --gemm-tile-m 16 --gemm-tile-n 16 --gemm-tile-k 16 `
  --gemm-reg-m 4 --gemm-reg-n 4 `
  --warmup 2 --iters 5
```

### End-to-End Timeline

```powershell
& $Nsys profile `
  --sample=none `
  --cpuctxsw=none `
  --trace=cuda,nvtx `
  --force-overwrite=true `
  -o "$Out\nsys_sgemm_e2e_4096_t16x16x16" `
  $Exe `
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 `
  --gemm-tile-m 16 --gemm-tile-n 16 --gemm-tile-k 16 `
  --gemm-reg-m 4 --gemm-reg-n 4 `
  --include-e2e `
  --warmup 2 --iters 5
```

### Nsight Systems Stats

```powershell
& $Nsys stats `
  --report nvtx_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum `
  "$Out\nsys_sgemm_kernel_only_4096_t16x16x16.nsys-rep"
```

```powershell
& $Nsys stats `
  --report nvtx_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum `
  "$Out\nsys_sgemm_e2e_4096_t16x16x16.nsys-rep"
```

## Nsight Compute

Use `ncu` after `nsys` to answer: why is one kernel slow?

For `tiled_gemm_block`, focus on:

- `Launch Statistics`: threads/block, registers/thread, shared memory/block
- `Occupancy`: theoretical occupancy, achieved occupancy, active warps per SM
- `Scheduler Stats`: issue slot utilization, eligible warps per scheduler
- `Warp State Statistics`: barrier stalls, scoreboard stalls
- `Memory Workload Analysis`: shared memory bank conflicts, global load efficiency, L1/L2 behavior

### Find Kernel Names

If a regex does not match, list all CUDA kernels from an `nsys` report first:

```powershell
& $Nsys stats `
  --report cuda_gpu_kern_sum `
  "$Out\nsys_sgemm_kernel_only_4096_t16x16x16.nsys-rep"
```

### Windows Nsight Compute Set Commands

Nsight Compute supports multiple metric sets. The actual CLI identifier for "detail" is `detailed`, so use `--set detailed` while keeping `detail` in the output filename.

| Goal | Command option | Output suffix |
| --- | --- | --- |
| Fast first look | `--set basic` | `_set_basic` |
| More workload detail | `--set detailed` | `_set_detail` |
| Everything, slowest and most fragile | `--set full` | `_set_full` |

The benchmark requires positive `--warmup` and `--iters` values. With `--warmup 1 --iters 1`, use `-s 2 -c 1` to skip the correctness launch and warmup launch, then collect the measured launch.

#### `tiled_gemm_block` With `--set basic`

```powershell
& $Ncu `
  --set basic `
  --target-processes all `
  --kernel-name-base demangled `
  -k regex:tiled_gemm_block_kernel `
  -s 2 `
  -c 1 `
  -f `
  -o "$Out\ncu_tiled_gemm_block_2048_t16x16x16_set_basic" `
  $Exe `
  --gemm-m 2048 --gemm-n 2048 --gemm-k 2048 `
  --gemm-tile-m 16 --gemm-tile-n 16 --gemm-tile-k 16 `
  --gemm-reg-m 4 --gemm-reg-n 4 `
  --warmup 1 --iters 1
```

#### `tiled_gemm_block` With `--set detailed`

```powershell
& $Ncu `
  --set detailed `
  --target-processes all `
  --kernel-name-base demangled `
  -k regex:tiled_gemm_block_kernel `
  -s 2 `
  -c 1 `
  -f `
  -o "$Out\ncu_tiled_gemm_block_2048_t16x16x16_set_detail" `
  $Exe `
  --gemm-m 2048 --gemm-n 2048 --gemm-k 2048 `
  --gemm-tile-m 16 --gemm-tile-n 16 --gemm-tile-k 16 `
  --gemm-reg-m 4 --gemm-reg-n 4 `
  --warmup 1 --iters 1
```

#### `tiled_gemm_block` With `--set full`

```powershell
& $Ncu `
  --set full `
  --target-processes all `
  --kernel-name-base demangled `
  -k regex:tiled_gemm_block_kernel `
  -s 2 `
  -c 1 `
  -f `
  -o "$Out\ncu_tiled_gemm_block_2048_t16x16x16_set_full" `
  $Exe `
  --gemm-m 2048 --gemm-n 2048 --gemm-k 2048 `
  --gemm-tile-m 16 --gemm-tile-n 16 --gemm-tile-k 16 `
  --gemm-reg-m 4 --gemm-reg-n 4 `
  --warmup 1 --iters 1
```

### Windows Nsight Compute Section Command

Use this when you want a stable, focused report for GEMM bottleneck analysis without collecting the entire `full` set.

```powershell
& $Ncu `
  --section SpeedOfLight `
  --section LaunchStats `
  --section Occupancy `
  --section WarpStateStats `
  --section SchedulerStats `
  --section MemoryWorkloadAnalysis `
  --section MemoryWorkloadAnalysis_Chart `
  --section ComputeWorkloadAnalysis `
  --section InstructionStats `
  --target-processes all `
  --kernel-name-base demangled `
  -k regex:tiled_gemm_block_kernel `
  -s 2 `
  -c 1 `
  -f `
  -o "$Out\ncu_tiled_gemm_block_4096_t32x32x32_sections" `
  $Exe `
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 `
  --gemm-tile-m 32 --gemm-tile-n 32 --gemm-tile-k 32 `
  --gemm-reg-m 4 --gemm-reg-n 4 `
  --warmup 1 --iters 1
```

#### `tiled_gemm_register` Section Command

```powershell
& $Ncu `
  --section SpeedOfLight `
  --section LaunchStats `
  --section Occupancy `
  --section WarpStateStats `
  --section SchedulerStats `
  --section MemoryWorkloadAnalysis `
  --section MemoryWorkloadAnalysis_Chart `
  --section ComputeWorkloadAnalysis `
  --section InstructionStats `
  --target-processes all `
  --kernel-name-base demangled `
  -k regex:tiled_gemm_register_kernel `
  -s 2 `
  -c 1 `
  -f `
  -o "$Out\ncu_tiled_gemm_register_4096_t64x64x32_4x4_sections" `
  $Exe `
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 `
  --gemm-tile-m 64 --gemm-tile-n 64 --gemm-tile-k 32 `
  --gemm-reg-m 4 --gemm-reg-n 4 `
  --warmup 1 --iters 1
```

#### `gemm_dbuffer_vload` Section Command

This captures the third-version kernel with shared-memory double buffering, register-fragment double buffering, and aligned `float4` global loads.

```powershell
& $Ncu `
  --section SpeedOfLight `
  --section LaunchStats `
  --section Occupancy `
  --section WarpStateStats `
  --section SchedulerStats `
  --section MemoryWorkloadAnalysis `
  --section MemoryWorkloadAnalysis_Chart `
  --section ComputeWorkloadAnalysis `
  --section InstructionStats `
  --target-processes all `
  --kernel-name-base demangled `
  -k regex:gemm_dbuffer_vload_kernel `
  -s 2 `
  -c 1 `
  -f `
  -o "$Out\ncu_gemm_dbuffer_vload_4096_t64x64x32_4x4_sections" `
  $Exe `
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 `
  --gemm-tile-m 64 --gemm-tile-n 64 --gemm-tile-k 32 `
  --gemm-reg-m 4 --gemm-reg-n 4 `
  --warmup 1 --iters 1
```

For the larger 128x128 CTA tile:

```powershell
& $Ncu `
  --set detailed `
  --target-processes all `
  --kernel-name-base demangled `
  -k regex:gemm_dbuffer_vload_kernel `
  -s 2 `
  -c 1 `
  -f `
  -o "$Out\ncu_gemm_dbuffer_vload_4096_t128x128x32_8x8_set_detail" `
  $Exe `
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 `
  --gemm-tile-m 128 --gemm-tile-n 128 --gemm-tile-k 32 `
  --gemm-reg-m 8 --gemm-reg-n 8 `
  --warmup 1 --iters 1
```

If `-k "regex:.*gemm.*" -s 10` captures the wrong kernel or no kernel, first list kernel names with `nsys stats --report cuda_gpu_kern_sum`, then narrow the regex or adjust `-s`.

In the GUI, look for:

- `Warp State Statistics`: barrier stalls are shown in the warp stall breakdown.
- `Memory Workload Analysis` / `Memory Workload Analysis Tables`: shared-memory bank conflict metrics are under the L1/TEX or shared-memory tables. If the table is still missing, open the `Raw` tab and search for `bank_conflict` or `shared`.

### Profile `cuda_naive`

Use this as the comparison point for launch configuration, occupancy, warp stalls, and memory behavior.

```powershell
& $Ncu `
  --set detailed `
  --target-processes all `
  --kernel-name-base demangled `
  -k regex:naive_gemm_kernel `
  -s 2 `
  -c 1 `
  -f `
  -o "$Out\ncu_cuda_naive_gemm_2048_set_detail" `
  $Exe `
  --gemm-m 2048 --gemm-n 2048 --gemm-k 2048 `
  --gemm-tile-m 16 --gemm-tile-n 16 --gemm-tile-k 16 `
  --gemm-reg-m 4 --gemm-reg-n 4 `
  --warmup 1 --iters 1
```

### Profile cuBLAS SGEMM

cuBLAS kernel names vary by GPU, CUDA, and cuBLAS version. Start with a broad SGEMM regex.

```powershell
& $Ncu `
  --set basic `
  --target-processes all `
  --kernel-name-base demangled `
  -k "regex:sgemm|gemm|ampere|sass|cutlass" `
  -s 2 `
  -c 1 `
  -f `
  -o "$Out\ncu_cublas_sgemm_2048_set_basic" `
  $Exe `
  --gemm-m 2048 --gemm-n 2048 --gemm-k 2048 `
  --gemm-tile-m 16 --gemm-tile-n 16 --gemm-tile-k 16 `
  --gemm-reg-m 4 --gemm-reg-n 4 `
  --warmup 1 --iters 1
```

If this captures the wrong kernel, use the `nsys stats --report cuda_gpu_kern_sum` output to replace the regex with the exact demangled kernel name.

## Tile Sweep For Diagnosis

Run a small sweep before deep profiling. If all tiled shapes are slower than `cuda_naive`, the issue is likely the v1 algorithm shape rather than a single bad tile.

```powershell
$Tiles = @(
  @(16, 16, 16),
  @(16, 16, 32),
  @(16, 32, 32),
  @(32, 16, 32),
  @(32, 32, 32)
)

foreach($Tile in $Tiles) {
  & $Exe `
    --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 `
    --gemm-tile-m $Tile[0] --gemm-tile-n $Tile[1] --gemm-tile-k $Tile[2] `
    --gemm-reg-m 4 --gemm-reg-n 4 `
    --warmup 2 --iters 5
}
```

## Linux Commands

The Linux commands use the `linux-make-cuda-release` preset from `CMakePresets.json`.
Run them from the repository root on a Linux machine with CUDA, Nsight Systems, and Nsight Compute installed.

### Linux Build And Paths

```bash
cd /path/to/AI_system

cmake --preset linux-make-cuda-release
cmake --build --preset linux-make-cuda-release

Repo="$(pwd)"
Exe="$Repo/out/build/linux-make-cuda-release/labs/gemm/sgemm_benchmark_lab"
Out="$Repo/out/reports/gemm"
Nsys="nsys"
Ncu="ncu"

mkdir -p "$Out"
```

If `nsys` or `ncu` is not in `PATH`, replace `Nsys` and `Ncu` with the full tool paths, for example:

```bash
Nsys="/opt/nvidia/nsight-systems/bin/nsys"
Ncu="/opt/nvidia/nsight-compute/ncu"
```

### Linux Baseline Benchmark

```bash
"$Exe" \
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 \
  --gemm-tile-m 16 --gemm-tile-n 16 --gemm-tile-k 16 \
  --gemm-reg-m 4 --gemm-reg-n 4 \
  --warmup 2 --iters 5
```

Third-version `gemm_dbuffer_vload` run:

```bash
"$Exe" \
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 \
  --gemm-tile-m 64 --gemm-tile-n 64 --gemm-tile-k 32 \
  --gemm-reg-m 4 --gemm-reg-n 4 \
  --warmup 2 --iters 5
```

End-to-end benchmark:

```bash
"$Exe" \
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 \
  --gemm-tile-m 16 --gemm-tile-n 16 --gemm-tile-k 16 \
  --gemm-reg-m 4 --gemm-reg-n 4 \
  --include-e2e \
  --warmup 2 --iters 5
```

### Linux Nsight Systems

Kernel-only timeline:

```bash
"$Nsys" profile \
  --sample=none \
  --cpuctxsw=none \
  --trace=cuda,nvtx \
  --force-overwrite=true \
  -o "$Out/nsys_sgemm_kernel_only_4096_t16x16x16" \
  "$Exe" \
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 \
  --gemm-tile-m 16 --gemm-tile-n 16 --gemm-tile-k 16 \
  --gemm-reg-m 4 --gemm-reg-n 4 \
  --warmup 2 --iters 5
```

End-to-end timeline:

```bash
"$Nsys" profile \
  --sample=none \
  --cpuctxsw=none \
  --trace=cuda,nvtx \
  --force-overwrite=true \
  -o "$Out/nsys_sgemm_e2e_4096_t16x16x16" \
  "$Exe" \
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 \
  --gemm-tile-m 16 --gemm-tile-n 16 --gemm-tile-k 16 \
  --gemm-reg-m 4 --gemm-reg-n 4 \
  --include-e2e \
  --warmup 2 --iters 5
```

View summarized tables in terminal:

```bash
"$Nsys" stats \
  --report nvtx_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  "$Out/nsys_sgemm_kernel_only_4096_t16x16x16.nsys-rep"
```

```bash
"$Nsys" stats \
  --report nvtx_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  "$Out/nsys_sgemm_e2e_4096_t16x16x16.nsys-rep"
```

List captured CUDA kernels:

```bash
"$Nsys" stats \
  --report cuda_gpu_kern_sum \
  "$Out/nsys_sgemm_kernel_only_4096_t16x16x16.nsys-rep"
```

Open the timeline in the GUI if the GUI tool is installed:

```bash
nsys-ui "$Out/nsys_sgemm_kernel_only_4096_t16x16x16.nsys-rep"
```

### Linux Nsight Compute

Nsight Compute supports multiple metric sets. The actual CLI identifier for "detail" is `detailed`, so use `--set detailed` while keeping `detail` in the output filename.

| Goal | Command option | Output suffix |
| --- | --- | --- |
| Fast first look | `--set basic` | `_set_basic` |
| More workload detail | `--set detailed` | `_set_detail` |
| Everything, slowest and most fragile | `--set full` | `_set_full` |

The benchmark requires positive `--warmup` and `--iters` values. With `--warmup 1 --iters 1`, use `-s 2 -c 1` to skip the correctness launch and warmup launch, then collect the measured launch.

#### `tiled_gemm_block` With `--set basic`

```bash
"$Ncu" \
  --set basic \
  --target-processes all \
  --kernel-name-base demangled \
  -k regex:tiled_gemm_block_kernel \
  -s 2 \
  -c 1 \
  -f \
  -o "$Out/ncu_tiled_gemm_block_2048_t16x16x16_set_basic" \
  "$Exe" \
  --gemm-m 2048 --gemm-n 2048 --gemm-k 2048 \
  --gemm-tile-m 16 --gemm-tile-n 16 --gemm-tile-k 16 \
  --gemm-reg-m 4 --gemm-reg-n 4 \
  --warmup 1 --iters 1
```

#### `tiled_gemm_block` With `--set detailed`

```bash
"$Ncu" \
  --set detailed \
  --target-processes all \
  --kernel-name-base demangled \
  -k regex:tiled_gemm_block_kernel \
  -s 2 \
  -c 1 \
  -f \
  -o "$Out/ncu_tiled_gemm_block_2048_t16x16x16_set_detail" \
  "$Exe" \
  --gemm-m 2048 --gemm-n 2048 --gemm-k 2048 \
  --gemm-tile-m 16 --gemm-tile-n 16 --gemm-tile-k 16 \
  --gemm-reg-m 4 --gemm-reg-n 4 \
  --warmup 1 --iters 1
```

#### `tiled_gemm_block` With `--set full`

```bash
"$Ncu" \
  --set full \
  --target-processes all \
  --kernel-name-base demangled \
  -k regex:tiled_gemm_block_kernel \
  -s 2 \
  -c 1 \
  -f \
  -o "$Out/ncu_tiled_gemm_block_2048_t16x16x16_set_full" \
  "$Exe" \
  --gemm-m 2048 --gemm-n 2048 --gemm-k 2048 \
  --gemm-tile-m 16 --gemm-tile-n 16 --gemm-tile-k 16 \
  --gemm-reg-m 4 --gemm-reg-n 4 \
  --warmup 1 --iters 1
```

#### Linux Nsight Compute Section Command

Use this when you want a stable, focused report for GEMM bottleneck analysis without collecting the entire `full` set.

```bash
"$Ncu" \
  --section SpeedOfLight \
  --section LaunchStats \
  --section Occupancy \
  --section WarpStateStats \
  --section SchedulerStats \
  --section MemoryWorkloadAnalysis \
  --section ComputeWorkloadAnalysis \
  --section InstructionStats \
  --target-processes all \
  --kernel-name-base demangled \
  -k "regex:.*gemm.*" \
  -s 10 \
  -c 1 \
  -f \
  -o "$Out/ncu_gemm_2048_t16x16x16_sections" \
  "$Exe" \
  --gemm-m 2048 --gemm-n 2048 --gemm-k 2048 \
  --gemm-tile-m 16 --gemm-tile-n 16 --gemm-tile-k 16 \
  --gemm-reg-m 4 --gemm-reg-n 4 \
  --warmup 1 --iters 1
```

If `-k "regex:.*gemm.*" -s 10` captures the wrong kernel or no kernel, first list kernel names with `nsys stats --report cuda_gpu_kern_sum`, then narrow the regex or adjust `-s`.

In the GUI, look for:

- `Warp State Statistics`: barrier stalls are shown in the warp stall breakdown.
- `Memory Workload Analysis` / `Memory Workload Analysis Tables`: shared-memory bank conflict metrics are under the L1/TEX or shared-memory tables. If the table is still missing, open the `Raw` tab and search for `bank_conflict` or `shared`.

Profile `gemm_dbuffer_vload`:

```bash
"$Ncu" \
  --section SpeedOfLight \
  --section LaunchStats \
  --section Occupancy \
  --section WarpStateStats \
  --section SchedulerStats \
  --section MemoryWorkloadAnalysis \
  --section ComputeWorkloadAnalysis \
  --section InstructionStats \
  --target-processes all \
  --kernel-name-base demangled \
  -k regex:gemm_dbuffer_vload_kernel \
  -s 2 \
  -c 1 \
  -f \
  -o "$Out/ncu_gemm_dbuffer_vload_4096_t64x64x32_4x4_sections" \
  "$Exe" \
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 \
  --gemm-tile-m 64 --gemm-tile-n 64 --gemm-tile-k 32 \
  --gemm-reg-m 4 --gemm-reg-n 4 \
  --warmup 1 --iters 1
```

Profile `cuda_naive`:

```bash
"$Ncu" \
  --set detailed \
  --target-processes all \
  --kernel-name-base demangled \
  -k regex:naive_gemm_kernel \
  -s 2 \
  -c 1 \
  -f \
  -o "$Out/ncu_cuda_naive_gemm_2048_set_detail" \
  "$Exe" \
  --gemm-m 2048 --gemm-n 2048 --gemm-k 2048 \
  --gemm-tile-m 16 --gemm-tile-n 16 --gemm-tile-k 16 \
  --gemm-reg-m 4 --gemm-reg-n 4 \
  --warmup 1 --iters 1
```

Profile cuBLAS SGEMM with a broad kernel regex:

```bash
"$Ncu" \
  --set basic \
  --target-processes all \
  --kernel-name-base demangled \
  -k "regex:sgemm|gemm|ampere|sass|cutlass" \
  -s 2 \
  -c 1 \
  -f \
  -o "$Out/ncu_cublas_sgemm_2048_set_basic" \
  "$Exe" \
  --gemm-m 2048 --gemm-n 2048 --gemm-k 2048 \
  --gemm-tile-m 16 --gemm-tile-n 16 --gemm-tile-k 16 \
  --gemm-reg-m 4 --gemm-reg-n 4 \
  --warmup 1 --iters 1
```

View an Nsight Compute report in terminal:

```bash
"$Ncu" --import "$Out/ncu_tiled_gemm_block_2048_t16x16x16_set_detail.ncu-rep" --page details
```

Open the Nsight Compute GUI if the GUI tool is installed:

```bash
ncu-ui "$Out/ncu_tiled_gemm_block_2048_t16x16x16_set_detail.ncu-rep"
```

On some Linux systems, hardware performance counters are restricted. If `ncu` reports a permission error, run the profiling command with `sudo` or configure NVIDIA performance counter access for the machine.

### Linux Tile Sweep

```bash
for Tile in 16,16,16 16,16,32 16,32,32 32,16,32 32,32,32; do
  IFS=',' read -r TileM TileN TileK <<< "$Tile"
  "$Exe" \
    --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 \
    --gemm-tile-m "$TileM" --gemm-tile-n "$TileN" --gemm-tile-k "$TileK" \
    --gemm-reg-m 4 --gemm-reg-n 4 \
    --warmup 2 --iters 5
done
```

## Reading The Results

Use this order when analyzing reports:

1. `nsys`: confirm kernel-only vs e2e cost and identify hot kernels.
2. `ncu` launch statistics: compare threads/block, registers/thread, shared memory/block.
3. `ncu` occupancy: check whether active warps are too low.
4. `ncu` warp state: look for high `barrier` stalls in `tiled_gemm_block`.
5. `ncu` memory workload: look for shared memory bank conflicts and inefficient global loads.

For the current `tiled_gemm_block`, a common slow pattern is high barrier cost plus low scheduler utilization. That means shared-memory tiling is adding synchronization and shared-memory traffic, but the kernel is not yet reusing data enough in registers.
