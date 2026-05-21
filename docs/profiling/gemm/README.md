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

```powershell
& $Exe `
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 `
  --gemm-tile-m 16 --gemm-tile-n 16 --gemm-tile-k 16 `
  --warmup 2 --iters 5
```

Run end-to-end only when you want to include allocation, H2D, D2H, and host-side overhead:

```powershell
& $Exe `
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 `
  --gemm-tile-m 16 --gemm-tile-n 16 --gemm-tile-k 16 `
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

For `tiled_gemm_v1`, focus on:

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

### Profile `tiled_gemm_v1`

This is the most important command when investigating why tiled v1 is slower than `cuda_naive`.

```powershell
& $Ncu `
  --set full `
  --target-processes all `
  --kernel-name-base demangled `
  --kernel-name "regex:tiled_gemm_v1_kernel" `
  --launch-skip 2 `
  --launch-count 1 `
  --force-overwrite `
  -o "$Out\ncu_tiled_gemm_v1_4096_t16x16x16" `
  $Exe `
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 `
  --gemm-tile-m 16 --gemm-tile-n 16 --gemm-tile-k 16 `
  --warmup 2 --iters 5
```

For a lighter first pass, use a smaller matrix. The benchmark requires positive `--warmup` and `--iters` values, so use `--warmup 1 --iters 1` and skip the first two matching launches: correctness and warmup.

```powershell
& $Ncu `
  --set basic `
  --target-processes all `
  --kernel-name-base demangled `
  --kernel-name "regex:tiled_gemm_v1_kernel" `
  --launch-skip 2 `
  --launch-count 1 `
  --force-overwrite `
  -o "$Out\ncu_tiled_gemm_v1_1024_analysis" `
  $Exe `
  --gemm-m 1024 --gemm-n 1024 --gemm-k 1024 `
  --gemm-tile-m 16 --gemm-tile-n 16 --gemm-tile-k 16 `
  --warmup 1 --iters 1
```

`--set basic` only collects a small default group. It usually shows `SpeedOfLight`, `LaunchStats`, `Occupancy`, and `WorkloadDistribution`, but it does not necessarily collect `WarpStateStats` or detailed shared-memory metrics. To inspect barrier stalls and shared-memory bank behavior, run a focused pass:

```powershell
& $Ncu `
  --section LaunchStats `
  --section Occupancy `
  --section SchedulerStats `
  --section WarpStateStats `
  --section MemoryWorkloadAnalysis `
  --section MemoryWorkloadAnalysis_Tables `
  --target-processes all `
  --kernel-name-base demangled `
  --kernel-name "regex:tiled_gemm_v1_kernel" `
  --launch-skip 2 `
  --launch-count 1 `
  --force-overwrite `
  -o "$Out\ncu_tiled_gemm_v1_1024_t16x16x16_detail" `
  $Exe `
  --gemm-m 1024 --gemm-n 1024 --gemm-k 1024 `
  --gemm-tile-m 16 --gemm-tile-n 16 --gemm-tile-k 16 `
  --warmup 1 --iters 1
```

In the GUI, look for:

- `Warp State Statistics`: barrier stalls are shown in the warp stall breakdown.
- `Memory Workload Analysis` / `Memory Workload Analysis Tables`: shared-memory bank conflict metrics are under the L1/TEX or shared-memory tables. If the table is still missing, open the `Raw` tab and search for `bank_conflict` or `shared`.

### Profile `cuda_naive`

Use this as the comparison point for launch configuration, occupancy, warp stalls, and memory behavior.

```powershell
& $Ncu `
  --set full `
  --target-processes all `
  --kernel-name-base demangled `
  --kernel-name "regex:naive_gemm_kernel" `
  --launch-skip 2 `
  --launch-count 1 `
  --force-overwrite `
  -o "$Out\ncu_cuda_naive_gemm_4096" `
  $Exe `
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 `
  --gemm-tile-m 16 --gemm-tile-n 16 --gemm-tile-k 16 `
  --warmup 2 --iters 5
```

### Profile cuBLAS SGEMM

cuBLAS kernel names vary by GPU, CUDA, and cuBLAS version. Start with a broad SGEMM regex.

```powershell
& $Ncu `
  --set basic `
  --target-processes all `
  --kernel-name-base demangled `
  --kernel-name "regex:sgemm|gemm|ampere|sass|cutlass" `
  --launch-skip 2 `
  --launch-count 1 `
  --force-overwrite `
  -o "$Out\ncu_cublas_sgemm_4096" `
  $Exe `
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 `
  --gemm-tile-m 16 --gemm-tile-n 16 --gemm-tile-k 16 `
  --warmup 2 --iters 5
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
  --warmup 2 --iters 5
```

End-to-end benchmark:

```bash
"$Exe" \
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 \
  --gemm-tile-m 16 --gemm-tile-n 16 --gemm-tile-k 16 \
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

Profile `tiled_gemm_v1`:

```bash
"$Ncu" \
  --set full \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name "regex:tiled_gemm_v1_kernel" \
  --launch-skip 2 \
  --launch-count 1 \
  --force-overwrite \
  -o "$Out/ncu_tiled_gemm_v1_4096_t16x16x16" \
  "$Exe" \
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 \
  --gemm-tile-m 16 --gemm-tile-n 16 --gemm-tile-k 16 \
  --warmup 2 --iters 5
```

For a lighter first pass, use a smaller matrix. The benchmark requires positive `--warmup` and `--iters` values, so use `--warmup 1 --iters 1` and skip the first two matching launches: correctness and warmup.

```bash
"$Ncu" \
  --set basic \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name "regex:tiled_gemm_v1_kernel" \
  --launch-skip 2 \
  --launch-count 1 \
  --force-overwrite \
  -o "$Out/ncu_tiled_gemm_v1_1024_t16x16x16_basic" \
  "$Exe" \
  --gemm-m 1024 --gemm-n 1024 --gemm-k 1024 \
  --gemm-tile-m 16 --gemm-tile-n 16 --gemm-tile-k 16 \
  --warmup 1 --iters 1
```

`--set basic` only collects a small default group. It usually shows `SpeedOfLight`, `LaunchStats`, `Occupancy`, and `WorkloadDistribution`, but it does not necessarily collect `WarpStateStats` or detailed shared-memory metrics. To inspect barrier stalls and shared-memory bank behavior, run a focused pass:

```bash
"$Ncu" \
  --section LaunchStats \
  --section Occupancy \
  --section SchedulerStats \
  --section WarpStateStats \
  --section MemoryWorkloadAnalysis \
  --section MemoryWorkloadAnalysis_Tables \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name "regex:tiled_gemm_v1_kernel" \
  --launch-skip 2 \
  --launch-count 1 \
  --force-overwrite \
  -o "$Out/ncu_tiled_gemm_v1_1024_t16x16x16_detail" \
  "$Exe" \
  --gemm-m 1024 --gemm-n 1024 --gemm-k 1024 \
  --gemm-tile-m 16 --gemm-tile-n 16 --gemm-tile-k 16 \
  --warmup 1 --iters 1
```

In the GUI, look for:

- `Warp State Statistics`: barrier stalls are shown in the warp stall breakdown.
- `Memory Workload Analysis` / `Memory Workload Analysis Tables`: shared-memory bank conflict metrics are under the L1/TEX or shared-memory tables. If the table is still missing, open the `Raw` tab and search for `bank_conflict` or `shared`.

Profile `cuda_naive`:

```bash
"$Ncu" \
  --set full \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name "regex:naive_gemm_kernel" \
  --launch-skip 2 \
  --launch-count 1 \
  --force-overwrite \
  -o "$Out/ncu_cuda_naive_gemm_4096" \
  "$Exe" \
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 \
  --gemm-tile-m 16 --gemm-tile-n 16 --gemm-tile-k 16 \
  --warmup 2 --iters 5
```

Profile cuBLAS SGEMM with a broad kernel regex:

```bash
"$Ncu" \
  --set basic \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name "regex:sgemm|gemm|ampere|sass|cutlass" \
  --launch-skip 2 \
  --launch-count 1 \
  --force-overwrite \
  -o "$Out/ncu_cublas_sgemm_4096" \
  "$Exe" \
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 \
  --gemm-tile-m 16 --gemm-tile-n 16 --gemm-tile-k 16 \
  --warmup 2 --iters 5
```

View an Nsight Compute report in terminal:

```bash
"$Ncu" --import "$Out/ncu_tiled_gemm_v1_4096_t16x16x16.ncu-rep" --page details
```

Open the Nsight Compute GUI if the GUI tool is installed:

```bash
ncu-ui "$Out/ncu_tiled_gemm_v1_4096_t16x16x16.ncu-rep"
```

On some Linux systems, hardware performance counters are restricted. If `ncu` reports a permission error, run the profiling command with `sudo` or configure NVIDIA performance counter access for the machine.

### Linux Tile Sweep

```bash
for Tile in 16,16,16 16,16,32 16,32,32 32,16,32 32,32,32; do
  IFS=',' read -r TileM TileN TileK <<< "$Tile"
  "$Exe" \
    --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 \
    --gemm-tile-m "$TileM" --gemm-tile-n "$TileN" --gemm-tile-k "$TileK" \
    --warmup 2 --iters 5
done
```

## Reading The Results

Use this order when analyzing reports:

1. `nsys`: confirm kernel-only vs e2e cost and identify hot kernels.
2. `ncu` launch statistics: compare threads/block, registers/thread, shared memory/block.
3. `ncu` occupancy: check whether active warps are too low.
4. `ncu` warp state: look for high `barrier` stalls in `tiled_gemm_v1`.
5. `ncu` memory workload: look for shared memory bank conflicts and inefficient global loads.

For the current `tiled_gemm_v1`, a common slow pattern is high barrier cost plus low scheduler utilization. That means shared-memory tiling is adding synchronization and shared-memory traffic, but the kernel is not yet reusing data enough in registers.
