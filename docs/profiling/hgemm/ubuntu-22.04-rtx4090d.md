# HGEMM Profiling On Ubuntu 22.04 + RTX 4090 D

This note is the Linux counterpart of `docs/profiling/hgemm/README.md`.
It targets:

- OS: `PRETTY_NAME="Ubuntu 22.04.2 LTS"`
- GPU: `NVIDIA GeForce RTX 4090 D`
- CUDA architecture: `sm_89`
- Repository root: `/workspace/AI_system`

Run commands from the repository root:

```bash
cd /workspace/AI_system
```

## Environment Check

```bash
cat /etc/os-release | grep PRETTY_NAME
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
nvcc --version
cmake --version
which ncu || true
which cuobjdump || true
which nvdisasm || true
```

Expected GPU line:

```text
NVIDIA GeForce RTX 4090 D, 8.9
```

## Build

Normal CUDA Release build for RTX 4090 D:

```bash
cmake --preset linux-make-cuda-release-4090d
cmake --build --preset linux-make-cuda-release-4090d --target hgemm_benchmark_lab -j"$(nproc)"
```

Build a separate `-lineinfo` configuration for Nsight Compute source/PTX/SASS correlation:

```bash
cmake --preset linux-make-cuda-release-4090d-lineinfo
cmake --build --preset linux-make-cuda-release-4090d-lineinfo --target hgemm_benchmark_lab -j"$(nproc)"
```

## Common Paths

```bash
export Repo=/workspace/AI_system
export Build=$Repo/out/build/linux-make-cuda-release-4090d
export BuildLineInfo=$Repo/out/build/linux-make-cuda-release-4090d-lineinfo
export Exe=$Build/labs/hgemm/hgemm_benchmark_lab
export ExeLineInfo=$BuildLineInfo/labs/hgemm/hgemm_benchmark_lab
export HgemmLibLineInfo=$BuildLineInfo/labs/hgemm/libai_system_hgemm_lab.a
export Out=$Repo/out/reports/hgemm/ubuntu-22.04-rtx4090d
export SassOut=$Repo/out/sass/hgemm/ubuntu-22.04-rtx4090d
export Ncu=${Ncu:-ncu}
export Cuobjdump=${Cuobjdump:-cuobjdump}
export Nvdisasm=${Nvdisasm:-nvdisasm}

mkdir -p "$Out" "$SassOut"
```

## Baseline Benchmark

The default shape is `M=N=K=4096`:

```bash
"$Exe" --warmup 2 --iters 5
```

List all compiled launcher names and Nsight Compute regex strings:

```bash
"$Exe" --list-kernels
```

Run one kernel by name:

```bash
"$Exe" \
  --kernel hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_swizzle \
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 \
  --warmup 2 --iters 5
```

Run a boundary-shape correctness pass before trusting a new implementation:

```bash
"$Exe" \
  --kernel all \
  --gemm-m 257 --gemm-n 263 --gemm-k 65 \
  --warmup 1 --iters 1
```

Correctness uses the same precision policy as the Windows build:

- CUDA-core, inline-MMA, WMMA, and cuBLAS HGEMM kernels use FP16 accumulation and write FP16 C.
- Tensor Core kernels compare against the cuBLAS Tensor Core half-accumulate reference.
- TN kernels compare against the cuBLAS TN half-accumulate reference.

For profiling, use `--no-correctness --warmup 1 --iters 1` to keep the launch stream simple.
With `-s 1 -c 1`, Nsight Compute skips the warmup launch and collects the measured launch.

## Nsight Compute Sections

This section set is a stable first pass for HGEMM:

- `SpeedOfLight`
- `LaunchStats`
- `Occupancy`
- `WarpStateStats`
- `SchedulerStats`
- `MemoryWorkloadAnalysis`
- `ComputeWorkloadAnalysis`
- `InstructionStats`

## Generate One NCU Report Per Kernel

This Bash loop emits one Nsight Compute report per HGEMM launcher.

```bash
declare -a Kernels=(
  "hgemm_naive_f16:hgemm_naive_f16_kernel"
  "hgemm_sliced_k_f16:hgemm_sliced_k_f16_kernel"
  "hgemm_t_8x8_sliced_k_f16x4:hgemm_t_8x8_sliced_k_f16x4_kernel"
  "hgemm_t_8x8_sliced_k_f16x4_pack:hgemm_t_8x8_sliced_k_f16x4_pack_kernel"
  "hgemm_t_8x8_sliced_k_f16x4_bcf:hgemm_t_8x8_sliced_k_f16x4_bcf_kernel"
  "hgemm_t_8x8_sliced_k_f16x4_pack_bcf:hgemm_t_8x8_sliced_k_f16x4_pack_bcf_kernel"
  "hgemm_t_8x8_sliced_k_f16x8_pack_bcf:hgemm_t_8x8_sliced_k_f16x8_pack_bcf_kernel"
  "hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf:hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf_kernel"
  "hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf:hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_kernel"
  "hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async:hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async_kernel"
  "hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf:hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_kernel"
  "hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_async:hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_async_kernel"
  "hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf:hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_kernel"
  "hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_async:hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_async_kernel"
  "hgemm_cublas_tensor_op_nn:gemm|hgemm|tensor"
  "hgemm_cublas_tensor_op_tn:gemm|hgemm|tensor"
  "hgemm_wmma_m16n16k16_naive:hgemm_wmma_m16n16k16_naive_kernel"
  "hgemm_wmma_m16n16k16_mma4x2:hgemm_wmma_m16n16k16_mma4x2_kernel"
  "hgemm_wmma_m16n16k16_mma4x2_warp2x4:hgemm_wmma_m16n16k16_mma4x2_warp2x4_kernel"
  "hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async:hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_kernel"
  "hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async:hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async_kernel"
  "hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages:hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_kernel"
  "hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem:hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem_kernel"
  "hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem:hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem_kernel"
  "hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem:hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem_kernel"
  "hgemm_mma_m16n8k16_naive:hgemm_mma_m16n8k16_naive_kernel"
  "hgemm_mma_m16n8k16_mma2x4_warp4x4:hgemm_mma_m16n8k16_mma2x4_warp4x4_kernel"
  "hgemm_mma_m16n8k16_mma2x4_warp4x4_stages:hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_kernel"
  "hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem:hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_kernel"
  "hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem:hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_kernel"
  "hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4:hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4_kernel"
  "hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr:hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr_kernel"
  "hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn:hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_kernel"
  "hgemm_mma_stages_block_swizzle_tn_cute:hgemm_mma_stages_block_swizzle_tn_cute_kernel"
  "hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_swizzle:hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_swizzle_kernel"
  "hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_swizzle:hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_kernel"
  "hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x2:hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x2_kernel"
  "hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x4:hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x4_kernel"
)

for Entry in "${Kernels[@]}"; do
  IFS=: read -r KernelName KernelRegex <<< "$Entry"
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
    -k "regex:${KernelRegex}" \
    -s 1 \
    -c 1 \
    -f \
    -o "$Out/ncu_${KernelName}_4096_sections" \
    "$Exe" \
    --kernel "$KernelName" \
    --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 \
    --no-correctness \
    --warmup 1 --iters 1
done
```

## Single-Kernel NCU Template

Use this when iterating on one kernel:

```bash
export KernelName=hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_swizzle
export KernelRegex=hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_swizzle_kernel

"$Ncu" \
  --set detailed \
  --import-source yes \
  --source-folders "$Repo/labs/hgemm,$Repo/include" \
  --target-processes all \
  --kernel-name-base demangled \
  -k "regex:${KernelRegex}" \
  -s 1 \
  -c 1 \
  -f \
  -o "$Out/ncu_${KernelName}_4096_set_detail" \
  "$ExeLineInfo" \
  --kernel "$KernelName" \
  --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 \
  --no-correctness \
  --warmup 1 --iters 1
```

Open the report in the Nsight Compute UI:

```bash
"$Ncu" --open-in-ui -i "$Out/ncu_${KernelName}_4096_set_detail.ncu-rep"
```

Print the source-correlated views in the terminal:

```bash
"$Ncu" -i "$Out/ncu_${KernelName}_4096_set_detail.ncu-rep" --page source --print-source cuda,sass
"$Ncu" -i "$Out/ncu_${KernelName}_4096_set_detail.ncu-rep" --page source --print-source ptx
"$Ncu" -i "$Out/ncu_${KernelName}_4096_set_detail.ncu-rep" --page source --print-source sass
```

Use `cuda,sass` when mapping instructions back to `labs/hgemm/*.cu`.
Use `ptx` when checking the PTX embedded in the cubin. The `sass` view is the final machine code.

## Focused Kernel Groups

Use this group to compare the best current Tensor Core paths on RTX 4090 D:

```bash
declare -a FocusKernels=(
  "hgemm_cublas_tensor_op_nn"
  "hgemm_cublas_tensor_op_tn"
  "hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem"
  "hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem"
  "hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_swizzle"
  "hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x2"
  "hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_tn_swizzle_x4"
  "hgemm_mma_stages_block_swizzle_tn_cute"
)

for Kernel in "${FocusKernels[@]}"; do
  "$Exe" \
    --kernel "$Kernel" \
    --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 \
    --no-correctness \
    --warmup 5 --iters 20
done
```

Run the same group with correctness enabled on an uneven shape before profiling a fresh edit:

```bash
for Kernel in "${FocusKernels[@]}"; do
  "$Exe" \
    --kernel "$Kernel" \
    --gemm-m 257 --gemm-n 263 --gemm-k 65 \
    --warmup 1 --iters 1
done
```

## Static Disassembly With cuobjdump

Dump ELF metadata, resource usage, PTX, and SASS from the line-info HGEMM library:

```bash
"$Cuobjdump" --list-elf "$HgemmLibLineInfo" > "$SassOut/cuobjdump_hgemm_elf_list.txt"
"$Cuobjdump" --dump-elf-symbols "$HgemmLibLineInfo" > "$SassOut/cuobjdump_hgemm_elf_symbols.txt"
"$Cuobjdump" --dump-resource-usage "$HgemmLibLineInfo" > "$SassOut/cuobjdump_hgemm_resource_usage.txt"
"$Cuobjdump" --dump-ptx "$HgemmLibLineInfo" > "$SassOut/cuobjdump_hgemm.ptx"
"$Cuobjdump" --dump-sass --sort-functions --gpu-architecture sm_89 "$HgemmLibLineInfo" > "$SassOut/cuobjdump_hgemm_sm89.sass"
```

Check for Tensor Core, async-copy, and memory instructions:

```bash
grep -E "LDGSTS|MMA|HMMA|LDSM|LDG|STS|BAR" "$SassOut/cuobjdump_hgemm_sm89.sass" | head -200
```

`cp.async` usually appears as `LDGSTS` in SASS.
The inline MMA paths should contain `MMA`/`HMMA` and `LDSM` instructions.

To dump one function only, first find the mangled symbol:

```bash
grep "hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_swizzle_kernel" \
  "$SassOut/cuobjdump_hgemm_elf_symbols.txt" | head -1
```

Then pass that symbol to `cuobjdump --function`:

```bash
export MangledSymbol='<paste-mangled-symbol-here>'
"$Cuobjdump" \
  --dump-sass \
  --gpu-architecture sm_89 \
  --function "$MangledSymbol" \
  "$HgemmLibLineInfo" > "$SassOut/cuobjdump_${KernelName}_sm89.sass"
```

## Static Disassembly With nvdisasm

Extract cubins from the HGEMM static library:

```bash
(
  cd "$SassOut"
  "$Cuobjdump" --extract-elf all "$HgemmLibLineInfo"
)
```

Disassemble every extracted cubin with line information:

```bash
for Cubin in "$SassOut"/*.cubin; do
  [ -e "$Cubin" ] || continue
  Base=$(basename "$Cubin" .cubin)
  "$Nvdisasm" \
    --print-code \
    --separate-functions \
    --print-line-info-ptx \
    --print-instruction-encoding \
    "$Cubin" > "$SassOut/nvdisasm_${Base}.sass"
done
```

If line annotations are missing, rebuild with the `linux-make-cuda-release-4090d-lineinfo` preset and re-run extraction.

## Direct Cubin Build With nvcc

When you want a standalone cubin for one translation unit, compile it directly.
This is useful for fast SASS inspection without walking the static library.

```bash
export Cubin=$SassOut/nvcc_hgemm_mma_basic_sm89.cubin

nvcc \
  -std=c++20 \
  --expt-relaxed-constexpr \
  -O3 \
  -lineinfo \
  -arch=sm_89 \
  -I"$Repo/labs/hgemm" \
  -I"$Repo/include" \
  -I"$BuildLineInfo/generated/include" \
  -cubin "$Repo/labs/hgemm/hgemm_mma_basic.cu" \
  -o "$Cubin"
```

Disassemble that cubin:

```bash
"$Nvdisasm" \
  --print-code \
  --separate-functions \
  --print-line-info-ptx \
  --print-instruction-encoding \
  "$Cubin" > "$SassOut/nvdisasm_hgemm_mma_basic_sm89.sass"
```

For the CuTe-style TN stage kernel, compile its standalone translation unit:

```bash
export Cubin=$SassOut/nvcc_hgemm_mma_stage_tn_cute_sm89.cubin

nvcc \
  -std=c++20 \
  --expt-relaxed-constexpr \
  -O3 \
  -lineinfo \
  -arch=sm_89 \
  -I"$Repo/labs/hgemm" \
  -I"$Repo/include" \
  -I"$BuildLineInfo/generated/include" \
  -cubin "$Repo/labs/hgemm/hgemm_mma_stage_tn_cute.cu" \
  -o "$Cubin"
```

## Reading Results

Start with:

1. `LaunchStats`: verify block size, register count, and dynamic shared memory.
2. `SpeedOfLight`: check whether the kernel is compute, memory, or latency limited.
3. `WarpStateStats` and `SchedulerStats`: identify stall reasons.
4. `MemoryWorkloadAnalysis`: compare global load efficiency and shared-memory behavior.
5. Source/SASS page: verify `LDGSTS`, `LDSM`, and `MMA/HMMA` appear where expected.

For RTX 4090 D, the most useful comparison set is:

- cuBLAS NN/TN half-accumulate reference.
- Best WMMA staged/dsmem kernels.
- Best inline-MMA swizzle kernels.
- The CuTe-style TN stage kernel, which is currently a correctness-first learning implementation.

Use the comparison to decide whether the bottleneck is shared-memory layout, pipeline staging, occupancy, or register pressure before changing code.
