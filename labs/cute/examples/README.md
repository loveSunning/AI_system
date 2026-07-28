# CuTe Examples

Current examples:

- `cute_layout_mapping.cu`: host-side CuTe `Layout` mapping smoke test for `(M,K)`, `(N,K)`, `(BM,BK,stage)`, and hierarchical `(3,(2,3)):(3,(12,1))` offsets.
- `cute_layout_algebra_demo.cu`: host-side CuTe layout algebra smoke test for `coalesce`, `composition`, `complement`, `logical_divide`, `zipped_divide`, `logical_product`, `blocked_product`, and `raked_product`.
- `cute_tensor_tile_demo.cu`: host-side GEMM-shaped CuTe `Tensor`, `local_tile`, shared-memory layout, MMA-sized register fragment, and `local_partition` mapping smoke test for `M=N=K=2048`, CTA `128x128x32`.
- `cute_copy_g2s_naive.cu`: real CUDA G2S demo comparing scalar `local_partition`, scalar `TiledCopy`, and 128-bit vectorized `TiledCopy`; prints the complete selected-thread TV mapping and validates aligned and ragged tiles.
- `cute_copy_g2s_cpasync.cu`: 128-bit `SM80_CP_ASYNC_CACHEALWAYS/CACHEGLOBAL` G2S demo with `fence`, `wait<0>`, CTA synchronization, and predicated edge-tile validation.
- `cute_copy_s2r.cu`: shared-to-register `TiledCopy` demo that writes every per-thread fragment and logical coordinate back for host verification.
- `cute_smem_swizzle_demo.cu`: compares plain, padded, and `Swizzle<5,0,5>` shared layouts, prints physical offsets/banks, and runs a real column-read bank-conflict microbenchmark.
- `cute_gemm_sm80_demo.cu`: Ampere MMA HGEMM with `cp.async`, layout-specific `ldmatrix` copy atoms, three shared-memory stages, and a register pipeline.
- `cute_gemm_sm70_demo.cu`: Volta `m8n8k4` MMA HGEMM with vectorized G2R/R2S copies and two-stage shared/register buffering; requires a compute-capability 7.x build and GPU.
- `cute_gemm_v2_fma_demo.cu`: non-MMA half-input GEMM using scalar FP32 FMA, vectorized G2R/R2S copies, and two-stage shared/register buffering.

The three GEMM demos use the same positional command line:

```text
<executable> [M] [N] [K] [nt|tn|both] [iterations] [warmups]
```

Their default problem is `4096x4096x4096`, both layouts, 20 measured iterations,
and 5 warmups. All three compare correctness and kernel-only performance with
the same FP32-accumulating cuBLAS HGEMM baseline. The teaching kernels require
whole CTA tiles, so M and N must be multiples of 128; K must be a multiple of
64, 32, or 8 for the SM80, SM70, and V2 demos respectively.

`A`, `B`, and `C` are represented as CuTe logical tensors `(M,K)`, `(N,K)`, and
`(M,N)`. `NT` uses strides `(1,M)` and `(1,N)` for A and B. `TN` uses strides
`(K,1)` for both. The logical product is unchanged between layouts:
`C(m,n) = sum_k A(m,k) * B(n,k)`.

| Demo | CTA tile | G2S path | S2R path | Compute | Pipeline |
|---|---:|---|---|---|---|
| `cute_gemm_sm80_demo` | `128x128x64` | 128-bit `cp.async` | `LDSM_N` for TN, `LDSM_T` for NT | `m16n8k16` MMA | 3-stage SMEM + register fragments |
| `cute_gemm_sm70_demo` | `128x128x32` | 128-bit G2R, then R2S | ordinary vector/register copy | `m8n8k4` MMA | 2-stage SMEM + G2R/S2R registers |
| `cute_gemm_v2_fma_demo` | `128x128x8` | 64-bit G2R, then R2S | half-to-float register copy | scalar FP32 FMA | 2-stage SMEM + G2R/S2R registers |

On Windows:

```powershell
cmake --preset windows-vs2022-cuda-release
cmake --build out/build/windows-vs2022-cuda-release --config Release --target cute_gemm_sm80_demo cute_gemm_sm70_demo cute_gemm_v2_fma_demo
.\out\build\windows-vs2022-cuda-release\labs\cute\Release\cute_gemm_sm80_demo.exe 1024 1024 1024 both 20 5
.\out\build\windows-vs2022-cuda-release\labs\cute\Release\cute_gemm_v2_fma_demo.exe 1024 1024 1024 both 20 5
```

The SM70 executable intentionally skips on non-7.x devices. CUDA 13 no longer
offers an `sm_70` target; its real kernel body can still be compile-checked with
`sm_75`, while a Volta performance run needs a toolkit/GPU combination that can
build and execute compute capability 7.0 code.

Planned examples:

- `cute_mma_atom_demo.cu`
- `cute_tiled_mma_demo.cu`
- `cute_hgemm_tn_baseline.cu`
