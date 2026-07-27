# CuTe Learning Plan

## W15: Layout, Tensor, and Tiling

Goal:

- Understand `Shape`, `Stride`, `Layout`, layout algebra, `Tensor`, `local_tile`, and `compose`.
- Manually explain coordinate-to-offset mapping.

Deliverables:

- `examples/cute_layout_mapping.cu`
- `examples/cute_layout_algebra_demo.cu`
- `examples/cute_tensor_tile_demo.cu`
- `notes/tensor-local-tile-partition.md`
- A note explaining `(M,K)`, `(N,K)`, `(BM,BK,stage)`, layout algebra, global tensor, shared tensor, register fragment, `local_tile`, and `local_partition` mappings.

## W16: TiledCopy

Goal:

- Map hand-written global-to-shared and shared-to-register copies onto CuTe abstractions.
- Understand `Copy_Atom`, `TiledCopy`, `partition_S`, and `partition_D`.

Deliverables:

- `examples/cute_copy_g2s_naive.cu`: scalar `local_partition`, scalar `TiledCopy`, 128-bit `TiledCopy`, TV coverage, and ragged `copy_if`.
- `examples/cute_copy_g2s_cpasync.cu`: `CACHEALWAYS`, `CACHEGLOBAL`, `fence`, `wait`, CTA synchronization, and ragged predication.
- `examples/cute_copy_s2r.cu`: shared-memory partitions and per-thread register fragments.
- `examples/cute_smem_swizzle_demo.cu`: plain, padded, and swizzled shared layouts with bank mapping.
- `notes/tiled-copy.md`
- `reports/w16-copy-report.md`

Completion criteria:

- Explain `(thread_id,value_id) -> tile coordinate`.
- Distinguish `Copy_Atom`, `TiledCopy`, and `ThrCopy`.
- Explain `partition_S`, `partition_D`, vector width, alignment, and coalescing.
- Explain `cp_async_fence`, `cp_async_wait`, and `__syncthreads`.
- Compute shared-memory bank IDs and explain padding versus swizzle.
- Validate both aligned `2048x2048` input and ragged `2053x2051` input.

## W17: MMA

Goal:

- Understand `MMA_Atom`, `TiledMMA`, `ThrMMA`, fragments, and `cute::gemm`.
- Connect CuTe MMA to earlier WMMA / `mma.sync` experience.

Deliverables:

- `cute_mma_atom_demo`
- `cute_tiled_mma_demo`
- `cute_hgemm_tn_baseline`

## W18: CuTe HGEMM v0.1

Goal:

- Build a complete CuTe HGEMM path: global -> shared -> register -> MMA -> C.
- Add multi-stage pipeline, block swizzle, and boundary handling.

Deliverables:

- `cute_hgemm_tn_pipeline`
- NCU benchmark
- Report comparing hand-written MMA / WMMA / Triton / cuBLAS
