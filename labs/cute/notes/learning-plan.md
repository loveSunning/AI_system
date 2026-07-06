# CuTe Learning Plan

## W15: Layout, Tensor, and Tiling

Goal:

- Understand `Shape`, `Stride`, `Layout`, `Tensor`, `local_tile`, and `compose`.
- Manually explain coordinate-to-offset mapping.

Deliverables:

- `examples/cute_layout_mapping.cu`
- A note explaining `(M,K)`, `(N,K)`, and `(BM,BK,stage)` offsets.

## W16: TiledCopy

Goal:

- Map hand-written global-to-shared and shared-to-register copies onto CuTe abstractions.
- Understand `Copy_Atom`, `TiledCopy`, `partition_S`, and `partition_D`.

Deliverables:

- `cute_copy_g2s_naive`
- `cute_copy_g2s_cpasync`
- `cute_copy_s2r`
- Shared-memory swizzle / bank-conflict demo

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
