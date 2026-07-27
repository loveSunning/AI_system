# CuTe Examples

Current examples:

- `cute_layout_mapping.cu`: host-side CuTe `Layout` mapping smoke test for `(M,K)`, `(N,K)`, `(BM,BK,stage)`, and hierarchical `(3,(2,3)):(3,(12,1))` offsets.
- `cute_layout_algebra_demo.cu`: host-side CuTe layout algebra smoke test for `coalesce`, `composition`, `complement`, `logical_divide`, `zipped_divide`, `logical_product`, `blocked_product`, and `raked_product`.
- `cute_tensor_tile_demo.cu`: host-side GEMM-shaped CuTe `Tensor`, `local_tile`, shared-memory layout, MMA-sized register fragment, and `local_partition` mapping smoke test for `M=N=K=2048`, CTA `128x128x32`.
- `cute_copy_g2s_naive.cu`: real CUDA G2S demo comparing scalar `local_partition`, scalar `TiledCopy`, and 128-bit vectorized `TiledCopy`; prints the complete selected-thread TV mapping and validates aligned and ragged tiles.
- `cute_copy_g2s_cpasync.cu`: 128-bit `SM80_CP_ASYNC_CACHEALWAYS/CACHEGLOBAL` G2S demo with `fence`, `wait<0>`, CTA synchronization, and predicated edge-tile validation.
- `cute_copy_s2r.cu`: shared-to-register `TiledCopy` demo that writes every per-thread fragment and logical coordinate back for host verification.
- `cute_smem_swizzle_demo.cu`: compares plain, padded, and `Swizzle<5,0,5>` shared layouts, prints physical offsets/banks, and runs a real column-read bank-conflict microbenchmark.

Planned examples:

- `cute_mma_atom_demo.cu`
- `cute_tiled_mma_demo.cu`
- `cute_hgemm_tn_baseline.cu`
