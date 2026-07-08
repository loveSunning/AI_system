# CuTe Examples

Current examples:

- `cute_layout_mapping.cu`: host-side CuTe `Layout` mapping smoke test for `(M,K)`, `(N,K)`, `(BM,BK,stage)`, and hierarchical `(3,(2,3)):(3,(12,1))` offsets.
- `cute_layout_algebra_demo.cu`: host-side CuTe layout algebra smoke test for `coalesce`, `composition`, `complement`, `logical_divide`, `zipped_divide`, `logical_product`, `blocked_product`, and `raked_product`.
- `cute_tensor_tile_demo.cu`: host-side GEMM-shaped CuTe `Tensor`, `local_tile`, shared-memory layout, MMA-sized register fragment, and `local_partition` mapping smoke test for `M=N=K=2048`, CTA `128x128x32`.

Planned examples:

- `cute_copy_g2s_naive.cu`
- `cute_copy_g2s_cpasync.cu`
- `cute_copy_s2r.cu`
- `cute_mma_atom_demo.cu`
- `cute_tiled_mma_demo.cu`
- `cute_hgemm_tn_baseline.cu`
