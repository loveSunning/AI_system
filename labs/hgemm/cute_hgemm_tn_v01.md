# CuTe HGEMM TN v0.1

This note explains `cute_hgemm_tn_v01.cu`, its tensor mappings, pipeline, boundary
policy, and the experiments that should be used to study it.

## Position In The HGEMM Lab

The benchmark name is:

```text
hgemm_cute_tn_v01
```

This is the first HGEMM implementation in this lab whose fast path is expressed
with CuTe abstractions:

- `Tensor` and `local_tile` describe the global CTA tiles.
- `TiledCopy` partitions 16-byte global-to-shared `cp.async` copies.
- a swizzled shared-memory `Layout` describes every pipeline stage.
- `TiledMMA`, `ThrMMA`, and register fragments describe the MMA ownership.
- `make_tiled_copy_A/B` and `retile_D` dispatch shared-to-register `ldmatrix`.
- `cute::gemm` dispatches `mma.sync.aligned.m16n8k16`.
- `axpby` writes the register accumulator to row-major C.

`hgemm_mma_stages_block_swizzle_tn_cute` remains useful as the hand-written
inline-PTX comparison. Despite its older name, that kernel spells out
`cp.async`, `ldmatrix`, register arrays, and `mma.sync` manually.

## Operation And Memory Layouts

The lab computes:

```text
C[M,N] = A[M,K] * transpose(B[N,K])
```

All three buffers are physically row-major:

```text
A(m,k) -> m * K + k      CuTe stride (K,1)
B(n,k) -> n * K + k      CuTe stride (K,1)
C(m,n) -> m * N + n      CuTe stride (N,1)
```

CuTe keeps the reduction dimension in mode 1 of both input tensors, so the
logical operation is written as `(M,K) x (N,K) -> (M,N)`. The `TN` name is the
BLAS-facing description of the same storage contract.

## Tile And Thread Configuration

v0.1 uses:

```text
CTA tile:       64 x 64 x 64
threads/CTA:    128 = 4 warps
MMA atom:       SM80 16x8x16 F16F16F16F16 TN
TiledMMA:       2x2 MMA atoms, tiled as 32x32x16
G2S copy atom:  SM80 cp.async, 16 bytes per copy instruction
pipeline:       2, 3, or 4 shared-memory stages
```

The original design candidate was `128x128x32`. v0.1 instead uses the official
CuTe `8x64` swizzle atom and a `64x64x64` CTA. This keeps the shared layout
injective and limits four A+B stages to approximately 64 KiB:

```text
4 * (64*64 + 64*64) * sizeof(half) = 65536 bytes
```

This is a correctness-first learning configuration. NCU results should decide
whether v0.2 changes the CTA tile rather than assuming that a larger tile wins.

## Global To CTA Mapping

The full tensors are:

```text
m_a: (M,K)
m_b: (N,K)
m_c: (M,N)
```

For CTA coordinate `(cta_m,cta_n,_)`, `local_tile` produces:

```text
g_a: (64,64,k_tile)
g_b: (64,64,k_tile)
g_c: (64,64)
```

The projections are important:

```cpp
Step<_1, X, _1>  // keep M and K for A; ignore N
Step<X, _1, _1>  // keep N and K for B; ignore M
Step<_1, _1, X>  // keep M and N for C; ignore K
```

## G2S TiledCopy

The G2S thread and value layouts are:

```text
ThrLayout: (16,8):(8,1)     128 threads, K-major thread numbering
ValLayout: (1,8)            eight adjacent half values per instruction
```

Eight `half` values are 16 bytes. The copy atom therefore dispatches one
`cp.async` instruction for each value group. `partition_S` and `partition_D`
apply the same logical ownership to global and shared tensors:

```text
t_ag_a / t_bg_b: this thread's global source vectors
t_as_a / t_bs_b: the matching shared destination vectors
```

In SASS, the async copy normally appears as `LDGSTS`, not as the PTX spelling
`cp.async`.

## Shared Layout And S2R

Each A/B stage uses a K-major swizzled layout built from:

```text
Swizzle<3,3,3>
Layout<Shape<8,Shape<8,8>>, Stride<8,Stride<1,64>>>
```

The layout preserves 16-byte K-contiguous G2S vectors while permuting shared
addresses to reduce bank conflicts for `ldmatrix`.

`make_tiled_copy_A/B` derives an S2R copy from the MMA layout. For each thread:

```text
t_xs_a / t_xs_b: shared-memory partitions read by ldmatrix
t_xr_a / t_xr_b: the same values retiled into MMA register fragments
t_cr_c:           FP16 register accumulator fragment
```

Expected final instructions include `LDSM` and `MMA`/`HMMA`.

## Multi-stage Pipeline

The pipeline has three concurrent data movements:

```text
global -> shared     cp.async
shared -> registers  ldmatrix
registers -> C frag  mma.sync
```

The prologue submits `Stages-1` global loads. The steady-state loop tracks:

```text
k_tile_next  next global K tile
smem_write   ring-buffer stage receiving cp.async
smem_read    ring-buffer stage consumed by ldmatrix
k_block      one of four 16-wide MMA blocks in a 64-wide CTA K tile
```

`cp_async_fence()` commits a producer group. `cp_async_wait<Stages-2>()`
prevents a shared stage from being consumed before its asynchronous copy is
safe. `__syncthreads()` then makes the stage visible to every thread in the CTA.

Compare stages with the same shape and swizzle setting:

```powershell
foreach($Stages in 2,3,4) {
  & $Exe --kernel hgemm_cute_tn_v01 `
    --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 `
    --stages $Stages --no-swizzle --no-correctness `
    --warmup 5 --iters 20
}
```

More stages increase latency-hiding distance but also consume more shared
memory. The fastest stage count is a measurement, not a constant.

## Block Swizzle

Without block swizzle, the launch grid is:

```text
grid.x = N tiles
grid.y = M tiles
```

With block swizzle, N tiles are split between `grid.x` and `grid.z`. The kernel
recovers the logical N coordinate with:

```text
cta_n = blockIdx.z * gridDim.x + blockIdx.x
```

`--swizzle-stride` is measured in N elements. For a 64-column CTA,
`--swizzle-stride 2048` groups up to 32 N tiles before advancing `grid.z`.
This is a CTA scheduling transform and is different from the shared-memory
swizzle used to avoid bank conflicts.

## Boundary Policy

The fast CuTe path runs on complete `64x64` output tiles when `K` is divisible
by 64. A half-accumulate boundary kernel handles everything else:

- if only M or N has a tail, full tiles use CuTe and the edge kernel writes only
  the uncovered rows or columns;
- if `K % 64 != 0`, v0.1 uses the boundary kernel for the whole result;
- all paths preserve the lab contract of FP16 accumulation and FP16 C.

This makes arbitrary M/N/K correct, but the fallback is intentionally not a
performance target. A future v0.2 can replace it with identity tensors,
predicated G2S copies, K-tail zero fill, and predicated C stores.

Use both tests:

```powershell
& $Exe --kernel hgemm_cute_tn_v01 `
  --gemm-m 256 --gemm-n 256 --gemm-k 256 `
  --stages 3 --swizzle --warmup 1 --iters 1

& $Exe --kernel hgemm_cute_tn_v01 `
  --gemm-m 257 --gemm-n 263 --gemm-k 65 `
  --stages 2 --swizzle --warmup 1 --iters 1
```

## What To Verify In NCU

Start with these questions:

1. Does the source page correlate `cute::copy` with `LDGSTS`?
2. Does S2R contain `LDSM`, and does compute contain `MMA` or `HMMA`?
3. How do shared bytes/CTA and achieved occupancy change for stages 2/3/4?
4. Are shared bank conflicts low with the swizzled layout?
5. Do long-scoreboard and barrier stalls fall as stages increase?
6. Does block swizzle change L2 hit rate for large rectangular matrices?

Runnable NCU, PTX, SASS, Nsight Systems, and Compute Sanitizer commands are in
the main `labs/hgemm/README.md`.
