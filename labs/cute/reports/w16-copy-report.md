# W16 CuTe TiledCopy Report

## Environment

```text
Date: 2026-07-27
OS: Windows 10
GPU: NVIDIA GeForce RTX 5060
Compute capability: 12.0
Driver: 580.88
CUDA Toolkit: 13.0 (nvcc 13.0.88)
CMake: 4.1.1
CUTLASS: local 4.5.2 checkout
CMake preset: windows-vs2022-cuda-release
CUDA architecture: sm_120
Build configuration: Release
```

## Delivered Targets

```text
cute_copy_g2s_naive
cute_copy_g2s_cpasync
cute_copy_s2r
cute_smem_swizzle_demo
```

All four targets compiled and ran successfully on the environment above.

## Correctness

### Global -> Shared

```text
Tile: 128x32 half
Threads: 128
Elements: 4096
Elements/thread: 32
128-bit instructions/thread: 4

TV coverage: 4096/4096
Duplicates: 0
Missing: 0

scalar local_partition: PASS
scalar TiledCopy: PASS
128-bit TiledCopy: PASS
```

Ragged case:

```text
Logical problem: M=2053, K=2051
Edge tile valid elements: 15
Zero-filled elements: 4081

ragged scalar copy_if: PASS
ragged vector copy_if: PASS
```

### cp.async

```text
cp.async CACHEALWAYS: PASS
cp.async CACHEGLOBAL: PASS
predicated cp.async: PASS
```

The demonstrated synchronization sequence is:

```text
copy -> cp_async_fence -> cp_async_wait<0> -> __syncthreads
```

### Shared -> Register

```text
Fragment coordinates covered: 4096/4096
Duplicates: 0
All fragment values: PASS
```

For selected thread 45:

```text
logical coordinates:
(11,8..15), (43,8..15), (75,8..15), (107,8..15)

shared offsets:
360..367, 1384..1391, 2408..2415, 3432..3439
```

### Shared-memory Layouts

The bank-conflict test uses one warp reading logical coordinates `(lane,column)` from a
`32x32 float` tile.

| Layout | `cosize` | Active banks | Maximum lanes per bank | Estimated conflict |
| --- | ---: | ---: | ---: | ---: |
| Plain `(32,32):(32,1)` | 1024 | 1 | 32 | 32-way |
| Padded `(32,32):(33,1)` | 1055 | 32 | 1 | 1-way |
| `Swizzle<5,0,5> o plain` | 1024 | 32 | 1 | 1-way |

All three layouts produced identical logical values.

## Microbenchmark Snapshot

The following numbers are one local run. They are educational one-CTA measurements, not
production bandwidth claims; kernel launch overhead dominates the copy tests.

### G2S round trip

| Implementation | Time | Reported round-trip bandwidth |
| --- | ---: | ---: |
| Scalar `local_partition` | 5.801 us | 2.825 GB/s |
| Scalar `TiledCopy` | 5.582 us | 2.935 GB/s |
| 128-bit `TiledCopy` | 5.455 us | 3.003 GB/s |
| `cp.async CACHEALWAYS` | 5.456 us | 3.003 GB/s |
| `cp.async CACHEGLOBAL` | 5.345 us | 3.065 GB/s |

### Shared-memory column read

Each kernel executes 4096 volatile shared-memory reads per lane.

| Layout | Time | Relative to plain |
| --- | ---: | ---: |
| Plain | 104.237 us | 1.000x |
| Padded | 12.854 us | 8.109x |
| Swizzled | 21.457 us | 4.858x |

These timings agree with the layout-derived bank distribution, but Nsight Compute should be
used before making hardware-counter claims.

## Interpretation

- `TiledCopy` makes the thread/value ownership explicit and verifies full tile coverage.
- A 128-bit `Copy_Atom` reduces each thread from 32 scalar half copies to four vector copies.
- `cp.async` is correct in isolation, but its main benefit requires overlap with S2R and MMA
  in a multi-stage mainloop.
- `partition_S` and `partition_D` preserve logical correspondence while allowing different
  source and destination physical layouts.
- Padding and swizzle both remove the demonstrated bank conflict. Padding increases `cosize`;
  the chosen swizzle does not.
- Predication is applied through an identity-coordinate tensor. Vector tails use padded,
  zero-initialized allocation because the predicate applies at copy-instruction granularity.

## Next Step

W17 should reuse these copy paths while introducing:

```text
MMA_Atom
TiledMMA
ThrMMA
partition_fragment_A/B
make_fragment_C
cute::gemm
```

The first W17 target should be `cute_mma_atom_demo`.
