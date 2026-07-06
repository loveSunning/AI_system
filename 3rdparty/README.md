# Third Party Dependencies

## CUTLASS

- Path: `3rdparty/cutlass`
- Version: NVIDIA CUTLASS `v4.5.2`
- Source: https://github.com/NVIDIA/cutlass.git

This checkout is intentionally ignored by the parent repository. The local CMake
default `AI_SYSTEM_CUTLASS_ROOT` points here, while scripts still allow overriding
the path with `CUTLASS_ROOT`, `-CutlassRoot`, or `--cutlass-root`.
