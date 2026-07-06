# CUTLASS Learning Plan

## W19: Quickstart and Profiler Smoke Test

Goal:

- Build the official CUTLASS tree.
- Run `cutlass_profiler`.
- Read one `device::Gemm` or `GemmUniversal` launch.

Deliverables:

- `cutlass_header_probe`
- `cutlass_profiler` smoke result
- Report explaining one GEMM launch.

## W20: Efficient GEMM and GEMM API 3.x

Goal:

- Understand threadblock / warp / instruction hierarchy.
- Map CuTe concepts onto CUTLASS mainloop and epilogue.

Deliverables:

- GEMM hierarchy diagram.
- Annotated config for one CUTLASS GEMM.

## W21: Profiler Sweep

Goal:

- Scan tile shape, stages, warp shape, instruction shape, alignment, and split-K.
- Build a shape-to-config selection method.

Deliverables:

- `cutlass_profiler_sweep.ps1`
- `cutlass_profiler_sweep.sh`
- `out/cutlass/profiler_sweep.csv`
- Sweep report with at least six shapes.

## W22: Fused Epilogue

Goal:

- Implement one explainable fused epilogue variant: `bias+relu` or `bias+silu`.
- Compare with separate epilogue kernels.

Deliverables:

- CUTLASS GEMM + fused epilogue example.
- Correctness and benchmark report.
