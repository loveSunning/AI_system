# CUTLASS Examples

Current examples:

- `cutlass_header_probe.cu`: verifies CUTLASS headers, CUDA runtime, and visible GPU compute capabilities.
- `cutlass_2x_gemm.cu`: FP16 Tensor Core GEMM written with the classic CUTLASS 2.x Device API.
- `cutlass_3x_gemm.cu`: the same math written with the CUTLASS 3.x Kernel/Collective API and CuTe atoms/layouts.
- `gemm_lab_common.hpp`: shared CLI, padded allocation, deterministic initialization, timing, and full-result verification.

The two GEMMs deliberately share the same inputs, output type, logical shape,
and verification path so their API structures can be compared without changing
the numerical problem.
