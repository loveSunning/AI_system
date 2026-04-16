# 性能工程基础

对应表格中的 `W01-W04`。

当前仓库已经把第一阶段落成了一个可编译的实验入口：`perf_engineering_lab`，包含：

- `vector add`
- `reduction`
- `naive GEMM`
- `cuBLAS SGEMM / HGEMM / Tensor Core GEMM` 对比

当前默认尺寸是：

- `vector-size=1048576`
- `reduction-size=1048576`
- `gemm-m=n=k=1024`

建议按周推进：

- `W01`：先把 benchmark、正确性校验、CLI 跑通。
- `W02`：用 Nsight Systems / Nsight Compute 抓一次热点。
- `W03`：把 `naive GEMM` 升级为 `tiled GEMM v1/v2`。
- `W04`：补齐 `softmax`、`warp reduction`、阶段总结。
