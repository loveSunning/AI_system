# GEMM 深入

对应 `W05-W08`。这个目录是 GEMM 专项实验区，用来从 naive GEMM 逐步推进到 shared memory tiling、register tiling、手写 Tensor Core 和 autotune。

## 设计边界

- `src/` / `include/ai_system/kernels`：稳定核心 kernel 和通用后端，例如 `cuda_naive`、`cublas_sgemm`。
- `include/ai_system/cuda`：core 和 labs 共享的 CUDA 底层工具，例如 device buffer、memcpy、event timing 和错误检查。
- `labs/gemm`：GEMM 专项实验，只隔离算法变化，不重复底层工具。
- `labs/perf_engineering`：横向性能总览工具，会调用 GEMM lab，但不拥有 GEMM 算法实现。

## 当前文件

- `gemm_lab.hpp`：GEMM lab 的公共接口，包含 `GemmLabBackend`、`PreparedGemmLabRunner`、`GemmLabTileConfig`。
- `gemm_lab.cu`：通用实验框架，负责输入检查、H2D/D2H、kernel dispatch、CUDA event timing 和 NVTX 标注。
- `gemm_lab_kernels.hpp`：lab 内部 launcher 声明，连接公共 runner 和具体算法文件。
- `tiled_gemm_v1.cu`：当前 shared-memory tiled GEMM v1 实现。
- `sgemm_benchmark_lab.cpp`：SGEMM 专项 benchmark 工具，对比 `cuda_naive`、`tiled_gemm_v1`、`cublas_sgemm`，后续继续追加 `tiled_gemm_v2` 等。

## SGEMM Benchmark

构建后可执行文件位于：

```powershell
out\build\windows-vs2022-cuda-release\labs\gemm\Release\sgemm_benchmark_lab.exe
```

默认只跑 `sgemm_kernel_only`，也就是数据已经准备在 GPU 上之后，只测 kernel 本体。这是调 kernel 时最常用的视角。

```powershell
out\build\windows-vs2022-cuda-release\labs\gemm\Release\sgemm_benchmark_lab.exe --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 --warmup 2 --iters 5
```

需要观察端到端成本时，显式打开 e2e：

```powershell
out\build\windows-vs2022-cuda-release\labs\gemm\Release\sgemm_benchmark_lab.exe --gemm-m 4096 --gemm-n 4096 --gemm-k 4096 --include-e2e --warmup 2 --iters 5
```

## Profiling

GEMM 的 Nsight Systems / Nsight Compute 分析命令集中维护在 [GEMM Profiling Commands](../../docs/profiling/gemm/README.md)。建议先用 `nsys` 看 timeline 和 NVTX 区间，再用 `ncu` 分别抓 `cuda_naive`、`tiled_gemm_v1` 和 `cublas_sgemm` 的 kernel 指标。

## Tile 参数

SGEMM benchmark 和 perf engineering lab 共用一套 GEMM lab tile 参数：

```powershell
--gemm-tile-m 16 --gemm-tile-n 16 --gemm-tile-k 16
```

默认是 `16x16x16`。当前每个维度支持 `8`、`16`、`32`。这些值会传给 `GemmLabTileConfig`，`tiled_gemm_v1` 再 dispatch 到对应的模板 kernel 实例，所以 `BLOCK_M`、`BLOCK_N`、`BLOCK_K` 仍然是编译期常量，shared memory 数组尺寸和 `#pragma unroll` 都能保持编译期展开。

非方形 tile 示例：

```powershell
out\build\windows-vs2022-cuda-release\labs\gemm\Release\sgemm_benchmark_lab.exe --gemm-m 1024 --gemm-n 2048 --gemm-k 1024 --gemm-tile-m 16 --gemm-tile-n 32 --gemm-tile-k 8

.\sgemm_benchmark_lab.exe --gemm-m 4096 --gemm-n 4096 --gemm-k 4096  --warmup 2 --iters 5 --gemm-tile-m 32 --gemm-tile-n 32 --gemm-tile-k 16
```

## 输出说明

`Benchmark Results` 包含 `shape` 和 `tileshape`：

- `shape`：矩阵规模，格式是 `MxNxK`。
- `tileshape`：GEMM lab 算法使用的 tile，格式是 `MxNxK`；没有 tile 概念的实现显示 `none`。
- `sgemm_kernel_only`：只测 kernel 执行时间，适合看算法本体。
- `sgemm_e2e`：包含准备、kernel 调用、拷回等端到端成本，默认关闭，需要 `--include-e2e`。

## 扩展流程

新增 GEMM 算子时，按这个顺序做：

1. 在 `GemmLabBackend` 中增加后端，例如 `TiledGemmV2`。
2. 新增算法文件，例如 `tiled_gemm_v2.cu`，只写该算法自己的 `__global__` kernel 和 launcher。
3. 在 `gemm_lab_kernels.hpp` 声明 internal launcher 和 `is_*_kernel_implemented` 查询。
4. 在 `gemm_lab.cu` 的 dispatch 和 `gemm_lab_backend_available` 中接入新后端。
5. 在 `sgemm_benchmark_lab.cpp` 中添加 e2e 和 kernel-only 对比项。
6. 如果需要横向总览，再同步接入 `labs/perf_engineering/perf_engineering_lab.cpp`。
