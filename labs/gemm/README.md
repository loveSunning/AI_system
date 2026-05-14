# GEMM 深入

对应 `W05-W08`。

这里作为 `naive GEMM -> tiled GEMM -> register blocking -> Tensor Core demo -> autotune` 的主战场。

## 当前入口

- `gemm_lab.hpp`：GEMM lab 的公共接口，包含 `GemmLabBackend`、`PreparedGemmLabRunner` 和 e2e helper。
- `gemm_lab.cu`：通用实验框架，负责输入检查、H2D/D2H、kernel dispatch、CUDA event timing 和 NVTX 标注。
- `gemm_lab_kernels.hpp`：lab 内部 launcher 声明，连接公共 runner 和具体算法文件。
- `tiled_gemm_v1.cu`：只放 tiled GEMM v1 的算法实验代码，当前保留 `tiled_gemm_v1_kernel` 接口和 TODO。
- `tiled_gemm_v1_lab`：最小实验入口，负责准备输入、调用 lab 库、结果校验和 CTest 覆盖。

## 扩展方式

新增 GEMM 算子时，推荐按下面的顺序扩展：

1. 在 `GemmLabBackend` 中增加一个后端枚举，例如 `TiledGemmV2` 或 `ManualTensorCoreV1`。
2. 新增独立算法文件，例如 `tiled_gemm_v2.cu` 或 `manual_tensor_core_v1.cu`，只写该算法自己的 `__global__` kernel 和 launcher。
3. 在 `gemm_lab_kernels.hpp` 声明新的 launcher。
4. 在 `gemm_lab.cu` 的 dispatch 中接入新后端。
5. 在 `labs/perf_engineering` 添加 e2e 和 kernel-only 对比项。

## 目录边界

- `src/` / `include/`：稳定核心基础设施和已经沉淀的基础 kernel。
- `include/ai_system/cuda`：core 和 labs 共享的 CUDA 底层工具，例如 device buffer、memcpy、event timing 和错误检查。
- `labs/gemm`：GEMM 专项实验代码，只隔离算法变化，不重复底层工具。
- `labs/perf_engineering`：横向 benchmark harness，链接 `ai_system_gemm_lab` 做对比，不拥有 GEMM 实验实现。

`tiled_gemm_v1_kernel` 当前仍未实现。实现 kernel 后，把 `tiled_gemm_v1.cu` 里的 `kTiledGemmV1KernelImplemented` 改为 `true`，`perf_engineering_lab` 和 `gemm_tiled_v1_lab` 会自动开始校验它。
